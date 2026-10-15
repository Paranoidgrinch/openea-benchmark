"""Automated electronic-EA stage after Stage-3 acquisition (first EA number).

Connects existing, separately validated layers that were so far only chained by
OH-specific scripts:

1. every Stage-3 job whose high-level identity/continuity review cleared with a
   bracketed sampled minimum is refined by the canonical Stage-3 loop until
   its bracket target is met (new points are cached and resumable);
2. each converged loop is bridged to an attachment record and its equilibrium
   energy interval is resolved;
3. an automated *high-level* D04 competition review per charge picks the
   lowest state only when it is separated from every other candidate;
4. atomic fragments are scanned over charge 0/-1 and all spin sectors up to a
   bound at the same CCSD(T)/basis level (no atomic term is assumed);
5. the canonical ``evaluate_stage3_electronic_ea`` decides BOUND / UNBOUND /
   UNRESOLVED for the selected neutral/anion pair.

The output is an ELECTRONIC EA in one finite basis: no CBS, core-valence,
relativity, SOC or ZPE.  It is never labelled a production EA and never issues a
BOUND/UNBOUND verdict: its interval only carries equilibrium-resolution
uncertainty, so a near-threshold sign at this level is not decisive (wave-0 NO:
EA_exp ~ +0.03 eV, single-basis electronic value negative).  No
experimental value is read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .adaptive.stage3_execution import (
    SCF_RESCUE_POLICY, SCF_SOSCF_RESCUE_WARNING, SEED_CONTINUOUS_BRANCH, PointExecutionStatus,
    Stage3ExecutionRequest, Stage3ExecutionSettings, Stage3PointResult,
    run_stage3_point,
)
from .adaptive.stage3_axial_symmetry import AxialSymmetryPolicy
from .adaptive.stage3_identity import resolve_high_level_identity_and_pec
from .adaptive.stage3_loop import Stage3LoopRetrySettings, run_stage3_refinement_loop
from .adaptive.stage3_refinement import Stage3RefinementSettings
from .attachment.asymptote import DissociationChannel
from .attachment.equilibrium import EquilibriumResolverSettings, EquilibriumStatus, resolve_stage3_equilibrium
from .attachment.fragment_execution import AtomicFragmentRequest, FragmentExecutionStatus, run_atomic_fragment
from .attachment.stage3_bridge import Stage3AttachmentBridgeStatus, bridge_stage3_loop_to_attachment
from .attachment.workflow import evaluate_stage3_electronic_ea
from .reference_branches import branch_parent

HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class ElectronicEAStagePolicy:
    """Explicit pilot policy; none of these values is a validated production threshold."""
    refinement_extension_step_angstrom: float = 0.025
    refinement_target_bracket_width_angstrom: float = 0.021
    refinement_min_separation_angstrom: float = 1.0e-7
    refinement_energy_tie_hartree: float = 1.0e-8
    max_refinement_rounds: int = 12
    equilibrium_max_side_points: int = 2
    equilibrium_min_models: int = 2
    equilibrium_min_curvature: float = 1.0e-3
    equilibrium_max_geometry_spread_angstrom: float = 0.01
    equilibrium_max_energy_spread_hartree: float = 5.0e-4
    equilibrium_point_tolerance_hartree: float = 1.0e-8
    # A competitor whose equilibrium interval (or, if unresolved, lowest
    # sampled CCSD(T) point) lies this far above the lowest candidate's upper
    # bound is recorded as excluded-by-margin instead of blocking D04.
    competitor_exclusion_margin_ev: float = 1.0
    max_atomic_2s: int = 6
    dissociation_safety_margin_hartree: float = 0.0
    target_half_width_ev: float | None = None
    # Single-reference collapse (Patch 43): a CCSD(T) point whose largest T2
    # amplitude reaches 1 (the reference determinant no longer dominates) or
    # whose (T) correction is this large a fraction of the CCSD correlation
    # energy is not an energy of the labelled state.  Healthy states of the
    # reference set stay below 0.35 / 0.14; collapsed C2 1Sigmag+ is 1.5 / 0.35.
    collapse_max_t2_amplitude: float = 1.0
    collapse_triples_fraction: float = 0.25
    # A collapsed candidate whose reference determinant (SCF) lies this far
    # above the lowest candidate's is not a ground-state competitor; closer
    # ones block D04 until a multireference treatment exists.
    collapsed_reference_scf_margin_ev: float = 3.0

    def refinement(self) -> Stage3RefinementSettings:
        return Stage3RefinementSettings(
            extension_step_angstrom=self.refinement_extension_step_angstrom,
            target_bracket_width_angstrom=self.refinement_target_bracket_width_angstrom,
            minimum_new_point_separation_angstrom=self.refinement_min_separation_angstrom,
            energy_tie_tolerance_hartree=self.refinement_energy_tie_hartree,
        )

    def equilibrium(self) -> EquilibriumResolverSettings:
        return EquilibriumResolverSettings(
            max_side_points=self.equilibrium_max_side_points,
            minimum_admissible_models=self.equilibrium_min_models,
            minimum_curvature_hartree_per_angstrom2=self.equilibrium_min_curvature,
            max_model_geometry_spread_angstrom=self.equilibrium_max_geometry_spread_angstrom,
            max_model_energy_spread_hartree=self.equilibrium_max_energy_spread_hartree,
            point_energy_tolerance_hartree=self.equilibrium_point_tolerance_hartree,
        )


def _save_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        json.dump(data, out, sort_keys=True, indent=2, allow_nan=False, default=str)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def _result_from_record(record: Mapping[str, Any]) -> Stage3PointResult:
    return Stage3PointResult(**{**record, "status": PointExecutionStatus(record["status"])})


_RESCUABLE_SCF_STATUSES = frozenset({PointExecutionStatus.SCF_NOT_CONVERGED.value,
                                     PointExecutionStatus.SCF_UNSTABLE.value})


def _rescue_policy_affects(cached: Mapping[str, Any]) -> bool:
    """A cached point whose outcome a newer SCF rescue policy can change."""
    result = cached["result"]
    if result["status"] in _RESCUABLE_SCF_STATUSES:
        return True
    # SOSCF_V1/V2 rescued seed-continuous points onto the other reference branch.
    return (cached["request"].get("reference_branch") == SEED_CONTINUOUS_BRANCH
            and SCF_SOSCF_RESCUE_WARNING in (result.get("reference_warnings") or ()))


class CachedPointRunner:
    """Stage-3 runner that persists every refinement point (resume without recompute).

    A class rather than a closure so it can be shipped to point-pool workers.
    """

    def __init__(self, cache_dir: Path, settings_signature: str,
                 backend: Callable[..., Stage3PointResult] | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings_signature = settings_signature
        self.backend = backend

    def __call__(self, request: Stage3ExecutionRequest, settings: Stage3ExecutionSettings) -> Stage3PointResult:
        fingerprint = sha256(json.dumps({"request": request.to_dict(), "settings": asdict(settings),
                                         "batch": self.settings_signature}, sort_keys=True, default=str)
                             .encode()).hexdigest()
        path = self.cache_dir / (sha256(request.request_id.encode()).hexdigest() + f"__{fingerprint[:16]}.json")
        if path.exists():
            cached = json.loads(path.read_text())
            # SCF failures cached under an older SCF rescue policy are recomputed once.
            if not (cached.get("scf_rescue_policy") != SCF_RESCUE_POLICY
                    and _rescue_policy_affects(cached)):
                return _result_from_record(cached["result"])
        from .point_pool import core_slot
        with core_slot():
            result = run_stage3_point(request, settings=settings, runner=self.backend)
        if result.status is PointExecutionStatus.ERROR:
            # Infrastructure errors (e.g. scratch disk full) are not a property of
            # the point: never cached, so a resumed run recomputes them.
            return result
        _save_json(path, {"request_id": request.request_id, "fingerprint": fingerprint,
                          "scf_rescue_policy": SCF_RESCUE_POLICY,
                          "request": request.to_dict(), "result": result.to_dict()})
        return result


def cached_point_runner(cache_dir: Path, settings_signature: str,
                        backend: Callable[..., Stage3PointResult] | None = None) -> CachedPointRunner:
    return CachedPointRunner(cache_dir, settings_signature, backend)


def _atomic_electrons(symbol: str, charge: int) -> int:
    from pyscf.data.elements import charge as nuclear_charge
    return int(nuclear_charge(symbol)) - int(charge)


def scan_atomic_fragments(*, atoms: Sequence[str], basis: str, settings: Stage3ExecutionSettings,
                          policy: ElectronicEAStagePolicy, cache_dir: Path,
                          runner: Callable[..., Any] | None = None) -> dict:
    """Lowest CCSD(T) state per (element, charge) over all scanned 2S values."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict] = {}
    for element in sorted(set(atoms)):
        for charge in (0, -1):
            n = _atomic_electrons(element, charge)
            scanned = []
            for spin_2s in range(n % 2, min(n, policy.max_atomic_2s) + 1, 2):
                fid = f"{element}_q{charge:+d}_2S{spin_2s}"
                path = cache_dir / f"{fid}__{basis}.json"
                if path.exists():
                    rec = json.loads(path.read_text())
                else:
                    result = run_atomic_fragment(
                        AtomicFragmentRequest(fid, element, charge, spin_2s, basis, state_label=None),
                        settings=settings, runner=runner)
                    rec = {k: (v.value if hasattr(v, "value") else v) for k, v in asdict(result).items()}
                    _save_json(path, rec)
                scanned.append(rec)
            completed = [r for r in scanned if r["status"] == FragmentExecutionStatus.COMPLETED.value
                         and r.get("ccsd_t_total_hartree") is not None]
            best = min(completed, key=lambda r: r["ccsd_t_total_hartree"], default=None)
            table[f"{element}{charge:+d}"] = {
                "element": element, "charge": charge,
                "lowest": None if best is None else {"spin_2s": best["spin_2s"],
                                                     "ccsd_t_total_hartree": best["ccsd_t_total_hartree"]},
                "scanned": [{"spin_2s": r["spin_2s"], "status": r["status"],
                             "ccsd_t_total_hartree": r.get("ccsd_t_total_hartree")} for r in scanned],
                "all_spins_completed": len(completed) == len(scanned),
            }
    return table


def anion_dissociation_channels(atoms: Sequence[str], fragments: Mapping[str, Mapping[str, Any]]):
    a, b = atoms
    channels = []
    for qa, qb in ((-1, 0), (0, -1)):
        fa, fb = fragments.get(f"{a}{qa:+d}"), fragments.get(f"{b}{qb:+d}")
        energy = None
        if fa and fb and fa["lowest"] and fb["lowest"]:
            energy = fa["lowest"]["ccsd_t_total_hartree"] + fb["lowest"]["ccsd_t_total_hartree"]
        channels.append(DissociationChannel(
            channel_id=f"{a}{qa:+d}+{b}{qb:+d}", fragment_a=f"{a}{qa:+d}", fragment_b=f"{b}{qb:+d}",
            asymptotic_energy_hartree=energy, source="AUTOMATED_ATOMIC_SPIN_SCAN_CCSD(T)"))
    return tuple(channels)


# Policy fields that only classify existing points; kept out of the
# refinement-point cache signature so adding them does not recompute points.
_CLASSIFICATION_ONLY_FIELDS = ("collapse_max_t2_amplitude", "collapse_triples_fraction",
                               "collapsed_reference_scf_margin_ev")


def reference_collapse_evidence(record: Mapping[str, Any], policy: "ElectronicEAStagePolicy") -> dict | None:
    """Return evidence if this CCSD(T) point is a single-reference collapse, else None."""
    t2 = record.get("max_abs_t2_amplitude")
    triples, corr = record.get("triples_correction_hartree"), record.get("ccsd_correlation_hartree")
    fraction = abs(triples / corr) if triples is not None and corr else None
    reasons = []
    if t2 is not None and t2 >= policy.collapse_max_t2_amplitude:
        reasons.append("MAX_T2_AMPLITUDE")
    if fraction is not None and fraction >= policy.collapse_triples_fraction:
        reasons.append("TRIPLES_FRACTION")
    if not reasons:
        return None
    return {"request_id": record.get("request_id"), "r_angstrom": record.get("r_angstrom"),
            "max_abs_t2_amplitude": t2, "triples_fraction": fraction, "reasons": reasons}


def _competition(label: str, entries: list[dict], margin_ev: float,
                 collapsed_scf_margin_ev: float = 3.0) -> dict:
    """Automated high-level D04 review for one charge group."""
    resolved = [e for e in entries if e["equilibrium_status"] == "RESOLVED"]
    if not resolved:
        return {"status": "UNRESOLVED", "selected_job_id": None,
                "reason": f"no {label} candidate has a resolved equilibrium interval"}
    lowest = min(resolved, key=lambda e: e["energy_central_hartree"])
    upper = lowest["energy_upper_hartree"]
    blocking, excluded, siblings, multireference_only = [], [], [], []
    for e in entries:
        if e is lowest:
            continue
        floor = e["energy_lower_hartree"] if e["equilibrium_status"] == "RESOLVED" else e["lowest_sampled_hartree"]
        if e["equilibrium_status"] == "SINGLE_REFERENCE_COLLAPSED" or (floor is None and e.get("scf_only_evidence")):
            # Its CCSD(T) energies are not energies of this state (collapse) or
            # do not exist (coupled cluster not converged at any point): no floor.
            # Only the reference determinant energy can separate it.
            ref, low_ref = e.get("scf_lowest_hartree"), lowest.get("scf_lowest_hartree")
            scf_gap_ev = None if ref is None or low_ref is None else (ref - low_ref) * HARTREE_TO_EV
            item = {"job_id": e["job_id"], "scf_gap_ev": scf_gap_ev,
                    "status": "MULTIREFERENCE_METHOD_REQUIRED",
                    "evidence": ("COLLAPSED_CCSD(T)" if e["equilibrium_status"] == "SINGLE_REFERENCE_COLLAPSED"
                                 else "SCF_ONLY_COUPLED_CLUSTER_NOT_CONVERGED")}
            if scf_gap_ev is not None and scf_gap_ev > collapsed_scf_margin_ev:
                multireference_only.append({**item, "basis": "REFERENCE_DETERMINANT_SCF_GAP"})
            else:
                blocking.append(item)
            continue
        if floor is None:
            blocking.append({"job_id": e["job_id"], "reason": "no energy evidence"})
            continue
        gap_ev = (floor - upper) * HARTREE_TO_EV
        if lowest.get("branch_of") and e.get("branch_of") == lowest["branch_of"] and (
                e["equilibrium_status"] == "RESOLVED" or gap_ev > 0.0):
            # Same symmetry-labelled state, other ROHF reference branch: not a
            # competitor; its CCSD(T) offset is the reference sensitivity.
            siblings.append({"job_id": e["job_id"], "equilibrium_status": e["equilibrium_status"],
                             "offset_ev": ((e["energy_central_hartree"] if e["energy_central_hartree"] is not None
                                            else floor) - lowest["energy_central_hartree"]) * HARTREE_TO_EV})
            continue
        if e["equilibrium_status"] == "RESOLVED" and gap_ev > 0.0:
            continue  # cleanly separated interval
        if e["equilibrium_status"] != "RESOLVED" and gap_ev > margin_ev:
            excluded.append({"job_id": e["job_id"], "gap_ev": gap_ev, "basis": "LOWEST_SAMPLED_CCSD(T)_POINT"})
            continue
        blocking.append({"job_id": e["job_id"], "gap_ev": gap_ev, "status": e["equilibrium_status"]})
    return {"status": "CLEARED" if not blocking else "UNRESOLVED",
            "selected_job_id": lowest["job_id"] if not blocking else None,
            "lowest_job_id": lowest["job_id"], "blocking": blocking, "excluded_by_margin": excluded,
            "reference_branch_siblings": siblings, "multireference_only_excluded": multireference_only,
            "reference_sensitivity_ev": max((abs(x["offset_ev"]) for x in siblings), default=0.0),
            "rule": "AUTOMATED_HIGH_LEVEL_D04: lowest resolved equilibrium interval must lie below every "
                    "other candidate's lower bound; unresolved candidates only by explicit margin"}


def single_basis_ea_interval(neutral: Mapping[str, Any], anion: Mapping[str, Any]) -> tuple[float, float, float, str]:
    """[L_N - U_A, U_N - L_A] in eV plus the sign *at this level only* (never a verdict)."""
    lower = (neutral["energy_lower_hartree"] - anion["energy_upper_hartree"]) * HARTREE_TO_EV
    upper = (neutral["energy_upper_hartree"] - anion["energy_lower_hartree"]) * HARTREE_TO_EV
    central = (neutral["energy_central_hartree"] - anion["energy_central_hartree"]) * HARTREE_TO_EV
    sign = "POSITIVE" if lower > 0.0 else ("NEGATIVE" if upper <= 0.0 else "STRADDLES_ZERO")
    return lower, upper, central, sign


def run_electronic_ea_stage(*, system: str, atoms: Sequence[str], requests: Sequence[Stage3ExecutionRequest],
                            point_records: Mapping[str, Mapping[str, Any]], pec_reviews: Mapping[str, Mapping[str, Any]],
                            stage3_settings: Stage3ExecutionSettings, output_dir: Path,
                            policy: ElectronicEAStagePolicy = ElectronicEAStagePolicy(),
                            point_backend: Callable[..., Stage3PointResult] | None = None,
                            fragment_runner: Callable[..., Any] | None = None,
                            identity_thresholds: Any = None, branch_thresholds: Any = None,
                            loops_out: dict | None = None) -> dict:
    """``loops_out`` (optional) receives the refinement loop result per job id, for
    downstream layers that need the identity-cleared curves (nuclear motion)."""
    output_dir = Path(output_dir)
    if identity_thresholds is None or branch_thresholds is None:
        from .workflow import WEEKEND_BRANCH_THRESHOLDS, WEEKEND_IDENTITY_THRESHOLDS
        identity_thresholds = identity_thresholds or WEEKEND_IDENTITY_THRESHOLDS
        branch_thresholds = branch_thresholds or WEEKEND_BRANCH_THRESHOLDS
    signature_fields = {k: v for k, v in asdict(policy).items() if k not in _CLASSIFICATION_ONLY_FIELDS}
    signature = sha256(json.dumps(signature_fields, sort_keys=True).encode()).hexdigest()
    runner = cached_point_runner(output_dir / "refinement_points", signature, point_backend)
    # Same identity semantics as the acquisition review: axial (SO(2)) orbit
    # equivalence of Pi/Delta initializations is admitted, nothing else.
    resolver = partial(resolve_high_level_identity_and_pec, axial_symmetry_policy=AxialSymmetryPolicy())

    by_job: dict[str, list[Stage3ExecutionRequest]] = {}
    for req in requests:
        by_job.setdefault(req.job_id, []).append(req)

    loops: dict[str, Any] = {}
    entries = {"neutral": [], "anion": []}
    work: list[tuple] = []

    def refine(job_id: str, entry: dict, done: list) -> None:
        # One job's refinement error is recorded on that candidate (fail closed in
        # D04) instead of aborting the refinement of every other job.
        try:
            _refine(job_id, entry, done)
        except Exception as exc:  # noqa: BLE001
            entry["equilibrium_status"] = "REFINEMENT_ERROR"
            entry["loop_rationale"] = f"{type(exc).__name__}: {exc}"

    def _refine(job_id: str, entry: dict, done: list) -> None:
        print(f"[ELECTRONIC EA] {system} refine {job_id}", flush=True)
        loop = run_stage3_refinement_loop(
            initial_requests=done, initial_results=[_result_from_record(point_records[r.request_id]) for r in done],
            refinement_settings=policy.refinement(), identity_thresholds=identity_thresholds,
            branch_thresholds=branch_thresholds, max_refinement_rounds=policy.max_refinement_rounds,
            execution_settings=stage3_settings,
            retry_settings=Stage3LoopRetrySettings(max_retries_per_request=1, cycle_multiplier=2.0),
            runner=runner, resolver=resolver)
        loops[job_id] = loop
        entry["loop_rationale"] = loop.rationale
        entry["loop_status"] = loop.status.value
        record = bridge_stage3_loop_to_attachment(loop)
        entry["bridge_status"] = record.status.value
        if record.status is Stage3AttachmentBridgeStatus.READY:
            eq = resolve_stage3_equilibrium(record, settings=policy.equilibrium())
            entry["equilibrium_status"] = eq.status.value
            if eq.status is EquilibriumStatus.RESOLVED:
                entry.update(energy_central_hartree=eq.energy_central_hartree,
                             energy_lower_hartree=eq.energy_interval_hartree.lower_hartree,
                             energy_upper_hartree=eq.energy_interval_hartree.upper_hartree,
                             r_central_angstrom=eq.geometry_central_angstrom)
        else:
            entry["equilibrium_status"] = "BRIDGE_NOT_READY"

    for job_id, job_reqs in sorted(by_job.items()):
        review = pec_reviews.get(job_id, {})
        done = [r for r in job_reqs if (point_records.get(r.request_id) or {}).get("status") == "COMPLETED"]
        sampled = [point_records[r.request_id]["ccsd_t_total_hartree"] for r in done
                   if point_records[r.request_id].get("ccsd_t_total_hartree") is not None]
        entry = {"job_id": job_id, "charge": job_reqs[0].charge, "spin_2s": job_reqs[0].spin_2s,
                 "lowest_sampled_hartree": min(sampled) if sampled else None,
                 "loop_status": None, "bridge_status": None, "equilibrium_status": "NOT_ATTEMPTED",
                 "energy_central_hartree": None, "energy_lower_hartree": None, "energy_upper_hartree": None,
                 "r_central_angstrom": None, "branch_of": branch_parent(job_id),
                 "reference_branch": job_reqs[0].reference_branch}
        scf = [point_records[r.request_id]["scf_energy_hartree"] for r in done
               if point_records[r.request_id].get("scf_energy_hartree") is not None]
        if not done:
            # No CCSD(T) point (e.g. CCSD not converged anywhere): the converged
            # reference determinants are the only energy evidence of this state.
            scf = [rec["scf_energy_hartree"] for rec in (point_records.get(r.request_id) or {} for r in job_reqs)
                   if rec.get("scf_converged") and rec.get("scf_energy_hartree") is not None]
            entry["scf_only_evidence"] = bool(scf)
        entry["scf_lowest_hartree"] = min(scf) if scf else None
        label = "neutral" if entry["charge"] == 0 else "anion"
        entries[label].append(entry)
        collapsed = [c for c in (reference_collapse_evidence(point_records[r.request_id], policy) for r in done) if c]
        if collapsed:
            entry.update(equilibrium_status="SINGLE_REFERENCE_COLLAPSED", lowest_sampled_hartree=None,
                         reference_collapse={"n_points": len(collapsed), "n_sampled": len(done),
                                             "points": collapsed})
            continue
        # An unbracketed high-level curve (minimum at the grid edge, e.g. because
        # the DFT and CCSD(T) minima differ) is extended by the refinement loop
        # rather than left as an unresolved D04 competitor.
        eligible = (review.get("initialization_review") == "CLEARED"
                    and review.get("geometry_continuity_review") == "CLEARED"
                    and (review.get("pec") or {}).get("minimum_scout", {}).get("status")
                    in ("BRACKETED_SINGLE_MINIMUM", "NO_BRACKETED_MINIMUM"))
        if not eligible:
            entry["equilibrium_status"] = "NOT_ELIGIBLE_REVIEW_NOT_CLEARED"
            continue
        work.append((job_id, entry, done))

    # Refinement loops of different jobs are independent: with a point pool they run
    # concurrently (threads here only wait; the calculations run in pool workers).
    from concurrent.futures import ThreadPoolExecutor
    from .point_pool import pool_workers
    with ThreadPoolExecutor(max_workers=max(1, min(len(work), pool_workers()))) as executor:
        for _ in executor.map(lambda item: refine(*item), work):
            pass

    if loops_out is not None:
        loops_out.update(loops)
    competition = {label: _competition(label, items, policy.competitor_exclusion_margin_ev,
                                      policy.collapsed_reference_scf_margin_ev)
                   for label, items in entries.items()}
    basis = requests[0].basis if requests else None
    fragments = scan_atomic_fragments(atoms=atoms, basis=basis, settings=stage3_settings, policy=policy,
                                      cache_dir=output_dir / "fragments", runner=fragment_runner) if requests else {}
    channels = anion_dissociation_channels(atoms, fragments) if fragments else ()

    record: dict[str, Any] = {
        "system": system, "stage": "ELECTRONIC_EA_SINGLE_BASIS", "basis": basis,
        "policy": asdict(policy), "candidates": entries, "high_level_competition": competition,
        "fragments": fragments, "anion_dissociation_channels": [asdict(c) for c in channels],
        "is_electronic_ea_only": True, "includes_zpe": False, "includes_cbs": False,
        "is_production_ea": False, "validation_metadata_used_in_computation": False,
    }
    n_id, a_id = competition["neutral"].get("selected_job_id"), competition["anion"].get("selected_job_id")
    record["verdict"] = "NOT_ISSUED_SINGLE_BASIS_INTERMEDIATE"
    if n_id is None or a_id is None:
        record.update(status="UNRESOLVED", reason="high-level state competition not cleared",
                      ea_lower_ev=None, ea_upper_ev=None, ea_central_ev=None, sign_at_this_level=None)
    else:
        neutral = next(e for e in entries["neutral"] if e["job_id"] == n_id)
        anion = next(e for e in entries["anion"] if e["job_id"] == a_id)
        # EA interval from the two equilibrium intervals.  It carries ONLY the
        # equilibrium-resolution uncertainty of this single basis; it is a
        # number for the next layers, not a bound/unbound verdict.
        lower, upper, central, sign = single_basis_ea_interval(neutral, anion)
        # Reference-branch sensitivity widens the interval on both sides; it is
        # an uncertainty of the single-reference treatment, not a correction.
        widen = (competition["neutral"].get("reference_sensitivity_ev", 0.0)
                 + competition["anion"].get("reference_sensitivity_ev", 0.0))
        if widen:
            lower, upper = lower - widen, upper + widen
            sign = "POSITIVE" if lower > 0.0 else ("NEGATIVE" if upper <= 0.0 else "STRADDLES_ZERO")
        result = evaluate_stage3_electronic_ea(
            neutral_loop_result=loops[n_id], anion_loop_result=loops[a_id],
            equilibrium_settings=policy.equilibrium(), anion_dissociation_channels=channels,
            dissociation_safety_margin_hartree=policy.dissociation_safety_margin_hartree,
            target_half_width_ev=policy.target_half_width_ev,
            neutral_state_id=n_id, anion_state_id=a_id)
        d = result.ea_decision
        record.update(status="ELECTRONIC_EA_EVALUATED", selected_neutral_job=n_id, selected_anion_job=a_id,
                      ea_lower_ev=lower, ea_upper_ev=upper, ea_central_ev=central, sign_at_this_level=sign,
                      interval_scope="EQUILIBRIUM_RESOLUTION_ONLY_NO_BASIS_ZPE_CORRELATION_UNCERTAINTY",
                      reference_sensitivity_ev=widen,
                      single_basis_decision_audit_only=result.status.value,
                      decision_rationale=list(d.rationale), evidence=list(result.evidence),
                      anion_binding=None if result.anion_binding is None else {
                          "status": result.anion_binding.status.value,
                          "evidence": list(result.anion_binding.evidence),
                          "margin_hartree": result.anion_binding.energy_margin_hartree})
    _save_json(output_dir / "electronic_ea.json", record)
    return record
