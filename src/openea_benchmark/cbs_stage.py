"""Automated frozen-core valence CBS layer after the electronic-EA stage (Patch 39).

Input is the single-basis electronic-EA record (selected neutral/anion jobs
and their resolved def2-TZVPPD equilibrium geometries).  For each selected
state, and for every other resolved candidate of the same charge within
``competitor_window_ev`` of it, frozen-core CCSD(T) single points are run at
that state's own equilibrium bond length in

* aug-cc-pV{Q,5}Z (aug-cc-pV(X+d)Z for Al-Ar, as recommended for second-row
  p-block atoms), and
* d-aug at the highest cardinal (PySCF's d-aug set where available, else the
  standard even-tempered construction: per angular momentum one more diffuse
  primitive a1**2/a2 from the two most diffuse ones),

each seeded from the state's own high-level reference nearest to the
geometry (projected into the new basis) inside the same symmetry block,
keeping the seeded reference (no instability following, so the basis series
stays on one reference branch; an instability is recorded).

The existing component-resolved model (``attachment.component_resolved_cbs``)
extrapolates SCF, CCSD correlation and (T) separately, adds the separate
d-aug-minus-aug diffuse correction, and fails closed (NEED_MORE_EVIDENCE) when
the model sensitivity exceeds its target.  Competitors are re-ranked at CBS:
if any competitor's CBS energy is not above the selected state's, the layer is
UNRESOLVED.  No experimental data is used; nothing here is a production EA
(core-valence, scalar relativity, SOC, post-CCSD(T), ZPE still missing).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .adaptive.stage3_execution import (
    SEED_CONTINUOUS_BRANCH, Stage3ExecutionRequest, Stage3ExecutionSettings, Stage3PointResult,
)
from .attachment.component_resolved_cbs import (
    PRIMARY_EXPONENTS, extrapolate_species, ComponentPair, resolve_cbs,
)
from .electronic_ea_stage import HARTREE_TO_EV, _save_json, cached_point_runner
from .point_pool import run_parallel

CARDINAL_LETTER = {2: "d", 3: "t", 4: "q", 5: "5", 6: "6"}
TIGHT_D_ELEMENTS = frozenset({"Al", "Si", "P", "S", "Cl", "Ar"})


@dataclass(frozen=True)
class CBSStagePolicy:
    """Explicit pilot policy, not a validated production threshold set."""
    low_cardinal: int = 4
    high_cardinal: int = 5
    diffuse_correction: bool = True
    competitor_window_ev: float = 0.5
    max_memory_mb: int = 48000
    model_spread_target_ev: float = 0.005  # component_resolved_cbs default


def family_label(cardinal: int, *, daug: bool = False) -> str:
    """Series label used by component_resolved_cbs (element mapping kept separately)."""
    return f"{'d-' if daug else ''}aug-cc-pv{CARDINAL_LETTER[cardinal]}z"


def _element_basis_name(element: str, cardinal: int) -> str:
    x = CARDINAL_LETTER[cardinal]
    return f"aug-cc-pv({x}+d)z" if element in TIGHT_D_ELEMENTS else f"aug-cc-pv{x}z"


def _even_tempered_extra_diffuse(shells: list) -> list:
    """One more diffuse primitive per angular momentum (d-aug construction)."""
    # A primitive counts when some contracted function consists of it alone
    # (PySCF stores uncontracted polarization sets as identity contractions).
    by_l: dict[int, list[float]] = {}
    for shell in shells:
        l, prims = int(shell[0]), shell[1:]
        for column in range(1, len(prims[0])):
            nonzero = [p[0] for p in prims if abs(float(p[column])) > 0.0]
            if len(nonzero) == 1:
                by_l.setdefault(l, []).append(float(nonzero[0]))
    extra = []
    for l, exps in sorted(by_l.items()):
        if len(exps) < 2:
            continue
        a1, a2 = sorted(exps)[:2]
        extra.append([l, [a1 * a1 / a2, 1.0]])
    return list(shells) + extra


def element_basis(element: str, cardinal: int, *, daug: bool = False) -> tuple[Any, str]:
    """(PySCF basis spec, provenance) for one element."""
    from pyscf import gto
    name = _element_basis_name(element, cardinal)
    if not daug:
        return name, name
    try:
        gto.basis.load("d-" + name, element)
        return "d-" + name, "d-" + name
    except Exception:
        shells = gto.basis.load(name, element)
        return _even_tempered_extra_diffuse(shells), f"d-{name}(EVEN_TEMPERED_CONSTRUCTED)"


def _job_point_records(job_id: str, point_records: Mapping[str, Mapping[str, Any]],
                       refinement_dir: Path) -> list[dict]:
    recs = [dict(r) for r in point_records.values() if r.get("job_id") == job_id]
    if refinement_dir.is_dir():
        for path in sorted(refinement_dir.glob("*.json")):
            res = json.loads(path.read_text())["result"]
            if res.get("job_id") == job_id:
                recs.append(res)
    return [r for r in recs if r.get("status") == "COMPLETED" and r.get("high_level_checkpoint_path")]


def _states_to_compute(ea: Mapping[str, Any], policy: CBSStagePolicy) -> list[dict]:
    states = []
    for label, sel_key in (("neutral", "selected_neutral_job"), ("anion", "selected_anion_job")):
        sel_id = ea[sel_key]
        cands = {c["job_id"]: c for c in ea["candidates"][label]}
        sel = cands[sel_id]
        states.append({"role": label, "job_id": sel_id, "r": sel["r_central_angstrom"], "selected": True})
        for c in cands.values():
            if c["job_id"] == sel_id or c.get("equilibrium_status") != "RESOLVED":
                continue
            if sel.get("branch_of") and c.get("branch_of") == sel["branch_of"]:
                continue  # other reference branch of the same state, not a competitor
            gap = (c["energy_central_hartree"] - sel["energy_central_hartree"]) * HARTREE_TO_EV
            if gap <= policy.competitor_window_ev:
                states.append({"role": label, "job_id": c["job_id"], "r": c["r_central_angstrom"],
                               "selected": False, "gap_tz_ev": gap})
    return states


def _run_point(runner, request, settings):
    return runner(request, settings)


def run_cbs_stage(*, electronic_ea: Mapping[str, Any], requests: Sequence[Stage3ExecutionRequest],
                  point_records: Mapping[str, Mapping[str, Any]], refinement_dir: Path,
                  stage3_settings: Stage3ExecutionSettings, output_dir: Path,
                  policy: CBSStagePolicy = CBSStagePolicy(),
                  point_backend: Callable[..., Stage3PointResult] | None = None) -> dict:
    if policy.low_cardinal != 4 or policy.high_cardinal != 5:
        raise ValueError("component_resolved_cbs exponents are defined for the Q/5 pair only")
    output_dir = Path(output_dir)
    record: dict[str, Any] = {
        "stage": "FROZEN_CORE_VALENCE_CBS", "policy": asdict(policy), "is_production_ea": False,
        "includes_core_valence": False, "includes_scalar_relativity": False, "includes_soc": False,
        "includes_post_ccsd_t": False, "includes_zpe": False, "validation_metadata_used_in_computation": False,
    }
    if electronic_ea.get("status") != "ELECTRONIC_EA_EVALUATED":
        record.update(status="NOT_ATTEMPTED", reason="single-basis electronic EA not evaluated")
        _save_json(output_dir / "cbs.json", record)
        return record

    settings = replace(stage3_settings, frozen_core=True, checkpoint_project=True,
                       max_memory_mb=policy.max_memory_mb, artifact_dir=str(output_dir / "checkpoints"))
    signature = sha256(json.dumps(asdict(policy), sort_keys=True).encode()).hexdigest()
    runner = cached_point_runner(output_dir / "points", signature, point_backend)
    template = {}
    for req in requests:
        template.setdefault(req.job_id, req)

    series = [(policy.low_cardinal, False), (policy.high_cardinal, False)]
    if policy.diffuse_correction:
        series.append((policy.high_cardinal, True))
    states = _states_to_compute(electronic_ea, policy)
    computed = []
    jobs: list[tuple[dict, str, dict, Stage3ExecutionRequest]] = []
    for state in states:
        src = template[state["job_id"]]
        recs = _job_point_records(state["job_id"], point_records, Path(refinement_dir))
        if not recs:
            state.update(status="NO_SEED_REFERENCE")
            computed.append(state)
            continue
        seed = min(recs, key=lambda r: abs(float(r["r_angstrom"]) - state["r"]))
        state["seed_r_angstrom"] = seed["r_angstrom"]
        state["points"] = {}
        for cardinal, daug in (series if state["selected"] else series[:2]):
            label = family_label(cardinal, daug=daug)
            mapping, prov = {}, {}
            for el in src.atoms:
                mapping[el], prov[el] = element_basis(el, cardinal, daug=daug)
            req = replace(src, request_id=f"{src.job_id}__CBS__{label}", job_id=src.job_id,
                          r_angstrom=float(state["r"]), basis=label, basis_by_element=mapping,
                          source_checkpoint_path=str(seed["high_level_checkpoint_path"]),
                          source_link_status="CBS_BASIS_CONTINUATION_FROM_HIGH_LEVEL_REFERENCE",
                          reference_branch=SEED_CONTINUOUS_BRANCH)
            print(f"[CBS] {src.system} {state['job_id']} {label} R={state['r']:.5f}", flush=True)
            jobs.append((state, label, prov, req))
        computed.append(state)

    # Every (state, basis) point is independent: concurrent on the point pool if configured.
    results = run_parallel(_run_point, [(runner, req, settings) for _, _, _, req in jobs])
    for (state, label, prov, _), result in zip(jobs, results):
        res = result.to_dict()
        state["points"][label] = {
            "status": res["status"], "basis_by_element": prov,
            "scf_energy_hartree": res["scf_energy_hartree"],
            "ccsd_correlation_hartree": res["ccsd_correlation_hartree"],
            "triples_correction_hartree": res["triples_correction_hartree"],
            "ccsd_t_total_hartree": res["ccsd_t_total_hartree"],
            "reference_warnings": res["reference_warnings"], "error_message": res.get("error_message")}
    record["states"] = computed

    def ok(state, label):
        p = (state.get("points") or {}).get(label)
        return p is not None and p["status"] == "COMPLETED"

    low, high = family_label(policy.low_cardinal), family_label(policy.high_cardinal)
    daug = family_label(policy.high_cardinal, daug=True)
    selected = {s["role"]: s for s in computed if s["selected"]}
    needed = [low, high] + ([daug] if policy.diffuse_correction else [])
    missing = [f"{role}:{label}" for role, s in selected.items() for label in needed if not ok(s, label)]
    if missing:
        record.update(status="UNRESOLVED", reason="missing CBS points: " + ", ".join(missing))
        _save_json(output_dir / "cbs.json", record)
        return record

    def total_cbs(state) -> float:
        p4, p5 = state["points"][low], state["points"][high]
        pair = ComponentPair(role=state["role"], basis_qz=low, basis_5z=high,
                             scf_qz=p4["scf_energy_hartree"], scf_5z=p5["scf_energy_hartree"],
                             ccsd_correlation_qz=p4["ccsd_correlation_hartree"],
                             ccsd_correlation_5z=p5["ccsd_correlation_hartree"],
                             triples_qz=p4["triples_correction_hartree"], triples_5z=p5["triples_correction_hartree"])
        return extrapolate_species(pair, model_name="PRIMARY", exponents=PRIMARY_EXPONENTS).total_cbs_hartree

    # Competitor re-ranking at CBS (fail closed when order is lost or unknown).
    ranking, order_ok = [], True
    for role, sel in selected.items():
        e_sel = total_cbs(sel)
        for s in computed:
            if s["role"] != role or s["selected"]:
                continue
            if not (ok(s, low) and ok(s, high)):
                order_ok = False
                ranking.append({"role": role, "job_id": s["job_id"], "status": "CBS_POINTS_MISSING"})
                continue
            gap = (total_cbs(s) - e_sel) * HARTREE_TO_EV
            order_ok &= gap > 0.0
            ranking.append({"role": role, "job_id": s["job_id"], "gap_tz_ev": s["gap_tz_ev"], "gap_cbs_ev": gap})
    record["competitor_ranking_at_cbs"] = ranking

    points = [{"role": role, "basis": label, "reusable": True, **s["points"][label]}
              for role, s in selected.items() for label in (low, high)]
    # Without the d-aug points the diffuse correction is zero by construction
    # (resolve_cbs forms it as EA[d-aug] - EA[aug] at the high cardinal).
    diffuse_label = daug if policy.diffuse_correction else high
    daug_ea = (selected["neutral"]["points"][diffuse_label]["ccsd_t_total_hartree"]
               - selected["anion"]["points"][diffuse_label]["ccsd_t_total_hartree"]) * HARTREE_TO_EV
    cbs = resolve_cbs(points=points, daug_fixed_ea_ev=daug_ea, full_pec_aug5_ea_ev=None,
                      diffuse_residual_estimate_ev=None,
                      model_spread_target_ev=policy.model_spread_target_ev).to_dict()
    record.update(
        status="UNRESOLVED_STATE_ORDER_AT_CBS" if not order_ok else cbs["status"],
        cbs=cbs, ea_cbs_ev=cbs["ea_cbs_plus_diffuse_primary_ev"],
        ea_cbs_lower_ev=cbs["lower_intermediate_ev"], ea_cbs_upper_ev=cbs["upper_intermediate_ev"],
        selected_neutral_job=selected["neutral"]["job_id"], selected_anion_job=selected["anion"]["job_id"],
        verdict="NOT_ISSUED_NO_CV_SR_ZPE_BINDING_LAYERS",
        fixed_basis_ea_ev={label: (selected["neutral"]["points"][label]["ccsd_t_total_hartree"]
                                   - selected["anion"]["points"][label]["ccsd_t_total_hartree"]) * HARTREE_TO_EV
                           for label in needed},
    )
    _save_json(output_dir / "cbs.json", record)
    return record
