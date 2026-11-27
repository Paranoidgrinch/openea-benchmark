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
UNRESOLVED.

Further fail-closed rules on the result (Patch 40):

* binding: the CBS interval (widened by the electronic-EA stage's reference-branch
  sensitivity) decides BOUND / UNBOUND / UNDETERMINED; a diffuse correction above
  ``diffuse_continuum_threshold_ev`` means the anion energy follows the basis into
  the continuum, so a negative EA is reported as UNBOUND and a positive one as
  undetermined -- never as a CBS number; a negative interval without that is UNBOUND
  only below -``unbound_decision_margin_ev`` (closer to zero the corrections could
  still bind the anion: UNDETERMINED, the EA_0 interval decides);
* multireference screening: one CCSD point per selected state at its def2-TZVPPD
  equilibrium gives T1, D1 (closed shell) and max |T2|.  Values above the
  literature screening thresholds are reported as a warning, not as a status: on the
  reference set they fire for CN, ClO, NO and AlO, which agree with experiment to
  0.04 eV, and do not single out SiC (+0.14 eV), so they cannot gate a result.

No experimental data is used; nothing here is a production EA
(core-valence, scalar relativity, SOC, post-CCSD(T), ZPE still missing).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .adaptive.stage3_execution import (
    SEED_CONTINUOUS_BRANCH, PointExecutionStatus, Stage3ExecutionRequest, Stage3ExecutionSettings,
    Stage3PointResult,
)
from .attachment.component_resolved_cbs import (
    PAIR_EXPONENTS,
    TQ_MODEL_SPREAD_TARGET_EV,
    extrapolate_species,
    ComponentPair,
    resolve_cbs,
)
from .electronic_ea_stage import HARTREE_TO_EV, _save_json, cached_point_runner, same_resolved_state
from .final_resolver import CBS_NEGATIVE_WITHIN_CORRECTION_MARGIN
from .point_pool import run_parallel

MR_DIAGNOSTIC_LABEL = "MRDIAG_STAGE3_BASIS"
CARDINAL_LETTER = {2: "d", 3: "t", 4: "q", 5: "5", 6: "6"}
TIGHT_D_ELEMENTS = frozenset({"Al", "Si", "P", "S", "Cl", "Ar"})
# Superset-basis variational check (d-aug vs aug at one cardinal): projection noise is ~1e-8 Eh.
SUBSET_BASIS_SCF_TOLERANCE_HARTREE = 1.0e-6
# Request-id suffix of the projected-orbital retry.  V2: complete orbital space of the larger
# basis (``complete_projected_orbitals``); the V1 points (suffix without version) had stayed
# in the smaller basis's span and must not be reused from the point cache.
PROJECTED_RETRY_SUFFIX = "__PROJECTED_SOSCF_V2"
# The CBS model-sensitivity bound is part of the interval half-width; above the 5 meV
# target it widens the interval instead of blocking, up to this cap.
CBS_MODEL_BOUND_CAP_EV = 0.02
# Geometry relaxation at the low CBS cardinal: probes around the single-basis
# equilibrium until the low-cardinal minimum is bracketed.  The first step is small
# (covalent curves); unbracketed minima are followed outward with growing steps
# (van der Waals curves, verify Mg2: def2-TZVPPD 4.40 A, aug-cc-pVQZ ~3.98 A).
GEOMETRY_FIRST_STEP_ANGSTROM = 0.02
GEOMETRY_MAX_STEP_ANGSTROM = 0.5
GEOMETRY_MAX_PROBES = 6
# Below this the curve counts as flat over the probes (CC convergence is 1e-8 Eh).
GEOMETRY_FLAT_TOLERANCE_HARTREE = 1.0e-7


@dataclass(frozen=True)
class CBSStagePolicy:
    """Explicit pilot policy, not a validated production threshold set."""
    low_cardinal: int = 4
    high_cardinal: int = 5
    diffuse_correction: bool = True
    competitor_window_ev: float = 0.5
    max_memory_mb: int = 48000
    model_spread_target_ev: float = 0.005  # component_resolved_cbs default
    # An anion whose energy still moves this much from aug- to d-aug-cc-pV5Z is not
    # converging to a bound state: the basis is discretizing the detachment continuum.
    diffuse_continuum_threshold_ev: float = 0.05
    # A negative CBS interval is decisive (UNBOUND, later layers skipped) only this far below
    # zero; closer to zero the later corrections could still bind the anion, so they run and
    # the EA_0 interval decides.  Bound on their positive sum: Delta ZPE <= ZPE(neutral)
    # <= 0.27 eV (H2, the stiffest diatomic), CV + SR + SO + post-CCSD(T) <= 0.19 eV on
    # verify_v1 (PbO), up to ~0.4 eV first-order SO for 6p atoms (small ZPE there).
    unbound_decision_margin_ev: float = 0.5
    # Multireference screening (CCSD amplitudes at the def2-TZVPPD equilibrium;
    # common literature values, reported as warnings only -- see module docstring).
    t1_closed_shell_threshold: float = 0.02
    t1_open_shell_threshold: float = 0.044
    d1_threshold: float = 0.05
    max_t2_threshold: float = 0.1
    # Diffuse check impossible (d-aug points not obtained): no d-aug correction, and the
    # aug-QZ -> aug-5Z change of the fixed-basis EA (at least this floor) is added to the
    # interval as the unknown diffuse residual.
    missing_diffuse_floor_ev: float = 0.01
    # Relax each selected state's bond length at the low cardinal (extra points there);
    # the relaxation energy corrects the CBS energies taken at the single-basis geometry.
    geometry_relaxation: bool = True


def family_label(cardinal: int, *, daug: bool = False, taug: bool = False) -> str:
    """Series label used by component_resolved_cbs (element mapping kept separately)."""
    return f"{'t-' if taug else 'd-' if daug else ''}aug-cc-pv{CARDINAL_LETTER[cardinal]}z"


# Diffuse convergence (2026-11-27).  The d-aug - aug change Delta_1 at the high cardinal
# was added without a bound.  Each further diffuse shell changes a valence anion's EA by a
# fraction r of the previous one (OH CCSD(T)/QZ at 0.97 A, 2026-11-27: Delta_1 = +0.01146,
# Delta_2 = +0.00010 eV, r = 0.009; t-aug overlap eigenvalue 1.3e-5, SCF stable); with
# r <= 1/3 the remaining tail is at most |Delta_1| r / (1 - r) = 0.5 |Delta_1|, added to
# the interval.  Above DIFFUSE_SECOND_STEP_TRIGGER_EV the next step is computed instead:
# Delta_2 = EA(t-aug) - EA(d-aug) at the low cardinal (even-tempered third diffuse shell),
# added to the correction, its geometric tail r / (1 - r) |Delta_2| (r = |Delta_2 / Delta_1|)
# to the interval; r >= 1 (not converging) adds |Delta_1| + |Delta_2|.
DIFFUSE_SECOND_STEP_TRIGGER_EV = 0.005
DIFFUSE_ASSUMED_RATIO = 1.0 / 3.0


def diffuse_tail_ev(delta1: float, delta2: float | None) -> tuple[float, str]:
    """(bound on the diffuse increments not computed, convergence status)."""
    if delta2 is None:
        r = DIFFUSE_ASSUMED_RATIO
        return abs(delta1) * r / (1.0 - r), "ONE_STEP_ASSUMED_RATIO"
    if abs(delta1) == 0.0:
        return abs(delta2), "SECOND_STEP"
    r = abs(delta2) / abs(delta1)
    if r >= 1.0:
        return abs(delta1) + abs(delta2), "NOT_CONVERGING"
    r_tail = max(r, 0.1)
    return abs(delta2) * r_tail / (1.0 - r_tail), "SECOND_STEP_GEOMETRIC"


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


def _has_cardinal(element: str, cardinal: int) -> bool:
    try:
        element_basis(element, cardinal)
    except Exception:  # noqa: BLE001 - not in any library
        return False
    return True


def element_basis(element: str, cardinal: int, *, daug: bool = False, taug: bool = False) -> tuple[Any, str]:
    """(PySCF basis spec, provenance) for one element.

    PP-route elements (beyond Kr) get the basis *name* aug-cc-pVXZ-PP (d-aug: its
    even-tempered extension), resolved together with its ECP when the molecule is built.
    ``taug``: the d-aug set plus one more even-tempered diffuse shell (all-electron route
    only; ValueError otherwise).
    """
    from pyscf import gto
    from .heavy_elements import ROUTE_PSEUDOPOTENTIAL, ROUTE_X2C, basis_name_for, element_route, resolve_named_basis
    if taug:
        if element_route(element)["route"] in (ROUTE_X2C, ROUTE_PSEUDOPOTENTIAL):
            raise ValueError(f"no t-aug construction for the {element_route(element)['route']} route")
        name = _element_basis_name(element, cardinal)
        try:
            shells = gto.basis.load("d-" + name, element)
            base = "d-" + name
        except Exception:
            shells = _even_tempered_extra_diffuse(gto.basis.load(name, element))
            base = f"d-{name}(EVEN_TEMPERED_CONSTRUCTED)"
        return _even_tempered_extra_diffuse(shells), f"t-{name}(EVEN_TEMPERED_FROM_{base})"
    if element_route(element)["route"] == ROUTE_X2C:
        name = basis_name_for(element, f"{'d-' if daug else ''}aug-cc-pV{CARDINAL_LETTER[cardinal].upper()}Z")
        loaded = resolve_named_basis(name, element)
        if loaded is None:
            raise ValueError(f"no {name} for {element}")
        return name, loaded.provenance
    if element_route(element)["route"] == ROUTE_PSEUDOPOTENTIAL:
        name = f"{'d-' if daug else ''}aug-cc-pV{CARDINAL_LETTER[cardinal].upper()}Z-PP"
        loaded = resolve_named_basis(name, element)
        if loaded is None or loaded.ecp is None:
            raise ValueError(f"no {name} with ECP for {element}")
        return name, loaded.provenance
    name = _element_basis_name(element, cardinal)
    if not daug:
        return name, name
    try:
        gto.basis.load("d-" + name, element)
        return "d-" + name, "d-" + name
    except Exception:
        shells = gto.basis.load(name, element)
        return _even_tempered_extra_diffuse(shells), f"d-{name}(EVEN_TEMPERED_CONSTRUCTED)"


def _second_diffuse_step(selected: Mapping[str, dict], template: Mapping[str, Any], policy: "CBSStagePolicy",
                         runner, settings, low_label: str) -> dict:
    """EA(t-aug) - EA(d-aug) at the low cardinal for the selected states (each state's chain:
    aug (cached) -> d-aug -> t-aug, so every step is checked against its subset basis)."""
    cardinal = policy.low_cardinal
    labels = (low_label, family_label(cardinal, daug=True), family_label(cardinal, taug=True))
    chains, roles = [], []
    for role, state in selected.items():
        src = template[state["job_id"]]
        reqs = []
        for label, daug, taug in zip(labels, (False, True, False), (False, False, True)):
            mapping, prov = {}, {}
            try:
                for el in src.atoms:
                    mapping[el], prov[el] = element_basis(el, cardinal, daug=daug, taug=taug)
            except ValueError as exc:
                return {"status": "NOT_CONSTRUCTIBLE", "reason": str(exc), "delta2_ev": None}
            reqs.append(replace(src, request_id=f"{src.job_id}__CBS__{label}", job_id=src.job_id,
                                r_angstrom=float(state["r"]), basis=label, basis_by_element=mapping,
                                source_checkpoint_path=str(state.get("seed_checkpoint") or src.source_checkpoint_path),
                                source_link_status="CBS_BASIS_CONTINUATION_FROM_HIGH_LEVEL_REFERENCE",
                                reference_branch=SEED_CONTINUOUS_BRANCH))
            if label != low_label:
                print(f"[CBS] {src.system} {state['job_id']} {label} R={state['r']:.5f} (second diffuse step)",
                      flush=True)
        chains.append((runner, reqs, settings))
        roles.append(role)
    outcomes = dict(zip(roles, run_parallel(_run_state_chain, chains)))
    energies, out = {}, {"status": "COMPLETED", "labels": list(labels[1:]), "points": {}}
    for role, outcome in outcomes.items():
        for label, (result, reseed) in zip(labels, outcome):
            res = result.to_dict()
            status = (reseed or {}).get("rejected") or res["status"]
            out["points"][f"{role}:{label}"] = {"status": status, "ccsd_t_total_hartree": res["ccsd_t_total_hartree"],
                                               "reseeded": reseed}
            if status == "COMPLETED":
                energies[(role, label)] = res["ccsd_t_total_hartree"]
    need = [(r, l) for r in roles for l in labels[1:]]
    if any(k not in energies for k in need):
        return {**out, "status": "FAILED", "delta2_ev": None}

    def ea(label):
        return (energies[("neutral", label)] - energies[("anion", label)]) * HARTREE_TO_EV
    return {**out, "delta2_ev": ea(labels[2]) - ea(labels[1])}


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
            if same_resolved_state(c, sel):
                continue  # same state reached by two symmetry blocks (verify_v1 LaO 2Sigma+/2Delta)
            gap = (c["energy_central_hartree"] - sel["energy_central_hartree"]) * HARTREE_TO_EV
            if gap <= policy.competitor_window_ev:
                states.append({"role": label, "job_id": c["job_id"], "r": c["r_central_angstrom"],
                               "selected": False, "gap_tz_ev": gap})
    return states


def _frozen_core_rule_changed(element: str) -> bool:
    from pyscf.data import elements
    from .adaptive.stage3_execution import _valence_core_orbitals
    z = int(elements.charge(element))
    n = _valence_core_orbitals(z)
    return n is not None and n != elements.chemcore_atm[z]


def _mr_diagnostic(res: Mapping[str, Any], policy: CBSStagePolicy) -> dict:
    """Screening values of one CCSD point; exceeded thresholds are listed by name."""
    out = {"status": res["status"], "basis": res["basis"], "r_angstrom": res["r_angstrom"],
           "t1": res.get("t1_diagnostic"), "t1_definition": res.get("t1_diagnostic_definition"),
           "d1": res.get("d1_diagnostic"), "max_abs_t2": res.get("max_abs_t2_amplitude"),
           "triples_fraction": None}
    if res.get("triples_correction_hartree") is not None and res.get("ccsd_correlation_hartree"):
        out["triples_fraction"] = res["triples_correction_hartree"] / res["ccsd_correlation_hartree"]
    t1_limit = policy.t1_closed_shell_threshold if res["spin_2s"] == 0 else policy.t1_open_shell_threshold
    checks = (("T1", out["t1"], t1_limit), ("D1", out["d1"], policy.d1_threshold),
              ("MAX_T2", out["max_abs_t2"], policy.max_t2_threshold))
    out["exceeded"] = [name for name, value, limit in checks if value is not None and value > limit]
    out["available"] = res["status"] == "COMPLETED" and out["t1"] is not None
    return out


def _binding(ea: float, lower: float, upper: float, diffuse_ev: float, policy: CBSStagePolicy) -> tuple[str, str]:
    """Binding at the CBS level.  UNBOUND here is final (the later layers are skipped):
    a continuum-discretizing anion has no localized state for corrections to act on, and a
    localized one must lie below -``unbound_decision_margin_ev``."""
    if abs(diffuse_ev) > policy.diffuse_continuum_threshold_ev:
        if upper < 0.0:
            return "UNBOUND", "NEGATIVE_EA_AND_DIFFUSE_SENSITIVE_ANION_CONTINUUM_DISCRETIZATION"
        return "UNDETERMINED", "DIFFUSE_CORRECTION_ABOVE_CONTINUUM_THRESHOLD"
    if lower > 0.0:
        return "BOUND", "CBS_INTERVAL_POSITIVE"
    if upper < -policy.unbound_decision_margin_ev:
        return "UNBOUND", "CBS_INTERVAL_NEGATIVE"
    if upper < 0.0:
        return "UNDETERMINED", CBS_NEGATIVE_WITHIN_CORRECTION_MARGIN
    return "UNDETERMINED", "CBS_INTERVAL_CONTAINS_ZERO"


def _diffuse_superset(label: str, previous: str) -> bool:
    """``label`` adds one diffuse shell to ``previous`` (aug -> d-aug -> t-aug)."""
    return label == "d-" + previous or (previous.startswith("d-") and label == "t-" + previous[2:])


def _run_state_chain(runner, requests, settings) -> list[tuple[Any, dict | None]]:
    """One state's basis series in order.  A point that fails from the projected TZ
    reference is retried once, seeded from the last completed larger-basis reference of
    the same state (same irrep occupation, no following) and with doubled SCF/CC cycles;
    the retry is recorded, never silent.

    A d-aug point contains the previous (aug) basis of the same cardinal, so its SCF
    energy cannot lie above that one.  When it fails or lies above (DIIS in a nearly
    linearly dependent diffuse basis jumps to another solution), it is recomputed by
    second-order SCF from the aug orbitals projected into the d-aug basis; a result
    still above the aug energy is rejected (``rejected`` in the retry record)."""
    out: list[tuple[Any, dict | None]] = []
    previous = None  # (basis label, completed result)

    def above(result, ref):
        return (ref is not None and result.status is PointExecutionStatus.COMPLETED
                and result.scf_energy_hartree is not None
                and result.scf_energy_hartree > ref.scf_energy_hartree + SUBSET_BASIS_SCF_TOLERANCE_HARTREE)

    for req in requests:
        superset_ref = (previous[1] if previous is not None and _diffuse_superset(str(req.basis), previous[0])
                        and previous[1].high_level_checkpoint_path else None)
        result, reseed = runner(req, settings), None
        if result.status is not PointExecutionStatus.COMPLETED or above(result, superset_ref):
            first = {"first_status": result.status.value, "first_scf_energy_hartree": result.scf_energy_hartree}
            if superset_ref is not None:
                seed = superset_ref.high_level_checkpoint_path
                retry = replace(req, request_id=req.request_id + PROJECTED_RETRY_SUFFIX, source_checkpoint_path=str(seed),
                                source_link_status="CBS_SUBSET_BASIS_ORBITALS_PROJECTED")
                retry_settings = replace(settings, scf_start="PROJECTED_SOSCF",
                                         cc_max_cycle=2 * settings.cc_max_cycle)
            else:
                seed = previous[1].high_level_checkpoint_path if previous is not None else req.source_checkpoint_path
                retry = replace(req, request_id=req.request_id + "__RESEED", source_checkpoint_path=str(seed),
                                source_link_status=("CBS_BASIS_CONTINUATION_FROM_PREVIOUS_CARDINAL"
                                                    if previous is not None else req.source_link_status))
                retry_settings = replace(settings, scf_max_cycle=2 * settings.scf_max_cycle,
                                         cc_max_cycle=2 * settings.cc_max_cycle)
            reseed = {**first, "seed_checkpoint": str(seed), "scf_start": retry_settings.scf_start,
                      "scf_max_cycle": retry_settings.scf_max_cycle}
            result = runner(retry, retry_settings)
            if above(result, superset_ref):
                reseed["rejected"] = "SCF_ABOVE_SUBSET_BASIS"
        if (result.status is PointExecutionStatus.COMPLETED and result.high_level_checkpoint_path
                and not (reseed or {}).get("rejected")):
            previous = (str(req.basis), result)
        out.append((result, reseed))
    return out


def _parabola(points: Sequence[tuple[float, float]]) -> tuple[float, float, float] | None:
    """(curvature a, vertex r, vertex energy) of the parabola through three points."""
    (x0, y0), (x1, y1), (x2, y2) = points
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(d) < 1e-14:
        return None
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / d
    if a <= 0.0:
        return a, float("nan"), float("nan")
    rv = -b / (2 * a)
    c = y0 - a * x0 ** 2 - b * x0
    return a, rv, c - b * b / (4 * a)


def relaxation_from_points(points: Mapping[float, float], r0: float) -> dict:
    """Relaxation energy E(min) - E(r0) of a sampled curve, or the next probe radius.

    Returns ``{"status": "BRACKETED"|"FLAT"|"NEXT", ...}``: BRACKETED gives the vertex of the
    parabola through the lowest point and its two neighbours (``relaxation_hartree`` <= 0,
    ``model_uncertainty_hartree`` = |vertex - lowest sampled|); NEXT gives ``next_r``."""
    rs = sorted(points)
    es = [points[r] for r in rs]
    e0 = points[r0]
    if max(es) - min(es) < GEOMETRY_FLAT_TOLERANCE_HARTREE:
        return {"status": "FLAT", "relaxation_hartree": 0.0, "model_uncertainty_hartree": 0.0, "r_min": r0}
    i = min(range(len(rs)), key=lambda k: es[k])
    if 0 < i < len(rs) - 1:
        fit = _parabola([(rs[i - 1], es[i - 1]), (rs[i], es[i]), (rs[i + 1], es[i + 1])])
        if fit is not None and fit[0] > 0.0 and rs[i - 1] <= fit[1] <= rs[i + 1]:
            ev = min(fit[2], es[i])
            return {"status": "BRACKETED", "relaxation_hartree": ev - e0,
                    "model_uncertainty_hartree": es[i] - ev, "r_min": fit[1]}
    # lowest point at an edge: step outward, by the parabola through the three points at
    # that edge when it curves upward, else by twice the last spacing
    edge = rs[:3] if i == 0 else rs[-3:]
    spacing = abs(edge[1] - edge[0]) if i == 0 else abs(edge[-1] - edge[-2])
    fit = _parabola([(r, points[r]) for r in edge]) if len(edge) == 3 else None
    step = 2.0 * spacing
    if fit is not None and fit[0] > 0.0:
        step = min(max(abs(fit[1] - rs[i]) * 1.2, spacing), 4.0 * spacing)
    step = min(max(step, GEOMETRY_FIRST_STEP_ANGSTROM), GEOMETRY_MAX_STEP_ANGSTROM)
    return {"status": "NEXT", "next_r": rs[i] - step if i == 0 else rs[i] + step,
            "lowest_hartree": es[i], "r_lowest": rs[i]}


def _relaxation_chain(runner, base_request, r0: float, e0: float, settings) -> dict:
    """Probe one state's curve at the low cardinal around ``r0`` (energy ``e0`` known)."""
    points = {r0: e0}
    probes: list[dict] = []
    pending = [r0 - GEOMETRY_FIRST_STEP_ANGSTROM, r0 + GEOMETRY_FIRST_STEP_ANGSTROM]
    verdict: dict = {"status": "NEXT"}
    while pending and len(probes) < GEOMETRY_MAX_PROBES:
        r = round(pending.pop(0), 5)
        if r <= 0.0 or r in points:
            break
        req = replace(base_request, request_id=f"{base_request.request_id}__GEOM_R{r:.5f}", r_angstrom=r)
        result = runner(req, settings)
        probes.append({"r_angstrom": r, "status": result.status.value,
                       "ccsd_t_total_hartree": result.ccsd_t_total_hartree})
        if result.status is not PointExecutionStatus.COMPLETED or result.ccsd_t_total_hartree is None:
            verdict = {"status": "PROBE_FAILED"}
            break
        points[r] = float(result.ccsd_t_total_hartree)
        if pending:
            continue
        verdict = relaxation_from_points(points, r0)
        if verdict["status"] == "NEXT":
            pending.append(verdict["next_r"])
    if verdict["status"] in ("NEXT", "PROBE_FAILED"):
        low = min(points.values())
        # not bracketed: the lowest sampled energy bounds the relaxation from one side only
        verdict = {**verdict, "status": "NOT_BRACKETED" if verdict["status"] == "NEXT" else "PROBE_FAILED",
                   "relaxation_hartree": low - e0, "model_uncertainty_hartree": abs(low - e0),
                   "r_min": min(points, key=points.get)}
    return {**verdict, "r0_angstrom": r0, "probes": probes}


def geometry_relaxation_correction(relax: Mapping[str, Mapping[str, Any]]) -> tuple[float, float]:
    """(EA correction, half-width) in eV from the neutral/anion relaxation records.

    EA = E(neutral) - E(anion): relaxing the neutral lowers the EA, the anion raises it.
    Half-width: the full relaxation of each state (the remaining low-cardinal -> CBS change
    of the bond length is not computed) plus the vertex-model uncertainty."""
    corr = (relax["neutral"]["relaxation_hartree"] - relax["anion"]["relaxation_hartree"]) * HARTREE_TO_EV
    half = sum(abs(r["relaxation_hartree"]) + r["model_uncertainty_hartree"] for r in relax.values()) * HARTREE_TO_EV
    return corr, half


def run_cbs_stage(*, electronic_ea: Mapping[str, Any], requests: Sequence[Stage3ExecutionRequest],
                  point_records: Mapping[str, Mapping[str, Any]], refinement_dir: Path,
                  stage3_settings: Stage3ExecutionSettings, output_dir: Path,
                  policy: CBSStagePolicy = CBSStagePolicy(),
                  point_backend: Callable[..., Stage3PointResult] | None = None) -> dict:
    if (policy.low_cardinal, policy.high_cardinal) not in PAIR_EXPONENTS:
        raise ValueError("component_resolved_cbs exponents are defined for the Q/5 and T/Q pairs only")
    atoms = requests[0].atoms if requests else ()
    if any(not _has_cardinal(el, policy.high_cardinal) for el in dict.fromkeys(atoms)):
        # largest basis QZ for an element (X2C route): T/Q pair, wider spread target
        policy = replace(policy, low_cardinal=3, high_cardinal=4,
                         model_spread_target_ev=max(policy.model_spread_target_ev, TQ_MODEL_SPREAD_TARGET_EV))
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
    # Point cache key: only the fields that existed when points were first cached
    # (Patch 39); result-interpretation thresholds must not invalidate computed points.
    computed_fields = ("low_cardinal", "high_cardinal", "diffuse_correction", "competitor_window_ev",
                       "max_memory_mb", "model_spread_target_ev")
    signature_payload: dict[str, Any] = {k: asdict(policy)[k] for k in computed_fields}
    # Points cached before the valence frozen-core rule (Patch 41) used PySCF's
    # set_frozen(); only where the two differ (e.g. Mg) must they be recomputed.
    atoms = requests[0].atoms if requests else ()
    changed = [el for el in dict.fromkeys(atoms) if _frozen_core_rule_changed(el)]
    if changed:
        signature_payload["frozen_core_rule"] = "VALENCE_NOBLE_GAS_CORE"
    signature = sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    runner = cached_point_runner(output_dir / "points", signature, point_backend)
    template = {}
    for req in requests:
        template.setdefault(req.job_id, req)

    series = [(policy.low_cardinal, False), (policy.high_cardinal, False)]
    if policy.diffuse_correction:
        series.append((policy.high_cardinal, True))
    states = _states_to_compute(electronic_ea, policy)
    computed = []
    diagnostics: dict[int, Any] = {}
    jobs: list[tuple[dict, str, dict, Stage3ExecutionRequest]] = []
    for state in states:
        src = template[state["job_id"]]
        recs = _job_point_records(state["job_id"], point_records, Path(refinement_dir))
        if not recs:
            state.update(status="NO_SEED_REFERENCE")
            computed.append(state)
            continue
        # Two initializations at the same R tie; the request id decides, not the record order.
        seed = min(recs, key=lambda r: (abs(float(r["r_angstrom"]) - state["r"]), str(r.get("request_id"))))
        state["seed_r_angstrom"] = seed["r_angstrom"]
        state["seed_checkpoint"] = str(seed["high_level_checkpoint_path"])
        state["points"] = {}
        if state["selected"]:
            jobs.append((state, MR_DIAGNOSTIC_LABEL, {},
                         replace(src, request_id=f"{src.job_id}__CBS__{MR_DIAGNOSTIC_LABEL}",
                                 r_angstrom=float(state["r"]),
                                 source_checkpoint_path=str(seed["high_level_checkpoint_path"]),
                                 source_link_status="CBS_BASIS_CONTINUATION_FROM_HIGH_LEVEL_REFERENCE",
                                 reference_branch=SEED_CONTINUOUS_BRANCH)))
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

    # States run concurrently on the point pool; the basis series of one state runs in
    # order so a failed point can be re-seeded from the previous basis of that state.
    chains: dict[int, list[int]] = {}
    for index, (state, _, _, _) in enumerate(jobs):
        chains.setdefault(id(state), []).append(index)
    chain_results = run_parallel(_run_state_chain, [(runner, [jobs[i][3] for i in idx], settings)
                                                    for idx in chains.values()])
    results: list = [None] * len(jobs)
    for idx, outcome in zip(chains.values(), chain_results):
        for i, item in zip(idx, outcome):
            results[i] = item
    for (state, label, prov, _), (result, reseed) in zip(jobs, results):
        res = result.to_dict()
        if label == MR_DIAGNOSTIC_LABEL:
            state["mr_diagnostic"] = _mr_diagnostic(res, policy)
            continue
        state["points"][label] = {
            "status": (reseed or {}).get("rejected") or res["status"], "basis_by_element": prov, "reseeded": reseed,
            "scf_energy_hartree": res["scf_energy_hartree"],
            "ccsd_correlation_hartree": res["ccsd_correlation_hartree"],
            "triples_correction_hartree": res["triples_correction_hartree"],
            "ccsd_t_total_hartree": res["ccsd_t_total_hartree"],
            "reference_warnings": res["reference_warnings"], "error_message": res.get("error_message")}
    record["states"] = computed
    requests_by_label = {(id(state), label): req for state, label, _, req in jobs}

    def ok(state, label):
        p = (state.get("points") or {}).get(label)
        return p is not None and p["status"] == "COMPLETED"

    low, high = family_label(policy.low_cardinal), family_label(policy.high_cardinal)
    daug = family_label(policy.high_cardinal, daug=True)
    selected = {s["role"]: s for s in computed if s["selected"]}
    needed = [low, high] + ([daug] if policy.diffuse_correction else [])
    diffuse_missing = [f"{role}:{daug}" for role, s in selected.items()
                       if policy.diffuse_correction and not ok(s, daug)]
    if diffuse_missing:
        needed = [low, high]
        record["diffuse_check"] = {"status": "MISSING", "missing": diffuse_missing,
                                   "reason": "d-aug SCF not obtained on the aug branch"}
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
        return extrapolate_species(pair, model_name="PRIMARY",
                                   exponents=PAIR_EXPONENTS[(policy.low_cardinal, policy.high_cardinal)][0],
                                   cardinals=(policy.low_cardinal, policy.high_cardinal)).total_cbs_hartree

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
    diffuse_label = daug if policy.diffuse_correction and not diffuse_missing else high

    def fixed_ea(label):
        return (selected["neutral"]["points"][label]["ccsd_t_total_hartree"]
                - selected["anion"]["points"][label]["ccsd_t_total_hartree"]) * HARTREE_TO_EV
    daug_ea = fixed_ea(diffuse_label)
    residual = None
    if diffuse_missing:
        residual = max(abs(fixed_ea(high) - fixed_ea(low)), policy.missing_diffuse_floor_ev)
        record["diffuse_check"]["residual_estimate_ev"] = residual
    elif policy.diffuse_correction:
        delta1 = daug_ea - fixed_ea(high)
        delta2, second = None, None
        if DIFFUSE_SECOND_STEP_TRIGGER_EV < abs(delta1) <= policy.diffuse_continuum_threshold_ev:
            second = _second_diffuse_step(selected, template, policy, runner, settings, low)
            delta2 = second.get("delta2_ev")
        residual, convergence = diffuse_tail_ev(delta1, delta2)
        if delta2 is None and second is not None:   # second step failed: the full first step as tail
            residual, convergence = abs(delta1), "SECOND_STEP_FAILED"
        daug_ea += delta2 or 0.0
        record["diffuse_check"] = {"status": convergence, "delta1_ev": delta1, "delta2_ev": delta2,
                                   "residual_estimate_ev": residual, "second_step": second}
    cbs = resolve_cbs(points=points, daug_fixed_ea_ev=daug_ea, full_pec_aug5_ea_ev=None,
                      diffuse_residual_estimate_ev=residual,
                      model_spread_target_ev=policy.model_spread_target_ev,
                      cardinals=(policy.low_cardinal, policy.high_cardinal), basis_labels=(low, high)).to_dict()
    # Two ROHF reference branches of one state: their single-basis offset is not
    # recomputed at CBS and widens the interval on both sides (as in the EA stage).
    sensitivity = float(electronic_ea.get("reference_sensitivity_ev") or 0.0)
    geo_corr = geo_half = 0.0
    if policy.geometry_relaxation:
        chains = [(runner, requests_by_label[(id(s), low)], float(s["r"]),
                   float(s["points"][low]["ccsd_t_total_hartree"]), settings) for s in selected.values()]
        relax = dict(zip(selected, run_parallel(_relaxation_chain, chains)))
        geo_corr, geo_half = geometry_relaxation_correction(relax)
        record["geometry_relaxation"] = {"basis": low, "states": relax, "ea_correction_ev": geo_corr,
                                         "half_width_ev": geo_half}
    ea_cbs = cbs["ea_cbs_plus_diffuse_primary_ev"] + geo_corr
    lower = cbs["lower_intermediate_ev"] - sensitivity + geo_corr - geo_half
    upper = cbs["upper_intermediate_ev"] + sensitivity + geo_corr + geo_half
    binding, binding_reason = _binding(ea_cbs, lower, upper, cbs["diffuse_correction_ev"], policy)
    mr = {role: s.get("mr_diagnostic") for role, s in selected.items()}
    mr_exceeded = sorted(f"{role}:{name}" for role, d in mr.items() if d for name in d["exceeded"])
    mr_missing = sorted(role for role, d in mr.items() if not (d and d["available"]))
    model_bound = cbs.get("cbs_model_sensitivity_bound_ev")
    model_widened = (cbs["status"] != "CLEARED" and model_bound is not None
                     and model_bound <= CBS_MODEL_BOUND_CAP_EV)
    if not order_ok:
        status = "UNRESOLVED_STATE_ORDER_AT_CBS"
    elif binding == "UNBOUND":
        status = "ANION_UNBOUND_AT_CBS"
    elif binding == "UNDETERMINED":
        status = "NEED_MORE_EVIDENCE_BINDING"
    else:
        status = "CLEARED" if model_widened else cbs["status"]
    record.update(cbs_model_bound_above_target_in_interval=model_widened,
        status=status, cbs=cbs, cbs_model_status=cbs["status"],
        ea_cbs_ev=ea_cbs if binding != "UNBOUND" else None, ea_cbs_raw_ev=ea_cbs,
        ea_cbs_lower_ev=lower, ea_cbs_upper_ev=upper,
        reference_sensitivity_ev=sensitivity,
        reference_sensitivity_source="SINGLE_BASIS_REFERENCE_BRANCH_OFFSET" if sensitivity else None,
        binding_at_cbs=binding, binding_reason=binding_reason,
        multireference_diagnostics=mr, multireference_exceeded=mr_exceeded,
        multireference_screening=("MISSING" if mr_missing else
                                  "ABOVE_SCREENING_VALUES_WARNING" if mr_exceeded else "BELOW_SCREENING_VALUES"),
        selected_neutral_job=selected["neutral"]["job_id"], selected_anion_job=selected["anion"]["job_id"],
        verdict="NOT_ISSUED_NO_CV_SR_ZPE_BINDING_LAYERS",
        fixed_basis_ea_ev={label: (selected["neutral"]["points"][label]["ccsd_t_total_hartree"]
                                   - selected["anion"]["points"][label]["ccsd_t_total_hartree"]) * HARTREE_TO_EV
                           for label in needed},
    )
    _save_json(output_dir / "cbs.json", record)
    return record
