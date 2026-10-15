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
  undetermined -- never as a CBS number;
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
    PRIMARY_EXPONENTS, extrapolate_species, ComponentPair, resolve_cbs,
)
from .electronic_ea_stage import HARTREE_TO_EV, _save_json, cached_point_runner
from .point_pool import run_parallel

MR_DIAGNOSTIC_LABEL = "MRDIAG_STAGE3_BASIS"
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
    # An anion whose energy still moves this much from aug- to d-aug-cc-pV5Z is not
    # converging to a bound state: the basis is discretizing the detachment continuum.
    diffuse_continuum_threshold_ev: float = 0.05
    # Multireference screening (CCSD amplitudes at the def2-TZVPPD equilibrium;
    # common literature values, reported as warnings only -- see module docstring).
    t1_closed_shell_threshold: float = 0.02
    t1_open_shell_threshold: float = 0.044
    d1_threshold: float = 0.05
    max_t2_threshold: float = 0.1


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
    """(PySCF basis spec, provenance) for one element.

    PP-route elements (beyond Kr) get the basis *name* aug-cc-pVXZ-PP (d-aug: its
    even-tempered extension), resolved together with its ECP when the molecule is built.
    """
    from pyscf import gto
    from .heavy_elements import ROUTE_PSEUDOPOTENTIAL, element_route, resolve_named_basis
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
    if abs(diffuse_ev) > policy.diffuse_continuum_threshold_ev:
        if upper < 0.0:
            return "UNBOUND", "NEGATIVE_EA_AND_DIFFUSE_SENSITIVE_ANION_CONTINUUM_DISCRETIZATION"
        return "UNDETERMINED", "DIFFUSE_CORRECTION_ABOVE_CONTINUUM_THRESHOLD"
    if lower > 0.0:
        return "BOUND", "CBS_INTERVAL_POSITIVE"
    if upper < 0.0:
        return "UNBOUND", "CBS_INTERVAL_NEGATIVE"
    return "UNDETERMINED", "CBS_INTERVAL_CONTAINS_ZERO"


def _run_state_chain(runner, requests, settings) -> list[tuple[Any, dict | None]]:
    """One state's basis series in order.  A point that fails from the projected TZ
    reference is retried once, seeded from the last completed larger-basis reference of
    the same state (same irrep occupation, no following) and with doubled SCF/CC cycles;
    the retry is recorded, never silent."""
    out: list[tuple[Any, dict | None]] = []
    previous_checkpoint = None
    for req in requests:
        result, reseed = runner(req, settings), None
        if result.status is not PointExecutionStatus.COMPLETED:
            seed = previous_checkpoint or req.source_checkpoint_path
            retry = replace(req, request_id=req.request_id + "__RESEED", source_checkpoint_path=str(seed),
                            source_link_status=("CBS_BASIS_CONTINUATION_FROM_PREVIOUS_CARDINAL"
                                                if previous_checkpoint else req.source_link_status))
            retry_settings = replace(settings, scf_max_cycle=2 * settings.scf_max_cycle,
                                     cc_max_cycle=2 * settings.cc_max_cycle)
            reseed = {"first_status": result.status.value, "seed_checkpoint": str(seed),
                      "scf_max_cycle": retry_settings.scf_max_cycle}
            result = runner(retry, retry_settings)
        if result.status is PointExecutionStatus.COMPLETED and result.high_level_checkpoint_path:
            previous_checkpoint = result.high_level_checkpoint_path
        out.append((result, reseed))
    return out


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
        seed = min(recs, key=lambda r: abs(float(r["r_angstrom"]) - state["r"]))
        state["seed_r_angstrom"] = seed["r_angstrom"]
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
            "status": res["status"], "basis_by_element": prov, "reseeded": reseed,
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
    # Two ROHF reference branches of one state: their single-basis offset is not
    # recomputed at CBS and widens the interval on both sides (as in the EA stage).
    sensitivity = float(electronic_ea.get("reference_sensitivity_ev") or 0.0)
    ea_cbs = cbs["ea_cbs_plus_diffuse_primary_ev"]
    lower, upper = cbs["lower_intermediate_ev"] - sensitivity, cbs["upper_intermediate_ev"] + sensitivity
    binding, binding_reason = _binding(ea_cbs, lower, upper, cbs["diffuse_correction_ev"], policy)
    mr = {role: s.get("mr_diagnostic") for role, s in selected.items()}
    mr_exceeded = sorted(f"{role}:{name}" for role, d in mr.items() if d for name in d["exceeded"])
    mr_missing = sorted(role for role, d in mr.items() if not (d and d["available"]))
    if not order_ok:
        status = "UNRESOLVED_STATE_ORDER_AT_CBS"
    elif binding == "UNBOUND":
        status = "ANION_UNBOUND_AT_CBS"
    elif binding == "UNDETERMINED":
        status = "NEED_MORE_EVIDENCE_BINDING"
    else:
        status = cbs["status"]
    record.update(
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
