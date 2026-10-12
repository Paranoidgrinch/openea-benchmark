"""Automated zero-point (v=0) layer and adiabatic-EA assembly (Patch 42).

Experimental electron affinities are adiabatic: anion v=0 -> neutral v=0.  The
electronic CBS value is therefore corrected by

    Delta_ZPE = ZPE(neutral) - ZPE(anion),   EA_0 = EA_e(CBS) + Delta_ZPE.

This module only wires existing, separately tested layers to the automated run:

* ``adaptive.nuclear_motion``: J=0 radial Schrodinger equation on the
  identity-cleared Stage-3 CCSD(T) curve (PCHIP + natural-cubic cross-check,
  nested grids, trimmed-domain and boundary-clearance checks) and the
  vibrational binding of the anion (v=0 below its dissociation channels);
* ``adaptive.nuclear_motion_orchestration`` (D12):
  - diagnosis -> tool: when the solver reports PEC_REFINEMENT_REQUIRED (the
    curve from the equilibrium search covers only about +-0.05 A, far less than
    the v=0 wavefunction) new Stage-3 points are added outward, seeded from the
    boundary checkpoint and re-reviewed by the canonical identity/continuity
    resolver, until the solve clears;
  - model sensitivity: the curve is recomputed at the same geometries in a
    larger basis (def2-QZVPPD, seeded from the primary checkpoints); the
    Delta_ZPE shift bounds the electronic-model error.

The cross-model identity evidence that D12 requires is generated here from
explicit checks, never from matching labels alone: the comparison points follow
no instability, their equilibrium bond length agrees with the primary curve and
the comparison-minus-primary energy offset has no kink (each point lies within
0.5 mEh of the line through its neighbours; a different state or reference
branch shows up as a jump).  If any check fails no evidence is
issued and the layer stays fail closed.

Masses: most abundant isotope of each element (PySCF table), i.e. the main
isotopologue that photodetachment experiments usually report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .adaptive.nuclear_motion import (
    DiatomicMassSpecification, NuclearMotionSettings, NuclearMotionStatus, _pchip_minimum,
    nuclear_motion_pec_from_high_level, run_diatomic_nuclear_motion,
)
from .adaptive.nuclear_motion_orchestration import (
    NuclearPECModelLevel, NuclearPECModelRunStatus, NuclearPECRefinementRunStatus, NuclearPECStage3Context,
    run_nuclear_pec_model_convergence, run_nuclear_pec_refinement,
)
from .adaptive.stage3_axial_symmetry import AxialSymmetryPolicy
from .adaptive.stage3_execution import Stage3ExecutionSettings, Stage3PointResult
from .adaptive.stage3_identity import resolve_high_level_identity_and_pec
from .adaptive.stage3_loop import Stage3LoopRetrySettings
from .adaptive.stage3_refinement import Stage3RefinementSettings
from .attachment.asymptote import DissociationChannel
from .reference_branches import FOLLOWED_SUFFIX, SEED_SUFFIX
from .electronic_ea_stage import (
    ElectronicEAStagePolicy, _save_json, anion_dissociation_channels, cached_point_runner, scan_atomic_fragments,
)

MASS_EVIDENCE = "PYSCF_DATA_COMMON_ISOTOPE_MASSES_MOST_ABUNDANT_ISOTOPE"


@dataclass(frozen=True)
class ZPEStagePolicy:
    """Explicit pilot policy, not a validated production threshold set."""
    # Stage-3 grid spacing: the continuity thresholds (occupied overlap >= 0.95) are
    # calibrated for it.  0.04 A gave overlaps of 0.946 for NO (AMBIGUOUS).
    extension_step_angstrom: float = 0.025
    max_pec_rounds: int = 40
    max_grid_doublings: int = 2
    coarse_grid_points: int = 301
    fine_grid_points: int = 601
    grid_convergence_tolerance_ev: float = 2.0e-5
    interpolation_sensitivity_tolerance_ev: float = 5.0e-4
    domain_sensitivity_tolerance_ev: float = 2.0e-4
    minimum_boundary_clearance_ev: float = 0.30
    trim_fraction: float = 0.08
    comparison_basis: str = "def2-QZVPPD"
    # Cross-model identity checks (primary vs comparison basis, same geometries).
    # Flat anion curves move their minimum with the basis: MgH- def2-TZVPPD 1.883 A,
    # def2-QZVPPD 1.858 A (literature CBS 1.865 A) with a smooth 0.1 mEh offset.
    max_cross_model_re_shift_angstrom: float = 0.05
    # Largest deviation of one comparison-minus-primary offset from the line through
    # its two neighbours: smooth basis-set curvature stays far below this, a jump to
    # another state or reference branch (mEh) does not.
    max_cross_model_offset_kink_hartree: float = 5.0e-4

    def nuclear_settings(self) -> NuclearMotionSettings:
        return NuclearMotionSettings(
            coarse_grid_points=self.coarse_grid_points, fine_grid_points=self.fine_grid_points,
            grid_convergence_tolerance_ev=self.grid_convergence_tolerance_ev,
            interpolation_sensitivity_tolerance_ev=self.interpolation_sensitivity_tolerance_ev,
            domain_sensitivity_tolerance_ev=self.domain_sensitivity_tolerance_ev,
            minimum_boundary_clearance_ev=self.minimum_boundary_clearance_ev,
            trim_fraction=self.trim_fraction, eigenvalue_tolerance_hartree=1.0e-12)


def main_isotopologue_masses(atoms) -> DiatomicMassSpecification:
    from pyscf.data import elements
    a, b = atoms
    mass = lambda el: float(elements.COMMON_ISOTOPE_MASSES[int(elements.charge(el))])
    return DiatomicMassSpecification(a, b, mass(a), mass(b), evidence_ids=(MASS_EVIDENCE,),
                                     mass_model="MOST_ABUNDANT_ISOTOPES")


def _solve_summary(result) -> dict:
    return {k: (v.value if hasattr(v, "value") else v) for k, v in (
        ("status", result.status), ("zpe_ev", result.zpe_ev), ("numerical_bound_ev", result.numerical_bound_ev),
        ("grid_difference_ev", result.grid_difference_ev),
        ("interpolation_difference_ev", result.interpolation_difference_ev),
        ("domain_difference_ev", result.domain_difference_ev),
        ("left_boundary_clearance_ev", result.left_boundary_clearance_ev),
        ("right_boundary_clearance_ev", result.right_boundary_clearance_ev),
        ("requires_lower_r_extension", result.requires_lower_r_extension),
        ("requires_upper_r_extension", result.requires_upper_r_extension),
        ("rationale", result.rationale))}


def _assessment_summary(assessment) -> dict:
    return {"status": assessment.status.value, "action": assessment.action,
            "correction_ev": assessment.correction_ev, "correction_half_width_ev": assessment.correction_half_width_ev,
            "neutral": _solve_summary(assessment.neutral), "anion": _solve_summary(assessment.anion),
            "anion_vibrational_binding": assessment.anion_vibrational_binding.status.value,
            "rationale": assessment.rationale}


def _pec_summary(context: NuclearPECStage3Context) -> dict:
    r = sorted(float(p.r_angstrom) for p in context.pec.points)
    return {"job_id": context.pec.job_id, "basis": context.pec.basis, "n_points": len(r),
            "r_min_angstrom": r[0], "r_max_angstrom": r[-1]}


def cross_model_identity_evidence(primary: NuclearPECStage3Context, comparison: NuclearPECStage3Context,
                                  policy: ZPEStagePolicy) -> tuple[tuple[str, ...], dict]:
    """Evidence IDs (empty = not established) that both curves describe one state."""
    prim = {round(float(p.r_angstrom), 6): float(p.energy_hartree) for p in primary.pec.points}
    comp = {round(float(p.r_angstrom), 6): float(p.energy_hartree) for p in comparison.pec.points}
    common = sorted(set(prim) & set(comp))
    checks: dict[str, Any] = {"common_points": len(common)}
    if len(common) < 5:
        checks["failed"] = "fewer than five common geometries"
        return (), checks
    r = np.array(common)
    offset = np.array([comp[x] - prim[x] for x in common])
    weight = (r[1:-1] - r[:-2]) / (r[2:] - r[:-2])
    between = offset[:-2] + weight * (offset[2:] - offset[:-2])
    checks["offset_kink_hartree"] = float(np.max(np.abs(offset[1:-1] - between)))
    re_p = _pchip_minimum(r, np.array([prim[x] for x in common]))[0]
    re_c = _pchip_minimum(r, np.array([comp[x] for x in common]))[0]
    checks["re_primary_angstrom"], checks["re_comparison_angstrom"] = float(re_p), float(re_c)
    checks["comparison_stability_following_steps"] = max(int(x.stability_following_steps) for x in comparison.results)
    failed = []
    if checks["comparison_stability_following_steps"] > 0:
        failed.append("a comparison point followed an SCF instability")
    if abs(re_p - re_c) > policy.max_cross_model_re_shift_angstrom:
        failed.append("equilibrium bond lengths differ")
    if checks["offset_kink_hartree"] > policy.max_cross_model_offset_kink_hartree:
        failed.append("comparison-minus-primary energy offset is not smooth")
    if failed:
        checks["failed"] = "; ".join(failed)
        return (), checks
    role = primary.role.upper()
    return (f"CROSS_MODEL_{role}_PROJECTED_SEED_SAME_IRREP_OCCUPATION_NO_FOLLOWING",
            f"CROSS_MODEL_{role}_EQUILIBRIUM_AGREES",
            f"CROSS_MODEL_{role}_SMOOTH_ENERGY_OFFSET"), checks


def _extension_failure_details(run, contexts, runner, settings, identity_thresholds, branch_thresholds,
                               resolver) -> list[dict]:
    """Why an extension batch failed: point statuses and the identity/continuity reviews."""
    details = []
    for batch in run.batches:
        results = [runner(req, settings) for req in batch.requests]  # cached: no recomputation
        entry = {"role": batch.role, "points": [
            {"r_angstrom": req.r_angstrom, "status": res.status.value, "scf_energy_hartree": res.scf_energy_hartree,
             "ccsd_t_total_hartree": res.ccsd_t_total_hartree, "error": res.error_message}
            for req, res in zip(batch.requests, results)]}
        if all(res.status.value == "COMPLETED" for res in results):
            ctx = contexts[batch.role]
            resolution = resolver(requests=tuple(ctx.requests) + tuple(batch.requests),
                                  results=tuple(ctx.results) + tuple(results),
                                  identity_thresholds=identity_thresholds, branch_thresholds=branch_thresholds,
                                  audit_settings=None, duplicate_energy_tolerance_hartree=1.0e-7)
            entry["initialization_review"] = {"status": resolution.initialization_review.status.value,
                                              "rationale": resolution.initialization_review.rationale}
            entry["continuity_review"] = {"status": resolution.geometry_continuity_review.status.value,
                                          "rationale": resolution.geometry_continuity_review.rationale}
            entry["continuity_comparisons"] = [
                {k: v for k, v in asdict(c).items() if not isinstance(v, (tuple, list)) or len(v) < 8}
                for c in resolution.continuity_comparisons
                if any(abs(float(c.r_a_angstrom) - float(req.r_angstrom)) < 1e-6
                       or abs(float(c.r_b_angstrom) - float(req.r_angstrom)) < 1e-6 for req in batch.requests)]
        details.append(entry)
    return details


def harmonic_curvature(pec: Any, half_window_angstrom: float = 0.03) -> float | None:
    """Second derivative (Eh/A^2) of a quadratic fit around the lowest sampled point."""
    pts = sorted((float(p.r_angstrom), float(p.energy_hartree)) for p in pec.points)
    if not pts:
        return None
    r0 = min(pts, key=lambda p: p[1])[0]
    near = [p for p in pts if abs(p[0] - r0) <= half_window_angstrom + 1e-9]
    if len(near) < 3:
        return None
    r, e = np.array(near).T
    k = 2.0 * float(np.polyfit(r - r0, e, 2)[0])
    return k if k > 0.0 else None


def _followed_sibling(job_id: str | None, loops: Mapping[str, Any]) -> str | None:
    """The STABILITY_FOLLOWED branch of a seed-continuous branch job, if it was refined."""
    if not job_id or not job_id.endswith(SEED_SUFFIX):
        return None
    sibling = job_id[: -len(SEED_SUFFIX)] + FOLLOWED_SUFFIX
    loop = loops.get(sibling)
    return sibling if loop is not None and getattr(loop, "final_pec", None) is not None else None


def run_vibrational_stage(*, electronic_ea: Mapping[str, Any], loops: Mapping[str, Any], atoms,
                          stage3_settings: Stage3ExecutionSettings, output_dir: Path,
                          identity_thresholds: Any = None, branch_thresholds: Any = None,
                          policy: ZPEStagePolicy = ZPEStagePolicy(),
                          point_backend: Callable[..., Stage3PointResult] | None = None) -> dict:
    """Zero-point layer on the selected curves, with a reference-branch fallback.

    A seed-continuous ROHF branch (``__refS``) can end inside the range the v=0
    wavefunction needs: the internal-saddle SCF solution disappears and the
    extension points do not converge.  The curve *shape* is then taken from the
    stability-followed sibling branch of the same symmetry-labelled state, which
    exists at all R.  The electronic EA keeps the selected branch; the relative
    difference of the two branches' harmonic frequencies at their minima,
    applied to that role's ZPE, is added to the Delta_ZPE half-width.
    """
    common = dict(electronic_ea=electronic_ea, loops=loops, atoms=atoms, stage3_settings=stage3_settings,
                  output_dir=Path(output_dir), identity_thresholds=identity_thresholds,
                  branch_thresholds=branch_thresholds, policy=policy, point_backend=point_backend)
    n_id, a_id = electronic_ea.get("selected_neutral_job"), electronic_ea.get("selected_anion_job")
    record = _run_vibrational_stage(**common, jobs={"neutral": n_id, "anion": a_id})
    if not str(record.get("status", "")).startswith("PEC_EXTENSION_"):
        return record
    record = _branch_fallback(record, common, electronic_ea, loops, n_id, a_id, output_dir)
    if str(record.get("status", "")).startswith("PEC_EXTENSION_"):
        record = _mr_fallback(record, electronic_ea, loops, atoms, policy, output_dir)
    return record


def _mr_fallback(record: dict, electronic_ea, loops, atoms, policy: ZPEStagePolicy, output_dir) -> dict:
    """No single-reference curve covers the v=0 range: Delta_ZPE from MR curves (Patch 50)."""
    from .mr_nuclear_curve import mr_zpe_fallback
    print(f"[ZPE] {electronic_ea.get('system')} single-reference curves failed; multireference curves", flush=True)
    channels = tuple(DissociationChannel(**c) for c in electronic_ea.get("anion_dissociation_channels") or ())
    mr = mr_zpe_fallback(electronic_ea=electronic_ea, loops=loops, masses=main_isotopologue_masses(atoms),
                         channels=channels, nuclear_settings=policy.nuclear_settings(), output_dir=Path(output_dir))
    out = dict(record)
    out["multireference_fallback"] = mr
    if mr.get("status") == "CLEARED":
        out.update(status="CLEARED", reason="Delta_ZPE from multireference (SC-NEVPT2) curves",
                   delta_zpe_ev=mr["delta_zpe_ev"], delta_zpe_half_width_ev=mr["delta_zpe_half_width_ev"],
                   zpe_neutral_ev=mr["zpe_neutral_ev"], zpe_anion_ev=mr["zpe_anion_ev"],
                   anion_vibrational_binding=mr["anion_vibrational_binding"], zpe_curve_method="MULTIREFERENCE")
    _save_json(Path(output_dir) / "nuclear_motion.json", out)
    return out


def _branch_fallback(record: dict, common: dict, electronic_ea, loops, n_id, a_id, output_dir) -> dict:
    # The failure details list every extended curve; only the ones that failed count.
    failed_roles = {entry["role"] for entry in record.get("failed_extension") or ()
                    if any(p.get("status") != "COMPLETED" for p in entry.get("points") or ())
                    or (entry.get("continuity_review") or {}).get("status") not in (None, "CLEARED")
                    or (entry.get("initialization_review") or {}).get("status") not in (None, "CLEARED")}
    if not failed_roles:
        return record
    jobs = {"neutral": n_id, "anion": a_id}
    substitution = {}
    for role in sorted(failed_roles):
        sibling = _followed_sibling(jobs[role], loops)
        if sibling is None:
            return record
        k_sel = harmonic_curvature(loops[jobs[role]].final_pec)
        k_sib = harmonic_curvature(loops[sibling].final_pec)
        if k_sel is None or k_sib is None:
            return record
        substitution[role] = {"selected_job": jobs[role], "shape_job": sibling,
                              "curvature_selected_hartree_per_angstrom2": k_sel,
                              "curvature_shape_hartree_per_angstrom2": k_sib,
                              "relative_frequency_difference": abs(float(np.sqrt(k_sib / k_sel)) - 1.0)}
        jobs[role] = sibling
    print(f"[ZPE] {electronic_ea.get('system')} seed branch ended; curve shape from "
          f"{', '.join(v['shape_job'] for v in substitution.values())}", flush=True)
    fallback = _run_vibrational_stage(**common, jobs=jobs)
    fallback["reference_branch_shape_substitution"] = {
        "reason": "SEED_CONTINUOUS_BRANCH_EXTENSION_FAILED",
        "selected_branch_attempt": {k: record.get(k) for k in ("status", "reason", "failed_extension")},
        "roles": substitution}
    if fallback.get("delta_zpe_half_width_ev") is not None:
        widening = sum(float(fallback.get(f"zpe_{role}_ev") or 0.0) * item["relative_frequency_difference"]
                       for role, item in substitution.items())
        fallback["reference_branch_shape_substitution"]["half_width_added_ev"] = widening
        fallback["delta_zpe_half_width_ev"] = float(fallback["delta_zpe_half_width_ev"]) + widening
    _save_json(Path(output_dir) / "nuclear_motion.json", fallback)
    return fallback


def _run_vibrational_stage(*, electronic_ea: Mapping[str, Any], loops: Mapping[str, Any], atoms,
                           stage3_settings: Stage3ExecutionSettings, output_dir: Path,
                           identity_thresholds: Any, branch_thresholds: Any, policy: ZPEStagePolicy,
                           point_backend: Callable[..., Stage3PointResult] | None,
                           jobs: Mapping[str, str | None]) -> dict:
    record: dict[str, Any] = {"stage": "NUCLEAR_MOTION_ZPE", "policy": asdict(policy),
                              "includes_rotation": False, "includes_dboc": False,
                              "validation_metadata_used_in_computation": False}
    n_id, a_id = jobs["neutral"], jobs["anion"]
    if electronic_ea.get("status") != "ELECTRONIC_EA_EVALUATED" or n_id not in loops or a_id not in loops:
        record.update(status="NOT_ATTEMPTED", reason="no evaluated electronic EA with refinement loops")
        _save_json(output_dir / "nuclear_motion.json", record)
        return record
    if identity_thresholds is None or branch_thresholds is None:
        from .workflow import WEEKEND_BRANCH_THRESHOLDS, WEEKEND_IDENTITY_THRESHOLDS
        identity_thresholds = identity_thresholds or WEEKEND_IDENTITY_THRESHOLDS
        branch_thresholds = branch_thresholds or WEEKEND_BRANCH_THRESHOLDS

    def resolver(**kwargs):
        return resolve_high_level_identity_and_pec(**kwargs, axial_symmetry_policy=AxialSymmetryPolicy())

    contexts = {}
    for role, job in (("neutral", n_id), ("anion", a_id)):
        loop = loops[job]
        try:
            contexts[role] = NuclearPECStage3Context(role, job, loop.final_pec, tuple(loop.requests),
                                                     tuple(loop.results))
        except ValueError as exc:
            record.update(status="NOT_ATTEMPTED", reason=f"{role} curve not usable: {exc}")
            _save_json(output_dir / "nuclear_motion.json", record)
            return record

    masses = main_isotopologue_masses(atoms)
    record["masses"] = {"atoms": list(atoms), "mass_a_u": masses.mass_a_u, "mass_b_u": masses.mass_b_u,
                        "evidence": MASS_EVIDENCE}
    channels = tuple(DissociationChannel(**c) for c in electronic_ea.get("anion_dissociation_channels") or ())
    settings = replace(stage3_settings, checkpoint_project=True,
                       artifact_dir=str(output_dir / "checkpoints"))
    # Point cache key: only what changes the computed points (geometries come from the
    # step, the comparison level from the basis); tolerances and budgets do not.
    signature = sha256(json.dumps({"extension_step_angstrom": policy.extension_step_angstrom,
                                   "comparison_basis": policy.comparison_basis},
                                  sort_keys=True).encode()).hexdigest()
    runner = cached_point_runner(output_dir / "points", signature, point_backend)
    retry = Stage3LoopRetrySettings(max_retries_per_request=1, cycle_multiplier=2.0)
    refinement = Stage3RefinementSettings(extension_step_angstrom=policy.extension_step_angstrom,
                                          target_bracket_width_angstrom=policy.extension_step_angstrom,
                                          minimum_new_point_separation_angstrom=1.0e-4,
                                          energy_tie_tolerance_hartree=1.0e-8)
    nuclear_settings = policy.nuclear_settings()

    def solve(model_evidence=None):
        pecs = {role: nuclear_motion_pec_from_high_level(ctx.pec, role=role, state_id=ctx.state_id,
                                                         evidence_ids=("OPENEA_ZPE_STAGE",))
                for role, ctx in contexts.items()}
        return run_diatomic_nuclear_motion(neutral_pec=pecs["neutral"], anion_pec=pecs["anion"], masses=masses,
                                           settings=nuclear_settings, anion_dissociation_channels=channels,
                                           model_evidence=model_evidence)

    # Diagnosis -> tool loop: the solver says what the curve lacks, D12 adds it.
    rounds, doublings = [], 0
    assessment = solve()
    for round_index in range(1, policy.max_pec_rounds + 1):
        rounds.append({"round": round_index, **_assessment_summary(assessment),
                       "pec": {role: _pec_summary(ctx) for role, ctx in contexts.items()}})
        if assessment.status is NuclearMotionStatus.PEC_REFINEMENT_REQUIRED:
            print(f"[ZPE] {electronic_ea.get('system')} extend nuclear PEC (round {round_index})", flush=True)
            run = run_nuclear_pec_refinement(
                assessment=assessment, neutral_context=contexts["neutral"], anion_context=contexts["anion"],
                nuclear_settings=nuclear_settings, refinement_settings=refinement,
                refinement_round=100 + round_index, execution_settings=settings,
                identity_thresholds=identity_thresholds, branch_thresholds=branch_thresholds,
                retry_settings=retry, point_runner=runner, resolver=resolver)
            if run.status is not NuclearPECRefinementRunStatus.COMPLETED:
                record.update(status=f"PEC_EXTENSION_{run.status.value}", reason=run.rationale, rounds=rounds,
                              failed_extension=_extension_failure_details(run, contexts, runner, settings,
                                                                          identity_thresholds, branch_thresholds,
                                                                          resolver))
                _save_json(output_dir / "nuclear_motion.json", record)
                return record
            contexts = {"neutral": run.neutral_context, "anion": run.anion_context}
        elif (assessment.status is NuclearMotionStatus.NUMERICAL_REFINEMENT_REQUIRED
              and doublings < policy.max_grid_doublings):
            doublings += 1
            nuclear_settings = replace(nuclear_settings, coarse_grid_points=2 * nuclear_settings.coarse_grid_points + 1,
                                       fine_grid_points=2 * nuclear_settings.fine_grid_points + 1)
        else:
            break
        assessment = solve()
    record["rounds"] = rounds
    record["primary_pec"] = {role: _pec_summary(ctx) for role, ctx in contexts.items()}

    if assessment.status is not NuclearMotionStatus.MODEL_CONVERGENCE_REQUIRED:
        record.update(status=assessment.status.value, assessment=_assessment_summary(assessment),
                      reason="primary nuclear solve did not converge numerically within the round budget")
        _save_json(output_dir / "nuclear_motion.json", record)
        return record

    # Model sensitivity: same geometries, larger basis.
    print(f"[ZPE] {electronic_ea.get('system')} comparison curve {policy.comparison_basis}", flush=True)
    model = NuclearPECModelLevel(
        model_id=policy.comparison_basis.replace("-", "").lower(), basis=policy.comparison_basis,
        basis_by_element=None, methods=("CCSD", "CCSD(T)"),
        authorization_evidence_ids=("OPENEA_ZPE_STAGE_POLICY_COMPARISON_BASIS",),
        rationale="Delta_ZPE basis sensitivity: primary Stage-3 basis vs a larger basis at the same geometries.")
    # Fragment asymptotes at the comparison basis (binding-margin model check).
    comparison_channels = channels
    if channels:
        cmp_fragments = scan_atomic_fragments(atoms=atoms, basis=policy.comparison_basis, settings=stage3_settings,
                                              policy=ElectronicEAStagePolicy(),
                                              cache_dir=output_dir / "comparison_fragments")
        comparison_channels = anion_dissociation_channels(atoms, cmp_fragments)
        record["comparison_dissociation_channels"] = [asdict(c) for c in comparison_channels]
    common = dict(primary_assessment=assessment, primary_neutral_context=contexts["neutral"],
                  primary_anion_context=contexts["anion"], model=model, masses=masses,
                  nuclear_settings=nuclear_settings, execution_settings=settings,
                  identity_thresholds=identity_thresholds, branch_thresholds=branch_thresholds,
                  anion_dissociation_channels=channels, comparison_anion_dissociation_channels=comparison_channels,
                  retry_settings=retry, point_runner=runner,
                  resolver=resolver)
    model_run = run_nuclear_pec_model_convergence(**common)
    if model_run.status is NuclearPECModelRunStatus.CROSS_MODEL_IDENTITY_REVIEW_REQUIRED:
        evidence, checks = [], {}
        for role, cmp_ctx in (("neutral", model_run.comparison_neutral_context),
                              ("anion", model_run.comparison_anion_context)):
            ids, checks[role] = cross_model_identity_evidence(contexts[role], cmp_ctx, policy)
            evidence.append(ids)
        record["cross_model_identity_checks"] = checks
        if all(evidence):
            model_run = run_nuclear_pec_model_convergence(
                **common, cross_model_identity_evidence_ids=tuple(x for ids in evidence for x in ids),
                existing_comparison_neutral_context=model_run.comparison_neutral_context,
                existing_comparison_anion_context=model_run.comparison_anion_context)
    record["model_run"] = {"status": model_run.status.value, "rationale": model_run.rationale,
                           "comparison": (None if model_run.comparison_assessment is None
                                          else _assessment_summary(model_run.comparison_assessment))}
    final = model_run.final_primary_assessment
    if model_run.status is not NuclearPECModelRunStatus.MODEL_EVIDENCE_READY or final is None:
        record.update(status=f"MODEL_{model_run.status.value}", assessment=_assessment_summary(assessment),
                      delta_zpe_ev=assessment.correction_ev, delta_zpe_half_width_ev=None)
    else:
        record.update(status=final.status.value, assessment=_assessment_summary(final),
                      delta_zpe_ev=final.correction_ev, delta_zpe_half_width_ev=final.correction_half_width_ev,
                      zpe_neutral_ev=final.neutral.zpe_ev, zpe_anion_ev=final.anion.zpe_ev,
                      anion_vibrational_binding=final.anion_vibrational_binding.status.value)
    _save_json(output_dir / "nuclear_motion.json", record)
    return record


def assemble_adiabatic_ea(*, cbs: Mapping[str, Any] | None, nuclear: Mapping[str, Any] | None,
                          core_valence: Mapping[str, Any] | None = None,
                          spin_orbit: Mapping[str, Any] | None = None,
                          scalar_relativity: Mapping[str, Any] | None = None,
                          post_ccsd_t: Mapping[str, Any] | None = None,
                          output_dir: Path | None = None) -> dict:
    """EA_0 = EA_e(CBS) + Delta_ZPE [+ CV] [+ SR] [+ SO] [+ post-CCSD(T)], worst-case intervals.

    Each correction enters only when its layer CLEARED; otherwise the status is
    recorded and the correction is left out (``includes_*`` false).
    """
    cv_ok = bool(core_valence) and core_valence.get("status") == "CORE_VALENCE_CLEARED"
    so_ok = bool(spin_orbit) and spin_orbit.get("status") == "CLEARED"
    sr_ok = bool(scalar_relativity) and scalar_relativity.get("status") == "SCALAR_RELATIVITY_CLEARED"
    pc_ok = bool(post_ccsd_t) and post_ccsd_t.get("status") == "CLEARED"
    missing = [name for name, ok in (("CV", cv_ok), ("SR", sr_ok), ("SOC", so_ok), ("POST_CCSD_T", pc_ok))
               if not ok]
    record: dict[str, Any] = {
        "stage": "ADIABATIC_EA_ASSEMBLY", "includes_cbs": True, "includes_zpe": True,
        "includes_core_valence": cv_ok, "includes_scalar_relativity": sr_ok, "includes_soc": so_ok,
        "includes_post_ccsd_t": pc_ok, "is_production_ea": False,
        "verdict": (f"NOT_ISSUED_NO_{'_'.join(missing)}_LAYERS" if missing
                    else "NOT_ISSUED_NO_FINAL_RESOLVER"),
        "core_valence_status": None if not core_valence else core_valence.get("status"),
        "spin_orbit_status": None if not spin_orbit else spin_orbit.get("status"),
        "scalar_relativity_status": None if not scalar_relativity else scalar_relativity.get("status"),
        "post_ccsd_t_status": None if not post_ccsd_t else post_ccsd_t.get("status"),
        "validation_metadata_used_in_computation": False}

    def done(**fields):
        record.update(fields)
        if output_dir is not None:
            _save_json(Path(output_dir) / "adiabatic_ea.json", record)
        return record

    if not cbs or cbs.get("binding_at_cbs") is None:
        return done(status="NOT_ASSEMBLED", reason="no CBS electronic EA")
    if cbs["binding_at_cbs"] == "UNBOUND":
        # A continuum anion has no v=0 level to correct; the verdict is the binding one.
        return done(status="ANION_UNBOUND", binding="UNBOUND", ea0_ev=None,
                    reason=f"CBS layer: {cbs.get('binding_reason')}")
    if not nuclear or nuclear.get("status") != "CLEARED" or nuclear.get("delta_zpe_half_width_ev") is None:
        return done(status="NEED_MORE_EVIDENCE_ZPE", ea0_ev=None,
                    reason=f"nuclear-motion layer status {None if not nuclear else nuclear.get('status')}")
    dz, hw = float(nuclear["delta_zpe_ev"]), float(nuclear["delta_zpe_half_width_ev"])
    dcv, cvb = ((float(core_valence["delta_cv_ev"]), float(core_valence["delta_cv_bound_ev"]))
                if cv_ok else (0.0, 0.0))
    dso, sob = ((float(spin_orbit["delta_ea_so_ev"]), float(spin_orbit["bound_ev"])) if so_ok else (0.0, 0.0))
    dsr, srb = ((float(scalar_relativity["delta_sr_ev"]), float(scalar_relativity["delta_sr_bound_ev"] or 0.0))
                if sr_ok else (0.0, 0.0))
    dpc, pcb = ((float(post_ccsd_t["delta_post_ev"]), float(post_ccsd_t["bound_ev"])) if pc_ok else (0.0, 0.0))
    shift, width = dz + dcv + dso + dsr + dpc, hw + cvb + sob + srb + pcb
    ea0 = float(cbs["ea_cbs_ev"]) + shift
    lower = float(cbs["ea_cbs_lower_ev"]) + shift - width
    upper = float(cbs["ea_cbs_upper_ev"]) + shift + width
    binding = "BOUND" if lower > 0.0 else ("UNBOUND" if upper < 0.0 else "UNDETERMINED")
    vib = nuclear.get("anion_vibrational_binding")
    if binding == "BOUND" and vib != "BOUND":
        binding = "UNDETERMINED"
    if cbs.get("status") not in ("CLEARED",):
        status = cbs.get("status")
    elif binding != "BOUND":
        status = f"NEED_MORE_EVIDENCE_BINDING_{binding}"
    else:
        status = "CLEARED"
    return done(status=status, binding=binding, ea0_ev=ea0, ea0_lower_ev=lower, ea0_upper_ev=upper,
                ea_cbs_electronic_ev=cbs["ea_cbs_ev"], delta_zpe_ev=dz, delta_zpe_half_width_ev=hw,
                delta_cv_ev=dcv if cv_ok else None, delta_cv_bound_ev=cvb if cv_ok else None,
                delta_so_ev=dso if so_ok else None, delta_so_bound_ev=sob if so_ok else None,
                delta_sr_ev=dsr if sr_ok else None, delta_sr_bound_ev=srb if sr_ok else None,
                delta_post_ccsd_t_ev=dpc if pc_ok else None, delta_post_ccsd_t_bound_ev=pcb if pc_ok else None,
                anion_vibrational_binding=vib, cbs_status=cbs.get("status"),
                multireference_screening=cbs.get("multireference_screening"))
