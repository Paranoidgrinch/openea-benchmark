"""Automatic benchmark acquisition from existing OpenEA scout to Stage-3 evidence.

This controller never awards a ground state, electron affinity, or scientific
method validation.  Open D04/D09 prerequisites remain open while useful
bracketed-PEC Stage-3 points may run *provisionally*.  All decisions and source
provenance are persisted so an interrupted run can resume safely.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .adaptive.stage3_bridge import build_adaptive_stage3_plan, Stage3ReleaseStatus
from .adaptive.stage3_execution import (
    PointExecutionStatus, Stage3ExecutionRequest, Stage3ExecutionSettings,
    Stage3PointResult, build_stage3_execution_requests,
)
from .adaptive.stage3_loop import execute_stage3_batch_with_retries
from .open_pec_extension import apply_open_pec_extension, asymptote_review_from_extension
from .reference_branches import (
    FOLLOWED_SUFFIX, MERGED_STATUS, SEED_SUFFIX, SUPERSEDED_STATUS, branches_identical,
    build_branch_requests, mixed_reference_jobs,
)
from .state_selection import build_state_selection_report


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _save(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(record, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    temp.replace(path)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _input_checked_summary(summary: Mapping[str, Any], system: str) -> None:
    if summary.get("system") != system or summary.get("validation_metadata_used_in_computation") is not False:
        raise ValueError("Scout summary identity/validation isolation not established")
    if not isinstance(summary.get("sectors"), list):
        raise ValueError("Scout summary has no sectors")


def _reference_warning_summary(
    requests: Sequence[Stage3ExecutionRequest], records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict]:
    """Per-job D02/D03 reference evidence; magnitudes only, no classification."""
    summary: dict[str, dict] = {}
    for req in requests:
        rec = records.get(req.request_id)
        if not rec or not rec.get("reference_warnings"):
            continue
        entry = summary.setdefault(req.job_id, {
            "points_with_followed_instability": 0,
            "max_following_lowering_hartree": None,
            "points_with_rhf_external_instability": 0,
            "max_uhf_lowering_hartree": None,
            "max_uhf_s2": None,
        })
        if "INTERNAL_INSTABILITY_FOLLOWED" in rec["reference_warnings"]:
            entry["points_with_followed_instability"] += 1
        if "RHF_EXTERNAL_INSTABILITY_RECORDED" in rec["reference_warnings"]:
            entry["points_with_rhf_external_instability"] += 1
        for key, field in (("max_following_lowering_hartree", "stability_following_lowering_hartree"),
                           ("max_uhf_lowering_hartree", "rhf_external_uhf_lowering_hartree"),
                           ("max_uhf_s2", "rhf_external_uhf_s2")):
            value = rec.get(field)
            if value is not None and (entry[key] is None or value > entry[key]):
                entry[key] = value
    return summary


def _job_purpose(bridge: Any) -> str:
    if bridge.release_status is Stage3ReleaseStatus.RELEASED:
        return "RELEASED_HIGH_LEVEL_EVIDENCE"
    return "PROVISIONAL_EVIDENCE_ONLY_NOT_SCIENTIFIC_RELEASE"


def build_benchmark_acquisition_plan(
    *, summary: Mapping[str, Any], manifest: Mapping[str, Any],
    open_pec_extension: Mapping[str, Any] | None = None,
) -> tuple[dict, Any]:
    system = str(summary.get("system", ""))
    _input_checked_summary(summary, system)
    asymptote_review = None
    if open_pec_extension is not None:
        summary = apply_open_pec_extension(summary, open_pec_extension)
    selection = build_state_selection_report(summary)
    if open_pec_extension is not None:
        asymptote_review = asymptote_review_from_extension(summary, open_pec_extension, selection)
    bridge = build_adaptive_stage3_plan(
        system_summary=summary, selection_report=selection, manifest=manifest,
        evidence_id="benchmark-scout:" + _digest(summary),
        asymptote_review=asymptote_review,
    )
    return selection, bridge


@dataclass(frozen=True)
class AcquisitionSettings:
    max_points: int = 100
    threads: int = 4
    memory_mb: int = 12000
    scalar_relativistic: str = "NONE"
    enable_provisional_points: bool = True
    compute_points: bool = True
    # D02 reoptimization of internally unstable HF references; 0 = legacy fail.
    stability_following_max_steps: int = 0
    # FAIL (legacy) or RECORD_AND_CONTINUE (quantified D03 warning, RHF kept).
    rhf_external_instability_policy: str = "FAIL"
    # Refine cleared Stage-3 PECs and assemble the single-basis electronic EA.
    assemble_electronic_ea: bool = False
    # Patch 38: split jobs mixing followed/unfollowed ROHF references.
    reference_branch_splitting: bool = True
    # Patch 39: frozen-core aug-cc-pV{Q,5}Z (+d-aug) CBS after the electronic EA.
    cbs_stage: bool = False
    cbs_memory_mb: int = 48000

    def __post_init__(self) -> None:
        if self.max_points < 0 or self.threads < 1 or self.memory_mb < 1:
            raise ValueError("Invalid point, thread or memory budget")
        if self.stability_following_max_steps < 0:
            raise ValueError("stability_following_max_steps must be non-negative")
        if self.scalar_relativistic not in ("NONE", "SFX2C1E"):
            raise ValueError("Unsupported scalar Hamiltonian")


def run_benchmark_acquisition(
    *, summary: Mapping[str, Any], manifest: Mapping[str, Any], output_dir: Path,
    settings: AcquisitionSettings = AcquisitionSettings(),
    point_runner: Callable[[Stage3ExecutionRequest, Stage3ExecutionSettings], Stage3PointResult] | None = None,
    allow_missing_checkpoints_for_tests: bool = False,
    open_pec_extension: Mapping[str, Any] | None = None,
) -> dict:
    """Advance one system automatically as far as available evidence permits.

    A provisional point never clears D04/D09.  The stage3 bridge remains the
    scientific authority for HIGH_ACCURACY release.  Provisional work only
    acquires data without claiming that release.
    """
    system = str(summary.get("system", ""))
    _input_checked_summary(summary, system)
    if system not in manifest.get("systems", {}):
        raise ValueError("Scout system is not in supplied manifest")
    manifest_system = manifest["systems"][system]
    if summary.get("atoms") is not None and tuple(summary["atoms"]) != tuple(manifest_system["atoms"]):
        raise ValueError("Scout atoms differ from supplied manifest")
    input_record = summary.get("computational_input") or {}
    for key in ("candidate_neutral_2S", "candidate_anion_2S"):
        if key in input_record and key in manifest_system and input_record[key] != manifest_system[key]:
            raise ValueError("Scout spin-sector coverage differs from supplied manifest")
    if input_record and "initial_R_angstrom" in input_record and "initial_R_angstrom" in manifest_system:
        if float(input_record["initial_R_angstrom"]) != float(manifest_system["initial_R_angstrom"]):
            raise ValueError("Scout initial geometry differs from supplied manifest")

    output_dir = Path(output_dir)
    selection, bridge = build_benchmark_acquisition_plan(
        summary=summary, manifest=manifest, open_pec_extension=open_pec_extension)
    manifest_fingerprint = _digest({"system": manifest["systems"][system], "protocol": manifest["protocol"]})
    source_fingerprint = (
        _digest(summary) if open_pec_extension is None
        else _digest({"scout": summary, "open_pec_extension": open_pec_extension})
    )
    identity = {"manifest_sha256": manifest_fingerprint, "scout_sha256": source_fingerprint}
    state_path = output_dir / "auto_state.json"
    if state_path.exists():
        previous = _load(state_path)
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("Existing acquisition uses a different manifest/scout summary; use a new output directory")

    _save(output_dir / "state_selection_report.json", selection)
    _save(output_dir / "stage3_plan.json", bridge.stage3_plan)
    jobs = tuple(bridge.stage3_plan["jobs"])
    open_state = {}
    for label, group in selection["charge_groups"].items():
        provisional = group.get("provisional_high_level_seed")
        reference = float(provisional["energy_hartree"]) if provisional else None
        open_state[label] = {
            "competition_status": group["competition_status"],
            "open_components": len(group["unresolved_components"]),
            "bracketed_candidates": len(group["bracketed_candidates"]),
            "resolved_open_components": len(group.get("resolved_open_components", ())),
            "open_diagnostics": [
                {"component_id": item.get("component_id"),
                 "spin_2s": item.get("spin_2s"),
                 "minimum_status": item.get("minimum_status"),
                 "observed_gap_to_bracketed_seed_ev_not_lower_bound": (
                     (float(item["observed_lowest_energy_hartree"]) - reference) * 27.211386245988
                     if reference is not None and item.get("observed_lowest_energy_hartree") is not None else None
                 )}
                for item in group["unresolved_components"]
            ],
        }
    stage3_purpose = _job_purpose(bridge)
    blocked = []
    requests: list[Stage3ExecutionRequest] = []
    checkpoint_hashes: dict[str, str] = {}

    if bridge.release_status is Stage3ReleaseStatus.RELEASED:
        candidate_ids = tuple(bridge.released_job_ids)
    elif settings.enable_provisional_points:
        # All bracketed Stage-2 candidates are eligible for provisional data
        # collection, with no energy-gap filtering, no pruning, no release.
        candidate_ids = tuple(str(job["job_id"]) for job in jobs)
    else:
        candidate_ids = ()

    for job_id in candidate_ids:
        try:
            reqs = build_stage3_execution_requests(
                stage3_plan=bridge.stage3_plan, manifest=manifest,
                authorized_job_ids=(job_id,),
            )
            for req in reqs:
                path = Path(req.source_checkpoint_path)
                if not allow_missing_checkpoints_for_tests:
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing Stage-2 source checkpoint: {path}")
                    if req.source_checkpoint_path not in checkpoint_hashes:
                        digest = sha256()
                        with path.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(chunk)
                        checkpoint_hashes[req.source_checkpoint_path] = digest.hexdigest()
            requests.extend(reqs)
        except (ValueError, TypeError, FileNotFoundError) as exc:
            blocked.append({"job_id": job_id, "reason": f"{type(exc).__name__}: {exc}"})

    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("Duplicate cross-job Stage-3 point request IDs")

    if settings.compute_points and requests and point_runner is None:
        from pyscf import lib
        lib.num_threads(settings.threads)

    stage3_settings = Stage3ExecutionSettings(
        max_memory_mb=settings.memory_mb,
        scalar_relativistic=settings.scalar_relativistic,
        artifact_dir=str(output_dir / "checkpoints"),
        stability_following_max_steps=settings.stability_following_max_steps,
        rhf_external_instability_policy=settings.rhf_external_instability_policy,
    )
    signature = _digest({"source": identity, "settings": asdict(stage3_settings), "purpose": stage3_purpose})
    point_dir = output_dir / "points"
    point_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    failures = []
    pending = []
    point_result_records: dict[str, dict] = {}
    ran = 0

    # Nothing gets silently skipped.  Each point is either cached, executed,
    # or explicitly pending, and the manifest fingerprint is verified on resume.
    def run_point(req: Stage3ExecutionRequest) -> None:
        nonlocal ran
        record_path = point_dir / (sha256(req.request_id.encode()).hexdigest() + ".json")
        fingerprint = _digest({"request": req.to_dict(), "batch": signature,
                               "source_checkpoint_sha256": checkpoint_hashes.get(req.source_checkpoint_path)})
        if record_path.exists():
            existing = _load(record_path)
            if existing.get("fingerprint") != fingerprint or existing.get("request_id") != req.request_id:
                raise ValueError(f"Stale/conflicting point artifact: {record_path}")
            status = existing["result"]["status"]
            point_result_records[req.request_id] = existing["result"]
            (completed if status == "COMPLETED" else failures).append(req.request_id)
            return
        if not settings.compute_points or ran >= settings.max_points:
            pending.append(req.request_id)
            return
        print(f"[AUTO STAGE-3] {req.system} q={req.charge:+d} 2S={req.spin_2s} "
              f"R={req.r_angstrom:.6f} A / {stage3_purpose}"
              + (f" / {req.reference_branch}" if req.reference_branch else ""), flush=True)
        (result,), attempts = execute_stage3_batch_with_retries(
            (req,), settings=stage3_settings, runner=point_runner,
        )
        _save(record_path, {
            "request_id": req.request_id, "fingerprint": fingerprint,
            "purpose": stage3_purpose, "request": req.to_dict(), "result": result.to_dict(),
            "execution_attempts": [attempt.to_dict() for attempt in attempts],
            "scientific_release": bridge.release_status is Stage3ReleaseStatus.RELEASED,
            "validation_reference_used": False,
        })
        point_result_records[req.request_id] = result.to_dict()
        ran += 1
        (completed if result.status is PointExecutionStatus.COMPLETED else failures).append(req.request_id)

    for req in requests:
        run_point(req)

    # Patch 38: a job mixing followed and unfollowed ROHF references is split
    # into two individually continuous reference branches (both computed at
    # every point); the parent is superseded.  Branch checkpoints are produced
    # in this run, so their (absent) source hashes stay out of the fingerprint.
    superseded: dict[str, str] = {}
    branch_jobs: list[dict] = []
    if settings.reference_branch_splitting and not pending:
        for parent in mixed_reference_jobs(requests, point_result_records):
            parent_reqs = [r for r in requests if r.job_id == parent]
            branch_reqs = build_branch_requests(parent_reqs, point_result_records, output_dir / "checkpoints")
            for req in branch_reqs:
                run_point(req)
            requests.extend(branch_reqs)
            superseded[parent] = SUPERSEDED_STATUS
            for suffix in (SEED_SUFFIX, FOLLOWED_SUFFIX):
                if any(r.job_id == parent + suffix for r in branch_reqs):
                    branch_jobs.append({"job_id": parent + suffix})
            seed_recs = [point_result_records.get(r.request_id, {}) for r in branch_reqs if r.job_id == parent + SEED_SUFFIX]
            fol_recs = [point_result_records.get(r.request_id, {}) for r in branch_reqs if r.job_id == parent + FOLLOWED_SUFFIX]
            if branches_identical(seed_recs, fol_recs):
                superseded[parent + FOLLOWED_SUFFIX] = MERGED_STATUS

    # Call existing high-level identity/PEC analysis automatically for each
    # completed job.  WEEKEND_* thresholds are explicitly a pilot policy;
    # never promote these diagnostics to a production identity decision.
    pec_reviews = {}
    from .adaptive.stage3_identity import resolve_high_level_identity_and_pec
    from .adaptive.stage3_axial_symmetry import AxialSymmetryPolicy
    from .workflow import WEEKEND_IDENTITY_THRESHOLDS, WEEKEND_BRANCH_THRESHOLDS
    for job in (*jobs, *branch_jobs):
        job_reqs = [r for r in requests if r.job_id == job["job_id"]]
        if not job_reqs:
            continue
        if job["job_id"] in superseded:
            pec_reviews[job["job_id"]] = {"status": superseded[job["job_id"]]}
            continue
        if any(r.request_id in pending for r in job_reqs):
            pec_reviews[job["job_id"]] = {"status": "WAITING_FOR_COMPLETE_POINTS"}
            continue
        # A failed point is never silently replaced by a sibling: the review
        # runs on the completed subset only, lists every excluded request and
        # is labelled PARTIAL, so it cannot be mistaken for full coverage.
        reviewed_reqs = [r for r in job_reqs if r.request_id in completed]
        excluded = [
            {"request_id": r.request_id, "r_angstrom": r.r_angstrom,
             "initialization_index": r.initialization_index,
             "status": point_result_records[r.request_id]["status"]}
            for r in job_reqs if r.request_id not in completed
        ]
        if not reviewed_reqs:
            pec_reviews[job["job_id"]] = {"status": "NO_COMPLETED_POINTS", "excluded_failed_points": excluded}
            continue
        job_results = [
            Stage3PointResult(**{**point_result_records[r.request_id],
                                 "status": PointExecutionStatus(point_result_records[r.request_id]["status"])})
            for r in reviewed_reqs
        ]
        try:
            resolution = resolve_high_level_identity_and_pec(
                requests=reviewed_reqs, results=job_results,
                identity_thresholds=WEEKEND_IDENTITY_THRESHOLDS,
                branch_thresholds=WEEKEND_BRANCH_THRESHOLDS,
                axial_symmetry_policy=AxialSymmetryPolicy(),
            )
            pec_reviews[job["job_id"]] = {
                "status": "DIAGNOSTIC_ONLY" if not excluded else "DIAGNOSTIC_ONLY_PARTIAL",
                "complete_point_coverage": not excluded,
                "excluded_failed_points": excluded,
                "initialization_review": resolution.initialization_review.status.value,
                "geometry_continuity_review": resolution.geometry_continuity_review.status.value,
                "pec": resolution.pec.to_dict(),
                "axial_symmetry_evidence": list(resolution.axial_symmetry_evidence),
                "axial_symmetry_status": "PILOT_DIAGNOSTIC_ONLY_NOT_SOC_MANIFOLD_CERTIFICATION",
                "threshold_policy": "WEEKEND_PILOT_NOT_PRODUCTION",
            }
        except Exception as exc:
            pec_reviews[job["job_id"]] = {
                "status": "UNRESOLVED", "reason": f"{type(exc).__name__}: {exc}",
                "excluded_failed_points": excluded}
    _save(output_dir / "stage3_pec_reviews.json", pec_reviews)

    electronic_ea = None
    if settings.assemble_electronic_ea and settings.compute_points and not pending:
        from .electronic_ea_stage import run_electronic_ea_stage
        electronic_ea = run_electronic_ea_stage(
            system=system, atoms=tuple(manifest_system["atoms"]),
            requests=[r for r in requests if r.job_id not in superseded],
            point_records=point_result_records, pec_reviews=pec_reviews,
            stage3_settings=stage3_settings, output_dir=output_dir / "electronic_ea",
            point_backend=point_runner,
        )
        electronic_ea["stage3_purpose"] = stage3_purpose

    cbs = None
    if electronic_ea is not None and settings.cbs_stage:
        from .cbs_stage import CBSStagePolicy, run_cbs_stage
        cbs = run_cbs_stage(
            electronic_ea=electronic_ea, requests=[r for r in requests if r.job_id not in superseded],
            point_records=point_result_records,
            refinement_dir=output_dir / "electronic_ea" / "refinement_points",
            stage3_settings=stage3_settings, output_dir=output_dir / "cbs",
            policy=CBSStagePolicy(max_memory_mb=settings.cbs_memory_mb), point_backend=point_runner)

    outcome = "UNRESOLVED"  # no complete adiabatic-EA assembler here yet
    state = {
        "system": system, **identity, "outcome": outcome,
        "run_stage": "STAGE3_POINT_EVIDENCE_ACQUISITION",
        "stage3_purpose": stage3_purpose,
        "stage3_release": bridge.release_status.value,
        "stage3_blocking_reasons": list(bridge.blocking_reasons),
        "open_pec_extension_applied": open_pec_extension is not None,
        "reference_branch_splits": {job: status for job, status in superseded.items()},
        # Wigner-Witmer labels that no single high-spin determinant represents
        # (e.g. 1Delta, open-shell singlets): not computed, an explicit gap.
        "mr_only_candidates_not_computed": [
            f"q={m.get('charge')} {m.get('label')}" for m in summary.get("mr_only_candidates", ())],
        "charge_groups": open_state,
        "candidate_job_count": len(jobs),
        "stage3_pec_reviews": {key: v["status"] for key, v in pec_reviews.items()},
        "stage3_identity_summary": {
            key: {
                "initialization_review": v.get("initialization_review", "UNAVAILABLE"),
                "geometry_continuity_review": v.get("geometry_continuity_review", "UNAVAILABLE"),
                "axial_symmetry_equivalent_comparisons": sum(
                    1 for item in v.get("axial_symmetry_evidence", [])
                    if item.get("status") == "EQUIVALENT"
                ),
                "axial_symmetry_unresolved_comparisons": sum(
                    1 for item in v.get("axial_symmetry_evidence", [])
                    if item.get("status") != "EQUIVALENT"
                ),
                "minimum_scout_status": (v.get("pec") or {}).get("minimum_scout", {}).get("status"),
                "complete_point_coverage": v.get("complete_point_coverage", False),
                "excluded_failed_point_count": len(v.get("excluded_failed_points", [])),
            }
            for key, v in pec_reviews.items()
        },
        "point_request_count": len(requests),
        "completed_point_count": len(completed), "failed_point_count": len(failures),
        "pending_point_count": len(pending),
        "completed_point_ids": completed, "failed_point_ids": failures,
        "failed_point_details": [
            {"request_id": request_id,
             "status": point_result_records[request_id]["status"],
             "error_type": point_result_records[request_id].get("error_type"),
             "error_message": point_result_records[request_id].get("error_message")}
            for request_id in failures
        ],
        "reference_warning_summary": _reference_warning_summary(requests, point_result_records),
        "pending_point_ids": pending, "blocked_jobs": blocked,
        "next_actions": [a.kind.value for a in bridge.adaptive_plan.actions],
        "electronic_ea": None if electronic_ea is None else {
            k: electronic_ea.get(k) for k in (
                "status", "verdict", "sign_at_this_level", "ea_lower_ev", "ea_upper_ev", "ea_central_ev",
                "selected_neutral_job", "selected_anion_job", "reason", "basis", "stage3_purpose")
        } | {"high_level_competition": {
            label: {k: v for k, v in c.items() if k in ("status", "selected_job_id", "lowest_job_id")}
            for label, c in electronic_ea["high_level_competition"].items()}},
        "cbs": None if cbs is None else {
            k: cbs.get(k) for k in ("status", "reason", "ea_cbs_ev", "ea_cbs_lower_ev", "ea_cbs_upper_ev",
                                    "fixed_basis_ea_ev", "competitor_ranking_at_cbs", "verdict")}
        | {"diffuse_correction_ev": (cbs.get("cbs") or {}).get("diffuse_correction_ev"),
           "model_sensitivity_ev": (cbs.get("cbs") or {}).get("cbs_model_sensitivity_bound_ev")},
        "ground_state_validated": False, "ea_computed": False,
        "state_competition_cleared": False,
        "automatic_pruning_performed": False,
        "validation_metadata_used_in_computation": False,
        "reason": "Point evidence is not a closed state competition, PEC/ZPE/CBS correction budget, or certified EA.",
    }
    _save(state_path, state)
    return state
