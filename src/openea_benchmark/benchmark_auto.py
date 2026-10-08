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


def _job_purpose(bridge: Any) -> str:
    if bridge.release_status is Stage3ReleaseStatus.RELEASED:
        return "RELEASED_HIGH_LEVEL_EVIDENCE"
    return "PROVISIONAL_EVIDENCE_ONLY_NOT_SCIENTIFIC_RELEASE"


def build_benchmark_acquisition_plan(*, summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[dict, Any]:
    system = str(summary.get("system", ""))
    _input_checked_summary(summary, system)
    selection = build_state_selection_report(summary)
    bridge = build_adaptive_stage3_plan(
        system_summary=summary, selection_report=selection, manifest=manifest,
        evidence_id="benchmark-scout:" + _digest(summary),
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

    def __post_init__(self) -> None:
        if self.max_points < 0 or self.threads < 1 or self.memory_mb < 1:
            raise ValueError("Invalid point, thread or memory budget")
        if self.scalar_relativistic not in ("NONE", "SFX2C1E"):
            raise ValueError("Unsupported scalar Hamiltonian")


def run_benchmark_acquisition(
    *, summary: Mapping[str, Any], manifest: Mapping[str, Any], output_dir: Path,
    settings: AcquisitionSettings = AcquisitionSettings(),
    point_runner: Callable[[Stage3ExecutionRequest, Stage3ExecutionSettings], Stage3PointResult] | None = None,
    allow_missing_checkpoints_for_tests: bool = False,
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
    selection, bridge = build_benchmark_acquisition_plan(summary=summary, manifest=manifest)
    manifest_fingerprint = _digest({"system": manifest["systems"][system], "protocol": manifest["protocol"]})
    source_fingerprint = _digest(summary)
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
    for req in requests:
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
            continue
        if not settings.compute_points or ran >= settings.max_points:
            pending.append(req.request_id)
            continue
        print(f"[AUTO STAGE-3] {req.system} q={req.charge:+d} 2S={req.spin_2s} "
              f"R={req.r_angstrom:.6f} A / {stage3_purpose}", flush=True)
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

    # Call existing high-level identity/PEC analysis automatically for each
    # completed job.  WEEKEND_* thresholds are explicitly a pilot policy;
    # never promote these diagnostics to a production identity decision.
    pec_reviews = {}
    from .adaptive.stage3_identity import resolve_high_level_identity_and_pec
    from .workflow import WEEKEND_IDENTITY_THRESHOLDS, WEEKEND_BRANCH_THRESHOLDS
    for job in jobs:
        job_reqs = [r for r in requests if r.job_id == job["job_id"]]
        if not job_reqs:
            continue
        if not all(r.request_id in completed for r in job_reqs):
            pec_reviews[job["job_id"]] = {"status": "WAITING_FOR_COMPLETE_POINTS"}
            continue
        job_results = [
            Stage3PointResult(**{**point_result_records[r.request_id],
                                 "status": PointExecutionStatus(point_result_records[r.request_id]["status"])})
            for r in job_reqs
        ]
        try:
            resolution = resolve_high_level_identity_and_pec(
                requests=job_reqs, results=job_results,
                identity_thresholds=WEEKEND_IDENTITY_THRESHOLDS,
                branch_thresholds=WEEKEND_BRANCH_THRESHOLDS,
            )
            pec_reviews[job["job_id"]] = {
                "status": "DIAGNOSTIC_ONLY",
                "initialization_review": resolution.initialization_review.status.value,
                "geometry_continuity_review": resolution.geometry_continuity_review.status.value,
                "pec": resolution.pec.to_dict(),
                "threshold_policy": "WEEKEND_PILOT_NOT_PRODUCTION",
            }
        except Exception as exc:
            pec_reviews[job["job_id"]] = {
                "status": "UNRESOLVED", "reason": f"{type(exc).__name__}: {exc}"}
    _save(output_dir / "stage3_pec_reviews.json", pec_reviews)

    outcome = "UNRESOLVED"  # no complete adiabatic-EA assembler here yet
    state = {
        "system": system, **identity, "outcome": outcome,
        "run_stage": "STAGE3_POINT_EVIDENCE_ACQUISITION",
        "stage3_purpose": stage3_purpose,
        "stage3_release": bridge.release_status.value,
        "stage3_blocking_reasons": list(bridge.blocking_reasons),
        "charge_groups": open_state,
        "candidate_job_count": len(jobs),
        "stage3_pec_reviews": {key: v["status"] for key, v in pec_reviews.items()},
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
        "pending_point_ids": pending, "blocked_jobs": blocked,
        "next_actions": [a.kind.value for a in bridge.adaptive_plan.actions],
        "ground_state_validated": False, "ea_computed": False,
        "state_competition_cleared": False,
        "automatic_pruning_performed": False,
        "validation_metadata_used_in_computation": False,
        "reason": "Point evidence is not a closed state competition, PEC/ZPE/CBS correction budget, or certified EA.",
    }
    _save(state_path, state)
    return state
