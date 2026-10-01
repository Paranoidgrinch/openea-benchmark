"""Bridge adaptive D04/D09 actions into the existing Stage-3 job planner.

This module deliberately keeps the existing ``stage3_plan.build_stage3_plan``
as the authority for Stage-2 checkpoint provenance, local geometry grids and
ROHF/CC job payloads.  The adaptive layer only decides whether those jobs are
scientifically requested now, deferred behind prerequisites, or blocked by a
contract/provenance problem.

Scientific invariants
---------------------
* No scout/DFT splitting authorizes pruning.
* Every adaptive Stage-3 seed must map one-to-one to an existing Stage-3 job.
* Every existing Stage-3 candidate job must be represented by the adaptive seed
  queue before a high-accuracy comparison can be released.
* D09/other prerequisite actions precede D04 high-accuracy comparison.
* Unresolved checkpoint provenance blocks release; it is never guessed.
* Experimental/validation metadata are not consulted by this bridge.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose
from typing import Mapping, Any

from openea_benchmark.stage3_plan import build_stage3_plan

from .model import Review
from .pec_selection_bridge import StateSelectionBridgeResult, bridge_state_selection_report
from .planner import AdaptivePlan, ActionKind, Stage3SeedCandidate, plan_from_state_selection_bridge


class Stage3ReleaseStatus(str, Enum):
    """Scientific release state for the existing Stage-3 job payloads."""

    NOT_REQUESTED = "NOT_REQUESTED"
    DEFERRED_PREREQUISITE = "DEFERRED_PREREQUISITE"
    BLOCKED_CONTRACT = "BLOCKED_CONTRACT"
    BLOCKED_PROVENANCE = "BLOCKED_PROVENANCE"
    RELEASED = "RELEASED"


@dataclass(frozen=True)
class AdaptiveStage3BridgeResult:
    selection_bridge: StateSelectionBridgeResult
    adaptive_plan: AdaptivePlan
    stage3_plan: Mapping[str, Any]
    release_status: Stage3ReleaseStatus
    ordered_job_ids: tuple[str, ...]
    released_job_ids: tuple[str, ...]
    blocked_job_ids: tuple[str, ...]
    blocking_reasons: tuple[str, ...]

    @property
    def high_accuracy_jobs_released(self) -> bool:
        return self.release_status is Stage3ReleaseStatus.RELEASED


def _assert_legacy_plan_invariants(plan: Mapping[str, Any]) -> None:
    if bool(plan.get("automatic_pruning_performed")):
        raise ValueError("Existing Stage-3 plan reports automatic pruning")
    if bool(plan.get("validation_metadata_used_in_planning")):
        raise ValueError("Validation metadata must not influence Stage-3 planning")
    if bool(plan.get("high_level_calculations_performed")):
        raise ValueError("Stage-3 planning bridge cannot consume executed calculations")


def _seed_matches_job(seed: Stage3SeedCandidate, job: Mapping[str, Any]) -> bool:
    if str(job.get("charge_group")) != seed.group_ref:
        return False
    if str(job.get("component_id")) != seed.component_id:
        return False
    try:
        return isclose(
            float(job["dft_center_r_angstrom"]),
            float(seed.r_angstrom),
            abs_tol=1e-8,
            rel_tol=0.0,
        ) and isclose(
            float(job["dft_center_energy_hartree"]),
            float(seed.sampled_energy_hartree),
            abs_tol=1e-9,
            rel_tol=0.0,
        )
    except (KeyError, TypeError, ValueError):
        return False


def _match_seed_jobs(
    adaptive_plan: AdaptivePlan,
    stage3_plan: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ordered job IDs plus fail-closed contract diagnostics."""

    jobs = tuple(stage3_plan.get("jobs", ()))
    if not all(isinstance(job, Mapping) for job in jobs):
        return (), ("STAGE3_JOBS_NOT_MAPPINGS",)

    ordered: list[str] = []
    reasons: list[str] = []
    matched_indices: set[int] = set()

    for seed in adaptive_plan.stage3_seeds.ready:
        matches = [
            (index, job)
            for index, job in enumerate(jobs)
            if index not in matched_indices and _seed_matches_job(seed, job)
        ]
        if len(matches) != 1:
            reasons.append(
                f"ADAPTIVE_SEED_JOB_MATCH_{len(matches)}:{seed.candidate_ref}"
            )
            continue
        index, job = matches[0]
        job_id = str(job.get("job_id") or "")
        if not job_id:
            reasons.append(f"STAGE3_JOB_ID_MISSING:{seed.candidate_ref}")
            continue
        matched_indices.add(index)
        ordered.append(job_id)

    # A legacy Stage-3 candidate that is absent from the adaptive queue would
    # silently bypass the adaptive scientific decision layer.  Fail closed.
    if len(matched_indices) != len(jobs):
        for index, job in enumerate(jobs):
            if index not in matched_indices:
                reasons.append(
                    "LEGACY_JOB_WITHOUT_ADAPTIVE_SEED:"
                    + str(job.get("job_id") or f"index={index}")
                )

    if adaptive_plan.stage3_seeds.blocked_candidate_refs:
        reasons.extend(
            f"ADAPTIVE_SEED_BLOCKED:{ref}"
            for ref in adaptive_plan.stage3_seeds.blocked_candidate_refs
        )

    return tuple(ordered), tuple(dict.fromkeys(reasons))


def _high_accuracy_action_index(plan: AdaptivePlan) -> int | None:
    indexes = [
        index
        for index, action in enumerate(plan.actions)
        if action.kind is ActionKind.HIGH_ACCURACY_CANDIDATE_COMPARISON
    ]
    if not indexes:
        return None
    if len(indexes) != 1:
        raise ValueError("Adaptive plan contains multiple high-accuracy comparison actions")
    return indexes[0]


def _unresolved_provenance_job_ids(
    stage3_plan: Mapping[str, Any],
    ordered_job_ids: tuple[str, ...],
) -> tuple[str, ...]:
    by_id = {str(job.get("job_id")): job for job in stage3_plan.get("jobs", ())}
    unresolved: list[str] = []
    for job_id in ordered_job_ids:
        job = by_id[job_id]
        source = job.get("source_provenance") or {}
        if source.get("link_status") == "UNRESOLVED":
            unresolved.append(job_id)
    return tuple(unresolved)


def bridge_adaptive_plan_to_stage3(
    *,
    adaptive_plan: AdaptivePlan,
    stage3_plan: Mapping[str, Any],
    selection_bridge: StateSelectionBridgeResult,
) -> AdaptiveStage3BridgeResult:
    """Validate and release/defer existing Stage-3 jobs under adaptive control."""

    _assert_legacy_plan_invariants(stage3_plan)
    ordered_job_ids, contract_reasons = _match_seed_jobs(adaptive_plan, stage3_plan)

    if contract_reasons:
        all_job_ids = tuple(str(job.get("job_id") or "") for job in stage3_plan.get("jobs", ()))
        return AdaptiveStage3BridgeResult(
            selection_bridge,
            adaptive_plan,
            stage3_plan,
            Stage3ReleaseStatus.BLOCKED_CONTRACT,
            ordered_job_ids,
            (),
            tuple(x for x in all_job_ids if x),
            contract_reasons,
        )

    high_accuracy_index = _high_accuracy_action_index(adaptive_plan)
    if high_accuracy_index is None:
        return AdaptiveStage3BridgeResult(
            selection_bridge,
            adaptive_plan,
            stage3_plan,
            Stage3ReleaseStatus.NOT_REQUESTED,
            ordered_job_ids,
            (),
            (),
            (),
        )

    action = adaptive_plan.actions[high_accuracy_index]
    ordered_refs = tuple(seed.candidate_ref for seed in adaptive_plan.stage3_seeds.ready)
    if action.target_refs != ordered_refs:
        return AdaptiveStage3BridgeResult(
            selection_bridge,
            adaptive_plan,
            stage3_plan,
            Stage3ReleaseStatus.BLOCKED_CONTRACT,
            ordered_job_ids,
            (),
            ordered_job_ids,
            ("HIGH_ACCURACY_TARGETS_DO_NOT_MATCH_COMPLETE_SEED_QUEUE",),
        )

    # The adaptive planner is already deterministically ordered.  Anything
    # before the high-accuracy comparison is a scientific prerequisite.
    prerequisites = adaptive_plan.actions[:high_accuracy_index]
    if prerequisites:
        return AdaptiveStage3BridgeResult(
            selection_bridge,
            adaptive_plan,
            stage3_plan,
            Stage3ReleaseStatus.DEFERRED_PREREQUISITE,
            ordered_job_ids,
            (),
            (),
            tuple(
                f"PREREQUISITE:{item.kind.value}:{item.route}"
                for item in prerequisites
            ),
        )

    unresolved = _unresolved_provenance_job_ids(stage3_plan, ordered_job_ids)
    if unresolved:
        return AdaptiveStage3BridgeResult(
            selection_bridge,
            adaptive_plan,
            stage3_plan,
            Stage3ReleaseStatus.BLOCKED_PROVENANCE,
            ordered_job_ids,
            (),
            unresolved,
            tuple(f"UNRESOLVED_CHECKPOINT_PROVENANCE:{job_id}" for job_id in unresolved),
        )

    return AdaptiveStage3BridgeResult(
        selection_bridge,
        adaptive_plan,
        stage3_plan,
        Stage3ReleaseStatus.RELEASED,
        ordered_job_ids,
        ordered_job_ids,
        (),
        (),
    )


def build_adaptive_stage3_plan(
    *,
    system_summary: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    evidence_id: str,
    higher_level_competition_review: Review | None = None,
    asymptote_review: Review | None = None,
) -> AdaptiveStage3BridgeResult:
    """End-to-end planning bridge from state-selection evidence to Stage-3 jobs.

    This function performs no electronic calculation.  It composes the
    existing conservative state-selection report, adaptive D04/D09 planner and
    existing Stage-3 provenance/job planner into one auditable decision record.
    """

    selection_bridge = bridge_state_selection_report(
        selection_report,
        evidence_id=evidence_id,
        higher_level_competition_review=higher_level_competition_review,
        asymptote_review=asymptote_review,
    )
    adaptive_plan = plan_from_state_selection_bridge(selection_bridge)
    legacy_plan = build_stage3_plan(
        system_summary=system_summary,
        selection_report=selection_report,
        manifest=manifest,
    )
    return bridge_adaptive_plan_to_stage3(
        adaptive_plan=adaptive_plan,
        stage3_plan=legacy_plan,
        selection_bridge=selection_bridge,
    )
