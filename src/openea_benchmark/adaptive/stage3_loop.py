"""Autonomous fail-closed orchestration of Stage-3 PEC refinement.

This module closes the control loop between the existing Stage-3 execution,
identity/continuity resolution, high-level PEC assembly and adaptive geometry
refinement layers.  It deliberately does *not* assign a ground state, compute
an electron affinity, or authorize pruning.

Scientific invariants
---------------------
* Every newly proposed geometry is executed as an explicit Stage3ExecutionRequest.
* Every completed batch is re-audited by the existing high-level identity and
  continuity resolver before another refinement decision is made.
* Numerical execution failure is terminal for this loop iteration and is never
  relabelled as scientific convergence.
* Unresolved identity/continuity is terminal and reported separately from a
  numerical failure.
* Hitting the configured round limit is not convergence.
* Only an explicit BRACKET_TARGET_MET decision is reported as CONVERGED.
* All accumulated request/result provenance is retained.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Any, Callable, Sequence

from .stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3ExecutionSettings,
    Stage3PointResult,
    run_stage3_point,
)
from .stage3_refinement import (
    RefinementAction,
    Stage3RefinementPlan,
    Stage3RefinementSettings,
    build_stage3_refinement_requests,
    plan_stage3_pec_refinement,
)


class Stage3LoopStatus(str, Enum):
    CONVERGED = "CONVERGED"
    SCIENTIFICALLY_UNRESOLVED = "SCIENTIFICALLY_UNRESOLVED"
    EXECUTION_BLOCKED = "EXECUTION_BLOCKED"
    ROUND_LIMIT_REACHED = "ROUND_LIMIT_REACHED"


@dataclass(frozen=True)
class Stage3LoopRound:
    evaluation_index: int
    completed_refinement_rounds: int
    pec_status: str
    minimum_status: str
    refinement_action: str
    proposed_r_angstrom: tuple[float, ...]
    new_request_ids: tuple[str, ...]
    new_result_statuses: tuple[str, ...]
    initialization_review_status: str
    geometry_continuity_review_status: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage3LoopResult:
    status: Stage3LoopStatus
    job_id: str
    rounds: tuple[Stage3LoopRound, ...]
    requests: tuple[Stage3ExecutionRequest, ...]
    results: tuple[Stage3PointResult, ...]
    final_pec: Any
    final_refinement_plan: Stage3RefinementPlan
    rationale: str
    is_production_ea: bool = False
    ground_state_assigned: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("Stage-3 loop result requires job_id")
        if not self.rationale.strip():
            raise ValueError("Stage-3 loop result requires rationale")
        if self.is_production_ea or self.ground_state_assigned or self.authorizes_pruning:
            raise ValueError("Stage-3 refinement loop cannot assign EA/ground state or authorize pruning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "job_id": self.job_id,
            "rounds": [item.to_dict() for item in self.rounds],
            "requests": [item.to_dict() for item in self.requests],
            "results": [item.to_dict() for item in self.results],
            "final_pec": self.final_pec.to_dict() if hasattr(self.final_pec, "to_dict") else self.final_pec,
            "final_refinement_plan": self.final_refinement_plan.to_dict(),
            "rationale": self.rationale,
            "is_production_ea": self.is_production_ea,
            "ground_state_assigned": self.ground_state_assigned,
            "authorizes_pruning": self.authorizes_pruning,
        }


def _validate_initial_contract(
    requests: Sequence[Stage3ExecutionRequest],
    results: Sequence[Stage3PointResult],
) -> str:
    if not requests:
        raise ValueError("Stage-3 refinement loop requires initial requests")
    request_ids = [item.request_id for item in requests]
    result_ids = [item.request_id for item in results]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Initial Stage-3 requests contain duplicate IDs")
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Initial Stage-3 results contain duplicate IDs")
    if set(request_ids) != set(result_ids):
        raise ValueError("Initial Stage-3 request/result IDs do not match")
    job_ids = {item.job_id for item in requests}
    if len(job_ids) != 1:
        raise ValueError("Stage-3 refinement loop accepts exactly one job")
    job_id = next(iter(job_ids))
    if any(item.job_id != job_id for item in results):
        raise ValueError("Initial Stage-3 results belong to another job")
    return job_id


def _resolution_status(resolution: Any, name: str) -> str:
    value = getattr(resolution, name, None)
    status = getattr(value, "status", None)
    return getattr(status, "value", str(status)) if status is not None else "UNKNOWN"


def _resolve(
    *,
    requests: Sequence[Stage3ExecutionRequest],
    results: Sequence[Stage3PointResult],
    identity_thresholds: Any,
    branch_thresholds: Any,
    audit_settings: Any,
    duplicate_energy_tolerance_hartree: float,
    resolver: Callable[..., Any] | None,
) -> Any:
    if resolver is None:
        from .stage3_identity import resolve_high_level_identity_and_pec

        resolver = resolve_high_level_identity_and_pec
    return resolver(
        requests=requests,
        results=results,
        identity_thresholds=identity_thresholds,
        branch_thresholds=branch_thresholds,
        audit_settings=audit_settings,
        duplicate_energy_tolerance_hartree=duplicate_energy_tolerance_hartree,
    )


def run_stage3_refinement_loop(
    *,
    initial_requests: Sequence[Stage3ExecutionRequest],
    initial_results: Sequence[Stage3PointResult],
    refinement_settings: Stage3RefinementSettings,
    identity_thresholds: Any,
    branch_thresholds: Any,
    max_refinement_rounds: int,
    execution_settings: Stage3ExecutionSettings | None = None,
    audit_settings: Any = None,
    duplicate_energy_tolerance_hartree: float = 1.0e-7,
    runner: Callable[[Stage3ExecutionRequest, Stage3ExecutionSettings], Stage3PointResult] | None = None,
    resolver: Callable[..., Any] | None = None,
) -> Stage3LoopResult:
    """Execute and re-audit adaptive Stage-3 PEC refinement until a terminal gate.

    ``max_refinement_rounds`` counts *new calculation batches*.  The initial
    request/result set is evaluated as round zero and does not consume this
    budget.  ``resolver`` and ``runner`` are injection seams for tests and
    future alternative open-source backends.
    """
    if not isinstance(max_refinement_rounds, int) or max_refinement_rounds < 0:
        raise ValueError("max_refinement_rounds must be an integer >= 0")
    tol = float(duplicate_energy_tolerance_hartree)
    if not isfinite(tol) or tol < 0.0:
        raise ValueError("duplicate_energy_tolerance_hartree must be finite and non-negative")

    job_id = _validate_initial_contract(initial_requests, initial_results)
    requests = list(initial_requests)
    results = list(initial_results)
    rounds: list[Stage3LoopRound] = []
    completed_refinement_rounds = 0
    settings = execution_settings or Stage3ExecutionSettings()

    while True:
        resolution = _resolve(
            requests=requests,
            results=results,
            identity_thresholds=identity_thresholds,
            branch_thresholds=branch_thresholds,
            audit_settings=audit_settings,
            duplicate_energy_tolerance_hartree=tol,
            resolver=resolver,
        )
        pec = resolution.pec
        if pec.job_id != job_id:
            raise ValueError("Identity resolver returned PEC for another Stage-3 job")

        plan = plan_stage3_pec_refinement(pec, settings=refinement_settings)
        proposed = tuple(float(item.r_angstrom) for item in plan.proposed_geometries)
        init_status = _resolution_status(resolution, "initialization_review")
        continuity_status = _resolution_status(resolution, "geometry_continuity_review")

        if plan.action is RefinementAction.BRACKET_TARGET_MET:
            rounds.append(Stage3LoopRound(
                evaluation_index=len(rounds),
                completed_refinement_rounds=completed_refinement_rounds,
                pec_status=pec.status.value,
                minimum_status=pec.minimum_scout.status.value,
                refinement_action=plan.action.value,
                proposed_r_angstrom=proposed,
                new_request_ids=(),
                new_result_statuses=(),
                initialization_review_status=init_status,
                geometry_continuity_review_status=continuity_status,
                rationale=plan.rationale,
            ))
            return Stage3LoopResult(
                status=Stage3LoopStatus.CONVERGED,
                job_id=job_id,
                rounds=tuple(rounds),
                requests=tuple(requests),
                results=tuple(results),
                final_pec=pec,
                final_refinement_plan=plan,
                rationale="Explicit high-level PEC bracket-width target is met after identity/continuity review",
            )

        if plan.action in (RefinementAction.WAIT_FOR_VALID_PEC, RefinementAction.UNRESOLVED):
            rounds.append(Stage3LoopRound(
                evaluation_index=len(rounds),
                completed_refinement_rounds=completed_refinement_rounds,
                pec_status=pec.status.value,
                minimum_status=pec.minimum_scout.status.value,
                refinement_action=plan.action.value,
                proposed_r_angstrom=proposed,
                new_request_ids=(),
                new_result_statuses=(),
                initialization_review_status=init_status,
                geometry_continuity_review_status=continuity_status,
                rationale=plan.rationale,
            ))
            return Stage3LoopResult(
                status=Stage3LoopStatus.SCIENTIFICALLY_UNRESOLVED,
                job_id=job_id,
                rounds=tuple(rounds),
                requests=tuple(requests),
                results=tuple(results),
                final_pec=pec,
                final_refinement_plan=plan,
                rationale="High-level PEC cannot be refined further without resolving a scientific identity/continuity or geometry ambiguity",
            )

        if completed_refinement_rounds >= max_refinement_rounds:
            rounds.append(Stage3LoopRound(
                evaluation_index=len(rounds),
                completed_refinement_rounds=completed_refinement_rounds,
                pec_status=pec.status.value,
                minimum_status=pec.minimum_scout.status.value,
                refinement_action=plan.action.value,
                proposed_r_angstrom=proposed,
                new_request_ids=(),
                new_result_statuses=(),
                initialization_review_status=init_status,
                geometry_continuity_review_status=continuity_status,
                rationale="Refinement remains actionable but the explicit round limit has been reached",
            ))
            return Stage3LoopResult(
                status=Stage3LoopStatus.ROUND_LIMIT_REACHED,
                job_id=job_id,
                rounds=tuple(rounds),
                requests=tuple(requests),
                results=tuple(results),
                final_pec=pec,
                final_refinement_plan=plan,
                rationale="Explicit refinement-round limit reached before the high-level bracket target was met",
            )

        refinement_round = completed_refinement_rounds + 1
        new_requests = build_stage3_refinement_requests(
            plan=plan,
            pec=pec,
            prior_requests=requests,
            prior_results=results,
            refinement_round=refinement_round,
        )
        if not new_requests:
            rounds.append(Stage3LoopRound(
                evaluation_index=len(rounds),
                completed_refinement_rounds=completed_refinement_rounds,
                pec_status=pec.status.value,
                minimum_status=pec.minimum_scout.status.value,
                refinement_action=plan.action.value,
                proposed_r_angstrom=proposed,
                new_request_ids=(),
                new_result_statuses=(),
                initialization_review_status=init_status,
                geometry_continuity_review_status=continuity_status,
                rationale="Actionable refinement plan produced no executable requests",
            ))
            return Stage3LoopResult(
                status=Stage3LoopStatus.SCIENTIFICALLY_UNRESOLVED,
                job_id=job_id,
                rounds=tuple(rounds),
                requests=tuple(requests),
                results=tuple(results),
                final_pec=pec,
                final_refinement_plan=plan,
                rationale="Adaptive refinement produced no new executable geometries under the configured separation policy",
            )

        new_results = tuple(
            run_stage3_point(item, settings=settings, runner=runner)
            for item in new_requests
        )
        rounds.append(Stage3LoopRound(
            evaluation_index=len(rounds),
            completed_refinement_rounds=refinement_round,
            pec_status=pec.status.value,
            minimum_status=pec.minimum_scout.status.value,
            refinement_action=plan.action.value,
            proposed_r_angstrom=proposed,
            new_request_ids=tuple(item.request_id for item in new_requests),
            new_result_statuses=tuple(item.status.value for item in new_results),
            initialization_review_status=init_status,
            geometry_continuity_review_status=continuity_status,
            rationale=plan.rationale,
        ))

        requests.extend(new_requests)
        results.extend(new_results)
        completed_refinement_rounds = refinement_round

        if any(item.status is not PointExecutionStatus.COMPLETED for item in new_results):
            # Re-resolution is deliberately not attempted after a failed batch:
            # the failure itself is the terminal evidence for this orchestration run.
            return Stage3LoopResult(
                status=Stage3LoopStatus.EXECUTION_BLOCKED,
                job_id=job_id,
                rounds=tuple(rounds),
                requests=tuple(requests),
                results=tuple(results),
                final_pec=pec,
                final_refinement_plan=plan,
                rationale="At least one newly requested Stage-3 point did not complete successfully",
            )
