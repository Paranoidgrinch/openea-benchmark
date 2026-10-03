from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceAction,
    BasisConvergenceSettings,
    DiffuseConvergenceAssessment,
    ElectronicEABasisPoint,
    assess_diffuse_convergence,
)


class AdaptiveDiffuseStatus(str, Enum):
    DIFFUSE_CLEARED = "DIFFUSE_CLEARED"
    DIFFUSE_LIMIT_REACHED = "DIFFUSE_LIMIT_REACHED"
    EXECUTION_BLOCKED = "EXECUTION_BLOCKED"
    POLICY_BLOCKED = "POLICY_BLOCKED"


@dataclass(frozen=True)
class AdaptiveDiffuseIteration:
    iteration_index: int
    evaluated_augmentation_levels: tuple[int, ...]
    assessment: DiffuseConvergenceAssessment
    requested_next_augmentation_level: int | None


@dataclass(frozen=True)
class AdaptiveDiffuseResult:
    status: AdaptiveDiffuseStatus
    points: tuple[ElectronicEABasisPoint, ...]
    iterations: tuple[AdaptiveDiffuseIteration, ...]
    final_assessment: DiffuseConvergenceAssessment | None
    execution_error_type: str | None
    execution_error_message: str | None
    is_production_ea: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if self.is_production_ea or self.authorizes_pruning:
            raise ValueError(
                "adaptive diffuse runner cannot claim production EA or authorize pruning"
            )


BasisPointEvaluator = Callable[[int], ElectronicEABasisPoint]


def correlation_consistent_diffuse_basis_name(
    cardinal_number: int,
    *,
    augmentation_level: int,
) -> str:
    labels = {2: "d", 3: "t", 4: "q", 5: "5", 6: "6"}
    prefixes = {0: "", 1: "aug-", 2: "d-aug-", 3: "t-aug-"}
    if cardinal_number not in labels:
        raise ValueError("validation resolver supports cardinal numbers 2..6")
    if augmentation_level not in prefixes:
        raise ValueError("validation resolver supports augmentation levels 0..3")
    return f"{prefixes[augmentation_level]}cc-pv{labels[cardinal_number]}z"


def _ordered_unique(points):
    by_level = {}
    for point in points:
        if point.augmentation_level in by_level:
            raise ValueError(
                f"duplicate augmentation point level={point.augmentation_level}"
            )
        by_level[point.augmentation_level] = point
    return tuple(by_level[x] for x in sorted(by_level))


def _requested_level(assessment: DiffuseConvergenceAssessment) -> int | None:
    if assessment.action is BasisConvergenceAction.COMPUTE_DOUBLE_AUGMENTED:
        return 2
    if assessment.action is BasisConvergenceAction.COMPUTE_MORE_DIFFUSE:
        if assessment.highest_augmentation_level is None:
            return None
        return assessment.highest_augmentation_level + 1
    return None


def run_adaptive_diffuse_series(
    *,
    cardinal_number: int,
    initial_augmentation_levels: tuple[int, ...],
    maximum_augmentation_level: int,
    convergence_settings: BasisConvergenceSettings,
    evaluate_point: BasisPointEvaluator,
) -> AdaptiveDiffuseResult:
    if len(initial_augmentation_levels) < 2:
        raise ValueError("initial_augmentation_levels must contain at least two levels")
    if tuple(sorted(initial_augmentation_levels)) != initial_augmentation_levels:
        raise ValueError("initial augmentation levels must be sorted")
    if len(set(initial_augmentation_levels)) != len(initial_augmentation_levels):
        raise ValueError("initial augmentation levels must be unique")
    if maximum_augmentation_level < max(initial_augmentation_levels):
        raise ValueError("maximum_augmentation_level is below the initial series")

    points = []
    iterations = []

    def execute(level):
        try:
            point = evaluate_point(level)
        except Exception as exc:
            return None, exc
        if point.cardinal_number != cardinal_number:
            return None, ValueError(
                f"evaluator returned X={point.cardinal_number}; expected X={cardinal_number}"
            )
        if point.augmentation_level != level:
            return None, ValueError(
                f"evaluator returned augmentation level {point.augmentation_level}; "
                f"expected {level}"
            )
        points.append(point)
        return point, None

    for level in initial_augmentation_levels:
        _, error = execute(level)
        if error is not None:
            return AdaptiveDiffuseResult(
                status=AdaptiveDiffuseStatus.EXECUTION_BLOCKED,
                points=_ordered_unique(points),
                iterations=tuple(iterations),
                final_assessment=None,
                execution_error_type=type(error).__name__,
                execution_error_message=str(error),
            )

    iteration_index = 0
    while True:
        ordered = _ordered_unique(points)
        try:
            assessment = assess_diffuse_convergence(
                ordered,
                cardinal_number=cardinal_number,
                settings=convergence_settings,
            )
        except Exception as exc:
            return AdaptiveDiffuseResult(
                status=AdaptiveDiffuseStatus.POLICY_BLOCKED,
                points=ordered,
                iterations=tuple(iterations),
                final_assessment=None,
                execution_error_type=type(exc).__name__,
                execution_error_message=str(exc),
            )

        requested = _requested_level(assessment)
        iterations.append(
            AdaptiveDiffuseIteration(
                iteration_index=iteration_index,
                evaluated_augmentation_levels=tuple(
                    p.augmentation_level for p in ordered
                ),
                assessment=assessment,
                requested_next_augmentation_level=requested,
            )
        )

        if assessment.status.value == "CLEARED":
            return AdaptiveDiffuseResult(
                status=AdaptiveDiffuseStatus.DIFFUSE_CLEARED,
                points=ordered,
                iterations=tuple(iterations),
                final_assessment=assessment,
                execution_error_type=None,
                execution_error_message=None,
            )

        if requested is None:
            return AdaptiveDiffuseResult(
                status=AdaptiveDiffuseStatus.POLICY_BLOCKED,
                points=ordered,
                iterations=tuple(iterations),
                final_assessment=assessment,
                execution_error_type=None,
                execution_error_message=(
                    "Diffuse convergence is not cleared and no next augmentation "
                    "level was identified"
                ),
            )

        if requested > maximum_augmentation_level:
            return AdaptiveDiffuseResult(
                status=AdaptiveDiffuseStatus.DIFFUSE_LIMIT_REACHED,
                points=ordered,
                iterations=tuple(iterations),
                final_assessment=assessment,
                execution_error_type=None,
                execution_error_message=None,
            )

        if requested in {p.augmentation_level for p in ordered}:
            return AdaptiveDiffuseResult(
                status=AdaptiveDiffuseStatus.POLICY_BLOCKED,
                points=ordered,
                iterations=tuple(iterations),
                final_assessment=assessment,
                execution_error_type=None,
                execution_error_message=(
                    f"Policy requested already evaluated augmentation level {requested}"
                ),
            )

        _, error = execute(requested)
        if error is not None:
            return AdaptiveDiffuseResult(
                status=AdaptiveDiffuseStatus.EXECUTION_BLOCKED,
                points=_ordered_unique(points),
                iterations=tuple(iterations),
                final_assessment=assessment,
                execution_error_type=type(error).__name__,
                execution_error_message=str(error),
            )
        iteration_index += 1
