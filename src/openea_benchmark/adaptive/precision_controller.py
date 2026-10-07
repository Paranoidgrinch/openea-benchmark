"""Precision-target assessment and information-gain planning.

Scientific status is intentionally outside this module.  Missing a requested
precision target never changes BOUND/UNBOUND/UNRESOLVED by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .decision import PrecisionStatus
from .model import ErrorBudget, Interval, MethodRole


class PrecisionPlanningStatus(str, Enum):
    TARGET_ALREADY_MET = 'TARGET_ALREADY_MET'
    RECOMMEND_CALCULATION = 'RECOMMEND_CALCULATION'
    BUDGET_INCOMPLETE = 'BUDGET_INCOMPLETE'
    NO_VALID_CALCULATION = 'NO_VALID_CALCULATION'
    UNDETERMINED = 'UNDETERMINED'


@dataclass(frozen=True)
class PrecisionTargetAssessment:
    requested_half_width_ev: float
    achieved_half_width_ev: float | None
    status: PrecisionStatus
    reason: str

    def __post_init__(self) -> None:
        if not isfinite(self.requested_half_width_ev) or self.requested_half_width_ev <= 0:
            raise ValueError('Requested precision must be finite and positive')
        if self.achieved_half_width_ev is not None:
            if not isfinite(self.achieved_half_width_ev) or self.achieved_half_width_ev < 0:
                raise ValueError('Achieved half-width must be finite and nonnegative')
        if not self.reason.strip():
            raise ValueError('Precision assessment requires a reason')


@dataclass(frozen=True)
class PrecisionActionCandidate:
    action_id: str
    addresses_component: str
    method_role: MethodRole
    expected_uncertainty_reduction_ev: float
    relative_cost: float
    evidence_ids: tuple[str, ...]
    scientifically_valid: bool = True

    def __post_init__(self) -> None:
        if not self.action_id.strip() or not self.addresses_component.strip():
            raise ValueError('Precision action requires action and component IDs')
        if not isfinite(self.expected_uncertainty_reduction_ev) or self.expected_uncertainty_reduction_ev <= 0:
            raise ValueError('Expected uncertainty reduction must be finite and positive')
        if not isfinite(self.relative_cost) or self.relative_cost <= 0:
            raise ValueError('Relative cost must be finite and positive')
        if self.scientifically_valid and not self.evidence_ids:
            raise ValueError('A scientifically valid precision action requires provenance')

    @property
    def information_gain_per_cost(self) -> float:
        return self.expected_uncertainty_reduction_ev / self.relative_cost


@dataclass(frozen=True)
class PrecisionPlan:
    assessment: PrecisionTargetAssessment
    status: PrecisionPlanningStatus
    dominant_component: str | None
    selected_action: PrecisionActionCandidate | None
    reason: str


def assess_precision_target(
    interval: Interval | None,
    requested_half_width_ev: float,
) -> PrecisionTargetAssessment:
    if not isfinite(requested_half_width_ev) or requested_half_width_ev <= 0:
        raise ValueError('Requested precision must be finite and positive')
    if interval is None:
        return PrecisionTargetAssessment(
            requested_half_width_ev,
            None,
            PrecisionStatus.UNDETERMINED,
            'No closed scientific interval is available for precision assessment.',
        )
    achieved = interval.half_width
    status = (
        PrecisionStatus.TARGET_MET
        if achieved <= requested_half_width_ev
        else PrecisionStatus.TARGET_NOT_MET
    )
    return PrecisionTargetAssessment(
        requested_half_width_ev,
        achieved,
        status,
        'Requested half-width is met.' if status is PrecisionStatus.TARGET_MET
        else 'Requested half-width is not met; scientific status remains separate.',
    )


def plan_precision_refinement(
    interval: Interval | None,
    requested_half_width_ev: float,
    error_budget: ErrorBudget,
    candidates: tuple[PrecisionActionCandidate, ...] = (),
) -> PrecisionPlan:
    """Recommend, but never execute, the best valid action for the dominant term."""

    assessment = assess_precision_target(interval, requested_half_width_ev)
    if assessment.status is PrecisionStatus.UNDETERMINED:
        return PrecisionPlan(
            assessment,
            PrecisionPlanningStatus.UNDETERMINED,
            None,
            None,
            'Precision planning requires a closed scientific interval.',
        )
    if assessment.status is PrecisionStatus.TARGET_MET:
        return PrecisionPlan(
            assessment,
            PrecisionPlanningStatus.TARGET_ALREADY_MET,
            None,
            None,
            'Do not launch additional expensive work solely because it is available.',
        )
    if not error_budget.is_closed:
        return PrecisionPlan(
            assessment,
            PrecisionPlanningStatus.BUDGET_INCOMPLETE,
            None,
            None,
            'Unknown error-budget components must be resolved or bounded before cost optimization.',
        )

    dominant = error_budget.dominant_uncertainty_source()
    if dominant is None:
        return PrecisionPlan(
            assessment,
            PrecisionPlanningStatus.NO_VALID_CALCULATION,
            None,
            None,
            'No documented dominant uncertainty source is available.',
        )

    eligible = tuple(
        candidate for candidate in candidates
        if candidate.scientifically_valid
        and candidate.addresses_component == dominant
        and candidate.method_role is not MethodRole.VALIDATION
    )
    if not eligible:
        return PrecisionPlan(
            assessment,
            PrecisionPlanningStatus.NO_VALID_CALCULATION,
            dominant,
            None,
            'No scientifically valid production/diagnostic/refinement action addresses the dominant uncertainty.',
        )

    selected = max(
        eligible,
        key=lambda item: (item.information_gain_per_cost, item.expected_uncertainty_reduction_ev, item.action_id),
    )
    return PrecisionPlan(
        assessment,
        PrecisionPlanningStatus.RECOMMEND_CALCULATION,
        dominant,
        selected,
        'Selected by expected uncertainty reduction per relative computational cost; execution is not automatic.',
    )
