from openea_benchmark.adaptive.decision import PrecisionStatus
from openea_benchmark.adaptive.model import (
    ErrorBudget,
    EvidenceQuality,
    Interval,
    MethodRole,
    UncertaintyComponent,
)
from openea_benchmark.adaptive.precision_controller import (
    PrecisionActionCandidate,
    PrecisionPlanningStatus,
    assess_precision_target,
    plan_precision_refinement,
)


def budget(*components):
    return ErrorBudget(
        baseline_offset_ev=Interval(-0.002, 0.002),
        baseline_evidence_ids=('base',),
        components=tuple(components),
    )


def term(name, half_width):
    return UncertaintyComponent(
        name,
        Interval(-half_width, half_width),
        EvidenceQuality.CONVERGENCE_ESTIMATED,
        (name + '-evidence',),
    )


def test_precision_target_is_independent_assessment():
    assessment = assess_precision_target(Interval(1.80, 1.84), 0.005)
    assert assessment.status is PrecisionStatus.TARGET_NOT_MET
    assert abs(assessment.achieved_half_width_ev - 0.02) < 1e-12


def test_target_met_stops_extra_work():
    plan = plan_precision_refinement(
        Interval(1.80, 1.804),
        0.005,
        budget(term('POST_CC', 0.010)),
        (),
    )
    assert plan.status is PrecisionPlanningStatus.TARGET_ALREADY_MET
    assert plan.selected_action is None


def test_unknown_component_blocks_cost_optimization():
    unknown = UncertaintyComponent('SOC', None, EvidenceQuality.UNKNOWN)
    plan = plan_precision_refinement(
        Interval(1.7, 1.9), 0.01, budget(unknown), (),
    )
    assert plan.status is PrecisionPlanningStatus.BUDGET_INCOMPLETE


def test_selects_action_for_dominant_component_by_information_gain_per_cost():
    b = budget(term('POST_CC', 0.030), term('SOC', 0.010))
    actions = (
        PrecisionActionCandidate(
            'CCSDT_X2_X3', 'POST_CC', MethodRole.DIAGNOSTIC,
            0.012, 6.0, ('t3-plan',),
        ),
        PrecisionActionCandidate(
            'CHEAPER_T3_CONTROL', 'POST_CC', MethodRole.DIAGNOSTIC,
            0.008, 2.0, ('t3-cheap',),
        ),
        PrecisionActionCandidate(
            'SOC_REFINEMENT', 'SOC', MethodRole.REFINEMENT,
            0.009, 1.0, ('soc-plan',),
        ),
    )
    plan = plan_precision_refinement(Interval(1.7, 1.9), 0.01, b, actions)
    assert plan.status is PrecisionPlanningStatus.RECOMMEND_CALCULATION
    assert plan.dominant_component == 'POST_CC'
    assert plan.selected_action.action_id == 'CHEAPER_T3_CONTROL'


def test_validation_method_is_not_auto_recommended():
    b = budget(term('POST_CC', 0.030))
    action = PrecisionActionCandidate(
        'CCSDTQ_BENCHMARK', 'POST_CC', MethodRole.VALIDATION,
        0.02, 1.0, ('validation-only',),
    )
    plan = plan_precision_refinement(Interval(1.7, 1.9), 0.01, b, (action,))
    assert plan.status is PrecisionPlanningStatus.NO_VALID_CALCULATION
    assert plan.selected_action is None
