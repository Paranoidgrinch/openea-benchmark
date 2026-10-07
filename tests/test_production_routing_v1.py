from openea_benchmark.adaptive.model import (
    MethodRole,
    ReferenceCharacterAssessment,
    ReferenceCharacterStatus,
    Review,
    ReviewStatus,
    ScientificResolutionStatus,
)
from openea_benchmark.adaptive.multireference import (
    MRCapabilityStatus,
    MRProductionCapability,
)
from openea_benchmark.adaptive.planner import (
    ProductionRouteStatus,
    plan_production_route,
)


def assessment(status):
    return ReferenceCharacterAssessment(status, ('ref',), ('reason',))


def test_safe_reference_authorizes_ccsdt_production_branch():
    plan = plan_production_route(assessment(ReferenceCharacterStatus.SAFE_SINGLE_REFERENCE))
    assert plan.status is ProductionRouteStatus.READY_SINGLE_REFERENCE
    assert plan.method_role is MethodRole.PRODUCTION
    assert plan.method_family == ('CCSD(T)',)


def test_borderline_reference_is_blocked_until_expanded_diagnostics_clear():
    plan = plan_production_route(assessment(ReferenceCharacterStatus.BORDERLINE))
    assert plan.status is ProductionRouteStatus.DIAGNOSTICS_REQUIRED
    assert 'EXPAND_REFERENCE_DIAGNOSTICS' in plan.required_actions
    assert plan.method_family == ()


def test_borderline_reference_can_proceed_after_review_but_keeps_uncertainty_action():
    expanded = Review(ReviewStatus.CLEARED, ('expanded',), 'expanded diagnostics reviewed')
    plan = plan_production_route(
        assessment(ReferenceCharacterStatus.BORDERLINE),
        expanded_reference_diagnostics=expanded,
    )
    assert plan.status is ProductionRouteStatus.READY_SINGLE_REFERENCE
    assert plan.method_family == ('CCSD(T)',)
    assert 'ENLARGE_REFERENCE_CHARACTER_UNCERTAINTY' in plan.required_actions


def test_mr_risk_fails_closed_without_validated_mr_method():
    plan = plan_production_route(assessment(ReferenceCharacterStatus.MULTIREFERENCE_RISK))
    assert plan.status is ProductionRouteStatus.TERMINAL_UNRESOLVED
    assert plan.scientific_status is ScientificResolutionStatus.UNRESOLVED
    assert plan.terminal_reason == 'MULTIREFERENCE_METHOD_REQUIRED'
    assert plan.method_family == ()


def test_installed_but_inadequate_mr_capability_still_fails_closed():
    capability = MRProductionCapability(
        MRCapabilityStatus.INADEQUATE,
        rationale='Exploratory CASSCF exists but no validated production EA protocol is registered.',
    )
    plan = plan_production_route(
        assessment(ReferenceCharacterStatus.MULTIREFERENCE_RISK),
        mr_capability=capability,
    )
    assert plan.status is ProductionRouteStatus.TERMINAL_UNRESOLVED
    assert plan.terminal_reason == 'MULTIREFERENCE_METHOD_REQUIRED'


def test_validated_mr_capability_may_authorize_mr_production():
    capability = MRProductionCapability(
        MRCapabilityStatus.VALIDATED_AVAILABLE,
        ('CASSCF', 'NEVPT2'),
        ('mr-validation',),
        'Validated for the intended state/PEC/EA role.',
    )
    plan = plan_production_route(
        assessment(ReferenceCharacterStatus.MULTIREFERENCE_RISK),
        mr_capability=capability,
    )
    assert plan.status is ProductionRouteStatus.READY_MULTIREFERENCE
    assert plan.method_role is MethodRole.PRODUCTION
    assert plan.method_family == ('CASSCF', 'NEVPT2')
    assert plan.scientific_status is None


def test_unresolved_reference_never_selects_a_production_method():
    plan = plan_production_route(assessment(ReferenceCharacterStatus.UNRESOLVED))
    assert plan.status is ProductionRouteStatus.DIAGNOSTICS_REQUIRED
    assert plan.method_family == ()
    assert plan.required_actions == ('RESOLVE_REFERENCE_CHARACTER',)
