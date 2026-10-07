from openea_benchmark.adaptive.model import MethodRole
from openea_benchmark.adaptive.production_evidence import (
    CBS_DIFFUSE_RESIDUAL,
    CORE_VALENCE,
    NUCLEAR_MOTION,
    POST_CC,
    REFERENCE_CHARACTER,
    SCALAR_RELATIVITY,
    SOC,
    ClosurePriority,
    ProductionClosureAction,
)
from openea_benchmark.adaptive.production_execution import (
    CapabilityImplementation,
    ExecutionAdapter,
    ExecutionAttemptStatus,
    ExecutionCapability,
    ExecutionDisposition,
    ProductionExecutionPlan,
    build_production_execution_plan,
    classify_closure_action,
    execute_next_closure_action,
)


def action(action_id, component, *, priority=ClosurePriority.BASIS_AND_CBS, role=MethodRole.REFINEMENT):
    return ProductionClosureAction(
        action_id,
        priority,
        'G3_TEST',
        component,
        role,
        ('evidence',),
        'test closure action',
    )


def test_cardinal_and_diffuse_map_only_to_existing_generic_runners():
    cardinal = classify_closure_action(action('COMPUTE_NEXT_CARDINAL', 'CARDINAL_CONVERGENCE'))
    assert cardinal.capability is ExecutionCapability.BASIS_CARDINAL
    assert cardinal.implementation is CapabilityImplementation.GENERIC_RUNNER_AVAILABLE
    assert cardinal.disposition is ExecutionDisposition.NEEDS_BOUND_CONTEXT
    assert cardinal.runner_id.endswith('run_adaptive_cardinal_series')

    diffuse = classify_closure_action(action('COMPUTE_MORE_DIFFUSE', 'DIFFUSE_CONVERGENCE'))
    assert diffuse.capability is ExecutionCapability.BASIS_DIFFUSE
    assert diffuse.implementation is CapabilityImplementation.GENERIC_RUNNER_AVAILABLE
    assert diffuse.runner_id.endswith('run_adaptive_diffuse_series')

    residual = classify_closure_action(action('BOUND_DIFFUSE_RESIDUAL', CBS_DIFFUSE_RESIDUAL))
    assert residual.capability is ExecutionCapability.BASIS_DIFFUSE


def test_cv_is_generic_but_scalar_and_postcc_remain_capability_gaps():
    cv = classify_closure_action(action('COMPUTE_X4', CORE_VALENCE))
    assert cv.capability is ExecutionCapability.CORE_VALENCE
    assert cv.implementation is CapabilityImplementation.GENERIC_RUNNER_AVAILABLE
    assert cv.disposition is ExecutionDisposition.NEEDS_BOUND_CONTEXT
    assert cv.runner_id.endswith('run_adaptive_core_valence_series')

    sr = classify_closure_action(action('COMPUTE_X4', SCALAR_RELATIVITY))
    post = classify_closure_action(action('COMPUTE_T3_X3', POST_CC, priority=ClosurePriority.CORRELATION, role=MethodRole.DIAGNOSTIC))
    for request in (sr, post):
        assert request.implementation is CapabilityImplementation.ASSESSMENT_ONLY
        assert request.disposition is ExecutionDisposition.CAPABILITY_GAP
        assert request.runner_id is None


def test_postcc_warning_returns_to_reference_review():
    req = classify_closure_action(
        action(
            'REASSESS_REFERENCE_CHARACTER',
            POST_CC,
            priority=ClosurePriority.METHOD_VALIDITY,
            role=MethodRole.DIAGNOSTIC,
        )
    )
    assert req.capability is ExecutionCapability.REFERENCE_CHARACTER
    assert req.disposition is ExecutionDisposition.MANUAL_OR_DIAGNOSTIC_REVIEW


def test_soc_nuclear_motion_are_explicit_capability_gaps():
    soc = classify_closure_action(action('ASSESS_SOC', SOC, priority=ClosurePriority.PHYSICAL_CORRECTION, role=MethodRole.PRODUCTION))
    nuc = classify_closure_action(action('SOLVE_NUCLEAR_MOTION', NUCLEAR_MOTION, priority=ClosurePriority.PHYSICAL_CORRECTION, role=MethodRole.PRODUCTION))
    assert soc.capability is ExecutionCapability.SOC
    assert nuc.capability is ExecutionCapability.NUCLEAR_MOTION
    assert soc.disposition is ExecutionDisposition.CAPABILITY_GAP
    assert nuc.disposition is ExecutionDisposition.CAPABILITY_GAP


def test_forbidden_high_order_escalation_is_policy_blocked():
    req = classify_closure_action(action('COMPUTE_CCSDTQ_X2', POST_CC, priority=ClosurePriority.CORRELATION, role=MethodRole.DIAGNOSTIC))
    assert req.implementation is CapabilityImplementation.POLICY_FORBIDDEN
    assert req.disposition is ExecutionDisposition.POLICY_BLOCKED


def test_execution_never_skips_higher_priority_manual_review():
    first = classify_closure_action(
        action('REASSESS_REFERENCE_CHARACTER', REFERENCE_CHARACTER, priority=ClosurePriority.METHOD_VALIDITY, role=MethodRole.DIAGNOSTIC)
    )
    second = classify_closure_action(action('COMPUTE_NEXT_CARDINAL', 'CARDINAL_CONVERGENCE'))
    plan = ProductionExecutionPlan((first, second))
    called = []
    adapter = ExecutionAdapter(
        ExecutionCapability.BASIS_CARDINAL,
        second.runner_id,
        lambda closure: called.append(closure.action_id),
    )
    attempt = execute_next_closure_action(plan, adapters={ExecutionCapability.BASIS_CARDINAL: adapter})
    assert attempt.status is ExecutionAttemptStatus.BLOCKED_MANUAL_REVIEW
    assert not called


def test_generic_runner_requires_explicit_bound_context_then_executes():
    request = classify_closure_action(action('COMPUTE_NEXT_CARDINAL', 'CARDINAL_CONVERGENCE'))
    plan = ProductionExecutionPlan((request,))
    blocked = execute_next_closure_action(plan)
    assert blocked.status is ExecutionAttemptStatus.BLOCKED_NEEDS_CONTEXT

    seen = []
    adapter = ExecutionAdapter(
        ExecutionCapability.BASIS_CARDINAL,
        request.runner_id,
        lambda closure: seen.append(closure.action_id) or {'status': 'ok'},
    )
    done = execute_next_closure_action(plan, adapters={ExecutionCapability.BASIS_CARDINAL: adapter})
    assert done.status is ExecutionAttemptStatus.EXECUTED
    assert done.result == {'status': 'ok'}
    assert seen == ['COMPUTE_NEXT_CARDINAL']


def test_plan_preserves_scientific_closure_order():
    # Minimal bundle-like object: execution plan only consumes closure_actions.
    class Bundle:
        closure_actions = (
            action('EXPAND_REFERENCE_DIAGNOSTICS', REFERENCE_CHARACTER, priority=ClosurePriority.METHOD_VALIDITY, role=MethodRole.DIAGNOSTIC),
            action('COMPUTE_MORE_DIFFUSE', 'DIFFUSE_CONVERGENCE'),
            action('ASSESS_SOC', SOC, priority=ClosurePriority.PHYSICAL_CORRECTION, role=MethodRole.PRODUCTION),
        )

    plan = build_production_execution_plan(Bundle())
    assert [x.action.action_id for x in plan.requests] == [
        'EXPAND_REFERENCE_DIAGNOSTICS', 'COMPUTE_MORE_DIFFUSE', 'ASSESS_SOC'
    ]
    assert plan.next_request.action.action_id == 'EXPAND_REFERENCE_DIAGNOSTICS'
    assert plan.has_capability_gap


def test_no_action_required_is_explicit():
    attempt = execute_next_closure_action(ProductionExecutionPlan(()))
    assert attempt.status is ExecutionAttemptStatus.NO_ACTION_REQUIRED
