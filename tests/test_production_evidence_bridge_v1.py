from dataclasses import replace

from openea_benchmark.adaptive.model import (
    EvidenceQuality,
    Interval,
    MethodRole,
    ReferenceCharacterAssessment,
    ReferenceCharacterStatus,
    Review,
    ReviewStatus,
    UncertaintyComponent,
)
from openea_benchmark.adaptive.multireference import MRCapabilityStatus, MRProductionCapability
from openea_benchmark.adaptive.precision_controller import PrecisionActionCandidate
from openea_benchmark.adaptive.production_evidence import (
    CORE_VALENCE,
    NUCLEAR_MOTION,
    POST_CC,
    REFERENCE_CHARACTER,
    SCALAR_RELATIVITY_REMAINDER,
    SOC,
    ProductionEvidencePlanningStatus,
    bounded_external_correction,
    build_single_reference_production_evidence_bundle,
    plan_production_evidence,
)
from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceAction,
    BasisConvergenceStatus,
    CardinalConvergenceAssessment,
    DiffuseConvergenceAssessment,
    EAIntervalEV,
)
from openea_benchmark.attachment.component_resolved_cbs import (
    CBSResolvedResult,
    CBSStatus,
    EACBSModel,
)
from openea_benchmark.attachment.core_valence_correction import (
    CoreValenceAssessment,
    CoreValenceStatus,
)
from openea_benchmark.attachment.post_ccsd_t import PostCCAssessment, PostCCPoint
from openea_benchmark.attachment.scalar_relativity import ScalarRelativityAssessment


def ref(status=ReferenceCharacterStatus.SAFE_SINGLE_REFERENCE):
    return ReferenceCharacterAssessment(status, ('ref',), ('reference assessment',))


def cardinal(status=BasisConvergenceStatus.CLEARED):
    return CardinalConvergenceAssessment(
        status=status,
        action=BasisConvergenceAction.NONE if status is BasisConvergenceStatus.CLEARED else BasisConvergenceAction.COMPUTE_NEXT_CARDINAL,
        augmentation_level=1,
        highest_cardinal=5,
        latest_increment_bound_ev=0.002,
        contraction_ratio=0.4,
        residual_estimate_ev=0.001,
        expanded_highest_interval=EAIntervalEV(1.79, 1.80, 1.81) if status is BasisConvergenceStatus.CLEARED else None,
        evidence_quality='CONVERGENCE_ESTIMATED',
        evidence=('cardinal',),
        rationale='cardinal evidence',
    )


def diffuse(status=BasisConvergenceStatus.CLEARED):
    return DiffuseConvergenceAssessment(
        status=status,
        action=BasisConvergenceAction.NONE if status is BasisConvergenceStatus.CLEARED else BasisConvergenceAction.COMPUTE_MORE_DIFFUSE,
        cardinal_number=5,
        highest_augmentation_level=2,
        latest_increment_bound_ev=0.001,
        contraction_ratio=0.3,
        residual_estimate_ev=0.001,
        expanded_best_interval=EAIntervalEV(1.79, 1.80, 1.81) if status is BasisConvergenceStatus.CLEARED else None,
        evidence_quality='CONVERGENCE_ESTIMATED',
        evidence=('diffuse',),
        rationale='diffuse evidence',
    )


def cbs(status=CBSStatus.CLEARED, *, diffuse_residual=0.001, geometry_check=0.0005, model_bound=0.003):
    primary = EACBSModel('primary', 0.0, 0.0, 0.0, 1.800)
    sensitivity = EACBSModel('sensitivity', 0.0, 0.0, 0.0, 1.802)
    return CBSResolvedResult(
        status=status,
        primary_model=primary,
        sensitivity_model=sensitivity,
        combined_correlation_sanity_ev=1.801,
        cbs_model_sensitivity_bound_ev=model_bound,
        diffuse_correction_ev=0.004,
        diffuse_residual_estimate_ev=diffuse_residual,
        fixed_geometry_transfer_check_ev=geometry_check,
        ea_cbs_plus_diffuse_primary_ev=1.804,
        conservative_intermediate_half_width_ev=model_bound + (diffuse_residual or 0.0) + (geometry_check or 0.0),
        lower_intermediate_ev=1.804 - model_bound,
        upper_intermediate_ev=1.804 + model_bound,
        evidence=('cbs',),
        next_actions=() if status is CBSStatus.CLEARED else ('COMPUTE_ADDITIONAL_CARDINAL_EVIDENCE_OR_REVISE_CBS_MODEL',),
    )


def cv():
    return CoreValenceAssessment(
        status=CoreValenceStatus.CLEARED,
        action='NONE',
        central_correction_ev=0.010,
        convergence_bound_ev=0.001,
        contraction_ratio=0.4,
        highest_cardinal=4,
        points=(),
        evidence=('cv',),
    )


def sr():
    return ScalarRelativityAssessment(
        status='CLEARED',
        action='NONE',
        central_correction_ev=0.002,
        convergence_bound_ev=0.0002,
        highest_cardinal=4,
        points=(),
        evidence=('sr', 'SFX2C1E_ONE_BODY_ONLY'),
    )


def post(status='CLEARED'):
    point = PostCCPoint(3, 'cc-pVTZ', 1.8, 1.8005, 0.0005)
    if status == 'CLEARED':
        return PostCCAssessment(
            status='CLEARED', action='NONE', central_correction_ev=0.0005,
            triples_correction_ev=0.0005, quadruples_correction_ev=None,
            triples_convergence_bound_ev=0.0001, quadruples_convergence_bound_ev=None,
            combined_bound_ev=0.0001, highest_triples_cardinal=3,
            highest_quadruples_cardinal=None, points=(point,), evidence=('post',),
        )
    return PostCCAssessment(
        status='POST_CC_WARNING', action='REASSESS_REFERENCE_CHARACTER', central_correction_ev=None,
        triples_correction_ev=0.02, quadruples_correction_ev=None,
        triples_convergence_bound_ev=0.01, quadruples_convergence_bound_ev=None,
        combined_bound_ev=None, highest_triples_cardinal=3,
        highest_quadruples_cardinal=None, points=(point,), evidence=('post-warning',),
    )


def external_physics():
    return {
        SCALAR_RELATIVITY_REMAINDER: bounded_external_correction(
            SCALAR_RELATIVITY_REMAINDER,
            central_ev=0.0, half_width_ev=0.0001,
            evidence_ids=('sr-rem',), rationale='bounded residual scalar relativity',
        ),
        SOC: bounded_external_correction(
            SOC,
            central_ev=-0.001, half_width_ev=0.0002,
            evidence_ids=('soc',), rationale='SOC evaluated',
        ),
        NUCLEAR_MOTION: bounded_external_correction(
            NUCLEAR_MOTION,
            central_ev=-0.002, half_width_ev=0.0005,
            evidence_ids=('nuclear',), rationale='nuclear motion evaluated',
        ),
    }


def complete_bundle(**kwargs):
    data = dict(
        reference_character=ref(), cardinal=cardinal(), diffuse=diffuse(), cbs=cbs(),
        core_valence=cv(), scalar_relativity=sr(), post_cc=post(),
        physical_correction_overrides=external_physics(),
    )
    data.update(kwargs)
    return build_single_reference_production_evidence_bundle(**data)


def test_missing_future_physics_fails_closed_and_requests_explicit_actions():
    bundle = build_single_reference_production_evidence_bundle(
        reference_character=ref(), cardinal=cardinal(), diffuse=diffuse(), cbs=cbs(),
        core_valence=cv(), scalar_relativity=sr(), post_cc=post(),
    )
    assert bundle.energy_gates.physical_corrections.status is ReviewStatus.UNRESOLVED
    assert SOC in bundle.error_budget.missing_components
    assert NUCLEAR_MOTION in bundle.error_budget.missing_components
    assert SCALAR_RELATIVITY_REMAINDER in bundle.error_budget.missing_components
    actions = {a.action_id for a in bundle.closure_actions}
    assert {'ASSESS_SOC', 'SOLVE_NUCLEAR_MOTION', 'BOUND_SCALAR_RELATIVITY_REMAINDER'} <= actions


def test_complete_existing_and_external_evidence_closes_g3():
    bundle = complete_bundle()
    assert bundle.energy_gates.is_closed
    assert bundle.error_budget.is_closed
    assert bundle.interval is not None
    # central = 1.804 + 0.010 + 0.002 + 0.0005 - 0.001 - 0.002 + 0.0
    assert abs(bundle.interval.midpoint - 1.8135) < 1e-12
    assert not bundle.closure_actions


def test_post_cc_warning_reopens_method_validity_and_never_requests_ccsdtq():
    bundle = complete_bundle(post_cc=post('POST_CC_WARNING'))
    assert bundle.energy_gates.reference_method_validity.status is ReviewStatus.UNRESOLVED
    assert bundle.energy_gates.correlation_reliability.status is ReviewStatus.UNRESOLVED
    actions = {a.action_id for a in bundle.closure_actions}
    assert 'REASSESS_REFERENCE_CHARACTER' in actions
    assert not any('T4' in action or 'CCSDTQ' in action for action in actions)


def test_unresolved_cbs_model_opens_g3b_and_emits_cbs_refinement_action():
    bundle = complete_bundle(cbs=cbs(CBSStatus.NEED_MORE_EVIDENCE))
    assert bundle.energy_gates.basis_diffuse_convergence.status is ReviewStatus.UNRESOLVED
    assert any(a.addresses_component == 'CBS_MODEL' for a in bundle.closure_actions)


def test_unknown_cbs_residual_is_not_silently_zero():
    bundle = complete_bundle(cbs=cbs(diffuse_residual=None))
    assert 'CBS_DIFFUSE_RESIDUAL' in bundle.error_budget.missing_components
    assert bundle.energy_gates.uncertainty_closure.status is ReviewStatus.UNRESOLVED
    assert bundle.interval is None
    assert any(a.action_id == 'BOUND_DIFFUSE_RESIDUAL' for a in bundle.closure_actions)


def test_borderline_requires_explicit_reference_uncertainty_component():
    expanded = Review(ReviewStatus.CLEARED, ('expanded',), 'expanded reference diagnostics')
    bundle = complete_bundle(
        reference_character=ref(ReferenceCharacterStatus.BORDERLINE),
        expanded_reference_diagnostics=expanded,
    )
    assert REFERENCE_CHARACTER in bundle.error_budget.missing_components
    assert bundle.energy_gates.uncertainty_closure.status is ReviewStatus.UNRESOLVED

    reference_uncertainty = UncertaintyComponent(
        REFERENCE_CHARACTER,
        Interval(-0.003, 0.003),
        EvidenceQuality.INDIRECTLY_ESTIMATED,
        ('borderline-bound',),
    )
    closed = complete_bundle(
        reference_character=ref(ReferenceCharacterStatus.BORDERLINE),
        expanded_reference_diagnostics=expanded,
        borderline_reference_uncertainty=reference_uncertainty,
    )
    assert closed.energy_gates.is_closed


def test_missing_optional_ccsdt_does_not_mean_zero_but_requests_correlation_review():
    bundle = build_single_reference_production_evidence_bundle(
        reference_character=ref(), cardinal=cardinal(), diffuse=diffuse(), cbs=cbs(),
        core_valence=cv(), scalar_relativity=sr(), post_cc=None,
        physical_correction_overrides=external_physics(),
    )
    assert POST_CC not in bundle.error_budget.missing_components
    assert bundle.energy_gates.correlation_reliability.status is ReviewStatus.UNRESOLVED
    assert any(a.action_id == 'ASSESS_CORRELATION_RELIABILITY' for a in bundle.closure_actions)


def test_precision_refinement_runs_only_after_gate_closure():
    incomplete = build_single_reference_production_evidence_bundle(
        reference_character=ref(), cardinal=cardinal(), diffuse=diffuse(), cbs=cbs(),
        core_valence=cv(), scalar_relativity=sr(), post_cc=post(),
    )
    blocked = plan_production_evidence(incomplete, requested_half_width_ev=0.001)
    assert blocked.status is ProductionEvidencePlanningStatus.GATE_CLOSURE_REQUIRED
    assert blocked.precision_plan is None

    bundle = complete_bundle()
    candidate = PrecisionActionCandidate(
        'REFINE_CBS_MODEL', 'VALENCE_CCSD(T)_CBS_MODEL', MethodRole.REFINEMENT,
        expected_uncertainty_reduction_ev=0.002, relative_cost=1.0,
        evidence_ids=('cbs-refinement-plan',),
    )
    plan = plan_production_evidence(
        bundle, requested_half_width_ev=0.001, precision_candidates=(candidate,),
    )
    assert plan.status is ProductionEvidencePlanningStatus.PRECISION_REFINEMENT_RECOMMENDED
    assert plan.precision_plan.selected_action.action_id == 'REFINE_CBS_MODEL'


def test_missed_precision_without_valid_action_remains_ready_for_scientific_decision():
    bundle = complete_bundle()
    plan = plan_production_evidence(bundle, requested_half_width_ev=0.001)
    assert plan.status is ProductionEvidencePlanningStatus.READY_FOR_SCIENTIFIC_DECISION
    assert plan.precision_plan.assessment.status.value == 'TARGET_NOT_MET'


def test_target_met_stops_precision_work_and_remains_ready_for_decision():
    bundle = complete_bundle()
    plan = plan_production_evidence(bundle, requested_half_width_ev=0.02)
    assert plan.status is ProductionEvidencePlanningStatus.READY_FOR_SCIENTIFIC_DECISION
    assert plan.precision_plan.status.value == 'TARGET_ALREADY_MET'


def test_mr_risk_cannot_be_mixed_with_single_reference_cbs_evidence():
    import pytest

    with pytest.raises(ValueError, match='READY_SINGLE_REFERENCE'):
        complete_bundle(reference_character=ref(ReferenceCharacterStatus.MULTIREFERENCE_RISK))


def test_even_validated_mr_route_requires_a_separate_mr_evidence_bridge():
    import pytest

    capability = MRProductionCapability(
        MRCapabilityStatus.VALIDATED_AVAILABLE,
        ('CASSCF', 'NEVPT2'),
        ('mr-validated',),
        'validated MR production route',
    )
    with pytest.raises(ValueError, match='READY_MULTIREFERENCE'):
        complete_bundle(
            reference_character=ref(ReferenceCharacterStatus.MULTIREFERENCE_RISK),
            mr_capability=capability,
        )
