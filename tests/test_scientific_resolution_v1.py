from openea_benchmark.adaptive import (
    CorrectionEvidence,
    EAEstimate,
    EnergyReliabilityGateSet,
    ErrorBudget,
    EvidenceQuality,
    Interval,
    MethodRole,
    NUCLEAR_MOTION,
    PhysicalValidityAssessment,
    PhysicalValidityStatus,
    PrecisionStatus,
    ProductionEvidenceBundle,
    ProductionRoutePlan,
    ProductionRouteStatus,
    Review,
    ReviewStatus,
    ScientificResolutionPath,
    ScientificResolutionStatus,
    UncertaintyComponent,
    physical_validity_from_binding,
    resolve_scientific_outcome,
)
from openea_benchmark.attachment.asymptote import BindingAssessment, BindingStatus


def cleared(tag: str) -> Review:
    return Review(ReviewStatus.CLEARED, (tag,), f'{tag} cleared')


def unresolved(tag: str) -> Review:
    return Review(ReviewStatus.UNRESOLVED, (tag,), f'{tag} unresolved')


def physical_bound() -> PhysicalValidityAssessment:
    return PhysicalValidityAssessment(
        PhysicalValidityStatus.PHYSICALLY_BOUND_ANION,
        ('g2-bound',),
        ('attachment and molecular binding cleared',),
    )


def physical_unbound() -> PhysicalValidityAssessment:
    return PhysicalValidityAssessment(
        PhysicalValidityStatus.NO_PHYSICALLY_BOUND_ANION,
        ('g2-unbound',),
        ('no physically bound anion',),
    )


def energy_gates() -> EnergyReliabilityGateSet:
    return EnergyReliabilityGateSet(
        reference_method_validity=cleared('g3a'),
        basis_diffuse_convergence=cleared('g3b'),
        correlation_reliability=cleared('g3c'),
        physical_corrections=cleared('g3d'),
        uncertainty_closure=cleared('g3e'),
    )


def bundle(
    *,
    baseline_ev: float = 1.0,
    baseline_half_width: float = 0.005,
    nuclear_center: float = 0.01,
    nuclear_half_width: float = 0.002,
    nuclear_status: ReviewStatus = ReviewStatus.CLEARED,
    closure_actions=(),
) -> ProductionEvidenceBundle:
    if nuclear_status is ReviewStatus.CLEARED:
        nuclear_component = UncertaintyComponent(
            NUCLEAR_MOTION,
            Interval(nuclear_center - nuclear_half_width, nuclear_center + nuclear_half_width),
            EvidenceQuality.CONVERGENCE_ESTIMATED,
            ('nuclear',),
        )
        nuclear = CorrectionEvidence(
            NUCLEAR_MOTION,
            cleared('nuclear'),
            nuclear_component,
            method_role=MethodRole.PRODUCTION,
        )
        corrections = (nuclear_component,)
    elif nuclear_status is ReviewStatus.NOT_APPLICABLE:
        nuclear = CorrectionEvidence(
            NUCLEAR_MOTION,
            Review(ReviewStatus.NOT_APPLICABLE, (), 'incorrectly declared N/A for test'),
            None,
            method_role=MethodRole.PRODUCTION,
        )
        corrections = ()
    else:
        nuclear = CorrectionEvidence(
            NUCLEAR_MOTION,
            Review(nuclear_status, ('nuclear-open',), 'nuclear motion open'),
            None,
            'SOLVE_NUCLEAR_MOTION',
            MethodRole.PRODUCTION,
        )
        corrections = (
            UncertaintyComponent(NUCLEAR_MOTION, None, EvidenceQuality.UNKNOWN, ('nuclear-open',)),
        )

    estimate = EAEstimate(
        baseline_ev,
        Interval(-baseline_half_width, baseline_half_width),
        ('baseline',),
        corrections,
    )
    budget = ErrorBudget(
        Interval(-baseline_half_width, baseline_half_width),
        ('baseline',),
        corrections,
        baseline_name='VALENCE_CCSD(T)_CBS_MODEL',
    )
    route = ProductionRoutePlan(
        ProductionRouteStatus.READY_SINGLE_REFERENCE,
        MethodRole.PRODUCTION,
        ('CCSD(T)',),
        (),
        ('route',),
        rationale='single-reference route ready',
    )
    return ProductionEvidenceBundle(
        route=route,
        estimate=estimate,
        error_budget=budget,
        energy_gates=energy_gates(),
        physical_corrections={NUCLEAR_MOTION: nuclear},
        correlation_review=cleared('corr'),
        closure_actions=tuple(closure_actions),
    )


def test_fragmentation_unbound_is_sufficient_for_early_unbound_g2():
    g2 = physical_validity_from_binding(
        BindingAssessment(BindingStatus.UNBOUND, ('frag',), -0.01)
    )
    assert g2.status is PhysicalValidityStatus.NO_PHYSICALLY_BOUND_ANION

    result = resolve_scientific_outcome(
        molecule='XH',
        state_completeness=cleared('g1'),
        physical_validity=g2,
    )
    assert result.status is ScientificResolutionStatus.UNBOUND
    assert result.precision_status is PrecisionStatus.NOT_APPLICABLE
    assert result.path is ScientificResolutionPath.EARLY_PHYSICAL_UNBOUND
    assert result.ea0_interval_ev is None


def test_early_unbound_requires_g1_state_completeness():
    result = resolve_scientific_outcome(
        molecule='XH',
        state_completeness=unresolved('g1'),
        physical_validity=physical_unbound(),
    )
    assert result.status is ScientificResolutionStatus.UNRESOLVED
    assert result.reasons == ('G1_STATE_COMPLETENESS_NOT_CLEARED',)


def test_bound_molecular_minimum_does_not_by_itself_clear_attachment_continuum():
    g2 = physical_validity_from_binding(
        BindingAssessment(BindingStatus.BOUND, ('frag-bound',), 0.02)
    )
    assert g2.status is PhysicalValidityStatus.UNRESOLVED
    assert 'attachment/continuum validity has not been reviewed' in g2.reasons[0]


def test_bound_g2_requires_cleared_attachment_review():
    binding = BindingAssessment(BindingStatus.BOUND, ('frag-bound',), 0.02)
    pending = physical_validity_from_binding(binding, electron_attachment_review=unresolved('d08'))
    assert pending.status is PhysicalValidityStatus.UNRESOLVED

    ready = physical_validity_from_binding(binding, electron_attachment_review=cleared('d08'))
    assert ready.status is PhysicalValidityStatus.PHYSICALLY_BOUND_ANION
    assert set(ready.evidence_ids) == {'frag-bound', 'd08'}


def test_physically_bound_path_without_g3_bundle_is_unresolved():
    result = resolve_scientific_outcome(
        molecule='OH',
        state_completeness=cleared('g1'),
        physical_validity=physical_bound(),
    )
    assert result.status is ScientificResolutionStatus.UNRESOLVED
    assert result.reasons == ('G3_PRODUCTION_EVIDENCE_MISSING',)


def test_closed_adiabatic_bundle_gives_bound_and_keeps_precision_separate():
    result = resolve_scientific_outcome(
        molecule='OH',
        state_completeness=cleared('g1'),
        physical_validity=physical_bound(),
        production_evidence=bundle(),
        requested_half_width_ev=0.010,
    )
    assert result.status is ScientificResolutionStatus.BOUND
    assert result.precision_status is PrecisionStatus.TARGET_MET
    assert result.path is ScientificResolutionPath.ADIABATIC_EA0
    assert result.ea0_interval_ev is not None
    assert result.ea0_interval_ev.lower > 0.0


def test_missed_precision_target_does_not_change_bound_status():
    result = resolve_scientific_outcome(
        molecule='OH',
        state_completeness=cleared('g1'),
        physical_validity=physical_bound(),
        production_evidence=bundle(baseline_half_width=0.03, nuclear_half_width=0.01),
        requested_half_width_ev=0.005,
    )
    assert result.status is ScientificResolutionStatus.BOUND
    assert result.precision_status is PrecisionStatus.TARGET_NOT_MET


def test_nuclear_motion_may_not_be_closed_as_not_applicable_for_final_ea0():
    result = resolve_scientific_outcome(
        molecule='OH',
        state_completeness=cleared('g1'),
        physical_validity=physical_bound(),
        production_evidence=bundle(nuclear_status=ReviewStatus.NOT_APPLICABLE),
    )
    assert result.status is ScientificResolutionStatus.UNRESOLVED
    assert result.reasons == ('NUCLEAR_MOTION_NOT_RESOLVED_FOR_ADIABATIC_EA0',)


def test_open_production_action_blocks_final_resolution_even_if_interval_exists():
    from openea_benchmark.adaptive import ClosurePriority, ProductionClosureAction

    action = ProductionClosureAction(
        'REVIEW_SOMETHING',
        ClosurePriority.UNCERTAINTY,
        'G3E_UNCERTAINTY_CLOSURE',
        'TEST_COMPONENT',
        MethodRole.DIAGNOSTIC,
        ('review',),
        'open review',
    )
    result = resolve_scientific_outcome(
        molecule='OH',
        state_completeness=cleared('g1'),
        physical_validity=physical_bound(),
        production_evidence=bundle(closure_actions=(action,)),
    )
    assert result.status is ScientificResolutionStatus.UNRESOLVED
    assert result.reasons == ('OPEN_PRODUCTION_CLOSURE_ACTION:REVIEW_SOMETHING',)


def test_g2_bound_and_nonpositive_closed_ea0_is_a_scientific_conflict():
    result = resolve_scientific_outcome(
        molecule='XH',
        state_completeness=cleared('g1'),
        physical_validity=physical_bound(),
        production_evidence=bundle(
            baseline_ev=-0.05,
            baseline_half_width=0.005,
            nuclear_center=0.0,
            nuclear_half_width=0.001,
        ),
    )
    assert result.status is ScientificResolutionStatus.UNRESOLVED
    assert result.reasons == ('G2_G3_CONFLICT_PHYSICALLY_BOUND_BUT_EA0_NONPOSITIVE',)


def test_ea0_interval_overlapping_zero_is_unresolved():
    result = resolve_scientific_outcome(
        molecule='XH',
        state_completeness=cleared('g1'),
        physical_validity=physical_bound(),
        production_evidence=bundle(
            baseline_ev=0.0,
            baseline_half_width=0.01,
            nuclear_center=0.0,
            nuclear_half_width=0.001,
        ),
    )
    assert result.status is ScientificResolutionStatus.UNRESOLVED
    assert result.path is ScientificResolutionPath.UNRESOLVED
    assert result.ea0_interval_ev is not None
    assert result.ea0_interval_ev.lower <= 0.0 < result.ea0_interval_ev.upper


def test_critical_open_question_blocks_even_apparent_early_unbound():
    result = resolve_scientific_outcome(
        molecule='XH',
        state_completeness=cleared('g1'),
        physical_validity=physical_unbound(),
        critical_open_questions=('ASYMPTOTIC_STATE_IDENTITY_CONFLICT',),
    )
    assert result.status is ScientificResolutionStatus.UNRESOLVED
    assert result.reasons == ('CRITICAL_OPEN_QUESTION:ASYMPTOTIC_STATE_IDENTITY_CONFLICT',)


def test_resolved_physical_validity_requires_provenance():
    try:
        PhysicalValidityAssessment(
            PhysicalValidityStatus.PHYSICALLY_BOUND_ANION,
            (),
            ('claimed bound',),
        )
    except ValueError as exc:
        assert 'evidence' in str(exc).lower()
    else:
        raise AssertionError('resolved G2 without provenance must fail')
