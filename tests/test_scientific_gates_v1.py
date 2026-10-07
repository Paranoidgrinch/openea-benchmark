from openea_benchmark.adaptive.model import (
    EnergyReliabilityGateSet,
    ErrorBudget,
    EvidenceQuality,
    GateSet,
    Interval,
    ReferenceCharacterAssessment,
    ReferenceCharacterStatus,
    Review,
    ReviewStatus,
    UncertaintyComponent,
)
from openea_benchmark.adaptive.scientific_gates import (
    basis_diffuse_convergence_review,
    build_energy_reliability_gates,
    physical_corrections_review,
    reference_method_validity_review,
    uncertainty_closure_review,
)
from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceAction,
    BasisConvergenceStatus,
    CardinalConvergenceAssessment,
    DiffuseConvergenceAssessment,
    EAIntervalEV,
)


def cleared(tag):
    return Review(ReviewStatus.CLEARED, (tag,), 'cleared')


def ref(status):
    return ReferenceCharacterAssessment(status, ('ref-evidence',), ('reason',))


def cardinal(status=BasisConvergenceStatus.CLEARED):
    return CardinalConvergenceAssessment(
        status,
        BasisConvergenceAction.NONE if status is BasisConvergenceStatus.CLEARED else BasisConvergenceAction.COMPUTE_NEXT_CARDINAL,
        1, 5, 0.003, 0.3, 0.002,
        EAIntervalEV(1.80, 1.81, 1.82) if status is BasisConvergenceStatus.CLEARED else None,
        'CONVERGENCE_ESTIMATED', ('cardinal',), 'cardinal review',
    )


def diffuse(status=BasisConvergenceStatus.CLEARED):
    return DiffuseConvergenceAssessment(
        status,
        BasisConvergenceAction.NONE if status is BasisConvergenceStatus.CLEARED else BasisConvergenceAction.COMPUTE_MORE_DIFFUSE,
        4, 2, 0.002, 0.2, 0.001,
        EAIntervalEV(1.80, 1.81, 1.82) if status is BasisConvergenceStatus.CLEARED else None,
        'CONVERGENCE_ESTIMATED', ('diffuse',), 'diffuse review',
    )


def closed_budget():
    return ErrorBudget(
        Interval(-0.002, 0.002),
        ('baseline',),
        (
            UncertaintyComponent(
                'SOC', Interval(-0.001, 0.001), EvidenceQuality.CONVERGENCE_ESTIMATED, ('soc-budget',)
            ),
        ),
    )


def test_safe_reference_closes_g3a():
    review = reference_method_validity_review(ref(ReferenceCharacterStatus.SAFE_SINGLE_REFERENCE))
    assert review.status is ReviewStatus.CLEARED


def test_borderline_reference_requires_expanded_diagnostics():
    assessment = ref(ReferenceCharacterStatus.BORDERLINE)
    assert reference_method_validity_review(assessment).status is ReviewStatus.UNRESOLVED
    assert reference_method_validity_review(
        assessment, expanded_diagnostics=cleared('expanded')
    ).status is ReviewStatus.CLEARED


def test_multireference_risk_never_clears_single_reference_g3a():
    review = reference_method_validity_review(ref(ReferenceCharacterStatus.MULTIREFERENCE_RISK))
    assert review.status is ReviewStatus.UNRESOLVED


def test_g3b_requires_both_cardinal_and_diffuse_clearance():
    assert basis_diffuse_convergence_review(cardinal(), diffuse()).status is ReviewStatus.CLEARED
    assert basis_diffuse_convergence_review(cardinal(), None).status is ReviewStatus.UNRESOLVED


def test_empty_physical_correction_inventory_does_not_mean_zero_missing_physics():
    assert physical_corrections_review({}).status is ReviewStatus.UNRESOLVED


def test_unknown_error_budget_component_keeps_g3e_open():
    budget = ErrorBudget(
        Interval(-0.002, 0.002), ('baseline',),
        (UncertaintyComponent('SOC', None, EvidenceQuality.UNKNOWN),),
    )
    assert uncertainty_closure_review(budget).status is ReviewStatus.UNRESOLVED


def test_component_resolved_g3_is_authoritative_over_legacy_composite():
    subgates = EnergyReliabilityGateSet(
        reference_method_validity=cleared('g3a'),
        basis_diffuse_convergence=cleared('g3b'),
        correlation_reliability=Review(ReviewStatus.UNRESOLVED, ('g3c',), 'open'),
        physical_corrections=cleared('g3d'),
        uncertainty_closure=cleared('g3e'),
    )
    gates = GateSet(
        cleared('g1'), cleared('g2'), cleared('legacy-g3'), subgates,
    )
    assert gates.open_gates() == ('G3C_CORRELATION_RELIABILITY',)


def test_all_g3_subgates_close_energy_reliability():
    bundle = build_energy_reliability_gates(
        reference_character=ref(ReferenceCharacterStatus.SAFE_SINGLE_REFERENCE),
        expanded_reference_diagnostics=None,
        cardinal=cardinal(),
        diffuse=diffuse(),
        correlation_reliability=cleared('corr'),
        physical_corrections={
            'CV': cleared('cv'),
            'SCALAR_REL': cleared('rel'),
            'SOC': cleared('soc'),
            'NUCLEAR_MOTION': cleared('nuc'),
        },
        error_budget=closed_budget(),
    )
    assert bundle.is_closed
    assert bundle.aggregate_review().status is ReviewStatus.CLEARED
    gates = GateSet(cleared('g1'), cleared('g2'), Review(ReviewStatus.PENDING), bundle)
    assert gates.open_gates() == ()


def test_validated_mr_capability_can_clear_g3a_on_mr_branch():
    from openea_benchmark.adaptive.multireference import (
        MRCapabilityStatus,
        MRProductionCapability,
    )
    capability = MRProductionCapability(
        MRCapabilityStatus.VALIDATED_AVAILABLE,
        ('CASSCF', 'NEVPT2'),
        ('mr-validated',),
        'validated MR EA route',
    )
    review = reference_method_validity_review(
        ref(ReferenceCharacterStatus.MULTIREFERENCE_RISK),
        mr_capability=capability,
    )
    assert review.status is ReviewStatus.CLEARED
    assert 'mr-validated' in review.evidence_ids


def test_borderline_g3e_requires_explicit_uncertainty_enlargement_review():
    bundle = build_energy_reliability_gates(
        reference_character=ref(ReferenceCharacterStatus.BORDERLINE),
        expanded_reference_diagnostics=cleared('expanded'),
        cardinal=cardinal(),
        diffuse=diffuse(),
        correlation_reliability=cleared('corr'),
        physical_corrections={'SOC': cleared('soc')},
        error_budget=closed_budget(),
    )
    assert bundle.reference_method_validity.status is ReviewStatus.CLEARED
    assert bundle.uncertainty_closure.status is ReviewStatus.UNRESOLVED

    closed = build_energy_reliability_gates(
        reference_character=ref(ReferenceCharacterStatus.BORDERLINE),
        expanded_reference_diagnostics=cleared('expanded'),
        cardinal=cardinal(),
        diffuse=diffuse(),
        correlation_reliability=cleared('corr'),
        physical_corrections={'SOC': cleared('soc')},
        error_budget=closed_budget(),
        borderline_uncertainty_review=cleared('borderline-budget'),
    )
    assert closed.uncertainty_closure.status is ReviewStatus.CLEARED
