import math

import numpy as np

from openea_benchmark.adaptive import (
    ANGSTROM_TO_BOHR,
    HARTREE_TO_EV,
    U_TO_ELECTRON_MASS,
    CapabilityImplementation,
    ClosurePriority,
    DiatomicMassSpecification,
    ExecutionCapability,
    ExecutionDisposition,
    MethodRole,
    NUCLEAR_MOTION,
    NuclearMotionPEC,
    NuclearMotionModelEvidence,
    NuclearMotionSettings,
    NuclearMotionStatus,
    ProductionClosureAction,
    ReviewStatus,
    VibrationalBindingStatus,
    VibrationalSolveStatus,
    classify_closure_action,
    derive_nuclear_motion_model_evidence,
    nuclear_motion_evidence,
    nuclear_motion_pec_from_high_level,
    run_diatomic_nuclear_motion,
    solve_vibrational_ground_state,
)
from openea_benchmark.attachment.asymptote import DissociationChannel
from openea_benchmark.adaptive.stage3_pec import (
    HighLevelMinimumCandidate, HighLevelMinimumScout, HighLevelMinimumStatus,
    HighLevelPEC, HighLevelPECPoint, HighLevelPECStatus, HighLevelPointStatus,
    IdentityReviewStatus,
)


SETTINGS = NuclearMotionSettings(
    coarse_grid_points=301,
    fine_grid_points=601,
    grid_convergence_tolerance_ev=2.0e-5,
    interpolation_sensitivity_tolerance_ev=5.0e-4,
    domain_sensitivity_tolerance_ev=1.0e-4,
    minimum_boundary_clearance_ev=0.30,
    trim_fraction=0.08,
    eigenvalue_tolerance_hartree=1.0e-12,
)

MASSES_H2 = DiatomicMassSpecification(
    'H', 'H', 1.0, 1.0, ('MASS:H1',), mass_model='TEST_EXPLICIT_ISOTOPE_MASS'
)

MODEL = NuclearMotionModelEvidence(
    delta_zpe_half_width_ev=2.0e-4,
    anion_binding_margin_half_width_ev=5.0e-4,
    evidence_ids=('NUCLEAR_MODEL:TEST',),
    rationale='synthetic cross-level convergence bound',
)


def harmonic_pec(role, *, omega, r0=1.0, shift=-75.0, mass_spec=MASSES_H2, n=23, width=1.1):
    r = np.linspace(r0 - width / 2.0, r0 + width / 2.0, n)
    x_bohr = (r - r0) * ANGSTROM_TO_BOHR
    mu = mass_spec.reduced_mass_electron_masses
    e = shift + 0.5 * mu * omega * omega * x_bohr * x_bohr
    return NuclearMotionPEC(
        role=role,
        state_id=f'{role}-state',
        r_angstrom=tuple(r),
        energy_hartree=tuple(e),
        method='CCSD(T)',
        basis='aug-cc-pVQZ',
        evidence_ids=(f'PEC:{role}',),
    )



def test_identity_cleared_stage3_pec_can_bind_directly_to_nuclear_contract():
    r = (0.8, 0.9, 1.0, 1.1, 1.2)
    e = (-10.00, -10.05, -10.08, -10.05, -10.00)
    points = tuple(
        HighLevelPECPoint(
            r_angstrom=ri, status=HighLevelPointStatus.ACCEPTED,
            energy_method='CCSD(T)', energy_hartree=ei, canonical_request_id=f'r{i}',
            request_ids=(f'r{i}',), source_root_ids=(f'root{i}',),
            initialization_energy_spread_hartree=0.0,
            high_level_checkpoint_paths=(f'/tmp/cp{i}.json',),
            evidence_ids=(f'point{i}',),
        )
        for i, (ri, ei) in enumerate(zip(r, e))
    )
    candidate = HighLevelMinimumCandidate(1.0, -10.08, 2, 0.9, 1.1, 'r2')
    pec = HighLevelPEC(
        job_id='job', system='XH', charge=0, spin_2s=1, component_id='c',
        basis='aug-cc-pVQZ', points=points,
        status=HighLevelPECStatus.READY_FOR_DISCRETE_MINIMUM_SCOUT,
        initialization_identity_status=IdentityReviewStatus.CLEARED,
        geometry_continuity_status=IdentityReviewStatus.CLEARED,
        minimum_scout=HighLevelMinimumScout(
            HighLevelMinimumStatus.BRACKETED_SINGLE_MINIMUM, (candidate,), 'single minimum'
        ),
    )
    bound = nuclear_motion_pec_from_high_level(
        pec, role='neutral', state_id='neutral-state', evidence_ids=('stage3-reviewed',)
    )
    assert bound.role == 'neutral'
    assert bound.method == 'CCSD(T)'
    assert bound.basis == 'aug-cc-pVQZ'
    assert bound.r_angstrom == r
    assert 'stage3-reviewed' in bound.evidence_ids


def test_harmonic_limit_recovers_half_omega_with_bounded_error():
    omega = 0.020
    result = solve_vibrational_ground_state(
        harmonic_pec('neutral', omega=omega),
        masses=MASSES_H2,
        settings=SETTINGS,
    )
    assert result.status is VibrationalSolveStatus.CLEARED
    expected = 0.5 * omega * HARTREE_TO_EV
    assert abs(result.zpe_ev - expected) < 5.0e-4
    assert result.numerical_bound_ev is not None
    assert abs(result.zpe_ev - expected) <= result.numerical_bound_ev + 5.0e-5


def test_morse_anharmonic_ground_state_matches_analytic_limit_within_bound():
    # Morse oscillator: E0 = omega/2 - omega^2/(16*De), measured from the minimum.
    de_hartree = 0.20
    omega = 0.020
    mu = MASSES_H2.reduced_mass_electron_masses
    a_bohr_inv = omega * math.sqrt(mu / (2.0 * de_hartree))
    r = np.linspace(0.35, 2.20, 61)
    x_bohr = (r - 1.0) * ANGSTROM_TO_BOHR
    e = -75.0 + de_hartree * (1.0 - np.exp(-a_bohr_inv * x_bohr)) ** 2
    pec = NuclearMotionPEC(
        'neutral', 'morse-state', tuple(r), tuple(e), 'CCSD(T)', 'aug-cc-pVQZ', ('PEC:MORSE',)
    )
    settings = NuclearMotionSettings(
        coarse_grid_points=401, fine_grid_points=801,
        grid_convergence_tolerance_ev=3.0e-5,
        interpolation_sensitivity_tolerance_ev=5.0e-4,
        domain_sensitivity_tolerance_ev=1.0e-4,
        minimum_boundary_clearance_ev=0.30, trim_fraction=0.08,
        eigenvalue_tolerance_hartree=1.0e-13,
    )
    result = solve_vibrational_ground_state(pec, masses=MASSES_H2, settings=settings)
    assert result.status is VibrationalSolveStatus.CLEARED
    expected_h = omega / 2.0 - omega * omega / (16.0 * de_hartree)
    error_ev = abs(result.zpe_hartree - expected_h) * HARTREE_TO_EV
    assert error_ev < 1.0e-4
    assert error_ev <= result.numerical_bound_ev


def test_constant_electronic_energy_shift_does_not_change_zpe():
    a = solve_vibrational_ground_state(
        harmonic_pec('neutral', omega=0.018, shift=-75.0),
        masses=MASSES_H2,
        settings=SETTINGS,
    )
    b = solve_vibrational_ground_state(
        harmonic_pec('neutral', omega=0.018, shift=-200.0),
        masses=MASSES_H2,
        settings=SETTINGS,
    )
    assert a.status is b.status is VibrationalSolveStatus.CLEARED
    assert abs(a.zpe_ev - b.zpe_ev) < 1.0e-8


def test_isotope_scaling_tracks_inverse_square_root_reduced_mass():
    # Keep the Born-Oppenheimer force constant fixed while changing nuclear mass.
    r = np.linspace(0.45, 1.55, 23)
    r0 = 1.0
    mu_h = MASSES_H2.reduced_mass_electron_masses
    omega_h = 0.020
    k = mu_h * omega_h * omega_h
    x_bohr = (r - r0) * ANGSTROM_TO_BOHR
    e = -10.0 + 0.5 * k * x_bohr * x_bohr
    pec = NuclearMotionPEC(
        'neutral', 'state', tuple(r), tuple(e), 'CCSD(T)', 'aug-cc-pVQZ', ('PEC',)
    )
    d2 = DiatomicMassSpecification('D', 'D', 2.0, 2.0, ('MASS:D2',))
    h = solve_vibrational_ground_state(pec, masses=MASSES_H2, settings=SETTINGS)
    d = solve_vibrational_ground_state(pec, masses=d2, settings=SETTINGS)
    assert h.status is d.status is VibrationalSolveStatus.CLEARED
    ratio = d.zpe_ev / h.zpe_ev
    assert abs(ratio - 1.0 / math.sqrt(2.0)) < 3.0e-3


def test_narrow_pec_fails_closed_and_requests_pec_refinement():
    narrow = harmonic_pec('neutral', omega=0.020, n=9, width=0.20)
    result = solve_vibrational_ground_state(narrow, masses=MASSES_H2, settings=SETTINGS)
    assert result.status is VibrationalSolveStatus.PEC_REFINEMENT_REQUIRED
    assert result.requires_lower_r_extension or result.requires_upper_r_extension


def test_state_identity_or_continuity_must_be_cleared_before_nuclear_solve():
    pec = harmonic_pec('neutral', omega=0.020)
    bad = NuclearMotionPEC(
        **{**pec.__dict__, 'identity_status': 'UNRESOLVED'}
    )
    result = solve_vibrational_ground_state(bad, masses=MASSES_H2, settings=SETTINGS)
    assert result.status is VibrationalSolveStatus.UNRESOLVED


def test_delta_zpe_is_neutral_minus_anion_and_uncertainty_is_conservative_sum():
    result = run_diatomic_nuclear_motion(
        neutral_pec=harmonic_pec('neutral', omega=0.022),
        anion_pec=harmonic_pec('anion', omega=0.016, shift=-75.5),
        masses=MASSES_H2,
        settings=SETTINGS,
        model_evidence=MODEL,
    )
    assert result.status is NuclearMotionStatus.CLEARED
    assert result.correction_ev > 0.0
    assert abs(result.correction_ev - (result.neutral.zpe_ev - result.anion.zpe_ev)) < 1.0e-12
    assert result.correction_half_width_ev == (
        result.neutral.numerical_bound_ev
        + result.anion.numerical_bound_ev
        + MODEL.delta_zpe_half_width_ev
    )
    assert result.anion_vibrational_binding.status is VibrationalBindingStatus.NOT_ASSESSED




def test_model_sensitivity_can_be_derived_from_two_numerically_cleared_pec_levels():
    primary = run_diatomic_nuclear_motion(
        neutral_pec=harmonic_pec('neutral', omega=0.022),
        anion_pec=harmonic_pec('anion', omega=0.016, shift=-75.5),
        masses=MASSES_H2,
        settings=SETTINGS,
        anion_dissociation_channels=(
            DissociationChannel('diss', 'A', 'B', -75.45, 'test-primary'),
        ),
    )
    comparison = run_diatomic_nuclear_motion(
        neutral_pec=harmonic_pec('neutral', omega=0.0218),
        anion_pec=harmonic_pec('anion', omega=0.0162, shift=-75.5),
        masses=MASSES_H2,
        settings=SETTINGS,
        anion_dissociation_channels=(
            DissociationChannel('diss', 'A', 'B', -75.4502, 'test-comparison'),
        ),
    )
    assert primary.status is NuclearMotionStatus.MODEL_CONVERGENCE_REQUIRED
    assert comparison.status is NuclearMotionStatus.MODEL_CONVERGENCE_REQUIRED
    assert primary.anion_vibrational_binding.margin_interval_hartree is not None
    assert comparison.anion_vibrational_binding.margin_interval_hartree is not None

    model = derive_nuclear_motion_model_evidence(
        primary,
        comparison,
        evidence_ids=('MODEL:QZ_vs_TZ',),
        rationale='synthetic two-level PEC comparison',
    )
    assert model.delta_zpe_half_width_ev > abs(primary.correction_ev - comparison.correction_ev)
    assert model.anion_binding_margin_half_width_ev is not None
    assert model.anion_binding_margin_half_width_ev > 0.0

    final = run_diatomic_nuclear_motion(
        neutral_pec=harmonic_pec('neutral', omega=0.022),
        anion_pec=harmonic_pec('anion', omega=0.016, shift=-75.5),
        masses=MASSES_H2,
        settings=SETTINGS,
        anion_dissociation_channels=(
            DissociationChannel('diss', 'A', 'B', -75.45, 'test-primary'),
        ),
        model_evidence=model,
    )
    assert final.status is NuclearMotionStatus.CLEARED
    assert final.anion_vibrational_binding.status is VibrationalBindingStatus.BOUND


def test_numerically_converged_v0_without_pec_model_bound_does_not_close_g3d():
    result = run_diatomic_nuclear_motion(
        neutral_pec=harmonic_pec('neutral', omega=0.022),
        anion_pec=harmonic_pec('anion', omega=0.016, shift=-75.5),
        masses=MASSES_H2,
        settings=SETTINGS,
    )
    assert result.status is NuclearMotionStatus.MODEL_CONVERGENCE_REQUIRED
    assert result.correction_ev is not None
    assert result.correction_half_width_ev is None
    assert result.action == 'ASSESS_NUCLEAR_PEC_MODEL_CONVERGENCE'
    evidence = nuclear_motion_evidence(result)
    assert evidence.review.status is ReviewStatus.UNRESOLVED


def test_mismatched_neutral_anion_methods_or_bases_are_rejected():
    neutral = harmonic_pec('neutral', omega=0.020)
    anion = harmonic_pec('anion', omega=0.018)
    bad_method = NuclearMotionPEC(**{**anion.__dict__, 'method': 'CCSD'})
    try:
        run_diatomic_nuclear_motion(
            neutral_pec=neutral,
            anion_pec=bad_method,
            masses=MASSES_H2,
            settings=SETTINGS,
            model_evidence=MODEL,
        )
    except ValueError as exc:
        assert 'same electronic method' in str(exc)
    else:
        raise AssertionError('mismatched method must fail')


def test_anion_v0_binding_is_decided_from_absolute_v0_interval():
    anion = harmonic_pec('anion', omega=0.016, shift=-75.5)
    neutral = harmonic_pec('neutral', omega=0.022, shift=-75.0)
    # The asymptote is safely above the anion v=0 level.
    bound = run_diatomic_nuclear_motion(
        neutral_pec=neutral,
        anion_pec=anion,
        masses=MASSES_H2,
        settings=SETTINGS,
        anion_dissociation_channels=(
            DissociationChannel('diss', 'A', 'B', -75.45, 'test'),
        ),
        model_evidence=MODEL,
    )
    assert bound.status is NuclearMotionStatus.CLEARED
    assert bound.anion_vibrational_binding.status is VibrationalBindingStatus.BOUND

    # Move the dissociation threshold below the v=0 interval: electronic well
    # may exist, but the molecular v=0 state is not bound.
    unbound = run_diatomic_nuclear_motion(
        neutral_pec=neutral,
        anion_pec=anion,
        masses=MASSES_H2,
        settings=SETTINGS,
        anion_dissociation_channels=(
            DissociationChannel('diss', 'A', 'B', -75.51, 'test'),
        ),
        model_evidence=MODEL,
    )
    assert unbound.anion_vibrational_binding.status is VibrationalBindingStatus.UNBOUND


def test_nuclear_motion_evidence_produces_bounded_delta_zpe_component():
    result = run_diatomic_nuclear_motion(
        neutral_pec=harmonic_pec('neutral', omega=0.022),
        anion_pec=harmonic_pec('anion', omega=0.016, shift=-75.5),
        masses=MASSES_H2,
        settings=SETTINGS,
        model_evidence=MODEL,
    )
    evidence = nuclear_motion_evidence(result)
    assert evidence.review.status is ReviewStatus.CLEARED
    assert evidence.component is not None
    assert evidence.component.name == NUCLEAR_MOTION
    assert evidence.component.correction_ev is not None
    assert abs(evidence.component.correction_ev.midpoint - result.correction_ev) < 1.0e-12


def test_nuclear_solver_is_execution_capability_but_pec_extension_is_not_yet_wired():
    solve = ProductionClosureAction(
        'SOLVE_NUCLEAR_MOTION',
        ClosurePriority.PHYSICAL_CORRECTION,
        'G3D_PHYSICAL_CORRECTIONS',
        NUCLEAR_MOTION,
        MethodRole.PRODUCTION,
        ('need-nuclear',),
        'nuclear motion needed',
    )
    req = classify_closure_action(solve)
    assert req.capability is ExecutionCapability.NUCLEAR_MOTION
    assert req.implementation is CapabilityImplementation.GENERIC_RUNNER_AVAILABLE
    assert req.disposition is ExecutionDisposition.NEEDS_BOUND_CONTEXT
    assert req.runner_id.endswith('run_diatomic_nuclear_motion')

    refine = ProductionClosureAction(
        'REFINE_NUCLEAR_PEC',
        ClosurePriority.PHYSICAL_CORRECTION,
        'G3D_PHYSICAL_CORRECTIONS',
        NUCLEAR_MOTION,
        MethodRole.REFINEMENT,
        ('need-more-pec',),
        'PEC needs extension',
    )
    refine_req = classify_closure_action(refine)
    assert refine_req.disposition is ExecutionDisposition.CAPABILITY_GAP
