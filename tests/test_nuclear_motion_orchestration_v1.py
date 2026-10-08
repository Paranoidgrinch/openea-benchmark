from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from openea_benchmark.adaptive import (
    ANGSTROM_TO_BOHR,
    CapabilityImplementation,
    ClosurePriority,
    DiatomicMassSpecification,
    ExecutionCapability,
    ExecutionDisposition,
    MethodRole,
    NUCLEAR_MOTION,
    NuclearMotionSettings,
    NuclearMotionStatus,
    NuclearPECModelLevel,
    NuclearPECModelRunStatus,
    NuclearPECRefinementRunStatus,
    NuclearPECStage3Context,
    ProductionClosureAction,
    VibrationalSolveStatus,
    classify_closure_action,
    nuclear_motion_pec_from_high_level,
    plan_nuclear_pec_refinement,
    run_diatomic_nuclear_motion,
    run_nuclear_pec_model_convergence,
    run_nuclear_pec_refinement,
)
from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3ExecutionSettings,
    Stage3PointResult,
)
from openea_benchmark.adaptive.stage3_pec import (
    IdentityReviewStatus,
    StateIdentityReview,
    assemble_high_level_pec,
)
from openea_benchmark.adaptive.stage3_refinement import Stage3RefinementSettings


MASS = DiatomicMassSpecification('H', 'H', 1.0, 1.0, ('MASS:H1',), mass_model='TEST')
NSET = NuclearMotionSettings(
    coarse_grid_points=101,
    fine_grid_points=201,
    grid_convergence_tolerance_ev=2.0e-4,
    interpolation_sensitivity_tolerance_ev=2.0e-3,
    domain_sensitivity_tolerance_ev=2.0e-3,
    minimum_boundary_clearance_ev=0.05,
    trim_fraction=0.08,
    eigenvalue_tolerance_hartree=1.0e-11,
)
RSET = Stage3RefinementSettings(
    extension_step_angstrom=0.10,
    target_bracket_width_angstrom=0.05,
    minimum_new_point_separation_angstrom=1.0e-8,
    energy_tie_tolerance_hartree=1.0e-9,
)
ESET = Stage3ExecutionSettings(
    checkpoint_project=True,
    frozen_core=False,
    scalar_relativistic='NONE',
    verbose=0,
)


def _energy(role, r, basis='aug-cc-pVQZ'):
    shift = -75.0 if role == 'neutral' else -75.50
    curvature = 0.40 if role == 'neutral' else 0.32
    if basis == 'aug-cc-pVTZ':
        curvature *= 0.985
        shift += 2.0e-5 if role == 'neutral' else -1.0e-5
    return shift + curvature * (float(r) - 1.0) ** 2


def _request(role, r, idx, *, basis='aug-cc-pVQZ', job=None, basis_map=True):
    charge = 0 if role == 'neutral' else -1
    spin = 1 if role == 'neutral' else 0
    job = job or f'XY__{role}'
    return Stage3ExecutionRequest(
        request_id=f'{job}__g{idx:03d}',
        job_id=job,
        system='XY',
        atoms=('H', 'H'),
        charge=charge,
        spin_2s=spin,
        component_id=f'{role}_component',
        r_angstrom=float(r),
        basis=basis,
        basis_by_element=({'H': basis} if basis_map else None),
        methods=('CCSD', 'CCSD(T)'),
        requested_reference='ROHF',
        scf_reference=('RHF' if spin == 0 else 'ROHF'),
        source_link_status='SINGLE_DFT_INITIALIZATION',
        source_root_id=f'{role}_root_{idx}',
        source_checkpoint_path=f'/tmp/{role}_source_{idx}.chk',
        source_origin_guess='minao',
        grid_index=idx,
        initialization_index=0,
        dft_center_r_angstrom=1.0,
        dft_center_energy_hartree=_energy(role, 1.0, basis),
        requires_independent_state_identity_validation=True,
    )


def _result(req, *, basis=None):
    basis = basis or req.basis
    role = 'neutral' if req.charge == 0 else 'anion'
    energy = _energy(role, req.r_angstrom, basis)
    return Stage3PointResult(
        request_id=req.request_id,
        job_id=req.job_id,
        status=PointExecutionStatus.COMPLETED,
        system=req.system,
        charge=req.charge,
        spin_2s=req.spin_2s,
        component_id=req.component_id,
        r_angstrom=req.r_angstrom,
        basis=req.basis,
        source_root_id=req.source_root_id,
        source_checkpoint_path=req.source_checkpoint_path,
        scf_reference=req.scf_reference,
        cc_reference=('RHF' if req.spin_2s == 0 else 'SEMICANONICAL_UHF_FROM_ROHF'),
        pyscf_version='TEST',
        scf_converged=True,
        scf_energy_hartree=energy + 0.1,
        s2=(0.0 if req.spin_2s == 0 else 0.75),
        multiplicity=(1.0 if req.spin_2s == 0 else 2.0),
        internal_stable=True,
        external_stable=(True if req.spin_2s == 0 else None),
        external_stability_available=(req.spin_2s == 0),
        semicanonicalization='TEST',
        cc_class='TEST_CCSD',
        ccsd_converged=True,
        ccsd_correlation_hartree=-0.1,
        ccsd_total_hartree=energy + 0.001,
        triples_correction_hartree=-0.001,
        ccsd_t_total_hartree=energy,
        t1_diagnostic=None,
        t1_diagnostic_definition=None,
        error_type=None,
        error_message=None,
        high_level_checkpoint_path=f'/tmp/{req.request_id}.hl.chk',
    )


def _review(requests, tag):
    return StateIdentityReview(
        IdentityReviewStatus.CLEARED,
        tuple(item.request_id for item in requests),
        (tag,),
        'synthetic cleared identity evidence',
    )


def _context(role, *, rs=None, basis='aug-cc-pVQZ', energy_override=None, basis_map=True):
    rs = tuple(rs or np.linspace(0.55, 1.45, 11))
    requests = tuple(_request(role, r, i, basis=basis, basis_map=basis_map) for i, r in enumerate(rs))
    results = []
    for req in requests:
        res = _result(req, basis=basis)
        if energy_override is not None:
            e = float(energy_override(req.r_angstrom))
            res = Stage3PointResult(**{**res.__dict__, 'ccsd_t_total_hartree': e})
        results.append(res)
    review = _review(requests, f'IDENTITY:{role}:{basis}')
    pec = assemble_high_level_pec(
        requests=requests,
        results=tuple(results),
        initialization_identity_review=review,
        geometry_continuity_review=review,
    )
    return NuclearPECStage3Context(role, f'{role}-state', pec, requests, tuple(results))


def _fake_runner(req, settings):
    return _result(req)


def _fake_resolver(**kwargs):
    requests = tuple(kwargs['requests'])
    results = tuple(kwargs['results'])
    review = _review(requests, f"FAKE_RESOLVE:{requests[0].job_id}")
    pec = assemble_high_level_pec(
        requests=requests,
        results=results,
        initialization_identity_review=review,
        geometry_continuity_review=review,
    )
    return SimpleNamespace(pec=pec)


def _assessment(neutral_ctx, anion_ctx, *, model=None):
    npec = nuclear_motion_pec_from_high_level(
        neutral_ctx.pec, role='neutral', state_id=neutral_ctx.state_id, evidence_ids=('PRIMARY:N',)
    )
    apec = nuclear_motion_pec_from_high_level(
        anion_ctx.pec, role='anion', state_id=anion_ctx.state_id, evidence_ids=('PRIMARY:A',)
    )
    return run_diatomic_nuclear_motion(
        neutral_pec=npec,
        anion_pec=apec,
        masses=MASS,
        settings=NSET,
        model_evidence=model,
    )


def test_stage3_refinement_preserves_element_specific_basis_policy():
    neutral = _context('neutral', basis_map=True)
    # Force a nuclear lower-boundary refinement request by constructing a PEC
    # whose electronic minimum is at the lower boundary.
    boundary = _context(
        'neutral',
        rs=(0.7, 0.8, 0.9, 1.0, 1.1),
        energy_override=lambda r: -75.0 + 0.1 * (r - 0.7),
        basis_map=True,
    )
    anion = _context('anion')
    assessment = _assessment(boundary, anion)
    assert assessment.status is NuclearMotionStatus.PEC_REFINEMENT_REQUIRED

    run = run_nuclear_pec_refinement(
        assessment=assessment,
        neutral_context=boundary,
        anion_context=anion,
        nuclear_settings=NSET,
        refinement_settings=RSET,
        refinement_round=1,
        execution_settings=ESET,
        identity_thresholds=object(),
        branch_thresholds=object(),
        point_runner=_fake_runner,
        resolver=_fake_resolver,
    )
    assert run.status is NuclearPECRefinementRunStatus.COMPLETED
    new_requests = [r for b in run.batches for r in b.requests]
    assert new_requests
    assert all(r.basis_by_element == {'H': 'aug-cc-pVQZ'} for r in new_requests)
    assert len(run.neutral_context.requests) > len(boundary.requests)
    assert run.anion_context.requests == anion.requests


def test_refine_nuclear_pec_is_now_a_generic_stage3_backed_capability():
    action = ProductionClosureAction(
        'REFINE_NUCLEAR_PEC', ClosurePriority.PHYSICAL_CORRECTION,
        'G3D_PHYSICAL_CORRECTIONS', NUCLEAR_MOTION, MethodRole.REFINEMENT,
        ('D12:REFINE',), 'need more electronic PEC support',
    )
    request = classify_closure_action(action)
    assert request.capability is ExecutionCapability.NUCLEAR_PEC_REFINEMENT
    assert request.implementation is CapabilityImplementation.GENERIC_RUNNER_AVAILABLE
    assert request.disposition is ExecutionDisposition.NEEDS_BOUND_CONTEXT
    assert request.runner_id.endswith('run_nuclear_pec_refinement')


def test_model_convergence_action_is_generic_but_requires_explicit_bound_context():
    action = ProductionClosureAction(
        'ASSESS_NUCLEAR_PEC_MODEL_CONVERGENCE', ClosurePriority.PHYSICAL_CORRECTION,
        'G3D_PHYSICAL_CORRECTIONS', NUCLEAR_MOTION, MethodRole.REFINEMENT,
        ('D12:MODEL',), 'bound DeltaZPE model sensitivity',
    )
    request = classify_closure_action(action)
    assert request.capability is ExecutionCapability.NUCLEAR_PEC_MODEL_CONVERGENCE
    assert request.implementation is CapabilityImplementation.GENERIC_RUNNER_AVAILABLE
    assert request.disposition is ExecutionDisposition.NEEDS_BOUND_CONTEXT
    assert request.runner_id.endswith('run_nuclear_pec_model_convergence')


def test_model_runner_requires_explicit_authorization_and_projection():
    neutral = _context('neutral')
    anion = _context('anion')
    primary = _assessment(neutral, anion)
    assert primary.status is NuclearMotionStatus.MODEL_CONVERGENCE_REQUIRED
    model = NuclearPECModelLevel(
        'TZ-check', 'aug-cc-pVTZ', {'H': 'aug-cc-pVTZ'}, ('CCSD', 'CCSD(T)'),
        ('PRECISION_CONTROLLER:AUTHORIZED_TZ_D12_COMPARISON',), 'explicit test comparison',
    )
    bad_settings = Stage3ExecutionSettings(
        checkpoint_project=False, frozen_core=False, scalar_relativistic='NONE', verbose=0,
    )
    try:
        run_nuclear_pec_model_convergence(
            primary_assessment=primary,
            primary_neutral_context=neutral,
            primary_anion_context=anion,
            model=model,
            masses=MASS,
            nuclear_settings=NSET,
            execution_settings=bad_settings,
            identity_thresholds=object(),
            branch_thresholds=object(),
            point_runner=_fake_runner,
            resolver=_fake_resolver,
        )
    except ValueError as exc:
        assert 'checkpoint_project=True' in str(exc)
    else:
        raise AssertionError('expected explicit basis-projection requirement')


def test_generated_comparison_level_stops_for_cross_model_identity_evidence():
    neutral = _context('neutral')
    anion = _context('anion')
    primary = _assessment(neutral, anion)
    model = NuclearPECModelLevel(
        'TZ-check', 'aug-cc-pVTZ', {'H': 'aug-cc-pVTZ'}, ('CCSD', 'CCSD(T)'),
        ('PRECISION_CONTROLLER:AUTHORIZED_TZ_D12_COMPARISON',), 'explicit test comparison',
    )
    run = run_nuclear_pec_model_convergence(
        primary_assessment=primary,
        primary_neutral_context=neutral,
        primary_anion_context=anion,
        model=model,
        masses=MASS,
        nuclear_settings=NSET,
        execution_settings=ESET,
        identity_thresholds=object(),
        branch_thresholds=object(),
        point_runner=_fake_runner,
        resolver=_fake_resolver,
    )
    assert run.status is NuclearPECModelRunStatus.CROSS_MODEL_IDENTITY_REVIEW_REQUIRED
    assert run.comparison_neutral_context is not None
    assert run.comparison_anion_context is not None
    assert run.model_evidence is None
    assert run.final_primary_assessment is None
    assert all(
        req.basis == 'aug-cc-pVTZ'
        for req in run.comparison_neutral_context.requests + run.comparison_anion_context.requests
    )


def test_cross_model_identity_evidence_allows_delta_zpe_model_bound_and_final_primary_clearance():
    neutral = _context('neutral')
    anion = _context('anion')
    primary = _assessment(neutral, anion)
    model = NuclearPECModelLevel(
        'TZ-check', 'aug-cc-pVTZ', {'H': 'aug-cc-pVTZ'}, ('CCSD', 'CCSD(T)'),
        ('PRECISION_CONTROLLER:AUTHORIZED_TZ_D12_COMPARISON',), 'explicit test comparison',
    )
    run = run_nuclear_pec_model_convergence(
        primary_assessment=primary,
        primary_neutral_context=neutral,
        primary_anion_context=anion,
        model=model,
        masses=MASS,
        nuclear_settings=NSET,
        execution_settings=ESET,
        identity_thresholds=object(),
        branch_thresholds=object(),
        cross_model_identity_evidence_ids=('STATE_IDENTITY:CROSS_BASIS_CLEARED',),
        point_runner=_fake_runner,
        resolver=_fake_resolver,
    )
    assert run.status is NuclearPECModelRunStatus.MODEL_EVIDENCE_READY
    assert run.model_evidence is not None
    assert run.model_evidence.delta_zpe_half_width_ev >= 0.0
    assert run.final_primary_assessment is not None
    assert run.final_primary_assessment.status is NuclearMotionStatus.CLEARED


def test_existing_comparison_pecs_can_be_reused_without_new_stage3_execution():
    neutral = _context('neutral')
    anion = _context('anion')
    primary = _assessment(neutral, anion)
    cmp_n = _context('neutral', basis='aug-cc-pVTZ')
    cmp_a = _context('anion', basis='aug-cc-pVTZ')
    model = NuclearPECModelLevel(
        'TZ-check', 'aug-cc-pVTZ', {'H': 'aug-cc-pVTZ'}, ('CCSD', 'CCSD(T)'),
        ('REUSE:EXISTING_STAGE3_TZ_PECS',), 'reuse explicit comparison PECs',
    )

    def forbidden_runner(req, settings):
        raise AssertionError('existing comparison PECs must be reused, not recomputed')

    run = run_nuclear_pec_model_convergence(
        primary_assessment=primary,
        primary_neutral_context=neutral,
        primary_anion_context=anion,
        model=model,
        masses=MASS,
        nuclear_settings=NSET,
        execution_settings=ESET,
        identity_thresholds=object(),
        branch_thresholds=object(),
        cross_model_identity_evidence_ids=('STATE_IDENTITY:CROSS_BASIS_CLEARED',),
        existing_comparison_neutral_context=cmp_n,
        existing_comparison_anion_context=cmp_a,
        point_runner=forbidden_runner,
        resolver=_fake_resolver,
    )
    assert run.status is NuclearPECModelRunStatus.MODEL_EVIDENCE_READY
    assert run.attempts == ()


def test_model_runner_rejects_implicit_same_level_comparison():
    neutral = _context('neutral')
    anion = _context('anion')
    primary = _assessment(neutral, anion)
    model = NuclearPECModelLevel(
        'same', 'aug-cc-pVQZ', None, ('CCSD', 'CCSD(T)'),
        ('BAD:AUTO',), 'same level must not pretend to be model convergence',
    )
    try:
        run_nuclear_pec_model_convergence(
            primary_assessment=primary,
            primary_neutral_context=neutral,
            primary_anion_context=anion,
            model=model,
            masses=MASS,
            nuclear_settings=NSET,
            execution_settings=ESET,
            identity_thresholds=object(),
            branch_thresholds=object(),
            point_runner=_fake_runner,
            resolver=_fake_resolver,
        )
    except ValueError as exc:
        assert 'must differ explicitly' in str(exc)
    else:
        raise AssertionError('same model must not count as convergence evidence')


def test_real_refinement_path_requires_checkpoint_artifact_directory_before_execution():
    boundary = _context(
        'neutral',
        rs=(0.7, 0.8, 0.9, 1.0, 1.1),
        energy_override=lambda r: -75.0 + 0.1 * (r - 0.7),
    )
    anion = _context('anion')
    assessment = _assessment(boundary, anion)
    try:
        run_nuclear_pec_refinement(
            assessment=assessment,
            neutral_context=boundary,
            anion_context=anion,
            nuclear_settings=NSET,
            refinement_settings=RSET,
            refinement_round=1,
            execution_settings=ESET,
            identity_thresholds=object(),
            branch_thresholds=object(),
            point_runner=_fake_runner,
            resolver=None,
        )
    except ValueError as exc:
        assert 'artifact_dir' in str(exc)
    else:
        raise AssertionError('real identity resolver requires retained high-level checkpoints')


def test_model_level_requires_explicit_authorization_evidence():
    try:
        NuclearPECModelLevel(
            'TZ-check', 'aug-cc-pVTZ', {'H': 'aug-cc-pVTZ'}, ('CCSD', 'CCSD(T)'),
            (), 'missing authorization must fail',
        )
    except ValueError as exc:
        assert 'authorization evidence' in str(exc)
    else:
        raise AssertionError('comparison electronic level must never be auto-authorized')


def test_interpolation_sensitivity_densifies_existing_pec_without_extrapolation():
    neutral = _context('neutral')
    anion = _context('anion')
    base = _assessment(neutral, anion)
    flagged_neutral = replace(
        base.neutral,
        status=VibrationalSolveStatus.PEC_REFINEMENT_REQUIRED,
        interpolation_difference_ev=NSET.interpolation_sensitivity_tolerance_ev * 2.0,
        requires_lower_r_extension=False,
        requires_upper_r_extension=False,
        rationale='synthetic interpolation sensitivity',
    )
    flagged = replace(
        base,
        status=NuclearMotionStatus.PEC_REFINEMENT_REQUIRED,
        neutral=flagged_neutral,
        correction_ev=None,
        correction_half_width_ev=None,
        action='REFINE_NUCLEAR_PEC',
        rationale='synthetic D12 PEC density refinement request',
    )
    batches = plan_nuclear_pec_refinement(
        assessment=flagged,
        neutral_context=neutral,
        anion_context=anion,
        nuclear_settings=NSET,
        refinement_settings=RSET,
        refinement_round=1,
    )
    assert len(batches) == 1
    proposed = [item.r_angstrom for item in batches[0].plan.proposed_geometries]
    source_r = [item.r_angstrom for item in neutral.pec.points]
    assert len(proposed) == len(source_r) - 1
    assert min(proposed) > min(source_r)
    assert max(proposed) < max(source_r)
