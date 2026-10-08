from dataclasses import replace

from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3PointResult,
)
from openea_benchmark.adaptive.stage3_pec import (
    IdentityReviewStatus,
    StateIdentityReview,
    assemble_high_level_pec,
)
from openea_benchmark.adaptive.stage3_refinement import (
    RefinementAction,
    Stage3RefinementSettings,
    build_stage3_refinement_requests,
    plan_stage3_pec_refinement,
)


SETTINGS = Stage3RefinementSettings(
    extension_step_angstrom=0.05,
    target_bracket_width_angstrom=0.06,
    minimum_new_point_separation_angstrom=1.0e-6,
    energy_tie_tolerance_hartree=1.0e-8,
)


def req(r, *, idx=0, root="root_A", job="job"):
    return Stage3ExecutionRequest(
        request_id=f"{job}__{root}__r{r:.6f}__i{idx}",
        job_id=job,
        system="OH",
        atoms=("O", "H"),
        charge=0,
        spin_2s=1,
        component_id="component_A",
        r_angstrom=r,
        basis="def2-TZVPPD",
        methods=("CCSD", "CCSD(T)"),
        requested_reference="ROHF",
        scf_reference="ROHF",
        source_link_status="MULTIPLE_DFT_INITIALIZATIONS",
        source_root_id=root,
        source_checkpoint_path=f"/tmp/{root}.chk",
        source_origin_guess="minao",
        grid_index=idx,
        initialization_index=0,
        dft_center_r_angstrom=1.0,
        dft_center_energy_hartree=-75.0,
        requires_independent_state_identity_validation=True,
    )


def result(request, energy, *, checkpoint=None):
    return Stage3PointResult(
        request_id=request.request_id,
        job_id=request.job_id,
        status=PointExecutionStatus.COMPLETED,
        system=request.system,
        charge=request.charge,
        spin_2s=request.spin_2s,
        component_id=request.component_id,
        r_angstrom=request.r_angstrom,
        basis=request.basis,
        source_root_id=request.source_root_id,
        source_checkpoint_path=request.source_checkpoint_path,
        scf_reference=request.scf_reference,
        cc_reference="SEMICANONICAL_UHF_FROM_ROHF",
        pyscf_version="2.14.0",
        scf_converged=True,
        scf_energy_hartree=energy + 0.1,
        s2=0.75,
        multiplicity=2.0,
        internal_stable=True,
        external_stable=None,
        external_stability_available=False,
        semicanonicalization="UHF_OCCUPIED_VIRTUAL_CANONICALIZATION",
        cc_class="UCCSD",
        ccsd_converged=True,
        ccsd_correlation_hartree=-0.1,
        ccsd_total_hartree=energy + 0.001,
        triples_correction_hartree=-0.001,
        ccsd_t_total_hartree=energy,
        t1_diagnostic=None,
        t1_diagnostic_definition=None,
        error_type=None,
        error_message=None,
        high_level_checkpoint_path=checkpoint,
    )


def cleared(ids, tag="continuity"):
    return StateIdentityReview(
        status=IdentityReviewStatus.CLEARED,
        covered_request_ids=tuple(ids),
        evidence_ids=(tag,),
        rationale="test evidence",
    )


def pec_from(rs, es, *, continuity=True, checkpoints=False):
    requests = [req(r, idx=i) for i, r in enumerate(rs)]
    results = [
        result(rq, e, checkpoint=(f"/tmp/hl_{i}.chk" if checkpoints else None))
        for i, (rq, e) in enumerate(zip(requests, es))
    ]
    review = cleared([item.request_id for item in requests]) if continuity else None
    pec = assemble_high_level_pec(
        requests=requests,
        results=results,
        geometry_continuity_review=review,
    )
    return pec, requests, results


def test_settings_have_no_zero_argument_production_defaults():
    try:
        Stage3RefinementSettings()
    except TypeError:
        pass
    else:
        raise AssertionError("expected explicit refinement settings")


def test_uncleared_pec_waits_instead_of_scheduling_geometry():
    pec, _, _ = pec_from((0.9, 1.0, 1.1), (-75.0, -75.1, -75.0), continuity=False)
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    assert plan.action is RefinementAction.WAIT_FOR_VALID_PEC
    assert plan.proposed_geometries == ()


def test_single_bracket_is_refined_on_both_sides():
    pec, _, _ = pec_from((0.9, 1.0, 1.1), (-75.0, -75.1, -75.0))
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    assert plan.action is RefinementAction.REFINE_SINGLE_BRACKET
    assert [round(x.r_angstrom, 6) for x in plan.proposed_geometries] == [0.95, 1.05]


def test_single_bracket_target_can_finish_geometry_refinement():
    pec, _, _ = pec_from((0.98, 1.0, 1.02), (-75.0, -75.1, -75.0))
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    assert plan.action is RefinementAction.BRACKET_TARGET_MET
    assert not plan.proposed_geometries


def test_multiple_minima_are_all_refined_not_pruned():
    pec, _, _ = pec_from(
        (0.8, 0.9, 1.0, 1.1, 1.2),
        (-75.0, -75.2, -75.0, -75.3, -75.0),
    )
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    assert plan.action is RefinementAction.REFINE_MULTIPLE_BRACKETS
    assert [round(x.r_angstrom, 6) for x in plan.proposed_geometries] == [0.85, 0.95, 1.05, 1.15]
    assert plan.authorizes_pruning is False


def test_lower_boundary_minimum_extends_lower_r():
    pec, _, _ = pec_from((0.9, 1.0, 1.1), (-75.2, -75.1, -75.0))
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    assert plan.action is RefinementAction.EXTEND_LOWER_R
    assert round(plan.proposed_geometries[0].r_angstrom, 6) == 0.85


def test_upper_boundary_minimum_extends_upper_r_like_oh_smoke():
    pec, _, _ = pec_from((0.95, 0.97, 0.99), (-75.0, -75.1, -75.2))
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    assert plan.action is RefinementAction.EXTEND_UPPER_R
    assert round(plan.proposed_geometries[0].r_angstrom, 6) == 1.04


def test_equal_lowest_boundaries_expand_both_directions():
    pec, _, _ = pec_from((0.9, 1.0, 1.1), (-75.2, -75.0, -75.2))
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    assert plan.action is RefinementAction.EXTEND_BOTH_SIDES
    assert [round(x.r_angstrom, 6) for x in plan.proposed_geometries] == [0.85, 1.15]


def test_nonstrict_interior_minimum_is_locally_resolved():
    pec, _, _ = pec_from((0.8, 0.9, 1.0, 1.1), (-75.0, -75.2, -75.2, -75.0))
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    assert plan.action is RefinementAction.REFINE_NONSTRICT_INTERIOR
    assert [round(x.r_angstrom, 6) for x in plan.proposed_geometries] == [0.85, 1.05]


def test_one_point_pec_requests_neighbors_when_minimum_scout_is_insufficient():
    pec, _, _ = pec_from((1.0,), (-75.0,))
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    assert plan.action is RefinementAction.EXTEND_BOTH_SIDES
    assert [round(x.r_angstrom, 6) for x in plan.proposed_geometries] == [0.95, 1.05]


def test_two_point_pec_extends_beyond_lower_energy_endpoint():
    pec, _, _ = pec_from((0.95, 1.0), (-75.0, -75.1))
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    assert plan.action is RefinementAction.EXTEND_UPPER_R
    assert round(plan.proposed_geometries[0].r_angstrom, 6) == 1.05


def test_refinement_plan_never_becomes_ea_or_ground_state():
    pec, _, _ = pec_from((0.9, 1.0, 1.1), (-75.0, -75.1, -75.0))
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    assert plan.is_production_ea is False
    assert plan.ground_state_assigned is False
    assert plan.authorizes_pruning is False


def test_interior_refinement_requests_are_seeded_from_both_sides():
    pec, requests, results = pec_from(
        (0.9, 1.0, 1.1),
        (-75.0, -75.1, -75.0),
        checkpoints=True,
    )
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    new = build_stage3_refinement_requests(
        plan=plan,
        pec=pec,
        prior_requests=requests,
        prior_results=results,
        refinement_round=1,
    )
    # Two new geometries, each continued independently from both bracketing sides.
    assert len(new) == 4
    by_r = {}
    for item in new:
        by_r.setdefault(round(item.r_angstrom, 6), []).append(item)
    assert sorted(by_r) == [0.95, 1.05]
    assert all(len(group) == 2 for group in by_r.values())
    assert all(item.source_link_status == "HIGH_LEVEL_REFINEMENT_SEED" for item in new)
    assert all(item.authorizes_pruning is False for item in new)


def test_boundary_extension_uses_nearest_high_level_checkpoint():
    pec, requests, results = pec_from(
        (0.95, 0.97, 0.99),
        (-75.0, -75.1, -75.2),
        checkpoints=True,
    )
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    new = build_stage3_refinement_requests(
        plan=plan,
        pec=pec,
        prior_requests=requests,
        prior_results=results,
        refinement_round=2,
    )
    assert len(new) == 1
    assert round(new[0].r_angstrom, 6) == 1.04
    assert new[0].source_checkpoint_path == "/tmp/hl_2.chk"
    assert new[0].source_root_id == requests[2].request_id


def test_missing_high_level_checkpoint_fails_closed():
    pec, requests, results = pec_from((0.9, 1.0, 1.1), (-75.0, -75.1, -75.0))
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    try:
        build_stage3_refinement_requests(
            plan=plan,
            pec=pec,
            prior_requests=requests,
            prior_results=results,
            refinement_round=1,
        )
    except ValueError as exc:
        assert "checkpoint" in str(exc).lower()
    else:
        raise AssertionError("expected missing checkpoint to fail closed")


def test_refinement_requests_preserve_basis_by_element_policy():
    base = req(0.9, idx=0)
    base = replace(base, basis='MIXED_TEST', basis_by_element={'O': 'aug-cc-pVTZ', 'H': 'aug-cc-pVQZ'})
    mid = replace(req(1.0, idx=1), basis='MIXED_TEST', basis_by_element={'O': 'aug-cc-pVTZ', 'H': 'aug-cc-pVQZ'})
    upper = replace(req(1.1, idx=2), basis='MIXED_TEST', basis_by_element={'O': 'aug-cc-pVTZ', 'H': 'aug-cc-pVQZ'})
    requests = [base, mid, upper]
    results = [
        result(base, -75.0, checkpoint='/tmp/mixed0.chk'),
        result(mid, -75.1, checkpoint='/tmp/mixed1.chk'),
        result(upper, -75.0, checkpoint='/tmp/mixed2.chk'),
    ]
    review = cleared([item.request_id for item in requests])
    pec = assemble_high_level_pec(
        requests=requests,
        results=results,
        geometry_continuity_review=review,
    )
    plan = plan_stage3_pec_refinement(pec, settings=SETTINGS)
    new = build_stage3_refinement_requests(
        plan=plan,
        pec=pec,
        prior_requests=requests,
        prior_results=results,
        refinement_round=1,
    )
    assert new
    assert all(item.basis == 'MIXED_TEST' for item in new)
    assert all(item.basis_by_element == {'O': 'aug-cc-pVTZ', 'H': 'aug-cc-pVQZ'} for item in new)
