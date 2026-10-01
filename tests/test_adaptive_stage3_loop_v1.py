from types import SimpleNamespace

from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3PointResult,
)
from openea_benchmark.adaptive.stage3_loop import (
    Stage3LoopStatus,
    run_stage3_refinement_loop,
)
from openea_benchmark.adaptive.stage3_pec import (
    IdentityReviewStatus,
    StateIdentityReview,
    assemble_high_level_pec,
)
from openea_benchmark.adaptive.stage3_refinement import Stage3RefinementSettings


REFINE = Stage3RefinementSettings(
    extension_step_angstrom=0.05,
    target_bracket_width_angstrom=0.06,
    minimum_new_point_separation_angstrom=1.0e-7,
    energy_tie_tolerance_hartree=1.0e-10,
)


def energy(r):
    return -75.1 + (float(r) - 1.0) ** 2


def request(r, index, *, request_id=None, checkpoint=None):
    rid = request_id or f"job__g{index:03d}"
    return Stage3ExecutionRequest(
        request_id=rid,
        job_id="job",
        system="OH",
        atoms=("O", "H"),
        charge=0,
        spin_2s=1,
        component_id="component_A",
        r_angstrom=r,
        basis="sto-3g",
        methods=("CCSD", "CCSD(T)"),
        requested_reference="ROHF",
        scf_reference="ROHF",
        source_link_status="SINGLE_DFT_INITIALIZATION",
        source_root_id=f"root_{index}",
        source_checkpoint_path=checkpoint or f"/tmp/source_{index}.chk",
        source_origin_guess="test",
        grid_index=index,
        initialization_index=0,
        dft_center_r_angstrom=1.0,
        dft_center_energy_hartree=-75.0,
        requires_independent_state_identity_validation=True,
    )


def completed(req, *, e=None, checkpoint=None):
    e = energy(req.r_angstrom) if e is None else e
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
        cc_reference="SEMICANONICAL_UHF_FROM_ROHF",
        pyscf_version="2.14.0",
        scf_converged=True,
        scf_energy_hartree=e + 0.1,
        s2=0.75,
        multiplicity=2.0,
        internal_stable=True,
        external_stable=None,
        external_stability_available=False,
        semicanonicalization="UHF_OCCUPIED_VIRTUAL_CANONICALIZATION",
        cc_class="UCCSD",
        ccsd_converged=True,
        ccsd_correlation_hartree=-0.1,
        ccsd_total_hartree=e + 0.001,
        triples_correction_hartree=-0.001,
        ccsd_t_total_hartree=e,
        t1_diagnostic=None,
        t1_diagnostic_definition=None,
        error_type=None,
        error_message=None,
        high_level_checkpoint_path=checkpoint or f"/tmp/high_{req.request_id}.chk",
    )


def clear_review(ids, tag):
    return StateIdentityReview(
        status=IdentityReviewStatus.CLEARED,
        covered_request_ids=tuple(ids),
        evidence_ids=(tag,),
        rationale="synthetic orchestration evidence",
    )


def resolver(*, requests, results, **_):
    ids = [item.request_id for item in requests]
    init = clear_review(ids, "init")
    continuity = clear_review(ids, "continuity")
    pec = assemble_high_level_pec(
        requests=requests,
        results=results,
        initialization_identity_review=init,
        geometry_continuity_review=continuity,
    )
    return SimpleNamespace(
        initialization_review=init,
        geometry_continuity_review=continuity,
        pec=pec,
    )


def initial():
    reqs = [request(r, i) for i, r in enumerate((0.9, 1.0, 1.1))]
    return reqs, [completed(item) for item in reqs]


def runner(req, _settings):
    return completed(req)


def test_two_refinement_batches_converge_to_explicit_bracket_target():
    reqs, results = initial()
    out = run_stage3_refinement_loop(
        initial_requests=reqs,
        initial_results=results,
        refinement_settings=REFINE,
        identity_thresholds=object(),
        branch_thresholds=object(),
        max_refinement_rounds=3,
        runner=runner,
        resolver=resolver,
    )
    assert out.status is Stage3LoopStatus.CONVERGED
    assert len(out.rounds) == 3
    assert out.rounds[0].proposed_r_angstrom == (0.95, 1.05)
    assert out.rounds[1].proposed_r_angstrom == (0.975, 1.025)
    assert out.rounds[2].refinement_action == "BRACKET_TARGET_MET"
    assert out.final_pec.minimum_scout.candidates[0].left_r_angstrom == 0.975
    assert out.final_pec.minimum_scout.candidates[0].right_r_angstrom == 1.025


def test_interior_new_geometries_keep_both_independent_seeds():
    reqs, results = initial()
    out = run_stage3_refinement_loop(
        initial_requests=reqs,
        initial_results=results,
        refinement_settings=REFINE,
        identity_thresholds=None,
        branch_thresholds=None,
        max_refinement_rounds=1,
        runner=runner,
        resolver=resolver,
    )
    assert out.status is Stage3LoopStatus.ROUND_LIMIT_REACHED
    refined = [r for r in out.requests if "__refine01" in r.request_id]
    assert len(refined) == 4
    assert sorted(round(r.r_angstrom, 6) for r in refined) == [0.95, 0.95, 1.05, 1.05]
    assert all(r.authorizes_pruning is False for r in refined)


def test_round_limit_is_not_reported_as_convergence():
    reqs, results = initial()
    out = run_stage3_refinement_loop(
        initial_requests=reqs,
        initial_results=results,
        refinement_settings=REFINE,
        identity_thresholds=None,
        branch_thresholds=None,
        max_refinement_rounds=0,
        runner=runner,
        resolver=resolver,
    )
    assert out.status is Stage3LoopStatus.ROUND_LIMIT_REACHED
    assert out.final_refinement_plan.action.value == "REFINE_SINGLE_BRACKET"


def test_execution_failure_is_terminal_and_distinct_from_scientific_unresolved():
    reqs, results = initial()
    calls = 0

    def failing_runner(req, settings):
        nonlocal calls
        calls += 1
        if calls == 1:
            base = completed(req)
            return Stage3PointResult(**{
                **base.__dict__,
                "status": PointExecutionStatus.ERROR,
                "ccsd_converged": None,
                "ccsd_t_total_hartree": None,
                "error_type": "SyntheticFailure",
                "error_message": "test",
            })
        return completed(req)

    out = run_stage3_refinement_loop(
        initial_requests=reqs,
        initial_results=results,
        refinement_settings=REFINE,
        identity_thresholds=None,
        branch_thresholds=None,
        max_refinement_rounds=3,
        runner=failing_runner,
        resolver=resolver,
    )
    assert out.status is Stage3LoopStatus.EXECUTION_BLOCKED
    assert "ERROR" in out.rounds[-1].new_result_statuses


def test_unresolved_identity_stops_without_new_calculation():
    reqs, results = initial()

    def unresolved_resolver(*, requests, results, **_):
        init = StateIdentityReview(
            status=IdentityReviewStatus.UNRESOLVED,
            covered_request_ids=tuple(r.request_id for r in requests),
            evidence_ids=(),
            rationale="ambiguous identity",
        )
        pec = assemble_high_level_pec(
            requests=requests,
            results=results,
            initialization_identity_review=init,
            geometry_continuity_review=None,
        )
        return SimpleNamespace(
            initialization_review=init,
            geometry_continuity_review=StateIdentityReview(
                status=IdentityReviewStatus.PENDING,
                covered_request_ids=(),
                evidence_ids=(),
                rationale="not attempted",
            ),
            pec=pec,
        )

    out = run_stage3_refinement_loop(
        initial_requests=reqs,
        initial_results=results,
        refinement_settings=REFINE,
        identity_thresholds=None,
        branch_thresholds=None,
        max_refinement_rounds=3,
        runner=runner,
        resolver=unresolved_resolver,
    )
    assert out.status is Stage3LoopStatus.SCIENTIFICALLY_UNRESOLVED
    assert len(out.requests) == len(reqs)
    assert out.rounds[-1].refinement_action == "WAIT_FOR_VALID_PEC"


def test_initial_request_result_mismatch_fails_closed():
    reqs, results = initial()
    try:
        run_stage3_refinement_loop(
            initial_requests=reqs,
            initial_results=results[:-1],
            refinement_settings=REFINE,
            identity_thresholds=None,
            branch_thresholds=None,
            max_refinement_rounds=1,
            resolver=resolver,
        )
    except ValueError as exc:
        assert "do not match" in str(exc)
    else:
        raise AssertionError("expected request/result mismatch to fail")


def test_loop_never_assigns_ea_ground_state_or_pruning_authority():
    reqs, results = initial()
    out = run_stage3_refinement_loop(
        initial_requests=reqs,
        initial_results=results,
        refinement_settings=REFINE,
        identity_thresholds=None,
        branch_thresholds=None,
        max_refinement_rounds=3,
        runner=runner,
        resolver=resolver,
    )
    assert out.is_production_ea is False
    assert out.ground_state_assigned is False
    assert out.authorizes_pruning is False
    assert all(r.authorizes_pruning is False for r in out.requests)


def test_accumulated_request_and_result_ids_remain_unique():
    reqs, results = initial()
    out = run_stage3_refinement_loop(
        initial_requests=reqs,
        initial_results=results,
        refinement_settings=REFINE,
        identity_thresholds=None,
        branch_thresholds=None,
        max_refinement_rounds=3,
        runner=runner,
        resolver=resolver,
    )
    request_ids = [x.request_id for x in out.requests]
    result_ids = [x.request_id for x in out.results]
    assert len(request_ids) == len(set(request_ids))
    assert len(result_ids) == len(set(result_ids))
    assert set(request_ids) == set(result_ids)


def test_negative_round_limit_is_rejected():
    reqs, results = initial()
    try:
        run_stage3_refinement_loop(
            initial_requests=reqs,
            initial_results=results,
            refinement_settings=REFINE,
            identity_thresholds=None,
            branch_thresholds=None,
            max_refinement_rounds=-1,
            resolver=resolver,
        )
    except ValueError as exc:
        assert "max_refinement_rounds" in str(exc)
    else:
        raise AssertionError("expected invalid round limit")


def test_duplicate_energy_tolerance_is_validated():
    reqs, results = initial()
    try:
        run_stage3_refinement_loop(
            initial_requests=reqs,
            initial_results=results,
            refinement_settings=REFINE,
            identity_thresholds=None,
            branch_thresholds=None,
            max_refinement_rounds=1,
            duplicate_energy_tolerance_hartree=-1.0,
            resolver=resolver,
        )
    except ValueError as exc:
        assert "duplicate_energy_tolerance" in str(exc)
    else:
        raise AssertionError("expected invalid tolerance")
