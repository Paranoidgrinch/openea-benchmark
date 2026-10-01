from dataclasses import replace

from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3PointResult,
)
from openea_benchmark.adaptive.stage3_pec import (
    HighLevelMinimumStatus,
    HighLevelPECStatus,
    HighLevelPointStatus,
    IdentityReviewStatus,
    StateIdentityReview,
    assemble_high_level_pec,
)


def req(r, init=0, root="root_A", *, methods=("CCSD", "CCSD(T)")):
    return Stage3ExecutionRequest(
        request_id=f"job__init{init:02d}_{root}__r{r:.3f}",
        job_id="job",
        system="X",
        atoms=("O", "H"),
        charge=0,
        spin_2s=1,
        component_id="component_A",
        r_angstrom=r,
        basis="def2-TZVPPD",
        methods=methods,
        requested_reference="ROHF",
        scf_reference="ROHF",
        source_link_status="MULTIPLE_DFT_INITIALIZATIONS",
        source_root_id=root,
        source_checkpoint_path=f"/tmp/{root}.chk",
        source_origin_guess="minao",
        grid_index=0,
        initialization_index=init,
        dft_center_r_angstrom=1.0,
        dft_center_energy_hartree=-75.0,
        requires_independent_state_identity_validation=True,
    )


def result(request, energy, *, status=PointExecutionStatus.COMPLETED, hlchk=None):
    return Stage3PointResult(
        request_id=request.request_id,
        job_id=request.job_id,
        status=status,
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
        scf_converged=(status is PointExecutionStatus.COMPLETED),
        scf_energy_hartree=-74.0,
        s2=0.75,
        multiplicity=2.0,
        internal_stable=True,
        external_stable=None,
        external_stability_available=False,
        semicanonicalization="UHF_OCCUPIED_VIRTUAL_CANONICALIZATION",
        cc_class="UCCSD",
        ccsd_converged=(status is PointExecutionStatus.COMPLETED),
        ccsd_correlation_hartree=-0.1,
        ccsd_total_hartree=(energy + 0.001 if energy is not None else None),
        triples_correction_hartree=(-0.001 if energy is not None else None),
        ccsd_t_total_hartree=energy,
        t1_diagnostic=None,
        t1_diagnostic_definition=None,
        error_type=None,
        error_message=None,
        high_level_checkpoint_path=hlchk,
    )


def cleared(ids, evidence="identity:test"):
    return StateIdentityReview(
        status=IdentityReviewStatus.CLEARED,
        covered_request_ids=tuple(ids),
        evidence_ids=(evidence,),
        rationale="Independent orbital/subspace comparison cleared these requests",
    )


def three_single_init(energies=(-75.0, -75.1, -75.0)):
    requests = [req(r, init=0) for r in (0.9, 1.0, 1.1)]
    results = [result(rq, en) for rq, en in zip(requests, energies)]
    continuity = cleared([r.request_id for r in requests], "continuity:test")
    return requests, results, continuity


def test_request_result_bijection_is_required():
    requests, results, continuity = three_single_init()
    try:
        assemble_high_level_pec(requests=requests, results=results[:-1], geometry_continuity_review=continuity)
    except ValueError as exc:
        assert "contract mismatch" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_incomplete_execution_blocks_pec():
    requests, results, continuity = three_single_init()
    results[1] = result(requests[1], None, status=PointExecutionStatus.CCSD_NOT_CONVERGED)
    pec = assemble_high_level_pec(requests=requests, results=results, geometry_continuity_review=continuity)
    assert pec.status is HighLevelPECStatus.EXECUTION_INCOMPLETE
    assert pec.points[1].status is HighLevelPointStatus.EXECUTION_INCOMPLETE
    assert pec.minimum_scout.status is HighLevelMinimumStatus.NOT_EVALUATED


def test_multiple_initializations_are_not_merged_without_identity_evidence():
    a, b = req(1.0, 0, "root_A"), req(1.0, 1, "root_B")
    pec = assemble_high_level_pec(requests=[a, b], results=[result(a, -75.0), result(b, -75.0)])
    assert pec.status is HighLevelPECStatus.INITIALIZATION_REVIEW_REQUIRED
    assert pec.points[0].status is HighLevelPointStatus.INITIALIZATION_IDENTITY_UNRESOLVED


def test_energy_agreement_is_not_identity_proof():
    a, b = req(1.0, 0, "root_A"), req(1.0, 1, "root_B")
    pec = assemble_high_level_pec(requests=[a, b], results=[result(a, -75.0), result(b, -75.0)])
    assert pec.points[0].energy_hartree is None


def test_cleared_duplicate_initializations_use_canonical_not_average():
    a, b = req(1.0, 0, "root_A"), req(1.0, 1, "root_B")
    review = cleared([a.request_id, b.request_id])
    pec = assemble_high_level_pec(
        requests=[a, b],
        results=[result(a, -75.00000001), result(b, -75.00000002)],
        initialization_identity_review=review,
    )
    point = pec.points[0]
    assert point.status is HighLevelPointStatus.ACCEPTED
    assert point.canonical_request_id == min(a.request_id, b.request_id)
    assert point.energy_hartree == -75.00000001
    assert point.initialization_energy_spread_hartree > 0.0


def test_identity_cleared_but_large_energy_disagreement_still_blocks():
    a, b = req(1.0, 0, "root_A"), req(1.0, 1, "root_B")
    review = cleared([a.request_id, b.request_id])
    pec = assemble_high_level_pec(
        requests=[a, b],
        results=[result(a, -75.0), result(b, -74.99)],
        initialization_identity_review=review,
        duplicate_energy_tolerance_hartree=1e-6,
    )
    assert pec.status is HighLevelPECStatus.EXECUTION_INCOMPLETE
    assert pec.points[0].status is HighLevelPointStatus.INITIALIZATION_ENERGY_DISAGREEMENT


def test_geometry_continuity_is_independent_prerequisite():
    requests, results, _ = three_single_init()
    pec = assemble_high_level_pec(requests=requests, results=results)
    assert pec.status is HighLevelPECStatus.GEOMETRY_CONTINUITY_REQUIRED
    assert pec.minimum_scout.status is HighLevelMinimumStatus.NOT_EVALUATED


def test_strict_discrete_minimum_is_found_only_after_continuity_clearance():
    requests, results, continuity = three_single_init()
    pec = assemble_high_level_pec(requests=requests, results=results, geometry_continuity_review=continuity)
    assert pec.status is HighLevelPECStatus.READY_FOR_DISCRETE_MINIMUM_SCOUT
    assert pec.minimum_scout.status is HighLevelMinimumStatus.BRACKETED_SINGLE_MINIMUM
    candidate = pec.minimum_scout.candidates[0]
    assert candidate.r_angstrom == 1.0
    assert candidate.energy_hartree == -75.1


def test_endpoint_low_energy_is_not_called_a_bracketed_minimum():
    requests, results, continuity = three_single_init(energies=(-75.2, -75.1, -75.0))
    pec = assemble_high_level_pec(requests=requests, results=results, geometry_continuity_review=continuity)
    assert pec.minimum_scout.status is HighLevelMinimumStatus.NO_BRACKETED_MINIMUM


def test_multiple_strict_minima_are_retained_not_pruned():
    rs = (0.8, 0.9, 1.0, 1.1, 1.2)
    es = (-75.0, -75.2, -75.0, -75.3, -75.0)
    requests = [req(r, 0) for r in rs]
    results = [result(rq, e) for rq, e in zip(requests, es)]
    continuity = cleared([r.request_id for r in requests], "continuity:test")
    pec = assemble_high_level_pec(requests=requests, results=results, geometry_continuity_review=continuity)
    assert pec.minimum_scout.status is HighLevelMinimumStatus.MULTIPLE_MINIMUM_CANDIDATES
    assert [x.r_angstrom for x in pec.minimum_scout.candidates] == [0.9, 1.1]
    assert pec.authorizes_pruning is False


def test_high_level_checkpoint_paths_are_preserved_for_future_identity_analysis():
    requests, results, continuity = three_single_init()
    results[1] = result(requests[1], -75.1, hlchk="/tmp/high_level.chk")
    pec = assemble_high_level_pec(requests=requests, results=results, geometry_continuity_review=continuity)
    assert pec.points[1].high_level_checkpoint_paths == ("/tmp/high_level.chk",)
    assert pec.is_production_ea is False
    assert pec.ground_state_assigned is False
