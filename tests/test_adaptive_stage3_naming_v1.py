from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3PointResult,
)
from openea_benchmark.adaptive.stage3_naming import (
    CHECKPOINT_BASENAME_MAX_CHARS,
    REQUEST_ID_MAX_CHARS,
    checkpoint_artifact_basename,
    refinement_request_id,
)
from openea_benchmark.adaptive.stage3_pec import (
    IdentityReviewStatus,
    StateIdentityReview,
    assemble_high_level_pec,
)
from openea_benchmark.adaptive.stage3_refinement import (
    Stage3RefinementSettings,
    build_stage3_refinement_requests,
    plan_stage3_pec_refinement,
)


def test_refinement_request_id_is_bounded_and_deterministic():
    seed = "seed__" + "ancestor_" * 200
    kwargs = dict(
        job_id="very_long_job__" + "x" * 400,
        refinement_round=57,
        geometry_index=12,
        seed_index=1,
        seed_request_id=seed,
        target_r_angstrom=1.23456789,
    )
    a = refinement_request_id(**kwargs)
    b = refinement_request_id(**kwargs)
    assert a == b
    assert len(a) <= REQUEST_ID_MAX_CHARS
    assert seed not in a
    assert "__refine57__g012__seed01__" in a


def test_refinement_request_id_changes_when_lineage_changes():
    common = dict(
        job_id="job",
        refinement_round=2,
        geometry_index=0,
        seed_index=0,
        target_r_angstrom=1.05,
    )
    a = refinement_request_id(seed_request_id="parent_A", **common)
    b = refinement_request_id(seed_request_id="parent_B", **common)
    assert a != b


def test_checkpoint_artifact_basename_is_bounded_and_collision_resistant():
    a = checkpoint_artifact_basename("request__" + "a" * 5000)
    b = checkpoint_artifact_basename("request__" + "a" * 4999 + "b")
    assert len(a) <= CHECKPOINT_BASENAME_MAX_CHARS
    assert len(b) <= CHECKPOINT_BASENAME_MAX_CHARS
    assert a.endswith(".hf.chk")
    assert a != b
    assert a == checkpoint_artifact_basename("request__" + "a" * 5000)


def _request(r, i, *, job, request_id):
    return Stage3ExecutionRequest(
        request_id=request_id,
        job_id=job,
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
        source_link_status="HIGH_LEVEL_REFINEMENT_SEED",
        source_root_id="root_A",
        source_checkpoint_path="/tmp/source.chk",
        source_origin_guess="test",
        grid_index=i,
        initialization_index=0,
        dft_center_r_angstrom=1.0,
        dft_center_energy_hartree=-75.0,
        requires_independent_state_identity_validation=True,
    )


def _result(req, energy, checkpoint):
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


def test_refinement_builder_preserves_full_parent_provenance_without_recursive_id_growth():
    job = "job__" + "J" * 350
    parent_ids = [
        "parent__" + ("lineage_" * 80) + str(i)
        for i in range(3)
    ]
    requests = [
        _request(r, i, job=job, request_id=parent_ids[i])
        for i, r in enumerate((0.9, 1.0, 1.1))
    ]
    results = [
        _result(req, energy, f"/tmp/high_{i}.chk")
        for i, (req, energy) in enumerate(zip(requests, (-75.0, -75.1, -75.0)))
    ]
    review = StateIdentityReview(
        status=IdentityReviewStatus.CLEARED,
        covered_request_ids=tuple(parent_ids),
        evidence_ids=("continuity",),
        rationale="test",
    )
    pec = assemble_high_level_pec(
        requests=requests,
        results=results,
        geometry_continuity_review=review,
    )
    plan = plan_stage3_pec_refinement(
        pec,
        settings=Stage3RefinementSettings(
            extension_step_angstrom=0.05,
            target_bracket_width_angstrom=0.06,
            minimum_new_point_separation_angstrom=1e-6,
            energy_tie_tolerance_hartree=1e-8,
        ),
    )
    new = build_stage3_refinement_requests(
        plan=plan,
        pec=pec,
        prior_requests=requests,
        prior_results=results,
        refinement_round=9,
    )
    assert len(new) == 4
    assert all(len(item.request_id) <= REQUEST_ID_MAX_CHARS for item in new)
    assert all(item.source_root_id in parent_ids for item in new)
    assert all(item.source_root_id not in item.request_id for item in new)
