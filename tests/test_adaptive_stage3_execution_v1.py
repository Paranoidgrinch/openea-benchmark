from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3ExecutionSettings,
    Stage3PointResult,
    build_stage3_execution_requests,
    run_stage3_point,
)


def fixture(*, multiple=False, open_shell=False):
    spin = 1 if open_shell else 0
    source_candidates = [
        {"root_id": "root_A", "checkpoint_path": "/tmp/a.chk", "origin_guess": "minao"}
    ]
    status = "SINGLE_DFT_INITIALIZATION"
    if multiple:
        status = "MULTIPLE_DFT_INITIALIZATIONS"
        source_candidates.append(
            {"root_id": "root_B", "checkpoint_path": "/tmp/b.chk", "origin_guess": "atom"}
        )
    job = {
        "job_id": f"X__q+0__2S{spin}__component_A",
        "system": "X",
        "charge": 0,
        "spin_2s": spin,
        "component_id": "component_A",
        "reference": "ROHF",
        "methods": ["CCSD", "CCSD(T)"],
        "basis": "def2-TZVPPD",
        "local_grid_angstrom": [1.0, 1.1, 1.2],
        "dft_center_r_angstrom": 1.1,
        "dft_center_energy_hartree": -10.0,
        "source_provenance": {
            "link_status": status,
            "checkpoint_candidates": source_candidates,
        },
        "requires_independent_state_identity_validation": True,
        "execution_status": "PLANNED_NOT_RUN",
    }
    plan = {
        "system": "X",
        "jobs": [job],
        "automatic_pruning_performed": False,
    }
    manifest = {"systems": {"X": {"atoms": ["O", "H"]}}}
    return plan, manifest, job


def test_authorization_is_explicit_and_empty_means_no_requests():
    plan, manifest, _ = fixture()
    assert build_stage3_execution_requests(
        stage3_plan=plan, manifest=manifest, authorized_job_ids=[]
    ) == ()


def test_unknown_authorized_job_fails_closed():
    plan, manifest, _ = fixture()
    try:
        build_stage3_execution_requests(
            stage3_plan=plan, manifest=manifest, authorized_job_ids=["missing"]
        )
    except ValueError as exc:
        assert "absent" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_closed_shell_maps_requested_rohf_contract_to_rhf_execution():
    plan, manifest, job = fixture(open_shell=False)
    reqs = build_stage3_execution_requests(
        stage3_plan=plan, manifest=manifest, authorized_job_ids=[job["job_id"]]
    )
    assert len(reqs) == 3
    assert all(r.requested_reference == "ROHF" for r in reqs)
    assert all(r.scf_reference == "RHF" for r in reqs)
    assert [r.r_angstrom for r in reqs] == [1.0, 1.1, 1.2]


def test_open_shell_keeps_rohf_scf_reference():
    plan, manifest, job = fixture(open_shell=True)
    reqs = build_stage3_execution_requests(
        stage3_plan=plan, manifest=manifest, authorized_job_ids=[job["job_id"]]
    )
    assert len(reqs) == 3
    assert all(r.scf_reference == "ROHF" for r in reqs)


def test_multiple_dft_initializations_are_all_preserved_on_all_grid_points():
    plan, manifest, job = fixture(multiple=True)
    reqs = build_stage3_execution_requests(
        stage3_plan=plan, manifest=manifest, authorized_job_ids=[job["job_id"]]
    )
    assert len(reqs) == 6
    assert {r.source_root_id for r in reqs} == {"root_A", "root_B"}
    assert sum(r.source_root_id == "root_A" for r in reqs) == 3
    assert sum(r.source_root_id == "root_B" for r in reqs) == 3
    assert all(not r.authorizes_pruning for r in reqs)


def test_unresolved_checkpoint_provenance_cannot_be_executed():
    plan, manifest, job = fixture()
    job["source_provenance"]["link_status"] = "UNRESOLVED"
    try:
        build_stage3_execution_requests(
            stage3_plan=plan, manifest=manifest, authorized_job_ids=[job["job_id"]]
        )
    except ValueError as exc:
        assert "unresolved checkpoint provenance" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_non_planned_execution_status_cannot_be_restarted_implicitly():
    plan, manifest, job = fixture()
    job["execution_status"] = "COMPLETED"
    try:
        build_stage3_execution_requests(
            stage3_plan=plan, manifest=manifest, authorized_job_ids=[job["job_id"]]
        )
    except ValueError as exc:
        assert "PLANNED_NOT_RUN" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_plan_reporting_pruning_is_rejected():
    plan, manifest, job = fixture()
    plan["automatic_pruning_performed"] = True
    try:
        build_stage3_execution_requests(
            stage3_plan=plan, manifest=manifest, authorized_job_ids=[job["job_id"]]
        )
    except ValueError as exc:
        assert "automatic pruning" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _fake_completed(req, settings):
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
        cc_reference="RHF",
        pyscf_version="2.14.0",
        scf_converged=True,
        scf_energy_hartree=-1.0,
        s2=0.0,
        multiplicity=1.0,
        internal_stable=True,
        external_stable=True,
        external_stability_available=True,
        semicanonicalization="RHF_CANONICAL_REFERENCE",
        cc_class="CCSD",
        ccsd_converged=True,
        ccsd_correlation_hartree=-0.02,
        ccsd_total_hartree=-1.02,
        triples_correction_hartree=-0.001,
        ccsd_t_total_hartree=-1.021,
        t1_diagnostic=0.01,
        t1_diagnostic_definition="test",
        error_type=None,
        error_message=None,
    )


def test_runner_injection_preserves_identity_and_never_creates_ea_claim():
    plan, manifest, job = fixture()
    req = build_stage3_execution_requests(
        stage3_plan=plan, manifest=manifest, authorized_job_ids=[job["job_id"]]
    )[0]
    result = run_stage3_point(req, runner=_fake_completed)
    assert result.status is PointExecutionStatus.COMPLETED
    assert result.ccsd_t_total_hartree == -1.021
    assert result.is_production_ea is False
    assert result.ground_state_assigned is False
    assert result.authorizes_pruning is False


def test_runner_exception_is_preserved_as_error_result():
    plan, manifest, job = fixture()
    req = build_stage3_execution_requests(
        stage3_plan=plan, manifest=manifest, authorized_job_ids=[job["job_id"]]
    )[0]

    def explode(req, settings):
        raise RuntimeError("boom")

    result = run_stage3_point(req, runner=explode)
    assert result.status is PointExecutionStatus.ERROR
    assert result.error_type == "RuntimeError"
    assert result.error_message == "boom"


def test_settings_reject_nonphysical_tolerances():
    try:
        Stage3ExecutionSettings(scf_conv_tol=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
