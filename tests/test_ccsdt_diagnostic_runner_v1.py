import json
from pathlib import Path

import pytest

from openea_benchmark.adaptive.ccsdt_diagnostic_runner import (
    AdaptiveCCSDTDiagnosticStatus,
    CCSDTDiagnosticAuthorization,
    CCSDTDiagnosticBasisSpec,
    CCSDTDiagnosticExecutionSettings,
    CCSDTDiagnosticMethod,
    CCSDTDiagnosticStateSpec,
    CCSDTMethodExecutionStatus,
    CCSDTMethodResult,
    run_adaptive_ccsdt_diagnostic_series,
)
from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV


def states(tmp_path, *, nfrozen=1):
    ncp = tmp_path / "neutral.chk"
    acp = tmp_path / "anion.chk"
    ncp.write_text("neutral", encoding="utf-8")
    acp.write_text("anion", encoding="utf-8")
    neutral = CCSDTDiagnosticStateSpec(
        role="neutral",
        system="XY",
        atoms=("X", "Y"),
        charge=0,
        spin_2s=1,
        state_id="N_DOUBLEt",
        r_angstrom=1.23,
        source_root_id="n-root",
        source_checkpoint_path=str(ncp),
        state_identity_validated=True,
        nfrozen_spatial_orbitals=nfrozen,
    )
    anion = CCSDTDiagnosticStateSpec(
        role="anion",
        system="XY",
        atoms=("X", "Y"),
        charge=-1,
        spin_2s=0,
        state_id="A_SINGLET",
        r_angstrom=1.28,
        source_root_id="a-root",
        source_checkpoint_path=str(acp),
        state_identity_validated=True,
        nfrozen_spatial_orbitals=nfrozen,
    )
    return neutral, anion


def bases():
    return (
        CCSDTDiagnosticBasisSpec(2, "aug-cc-pVXZ", "aug-cc-pVDZ", {"X": "x-dz", "Y": "y-dz"}),
        CCSDTDiagnosticBasisSpec(3, "aug-cc-pVXZ", "aug-cc-pVTZ", {"X": "x-tz", "Y": "y-tz"}),
        CCSDTDiagnosticBasisSpec(4, "aug-cc-pVXZ", "aug-cc-pVQZ", {"X": "x-qz", "Y": "y-qz"}),
    )


def authorization(ok=True, cardinals=(2, 3)):
    return CCSDTDiagnosticAuthorization(
        authorized=ok,
        reason="post-CC correlation is the active diagnostic target" if ok else "not justified",
        evidence_ids=("G3C_POST_CC_DOMINANT",) if ok else (),
        authorized_cardinals=cardinals if ok else (),
    )


def fake_runner(delta_by_x, calls, *, fail=None, reference_shift=None):
    def run(request, settings):
        calls.append(request)
        if fail == (request.cardinal, request.role, request.method.value):
            return CCSDTMethodResult(
                request_id=request.request_id,
                status=CCSDTMethodExecutionStatus.ERROR,
                method=request.method,
                reference_energy_hartree=None,
                correlation_energy_hartree=None,
                total_energy_hartree=None,
                ccpy_version="fake",
                scf_converged=False,
                internal_stable=None,
                external_stable=None,
                external_stability_available=request.scf_reference == "RHF",
                symmetry_group="C1",
                nfrozen_spatial_orbitals=request.nfrozen_spatial_orbitals,
                error_type="SyntheticFailure",
                error_message="boom",
            )

        base_ea = 1.8000
        delta = delta_by_x[request.cardinal]
        ea = base_ea if request.method is CCSDTDiagnosticMethod.CCSD_T else base_ea + delta
        ref = -99.0 if request.role == "neutral" else -99.1
        if reference_shift and (request.cardinal, request.role, request.method.value) in reference_shift:
            ref += reference_shift[(request.cardinal, request.role, request.method.value)]
        neutral_total = -100.0
        total = neutral_total if request.role == "neutral" else neutral_total - ea / HARTREE_TO_EV
        return CCSDTMethodResult(
            request_id=request.request_id,
            status=CCSDTMethodExecutionStatus.COMPLETED,
            method=request.method,
            reference_energy_hartree=ref,
            correlation_energy_hartree=total - ref,
            total_energy_hartree=total,
            ccpy_version="fake",
            scf_converged=True,
            internal_stable=True,
            external_stable=True if request.scf_reference == "RHF" else None,
            external_stability_available=request.scf_reference == "RHF",
            symmetry_group="C1",
            nfrozen_spatial_orbitals=request.nfrozen_spatial_orbitals,
        )

    return run


def no_basis_io(element, basis):
    return None


def test_small_stable_delta_t3_clears_as_diagnostic_only(tmp_path):
    neutral, anion = states(tmp_path)
    calls = []
    result = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        method_runner=fake_runner({2: 0.0004, 3: 0.0005, 4: 0.00055}, calls),
        basis_validator=no_basis_io,
    )
    assert result.status is AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_CLEARED
    assert result.assessment.status == "CLEARED"
    assert result.assessment.action == "NONE"
    assert result.assessment.central_correction_ev == pytest.approx(0.0005)
    assert result.assessment.combined_bound_ev == pytest.approx(0.0001)
    assert result.method_role == "DIAGNOSTIC"
    assert result.is_production_ea is False
    assert result.includes_scalar_relativity is False
    assert result.includes_soc is False
    assert len(calls) == 8
    assert {x.method for x in calls} == {CCSDTDiagnosticMethod.CCSD_T, CCSDTDiagnosticMethod.CCSDT}
    assert all(x.symmetry_group == "C1" for x in calls)
    assert all(x.hamiltonian == "NONRELATIVISTIC" for x in calls)
    assert all(x.correlation_space == "FROZEN_CORE_VALENCE" for x in calls)
    assert all(x.nfrozen_spatial_orbitals == 1 for x in calls)


def test_large_delta_t3_returns_reference_character_warning_never_higher_cc(tmp_path):
    neutral, anion = states(tmp_path)
    calls = []
    result = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        method_runner=fake_runner({2: 0.0100, 3: 0.0102, 4: 0.0101}, calls),
        basis_validator=no_basis_io,
    )
    assert result.status is AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_WARNING
    assert result.assessment.status == "POST_CC_WARNING"
    assert result.assessment.action == "REASSESS_REFERENCE_CHARACTER"
    assert "CCSDTQ" not in result.assessment.action
    assert all(x.method.value in {"CCSD(T)", "CCSDT"} for x in calls)


def test_runner_refuses_to_spend_ccsdt_without_explicit_authorization(tmp_path):
    neutral, anion = states(tmp_path)
    calls = []
    result = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(False),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        method_runner=fake_runner({2: 0.0, 3: 0.0, 4: 0.0}, calls),
        basis_validator=no_basis_io,
    )
    assert result.status is AdaptiveCCSDTDiagnosticStatus.POLICY_BLOCKED
    assert result.execution_error_type == "CCSDTDiagnosticNotAuthorized"
    assert calls == []


def test_state_identity_and_same_frozen_core_definition_are_required(tmp_path):
    neutral, anion = states(tmp_path)
    bad_anion = CCSDTDiagnosticStateSpec(**{**anion.to_dict(), "nfrozen_spatial_orbitals": 2})
    with pytest.raises(ValueError, match="same explicitly declared frozen-core"):
        run_adaptive_ccsdt_diagnostic_series(
            authorization=authorization(),
            neutral=neutral,
            anion=bad_anion,
            basis_specs=bases(),
            method_runner=fake_runner({2: 0.0, 3: 0.0, 4: 0.0}, []),
            basis_validator=no_basis_io,
        )

    data = neutral.to_dict()
    data["state_identity_validated"] = False
    with pytest.raises(ValueError, match="state identity"):
        CCSDTDiagnosticStateSpec(**data)


def test_ecp_or_core_replacement_basis_policy_fails_closed():
    with pytest.raises(ValueError, match="all-electron"):
        CCSDTDiagnosticBasisSpec(
            2,
            "ecp-family",
            "ECP-DZ",
            {"X": "ecp-dz", "Y": "ecp-dz"},
            electron_model="ECP",
        )


def test_basis_preflight_happens_before_any_expensive_method_call(tmp_path):
    neutral, anion = states(tmp_path)
    calls = []

    def broken_basis(element, basis):
        if basis == "y-tz":
            raise KeyError("missing")

    result = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        method_runner=fake_runner({2: 0.0, 3: 0.0, 4: 0.0}, calls),
        basis_validator=broken_basis,
    )
    assert result.status is AdaptiveCCSDTDiagnosticStatus.POLICY_BLOCKED
    assert "basis preflight failed" in result.execution_error_message
    assert calls == []


def test_one_initial_cardinal_can_request_one_explicit_next_cardinal(tmp_path):
    neutral, anion = states(tmp_path)
    calls = []
    result = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(cardinals=(2, 3)),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        initial_cardinals=(2,),
        maximum_cardinal=3,
        method_runner=fake_runner({2: 0.0004, 3: 0.0005, 4: 0.0005}, calls),
        basis_validator=no_basis_io,
    )
    assert result.status is AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_CLEARED
    assert [x.cardinal for x in result.evidence] == [2, 3]
    assert len(calls) == 8


def test_missing_authorized_next_basis_is_unresolved_not_guessed(tmp_path):
    neutral, anion = states(tmp_path)
    calls = []
    result = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=(bases()[0],),
        initial_cardinals=(2,),
        maximum_cardinal=2,
        method_runner=fake_runner({2: 0.0004}, calls),
        basis_validator=no_basis_io,
    )
    assert result.status is AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_UNRESOLVED
    assert result.assessment.action == "COMPUTE_T3_X3"
    assert result.execution_error_type == "MaximumCCSDTDiagnosticCardinalReached"


def test_ccsd_t_and_ccsdt_must_share_matched_reference_energy(tmp_path):
    neutral, anion = states(tmp_path)
    calls = []
    shift = {(2, "neutral", "CCSDT"): 1.0e-5}
    result = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        method_runner=fake_runner(
            {2: 0.0004, 3: 0.0005, 4: 0.0005}, calls, reference_shift=shift
        ),
        basis_validator=no_basis_io,
    )
    assert result.status is AdaptiveCCSDTDiagnosticStatus.EXECUTION_BLOCKED
    assert "matched SCF reference energy" in result.execution_error_message


def test_checkpoint_resume_reuses_completed_subpoints_and_rejects_stale_signature(tmp_path):
    neutral, anion = states(tmp_path)
    checkpoint_dir = tmp_path / "ccsdt_cp"
    first_calls = []
    first = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        method_runner=fake_runner({2: 0.0004, 3: 0.0005, 4: 0.0005}, first_calls),
        basis_validator=no_basis_io,
        checkpoint_dir=checkpoint_dir,
    )
    assert first.status is AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_CLEARED
    assert len(first_calls) == 8

    second_calls = []
    second = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        method_runner=fake_runner({2: 0.0004, 3: 0.0005, 4: 0.0005}, second_calls),
        basis_validator=no_basis_io,
        checkpoint_dir=checkpoint_dir,
    )
    assert second.status is AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_CLEARED
    assert second_calls == []
    assert all(calc.checkpoint_reused for ev in second.evidence for calc in ev.calculations)

    changed = CCSDTDiagnosticExecutionSettings(cc_amp_convergence=5.0e-9)
    third = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        execution_settings=changed,
        method_runner=fake_runner({2: 0.0004, 3: 0.0005, 4: 0.0005}, []),
        basis_validator=no_basis_io,
        checkpoint_dir=checkpoint_dir,
    )
    assert third.status is AdaptiveCCSDTDiagnosticStatus.EXECUTION_BLOCKED
    assert "signature mismatch" in third.execution_error_message


def test_partial_checkpoint_resume_recomputes_only_failed_subpoint(tmp_path):
    neutral, anion = states(tmp_path)
    checkpoint_dir = tmp_path / "partial"
    first_calls = []
    first = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        method_runner=fake_runner(
            {2: 0.0004, 3: 0.0005, 4: 0.0005},
            first_calls,
            fail=(3, "anion", "CCSDT"),
        ),
        basis_validator=no_basis_io,
        checkpoint_dir=checkpoint_dir,
    )
    assert first.status is AdaptiveCCSDTDiagnosticStatus.EXECUTION_BLOCKED

    second_calls = []
    second = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        method_runner=fake_runner({2: 0.0004, 3: 0.0005, 4: 0.0005}, second_calls),
        basis_validator=no_basis_io,
        checkpoint_dir=checkpoint_dir,
    )
    assert second.status is AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_CLEARED
    assert [(x.cardinal, x.role, x.method.value) for x in second_calls] == [(3, "anion", "CCSDT")]


def test_checkpoint_method_tokens_cannot_collide(tmp_path):
    neutral, anion = states(tmp_path)
    cp = tmp_path / "tokens"
    calls = []
    result = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        method_runner=fake_runner({2: 0.0004, 3: 0.0005, 4: 0.0005}, calls),
        basis_validator=no_basis_io,
        checkpoint_dir=cp,
    )
    assert result.status is AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_CLEARED
    paths = list(cp.rglob("*.json"))
    names = {p.name for p in paths}
    assert "neutral__CCSD_pT.json" in names
    assert "neutral__CCSDT.json" in names
    assert len(paths) == 8


def test_checkpoint_payload_contains_no_ccsdtq_request(tmp_path):
    neutral, anion = states(tmp_path)
    cp = tmp_path / "payload"
    run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        method_runner=fake_runner({2: 0.0004, 3: 0.0005, 4: 0.0005}, []),
        basis_validator=no_basis_io,
        checkpoint_dir=cp,
    )
    for path in cp.rglob("*.json"):
        raw = path.read_text(encoding="utf-8")
        assert '"CCSDTQ"' not in raw
        payload = json.loads(raw)
        assert payload["request"]["method"] in {"CCSD(T)", "CCSDT"}


def test_new_cardinal_requires_explicit_authorization_but_prior_checkpoint_can_be_reused(tmp_path):
    neutral, anion = states(tmp_path)
    cp = tmp_path / "authorized-cardinals"

    # First run: only X2 is authorized.  It is computed and stored, but X3 is
    # outside the current authorization and therefore cannot be launched.
    first_calls = []
    first = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(cardinals=(2,)),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        initial_cardinals=(2,),
        maximum_cardinal=3,
        method_runner=fake_runner({2: 0.0004, 3: 0.0005, 4: 0.0005}, first_calls),
        basis_validator=no_basis_io,
        checkpoint_dir=cp,
    )
    assert first.status is AdaptiveCCSDTDiagnosticStatus.EXECUTION_BLOCKED
    assert "X=3 is not authorized" in first.execution_error_message
    assert len(first_calls) == 4

    # Second run: only the new X3 cost is authorized.  X2 is reused from its
    # compatible checkpoint and no longer consumes authorization/compute.
    second_calls = []
    second = run_adaptive_ccsdt_diagnostic_series(
        authorization=authorization(cardinals=(3,)),
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        initial_cardinals=(2, 3),
        maximum_cardinal=3,
        method_runner=fake_runner({2: 0.0004, 3: 0.0005, 4: 0.0005}, second_calls),
        basis_validator=no_basis_io,
        checkpoint_dir=cp,
    )
    assert second.status is AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_CLEARED
    assert [(x.cardinal, x.role, x.method.value) for x in second_calls] == [
        (3, "neutral", "CCSD(T)"),
        (3, "neutral", "CCSDT"),
        (3, "anion", "CCSD(T)"),
        (3, "anion", "CCSDT"),
    ]
    x2 = next(ev for ev in second.evidence if ev.cardinal == 2)
    assert all(calc.checkpoint_reused for calc in x2.calculations)


def test_scalar_relativistic_hamiltonian_cannot_leak_into_post_cc_diagnostic():
    with pytest.raises(ValueError, match="scalar relativity is a separate correction"):
        CCSDTDiagnosticExecutionSettings(scalar_relativistic="SFX2C1E")
