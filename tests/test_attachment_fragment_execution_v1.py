from dataclasses import replace

import pytest

from openea_benchmark.adaptive.stage3_execution import Stage3ExecutionSettings
from openea_benchmark.attachment.fragment_execution import (
    AtomicFragmentRequest,
    AtomicFragmentResult,
    FragmentExecutionStatus,
    run_atomic_fragment,
)


def request(fragment_id="O_minus"):
    return AtomicFragmentRequest(
        fragment_id=fragment_id,
        element="O",
        charge=-1,
        spin_2s=1,
        basis="sto-3g",
        state_label="2P",
    )


def completed(req, energy=-74.5):
    return AtomicFragmentResult(
        fragment_id=req.fragment_id,
        status=FragmentExecutionStatus.COMPLETED,
        element=req.element,
        charge=req.charge,
        spin_2s=req.spin_2s,
        basis=req.basis,
        state_label=req.state_label,
        scf_reference="ROHF",
        cc_reference="SEMICANONICAL_UHF_FROM_ROHF",
        scf_converged=True,
        scf_energy_hartree=energy + 0.1,
        internal_stable=True,
        external_stable=None,
        external_stability_available=False,
        ccsd_converged=True,
        ccsd_total_hartree=energy + 0.01,
        triples_correction_hartree=-0.01,
        ccsd_t_total_hartree=energy,
        s2=0.75,
        multiplicity=2.0,
        pyscf_version="test",
        error_type=None,
        error_message=None,
    )


def test_request_rejects_negative_spin():
    with pytest.raises(ValueError):
        AtomicFragmentRequest(
            "bad", "O", 0, -1, "sto-3g"
        )


def test_runner_injection_preserves_completed_result():
    req = request()

    def runner(item, settings):
        assert item == req
        assert isinstance(settings, Stage3ExecutionSettings)
        return completed(item)

    result = run_atomic_fragment(req, runner=runner)
    assert result.status is FragmentExecutionStatus.COMPLETED
    assert result.ccsd_t_total_hartree == -74.5


def test_runner_exception_is_preserved_as_error_evidence():
    req = request()

    def runner(item, settings):
        raise OSError("synthetic failure")

    result = run_atomic_fragment(req, runner=runner)
    assert result.status is FragmentExecutionStatus.ERROR
    assert result.error_type == "OSError"
    assert "synthetic failure" in result.error_message


def test_fragment_result_cannot_claim_production_threshold():
    req = request()
    with pytest.raises(ValueError):
        replace(
            completed(req),
            is_production_threshold=True,
        )


def test_two_electron_waiver_audit_fields_default_fail_closed():
    req = request()
    result = completed(req)
    assert result.external_instability_waived is False
    assert result.external_instability_waiver_reason is None
    assert result.two_electron_ccsd_exact_space is False


def test_two_electron_waiver_can_be_recorded_without_promoting_threshold():
    req = AtomicFragmentRequest(
        "H_minus_1S",
        "H",
        -1,
        0,
        "aug-cc-pvdz",
        state_label="1S",
    )
    result = replace(
        completed(req, -0.52),
        scf_reference="RHF",
        cc_reference="RHF",
        external_stability_available=True,
        external_stable=False,
        external_instability_waived=True,
        external_instability_waiver_reason=(
            "ATOMIC_CLOSED_SHELL_TWO_ELECTRON_CCSD_COMPLETE_EXCITATION_SPACE"
        ),
        two_electron_ccsd_exact_space=True,
    )
    assert result.status is FragmentExecutionStatus.COMPLETED
    assert result.external_stable is False
    assert result.external_instability_waived is True
    assert result.two_electron_ccsd_exact_space is True
    assert result.is_production_threshold is False
