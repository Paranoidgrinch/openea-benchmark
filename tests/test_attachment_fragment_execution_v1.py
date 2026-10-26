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


def test_closed_shell_ns2_atom_keeps_its_rhf_ccsd_t_energy():
    # verify_v1 Li-/Na-: RHF 1S is externally (RHF -> UHF) unstable; dropping it made 3P
    # the anion ground state, 0.88 eV too high for Li-.
    req = AtomicFragmentRequest("Li_minus_1S", "Li", -1, 0, "def2-svpd", state_label="1S")
    result = run_atomic_fragment(req)
    assert result.status is FragmentExecutionStatus.COMPLETED
    assert result.external_stable is False and result.external_instability_waived is True
    assert result.external_instability_waiver_reason == "ATOMIC_CLOSED_SHELL_1S_RHF_REFERENCE"
    neutral = run_atomic_fragment(AtomicFragmentRequest("Li_2S", "Li", 0, 1, "def2-svpd", state_label="2S"))
    assert 0.2 < (neutral.ccsd_t_total_hartree - result.ccsd_t_total_hartree) * 27.211386 < 0.8   # EA(Li) 0.62 eV


def test_closed_shell_waiver_only_for_complete_subshells():
    from openea_benchmark.attachment.fragment_execution import closed_subshell_electron_count
    assert all(closed_subshell_electron_count(el, q) for el, q in
               (("Li", -1), ("Na", -1), ("Cu", -1), ("Ag", -1), ("Au", -1), ("Be", 0), ("Mg", 0), ("Zn", 0)))
    # closed-shell determinant of an open p/d atom is a 1D component, not a 1S state
    assert not any(closed_subshell_electron_count(el, q) for el, q in
                   (("O", 0), ("C", 0), ("S", 0), ("Si", 0), ("N", -1), ("P", -1), ("B", -1), ("Pb", 0), ("Ni", 0)))
