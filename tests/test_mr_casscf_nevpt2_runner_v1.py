from dataclasses import replace
from pathlib import Path

import pytest

from openea_benchmark.adaptive.mr_casscf_nevpt2_runner import (
    MRPointAuthorization, MRPointRequest, MRPointSettings, MRPointStatus,
    MRPointResult, MRRootEnergy, run_mr_casscf_nevpt2_point,
)
from openea_benchmark.adaptive.multireference import (
    MRBranchStatus, resolve_multireference_branch,
)


def req(tmp_path):
    chk = tmp_path / "hf.chk"
    chk.write_bytes(b"checkpoint")
    return MRPointRequest(
        request_id="a", system="LiH", role="neutral", atoms=("Li", "H"),
        charge=0, spin_2s=0, state_manifold_id="X_1Sigma",
        r_angstrom=1.6, basis_label="def2-svp",
        basis_by_element={"Li": "def2-svp", "H": "def2-svp"},
        source_root_id="root0", source_checkpoint_path=str(chk),
        active_orbital_indices=(1, 2), active_electrons_alpha=1,
        active_electrons_beta=1, nroots=2,
        active_space_review_ids=("active-space-review",),
        state_manifold_review_ids=("manifold-review",),
    )


def auth():
    return MRPointAuthorization(True, "Only authorized MR study", ("authorization",))


def fake_backend(r, s):
    return MRPointResult(
        r.request_id, MRPointStatus.COMPLETE_REVIEW_REQUIRED, "root review open",
        pyscf_version="FAKE", scf_energy_hartree=-8.0,
        cas_orbital_optimization_converged=True,
        roots=(MRRootEnergy(0, -8, -0.02, -8.02, 0.0),
               MRRootEnergy(1, -7.9, -0.03, -7.93, 0.0)),
        active_space_review_ids=r.active_space_review_ids,
        state_manifold_review_ids=r.state_manifold_review_ids,
    )


def test_success_is_diagnostic_and_not_production(tmp_path):
    result = run_mr_casscf_nevpt2_point(req(tmp_path), auth(), backend=fake_backend)
    assert result.status is MRPointStatus.COMPLETE_REVIEW_REQUIRED
    assert [x.sc_nevpt2_total_hartree for x in result.roots] == [-8.02, -7.93]
    assert result.method_role == "DIAGNOSTIC_METHOD_DEVELOPMENT"
    assert not result.is_production_ea and not result.mr_production_validated
    assert all(x.root_identity_review_required for x in result.roots)
    assert len(result.source_checkpoint_sha256) == 64
    assert len(result.result_signature) == 64
    assert resolve_multireference_branch(None).status is MRBranchStatus.TERMINAL_UNRESOLVED


def test_no_autorun_without_authorization(tmp_path):
    def fail(*_):
        raise AssertionError("backend ran")
    a = MRPointAuthorization(False, "not approved")
    result = run_mr_casscf_nevpt2_point(req(tmp_path), a, backend=fail)
    assert result.status is MRPointStatus.BLOCKED_NOT_AUTHORIZED


def test_requires_real_source_checkpoint(tmp_path):
    r = replace(req(tmp_path), source_checkpoint_path=str(tmp_path / "not-found"))
    result = run_mr_casscf_nevpt2_point(r, auth(), backend=fake_backend)
    assert result.status is MRPointStatus.BLOCKED_INVALID_INPUT


def test_deterministic_signature_and_checkpoint_digest(tmp_path):
    r = req(tmp_path)
    first = run_mr_casscf_nevpt2_point(r, auth(), backend=fake_backend)
    second = run_mr_casscf_nevpt2_point(r, auth(), backend=fake_backend)
    assert first.result_signature == second.result_signature
    Path(r.source_checkpoint_path).write_bytes(b"changed checkpoint")
    third = run_mr_casscf_nevpt2_point(r, auth(), backend=fake_backend)
    assert first.result_signature != third.result_signature


def test_determinant_budget_blocks_backend(tmp_path):
    a = MRPointAuthorization(True, "budget", ("budget",), max_fci_determinants=3)
    result = run_mr_casscf_nevpt2_point(req(tmp_path), a, backend=fake_backend)
    assert result.status is MRPointStatus.BLOCKED_INVALID_INPUT
    assert "budget" in result.reason


@pytest.mark.parametrize("change", [
    {"role": "other"}, {"active_orbital_indices": ()},
    {"active_orbital_indices": (1, 1)},
    {"active_orbital_indices": (-1, 2)},
    {"active_electrons_alpha": 3}, {"spin_2s": 1},
    {"nroots": 0}, {"nroots": 11},
    {"active_space_review_ids": ()}, {"state_manifold_review_ids": ()},
    {"r_angstrom": -0.5}, {"basis_by_element": {"H": "sto-3g"}},
    {"source_checkpoint_path": ""},
])
def test_invalid_request_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        replace(req(tmp_path), **change)


def test_spin_1_2_odd_anion_is_supported(tmp_path):
    r = replace(req(tmp_path), role="anion", charge=-1, spin_2s=1,
                active_electrons_alpha=2, active_electrons_beta=1)
    assert r.nelecas == (2, 1)
    assert r.fci_determinants == 2


def test_inconsistent_root_energy_is_rejected():
    with pytest.raises(ValueError, match="Inconsistent"):
        MRRootEnergy(0, -8.0, -0.02, -8.2, 0.0)


def test_cannot_certify_production_from_point_result():
    with pytest.raises(ValueError):
        MRPointResult("x", MRPointStatus.COMPLETE_REVIEW_REQUIRED, "no", is_production_ea=True)


def test_backend_exception_is_fail_closed(tmp_path):
    def raise_exc(*_):
        raise RuntimeError("backend failure")
    result = run_mr_casscf_nevpt2_point(req(tmp_path), auth(), backend=raise_exc)
    assert result.status is MRPointStatus.ERROR and not result.roots


def test_backend_wrong_request_id_rejected(tmp_path):
    with pytest.raises(ValueError, match="mismatched"):
        run_mr_casscf_nevpt2_point(req(tmp_path), auth(),
                                   backend=lambda r, s: replace(fake_backend(r, s), request_id="other"))


def test_settings_validate_inputs():
    with pytest.raises(ValueError):
        MRPointSettings(casscf_conv_tol=0)
    with pytest.raises(ValueError):
        MRPointSettings(scalar_relativistic="DHF")


def test_authorization_needs_provenance():
    with pytest.raises(ValueError):
        MRPointAuthorization(True, "why")


def test_no_fake_root_in_error_result():
    with pytest.raises(ValueError):
        MRPointResult("x", MRPointStatus.ERROR, "error", roots=(MRRootEnergy(0, -8, -0.1, -8.1, 0.0),))


def test_missing_backend_root_is_rejected(tmp_path):
    def missing(r, settings):
        return replace(fake_backend(r, settings), roots=(fake_backend(r, settings).roots[0],))
    with pytest.raises(ValueError, match="incomplete"):
        run_mr_casscf_nevpt2_point(req(tmp_path), auth(), backend=missing)


def test_unphysical_backend_spin_is_rejected(tmp_path):
    def wrong_spin(r, settings):
        return replace(fake_backend(r, settings), roots=(
            MRRootEnergy(0, -8, -0.02, -8.02, 2.0),
            MRRootEnergy(1, -7.9, -0.03, -7.93, 2.0),
        ))
    with pytest.raises(ValueError, match="incorrect spin"):
        run_mr_casscf_nevpt2_point(req(tmp_path), auth(), backend=wrong_spin)


def test_request_change_alters_signature(tmp_path):
    r = req(tmp_path)
    a = run_mr_casscf_nevpt2_point(r, auth(), backend=fake_backend)
    b = run_mr_casscf_nevpt2_point(replace(r, r_angstrom=1.7), auth(), backend=fake_backend)
    assert a.result_signature != b.result_signature


def test_settings_change_alters_signature(tmp_path):
    r = req(tmp_path)
    a = run_mr_casscf_nevpt2_point(r, auth(), backend=fake_backend)
    b = run_mr_casscf_nevpt2_point(r, auth(), settings=MRPointSettings(casscf_conv_tol=1e-9), backend=fake_backend)
    assert a.result_signature != b.result_signature


def test_real_backend_sort_mo_contract_regression(monkeypatch, tmp_path):
    """Exercise the *actual* _run_pyscf call up to PySCF's sort_mo boundary.

    PySCF indexes a one-dimensional mask by ``caslst``: a tuple (1, 2)
    requests two axes and raises IndexError; a list [1, 2] is required.
    The fake backend deliberately performs that NumPy indexing instead of
    pretending that any argument shape is accepted.
    """
    import sys
    from types import ModuleType, SimpleNamespace
    import numpy as np
    from openea_benchmark.adaptive.mr_casscf_nevpt2_runner import _run_pyscf

    class ReachedCorrectSortCall(Exception):
        pass

    class FakeMol:
        nelectron = 4
        _ecp = {}

        def nao_nr(self):
            return 6

    class FakeHF:
        def __init__(self, mol):
            self.mo_coeff = np.eye(6)
            self.converged = False

        def kernel(self, dm0=None):
            assert dm0 == "checkpoint-density"
            self.converged = True

    class FakeCASSCF:
        def __init__(self, mf, ncas, nelecas):
            assert ncas == 2 and nelecas == (1, 1)
            self.mo = mf.mo_coeff

        def sort_mo(self, caslst, mo_coeff=None, base=1):
            assert base == 0
            # PySCF allows mo_coeff=None and defaults to the object's MOs.
            # The old runner passed an index tuple, so this exact 1-D
            # NumPy mask access must reproduce the Artemis IndexError.
            coefficients = self.mo if mo_coeff is None else mo_coeff
            mask = np.ones(coefficients.shape[1], dtype=bool)
            mask[caslst] = False
            assert mo_coeff is self.mo
            assert isinstance(caslst, list)
            assert caslst == [1, 2]
            assert int(mask.sum()) == 4
            raise ReachedCorrectSortCall

    fake_pyscf = ModuleType("pyscf")
    fake_pyscf.__version__ = "synthetic-api-boundary"
    fake_pyscf.gto = SimpleNamespace(M=lambda **kwargs: FakeMol())
    fake_pyscf.scf = SimpleNamespace(
        RHF=FakeHF, ROHF=FakeHF,
        hf=SimpleNamespace(init_guess_by_chkfile=lambda *args, **kwargs: "checkpoint-density"),
        rohf=SimpleNamespace(init_guess_by_chkfile=lambda *args, **kwargs: "checkpoint-density"),
    )
    fake_pyscf.mcscf = SimpleNamespace(CASSCF=FakeCASSCF)
    fake_pyscf.fci = SimpleNamespace()
    fake_pyscf.mrpt = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)

    with pytest.raises(ReachedCorrectSortCall):
        _run_pyscf(req(tmp_path), MRPointSettings())
