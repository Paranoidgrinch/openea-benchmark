"""SOSCF rescue of the Stage-3 SCF, also inside internal-stability following (NO- QZVPPD, 2026-10-12)."""
from types import SimpleNamespace

from openea_benchmark.adaptive.stage3_execution import Stage3ExecutionSettings, _scf_kernel_with_rescue


class FakeSCF:
    def __init__(self, diis_converges, soscf_converges):
        self.diis_converges, self.soscf_converges = diis_converges, soscf_converges
        self.converged, self.e_tot, self.mo_coeff, self.mo_occ, self.mo_energy = False, -1.0, "c", "o", "e"
        self.soscf_starts = []

    def kernel(self, dm0=None):
        self.converged = self.diis_converges

    def newton(self):
        outer = self

        class Newton(SimpleNamespace):
            def kernel(self, mo_coeff, mo_occ):
                outer.soscf_starts.append(mo_coeff)
                self.converged = outer.soscf_converges
                self.mo_coeff, self.mo_occ, self.mo_energy, self.e_tot = "c2", "o2", "e2", -2.0

        return Newton()


def test_converged_diis_needs_no_rescue():
    mf = FakeSCF(True, True)
    assert _scf_kernel_with_rescue(mf, None, Stage3ExecutionSettings()) is False
    assert mf.soscf_starts == [] and mf.e_tot == -1.0


def test_stalled_diis_is_finished_by_soscf_from_its_last_orbitals():
    mf = FakeSCF(False, True)
    assert _scf_kernel_with_rescue(mf, None, Stage3ExecutionSettings()) is True
    assert mf.soscf_starts == ["c"] and mf.converged and (mf.mo_coeff, mf.e_tot) == ("c2", -2.0)


def test_failed_rescue_leaves_scf_unconverged():
    mf = FakeSCF(False, False)
    assert _scf_kernel_with_rescue(mf, None, Stage3ExecutionSettings()) is False
    assert not mf.converged and mf.e_tot == -1.0


def test_following_step_on_real_molecule_uses_rescue_path():
    """O2- like open shell: following with rescue ends converged and internally stable."""
    from pyscf import gto, scf

    mol = gto.M(atom="O 0 0 0; O 0 0 1.35", basis="aug-cc-pvdz", charge=-1, spin=1, verbose=0)
    mf = scf.ROHF(mol)
    mf.max_cycle = 3  # force a DIIS stall so every step needs the rescue
    settings = Stage3ExecutionSettings()
    _scf_kernel_with_rescue(mf, None, settings)
    for _ in range(3):
        mo, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)
        if stable and mf.converged:
            break
        _scf_kernel_with_rescue(mf, mf.make_rdm1(mo, mf.mo_occ), settings)
    assert mf.converged and stable


def test_seed_branch_is_never_rescued_onto_another_branch():
    mf = FakeSCF(False, True)
    assert _scf_kernel_with_rescue(mf, None, Stage3ExecutionSettings(), allow_rescue=False) is False
    assert mf.soscf_starts == [] and not mf.converged
