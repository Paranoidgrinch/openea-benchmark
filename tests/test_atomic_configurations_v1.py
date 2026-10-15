"""Configuration control for atomic fragments (Patch 60)."""
import numpy as np
from pyscf import gto, scf

from openea_benchmark.attachment.fragment_execution import (
    AtomicConfigurationPolicy, AtomicFragmentRequest, _core_shells_removed, _maximum_overlap,
    configuration_label, run_atomic_fragment, scan_scf_configurations, target_occupations,
)
from openea_benchmark.adaptive.stage3_execution import Stage3ExecutionSettings


def test_pseudopotential_core_shells_in_n_l_order():
    assert _core_shells_removed(28) == [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
    assert _core_shells_removed(60)[-1] == (4, 3)
    assert _core_shells_removed(0) == []


def _ti(spin):
    mol = gto.M(atom="Ti 0 0 0", basis="def2-svp", spin=spin, verbose=0)
    mf = scf.ROHF(mol)
    mf.init_guess = "atom"
    mf.kernel()
    return mol, mf


def test_target_occupations_follow_configuration_and_spin():
    mol, mf = _ti(2)
    occs = dict(target_occupations(mol, mf.mo_coeff, mf.mo_energy, "Ti", 0, 2))
    ground = next(c for c in occs if c.endswith("4s2 3d2"))
    a, b = occs[ground]
    assert a.sum() - b.sum() == 2 and a.sum() + b.sum() == mol.nelectron
    # a quintet needs four open shells: 4s1 3d3 only
    mol5, mf5 = _ti(4)
    labels = [c.split("3p6 ")[-1] for c, _ in target_occupations(mol5, mf5.mo_coeff, mf5.mo_energy, "Ti", 0, 4)]
    assert labels and all("4s1" in c for c in labels)


def test_maximum_overlap_keeps_closed_shell_occupation_two():
    mol = gto.M(atom="Ne 0 0 0", basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol).run()
    m2 = _maximum_overlap(scf.RHF(mol), mf.mo_coeff, mf.mo_occ)
    m2.kernel(mf.make_rdm1())
    assert m2.converged and abs(m2.e_tot - mf.e_tot) < 1e-8 and set(np.unique(m2.mo_occ)) == {0.0, 2.0}


def test_configuration_label_and_scan_light_atom():
    req = AtomicFragmentRequest("O", "O", 0, 2, "cc-pvdz")
    st = Stage3ExecutionSettings(verbose=0)
    sols = scan_scf_configurations(req, st, AtomicConfigurationPolicy())
    assert sols and sols[0]["label"] == "p2"
    res = run_atomic_fragment(req, settings=st)
    assert res.status.value == "COMPLETED" and res.configuration_label == "p2" and res.configuration_candidates
