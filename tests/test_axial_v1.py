"""Axial (Lz-adapted) orbitals: real integrals, ROHF and CC identical to the real-orbital
codes where both represent the state; 1Delta as one determinant (pi+^2)."""
import numpy as np
import pytest
from pyscf import cc, gto, scf

from openea_benchmark.axial.basis import axial_ao_transform, axial_integrals
from openea_benchmark.axial.cc import axial_cc
from openea_benchmark.axial.scf import axial_rohf, minao_density

X3 = {0: (3, 3), 1: (1, 0), -1: (1, 0)}       # NH X 3Sigma-: pi+ pi- (alpha)
A1 = {0: (3, 3), 1: (1, 1)}                    # NH a 1Delta, M_L = 2: pi+^2


def nh(basis="aug-cc-pvdz", spin=2):
    return gto.M(atom="N 0 0 0; H 0 0 1.037", basis=basis, spin=spin, verbose=0)


def test_axial_functions_are_lz_eigenfunctions_up_to_g():
    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="cc-pvqz", verbose=0)   # s..g shells
    u, m = axial_ao_transform(mol)
    with mol.with_common_origin((0.0, 0.0, 0.0)):
        lz = -1j * mol.intor("int1e_cg_irxp", comp=3)[2]
    s = mol.intor("int1e_ovlp")
    lz_ax, s_ax = u.conj().T @ lz @ u, u.conj().T @ s @ u
    # generalized eigenproblem: Lz chi = m S chi, block diagonal in m
    assert np.allclose(lz_ax, s_ax * m[None, :], atol=1e-10)
    assert set(np.abs(m)) == set(range(5))
    assert np.allclose(u.conj().T @ u, np.eye(len(m)))


def test_integrals_are_real_and_conserve_m():
    mol = nh("cc-pvtz")
    ints = axial_integrals(mol)            # raises if any imaginary part survives
    m = ints.m
    p, q, r, s = np.nonzero(np.abs(ints.eri) > 1e-12)
    assert np.all(-m[p] + m[q] - m[r] + m[s] == 0)
    assert np.allclose(ints.eri, ints.eri.transpose(2, 3, 0, 1))      # (pq|rs) = (rs|pq)
    assert np.allclose(ints.eri, ints.eri.transpose(1, 0, 3, 2))      # (pq|rs) = (qp|sr)
    assert not np.allclose(ints.eri, ints.eri.transpose(1, 0, 2, 3))  # but not (qp|rs): 4-fold only


def test_axial_rohf_equals_pyscf_rohf_for_a_real_representable_state():
    mol = nh()
    ref = scf.ROHF(gto.M(atom=mol.atom, basis=mol.basis, spin=2, symmetry="C2v", verbose=0))
    ref.irrep_nelec = {"A1": (3, 3), "B1": (1, 0), "B2": (1, 0)}
    ref.conv_tol = 1e-11
    ref.kernel()
    ints = axial_integrals(mol)
    res = axial_rohf(ints, X3, minao_density(mol, ints))
    assert res.converged and res.e_tot == pytest.approx(ref.e_tot, abs=1e-9)
    assert sum(k * o for k, o in zip(res.mo_m, res.mo_occ)) == 0


def test_semicanonical_axial_ccsd_t_equals_the_stage3_definition():
    mol = nh()
    ref = scf.ROHF(gto.M(atom=mol.atom, basis=mol.basis, spin=2, symmetry="C2v", verbose=0))
    ref.irrep_nelec = {"A1": (3, 3), "B1": (1, 0), "B2": (1, 0)}
    ref.conv_tol = 1e-11
    ref.kernel()
    u = ref.to_uhf()
    u.mo_energy, u.mo_coeff = u.canonicalize(u.mo_coeff, u.mo_occ)
    pc = cc.UCCSD(u)
    pc.frozen, pc.conv_tol, pc.conv_tol_normt = 1, 1e-10, 1e-8
    pc.kernel()
    ints = axial_integrals(mol)
    res = axial_cc(ints, axial_rohf(ints, X3, minao_density(mol, ints)), mol.nelectron, 1, ("ccsd(t)",))
    assert res["ccsd"] == pytest.approx(pc.e_corr, abs=1e-8)
    assert res["ccsd(t)"] == pytest.approx(pc.e_corr + pc.ccsd_t(), abs=1e-8)


def test_one_delta_is_a_single_axial_determinant():
    mol = nh(spin=0)
    ints = axial_integrals(mol)
    delta = axial_rohf(ints, A1, minao_density(mol, ints))
    assert delta.converged and sum(k * o for k, o in zip(delta.mo_m, delta.mo_occ)) == 2
    mol3 = nh()
    sigma = axial_rohf(axial_integrals(mol3), X3, minao_density(mol3, axial_integrals(mol3)))
    # Hund: 3Sigma- below 1Delta (HF overestimates the gap; measured T0 1.56 eV)
    assert 1.2 < (delta.e_tot - sigma.e_tot) * 27.211386 < 2.2
    corr = axial_cc(ints, delta, mol.nelectron, 1, ("ccsd(t)",))
    assert corr["ccsd(t)"] < corr["ccsd"] < 0.0
    with pytest.raises(ValueError):
        axial_rohf(ints, {0: (3, 3), 1: (0, 1)}, minao_density(mol, ints))   # not high spin
