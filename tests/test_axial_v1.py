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


def test_axial_uhf_and_its_cc_equal_pyscf_uhf():
    from openea_benchmark.axial.scf import axial_uhf
    mol = nh()
    ref = scf.UHF(gto.M(atom=mol.atom, basis=mol.basis, spin=2, symmetry="C2v", verbose=0))
    ref.irrep_nelec = {"A1": (3, 3), "B1": (1, 0), "B2": (1, 0)}
    ref.conv_tol = 1e-11
    ref.kernel()
    pc = cc.UCCSD(ref)
    pc.frozen, pc.conv_tol, pc.conv_tol_normt = 1, 1e-10, 1e-8
    pc.kernel()
    ints = axial_integrals(mol)
    res = axial_uhf(ints, X3, minao_density(mol, ints))
    assert res.converged and res.e_tot == pytest.approx(ref.e_tot, abs=1e-9)
    corr = axial_cc(ints, res, mol.nelectron, 1, ("ccsd(t)",))
    assert corr["ccsd"] == pytest.approx(pc.e_corr, abs=1e-8)
    assert corr["ccsd(t)"] == pytest.approx(pc.e_corr + pc.ccsd_t(), abs=1e-8)


def test_atomic_lm_blocks_keep_l_pure_orbitals():
    from openea_benchmark.axial.scf import atomic_blocks, atomic_occupation, axial_uhf
    mol = gto.M(atom="N 0 0 0", basis="aug-cc-pvtz", spin=3, verbose=0)    # 4S: p m = 1, 0, -1 (alpha)
    ref = scf.UHF(mol)
    ref.conv_tol = 1e-11
    ref.kernel()
    ints = axial_integrals(mol)
    occ = atomic_occupation({0: 2}, {1: [1, 0, -1]})
    assert occ[(0, 0)] == (2, 2) and occ[(1, 1)] == (1, 0)
    res = axial_uhf(ints, occ, minao_density(mol, ints), block_of_ao=atomic_blocks(mol, ints))
    assert res.converged and res.e_tot == pytest.approx(ref.e_tot, abs=1e-8)
    assert res.s2 == pytest.approx(ref.spin_square()[0], abs=1e-6)   # UHF core spin polarization
    with pytest.raises(ValueError):
        atomic_blocks(nh(), ints)


def test_embedded_atomic_start_reaches_the_molecular_solution():
    # NH X3Sigma- from the l-pure N atom (4S): the open 2p m = +-1 orbitals are followed
    from openea_benchmark.axial.scf import atomic_blocks, atomic_occupation, axial_uhf, embed_atomic_orbitals
    mol = nh()
    ints = axial_integrals(mol)
    n_atom = gto.M(atom="N 0 0 0", basis="aug-cc-pvdz", spin=3, verbose=0)
    ia = axial_integrals(n_atom)
    atom = axial_uhf(ia, atomic_occupation({0: 2}, {1: [1, 0, -1]}), minao_density(n_atom, ia),
                     block_of_ao=atomic_blocks(n_atom, ia))
    dm, ref = embed_atomic_orbitals(mol, ints, 0, n_atom, atom,
                                    lambda spin, j, m, w: spin == 0 and m != 0 and w.get(1, 0) > 0.5)
    assert {m: v[0].shape[1] for m, v in ref.items()} == {1: 1, -1: 1}
    res = axial_uhf(ints, X3, dm, reference=ref)
    plain = axial_uhf(ints, X3, minao_density(mol, ints))
    assert res.converged and res.e_tot == pytest.approx(plain.e_tot, abs=1e-8)


def test_cholesky_integrals_reproduce_dense_scf_and_cc():
    """Cholesky vectors (R - I form, cholesky.py): SCF, J/K and CCSD(T)/CCSDT equal the dense
    integrals to the decomposition threshold, for ROHF (semicanonical) and UHF references."""
    from openea_benchmark.axial.cholesky import ao_cholesky
    from openea_benchmark.axial.scf import axial_uhf, coulomb_exchange
    mol = gto.M(atom="N 0 0 0; H 0 0 1.037", basis="cc-pvdz", spin=2, symmetry="C1", verbose=0)
    b = ao_cholesky(mol, tol=1e-8)
    assert np.abs(np.einsum("Ppq,Prs->pqrs", b, b) - mol.intor("int2e")).max() < 1e-8
    dense, chol = axial_integrals(mol), axial_integrals(mol, cholesky_tol=1e-8)
    assert np.abs(chol.eri.dense() - dense.eri).max() < 1e-8
    d = np.random.default_rng(0).normal(size=dense.hcore.shape)
    for x, y in zip(coulomb_exchange(dense.eri, d + d.T), coulomb_exchange(chol.eri, d + d.T)):
        assert np.abs(x - y).max() < 1e-6
    for occ in (X3, A1):
        r_d = axial_rohf(dense, occ, minao_density(mol, dense))
        r_c = axial_rohf(chol, occ, minao_density(mol, chol))
        assert r_c.e_tot == pytest.approx(r_d.e_tot, abs=1e-7)
        e_d = axial_cc(dense, r_d, mol.nelectron, 1, ("ccsd(t)", "ccsdt"))
        e_c = axial_cc(chol, r_c, mol.nelectron, 1, ("ccsd(t)", "ccsdt"))
        for k in e_d:
            assert r_c.e_tot + e_c[k] == pytest.approx(r_d.e_tot + e_d[k], abs=1e-7)
    u_d, u_c = (axial_uhf(i, X3, minao_density(mol, i)) for i in (dense, chol))
    e_d, e_c = (axial_cc(i, u, mol.nelectron, 1) for i, u in ((dense, u_d), (chol, u_c)))
    assert u_c.e_tot + e_c["ccsd(t)"] == pytest.approx(u_d.e_tot + e_d["ccsd(t)"], abs=1e-7)
    with pytest.raises(NotImplementedError):
        axial_cc(chol, r_c, mol.nelectron, 1, semicanonical=False)
