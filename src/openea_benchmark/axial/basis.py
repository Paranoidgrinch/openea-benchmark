"""Axial AO basis: chi_{+-m} = (cos-type +- i sin-type)/sqrt(2) within each real spherical shell."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IMAG_TOLERANCE = 1.0e-10


@dataclass
class AxialIntegrals:
    """Hamiltonian of one molecule in the axial AO basis (all arrays real)."""
    m: np.ndarray            # Lz quantum number of every axial AO
    transform: np.ndarray    # AO -> axial (nao x nao complex, unitary in coefficient space)
    hcore: np.ndarray
    ovlp: np.ndarray
    eri: np.ndarray          # chemists' (pq|rs) = int chi_p* chi_q (1/r12) chi_r* chi_s
    energy_nuc: float


def axial_ao_transform(mol) -> tuple[np.ndarray, np.ndarray]:
    """(U, m): columns of U are the axial AOs in the real AO basis, m their Lz eigenvalues.

    PySCF orders a spherical shell x, y, z for l = 1 and m = -l..l (sin-type m < 0, cos-type
    m > 0) for l >= 2.  The sign of each pair's m is taken from <Lz>, so any sign convention
    of the real harmonics is followed; all functions share the x axis, which keeps every
    integral real."""
    if mol.cart:
        raise ValueError("axial basis needs spherical AOs")
    if np.abs(mol.atom_coords()[:, :2]).max() > 1e-8:
        raise ValueError("axial basis needs every atom on the z axis")
    nao = mol.nao_nr()
    with mol.with_common_origin((0.0, 0.0, 0.0)):
        lz = -1j * mol.intor("int1e_cg_irxp", comp=3)[2]
    s = mol.intor("int1e_ovlp")
    u = np.zeros((nao, nao), dtype=complex)
    m = np.zeros(nao, dtype=int)
    offset = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        for _ in range(mol.bas_nctr(ib)):
            idx = offset + np.arange(2 * l + 1)
            if l == 0:
                zero, pairs = idx[0], []
            elif l == 1:
                zero, pairs = idx[2], [(idx[0], idx[1])]
            else:
                zero, pairs = idx[l], [(idx[l + k], idx[l - k]) for k in range(1, l + 1)]
            u[zero, zero] = 1.0
            for k, (c, sn) in enumerate(pairs, start=1):
                plus = np.zeros(nao, dtype=complex)
                plus[c], plus[sn] = 1.0 / np.sqrt(2.0), 1j / np.sqrt(2.0)
                value = (plus.conj() @ lz @ plus).real / (plus.conj() @ s @ plus).real
                if abs(abs(value) - k) > 1e-6:
                    raise ValueError(f"shell {ib}: <Lz> = {value} for a |m| = {k} pair")
                sign = 1 if value > 0 else -1
                u[:, c], m[c] = plus, sign * k
                u[:, sn], m[sn] = plus.conj(), -sign * k
            offset += 2 * l + 1
    return u, m


def _transform_index(a: np.ndarray, u: np.ndarray, axis: int, conj: bool) -> np.ndarray:
    """Contract one AO index of ``a`` with the (at most two nonzeros per column) matrix u."""
    a = np.moveaxis(a, axis, -1)
    nz = [np.flatnonzero(u[:, j]) for j in range(u.shape[1])]
    i1 = np.array([z[0] for z in nz])
    i2 = np.array([z[-1] for z in nz])
    c1 = u[i1, np.arange(u.shape[1])]
    c2 = np.where(i2 != i1, u[i2, np.arange(u.shape[1])], 0.0)
    if conj:
        c1, c2 = c1.conj(), c2.conj()
    out = a[..., i1] * c1 + a[..., i2] * c2
    return np.moveaxis(out, -1, axis)


def axial_integrals(mol, hcore_ao: np.ndarray | None = None) -> AxialIntegrals:
    """Integrals in the axial AO basis; their imaginary parts must vanish (checked).

    ``hcore_ao`` lets a scalar-relativistic one-electron operator (sfX2C1e) in."""
    u, m = axial_ao_transform(mol)
    if hcore_ao is None:
        h = mol.intor("int1e_kin") + mol.intor("int1e_nuc") + (mol.intor("ECPscalar") if mol.has_ecp() else 0.0)
    else:
        h = hcore_ao
    h_ax = u.conj().T @ h @ u
    s_ax = u.conj().T @ mol.intor("int1e_ovlp") @ u
    eri = mol.intor("int2e", aosym="s1")
    for axis, conj in ((0, True), (1, False), (2, True), (3, False)):
        eri = _transform_index(eri, u, axis, conj)
    imag = max(np.abs(h_ax.imag).max(), np.abs(s_ax.imag).max(), np.abs(eri.imag).max())
    if imag > IMAG_TOLERANCE:
        raise ValueError(f"axial integrals not real (max |Im| {imag:.1e})")
    return AxialIntegrals(m=m, transform=u, hcore=np.ascontiguousarray(h_ax.real),
                          ovlp=np.ascontiguousarray(s_ax.real), eri=np.ascontiguousarray(eri.real),
                          energy_nuc=float(mol.energy_nuc()))
