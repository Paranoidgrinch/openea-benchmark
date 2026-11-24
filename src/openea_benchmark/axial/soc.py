"""Spin-orbit increment for axial determinants: Omega-blocked GHF + GCCSD(T), real arithmetic.

Spin-orbital basis: alpha and beta copies of the axial AOs.  The X2C-1e two-component
one-electron Hamiltonian is real in this basis (checked) and conserves
Omega = m + sigma: the alpha-alpha and beta-beta blocks are diagonal in m, the
alpha-beta block couples alpha m only with beta m + 1 (PrO aDZ: every other element
< 1e-9).  With the spin-free Coulomb interaction the GHF Fock, the spinors and the
GCCSD amplitudes are all real (generalization of the UO pilot, which ran complex GHF in
Jz blocks).

Increment: Delta_SO = E[h_2c] - E[h_sf'] with the same GHF/GCCSD(T) code, where h_sf'
= (h_aa + h_bb) / 2 is the scalar part of the same two-component Hamiltonian (the
one-component sfX2C differs from it by a picture-change term, PrO core elements up to
0.39 Eh, that is not spin-orbit).  Spin-orbit from X2C-1e only (no two-electron
spin-orbit screening): an overestimate, to be carried as an uncertainty.

Lowest Omega component: Omega = Lambda - S for a less than half-filled f shell (regular
multiplet: M_S = -S), Lambda + S for more than half filled (M_S = +S); the start is the
axial UHF determinant with alpha and beta exchanged when M_S = -S is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .basis import AxialIntegrals, axial_ao_transform

IMAG_TOLERANCE = 1.0e-11   # relative


def spinor_hcore(mol, u: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(h_2c, h_sf'): X2C-1e two-component hcore over [alpha axial AOs, beta axial AOs] and
    its scalar part (h_aa + h_bb)/2, both real."""
    from pyscf import scf
    if u is None:
        u, _ = axial_ao_transform(mol)
    n = u.shape[0]
    h = scf.GHF(mol).x2c1e().get_hcore()
    ug = np.zeros((2 * n, 2 * n), complex)
    ug[:n, :n] = ug[n:, n:] = u
    hax = ug.conj().T @ h @ ug
    scale = np.abs(hax.real).max()
    if np.abs(hax.imag).max() > IMAG_TOLERANCE * scale:
        raise ValueError(f"two-component hcore not real in the axial basis ({np.abs(hax.imag).max():.1e})")
    h2 = np.ascontiguousarray(hax.real)
    h2 = 0.5 * (h2 + h2.T)
    return h2, 0.5 * (h2[:n, :n] + h2[n:, n:])


def _j(eri, dm):
    if hasattr(eri, "jk"):
        return eri.jk(dm)[0]
    n = dm.shape[0]
    return (eri.reshape(n * n, n * n) @ (0.5 * (dm + dm.T)).reshape(n * n)).reshape(n, n)


def _k(eri, dm):
    """K_pq = sum_rs (pr|sq) D_rs for any real D (also the non-symmetric alpha-beta block)."""
    if hasattr(eri, "r"):
        out = np.zeros_like(dm)
        for start in range(0, eri.nvec, 256):
            r, i = eri.r[start:start + 256], eri.i[start:start + 256]
            out += (r @ dm @ r).sum(axis=0) - (i @ dm @ i).sum(axis=0)
        return out
    return np.einsum("prsq,rs->pq", eri, dm, optimize=True)


def ghf_fock(ints: AxialIntegrals, h2: np.ndarray, dm: np.ndarray) -> tuple[np.ndarray, float]:
    n = ints.hcore.shape[0]
    daa, dab, dbb = dm[:n, :n], dm[:n, n:], dm[n:, n:]
    j = _j(ints.eri, daa + dbb)
    kab = _k(ints.eri, dab)
    g = np.block([[j - _k(ints.eri, daa), -kab], [-kab.T, j - _k(ints.eri, dbb)]])
    f = h2 + g
    e = float(np.sum(h2 * dm) + 0.5 * np.sum(g * dm) + ints.energy_nuc)
    return f, e


@dataclass
class AxialGHFResult:
    e_tot: float
    converged: bool
    cycles: int
    mo_coeff: np.ndarray      # (2n, 2n) real spinors, occupied first, energy order within
    mo_energy: np.ndarray
    nocc: int
    two_omega: np.ndarray     # 2 Omega of every spinor


def spinor_start(uhf, flip: bool) -> tuple[np.ndarray, list[int]]:
    """Occupied spinors (2n x nocc) of an axial UHF determinant; ``flip`` exchanges alpha
    and beta (M_S -> -M_S).  Returns the spinors and their 2 Omega."""
    ca, cb = uhf.mo_coeff
    na, nb = uhf.nocc
    ma, mb = uhf.mo_m
    n = ca.shape[0]
    cols, w = [], []
    for spin, (c, nocc, ms) in enumerate(((ca, na, ma), (cb, nb, mb))):
        to_beta = (spin == 0) == flip
        for j in range(nocc):
            v = np.zeros(2 * n)
            if to_beta:
                v[n:] = c[:, j]
                w.append(2 * int(ms[j]) - 1)
            else:
                v[:n] = c[:, j]
                w.append(2 * int(ms[j]) + 1)
            cols.append(v)
    return np.array(cols).T, w


def axial_ghf(ints: AxialIntegrals, h2: np.ndarray, start: np.ndarray, follow: list[int] | None = None, *,
              max_cycle: int = 200, conv_tol: float = 1e-10, conv_tol_grad: float = 1e-7,
              diis_space: int = 12, level_shift: float = 0.0, verbose: bool = False) -> AxialGHFResult:
    """GHF of one Omega-blocked determinant: the number of spinors per Omega block is that
    of ``start`` (columns: occupied spinors, each in one block).  ``follow``: columns of
    ``start`` to keep by maximum overlap with the previous iteration (open f / 6s)."""
    from pyscf.lib.diis import DIIS
    n = ints.hcore.shape[0]
    two_omega = np.concatenate([2 * ints.m + 1, 2 * ints.m - 1])
    blocks = {int(w): np.flatnonzero(two_omega == w) for w in np.unique(two_omega)}
    s = ints.ovlp
    sg = np.zeros((2 * n, 2 * n))
    sg[:n, :n] = sg[n:, n:] = s
    count, targets = {}, {}
    for col in range(start.shape[1]):
        w = [k for k, idx in blocks.items()
             if start[idx, col] @ sg[np.ix_(idx, idx)] @ start[idx, col] > 0.99]
        if len(w) != 1:
            raise ValueError(f"start spinor {col} not in one Omega block")
        count[w[0]] = count.get(w[0], 0) + 1
        if follow and col in follow:
            targets.setdefault(w[0], []).append(start[blocks[w[0]], col])
    dm = start @ start.T
    diis = DIIS()
    diis.space = diis_space
    e_old, converged = None, False

    def diagonalize(f):
        coeff, energy, occ, wl = np.zeros((2 * n, 2 * n)), np.zeros(2 * n), np.zeros(2 * n, bool), np.zeros(2 * n, int)
        col = 0
        for k, idx in blocks.items():
            e_k, c_k = eigh(f[np.ix_(idx, idx)], sg[np.ix_(idx, idx)])
            nk = count.get(k, 0)
            pick = []
            for t in targets.get(k, [])[:nk]:
                ov = (t @ sg[np.ix_(idx, idx)] @ c_k) ** 2
                ov[pick] = -1.0
                pick.append(int(np.argmax(ov)))
            if k in targets:
                targets[k] = [c_k[:, j] for j in pick]
            pick += [j for j in range(len(idx)) if j not in pick][:nk - len(pick)]
            order = sorted(pick) + [j for j in range(len(idx)) if j not in pick]
            w_ = len(idx)
            coeff[idx, col:col + w_] = c_k[:, order]
            energy[col:col + w_] = e_k[order]
            occ[col:col + nk] = True
            wl[col:col + w_] = k
            col += w_
        return coeff, energy, occ, wl

    for cycle in range(1, max_cycle + 1):
        f, e = ghf_fock(ints, h2, dm)
        err = f @ dm @ sg - sg @ dm @ f
        grad = float(np.abs(err).max())
        if verbose:
            print(f"GHF cycle {cycle:3d}  E = {e:.12f}  grad = {grad:.2e}", flush=True)
        if e_old is not None and abs(e - e_old) < conv_tol and grad < conv_tol_grad:
            converged = True
            break
        e_old = e
        if cycle > 1:
            f = diis.update(f, err)
        if level_shift:
            f = f + level_shift * (sg - sg @ dm @ sg)
        coeff, energy, occ, wl = diagonalize(f)
        dm = coeff[:, occ] @ coeff[:, occ].T
    f, e = ghf_fock(ints, h2, dm)
    coeff, energy, occ, wl = diagonalize(f)
    order = np.lexsort((energy, ~occ))
    return AxialGHFResult(e_tot=e, converged=converged, cycles=cycle, mo_coeff=coeff[:, order],
                          mo_energy=energy[order], nocc=int(occ.sum()), two_omega=wl[order])


def ghf_cc(ints: AxialIntegrals, h2: np.ndarray, ghf: AxialGHFResult, nfrozen: int,
           with_t: bool = True, conv_tol: float = 1e-9) -> dict:
    """GCCSD(T) on the GHF spinors (``nfrozen`` lowest spinors frozen), PySCF GCCSD with
    integrals built from the axial (Cholesky or dense) integrals."""
    from pyscf import cc, gto, scf
    from pyscf.cc import gccsd, gccsd_t
    from .cholesky import AxialCholesky, mo_block
    n = ints.hcore.shape[0]
    c = ghf.mo_coeff[:, nfrozen:]
    nocc = ghf.nocc - nfrozen
    nmo = c.shape[1]
    f_ao, _ = ghf_fock(ints, h2, ghf.mo_coeff[:, :ghf.nocc] @ ghf.mo_coeff[:, :ghf.nocc].T)
    fock = c.T @ f_ao @ c
    ca, cb = c[:n], c[n:]
    if isinstance(ints.eri, AxialCholesky):
        ra, ia = ints.eri.mo(ca)
        rb, ib = ints.eri.mo(cb)
        vec = (ra + rb, ia + ib)
        o, v = slice(0, nocc), slice(nocc, nmo)

        def block(x1, x2, x3, x4):
            return np.ascontiguousarray(mo_block(vec, vec, x1, x2, x3, x4, antisymmetrize=True))
    else:
        eri_mo = sum(np.einsum("ijkl,ip,jq,kr,ls->pqrs", ints.eri, c1, c1, c2, c2, optimize=True)
                     for c1 in (ca, cb) for c2 in (ca, cb))
        phys = eri_mo.transpose(0, 2, 1, 3) - eri_mo.transpose(0, 2, 3, 1)
        o, v = slice(0, nocc), slice(nocc, nmo)

        def block(x1, x2, x3, x4):
            return np.ascontiguousarray(phys[x1, x2, x3, x4])
    mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)   # carrier only
    mol.nelectron = nocc
    mf = scf.GHF(mol)
    mf.mo_occ = np.array([1.0] * nocc + [0.0] * (nmo - nocc))
    mf.mo_energy = np.diag(fock).copy()
    mf.mo_coeff = np.eye(nmo)
    mf.e_tot = ghf.e_tot
    mf.converged = True
    mycc = cc.GCCSD(mf)
    mycc.conv_tol = conv_tol
    mycc.max_cycle = 200
    mycc.verbose = 0
    mycc.get_e_hf = lambda *args, **kwargs: ghf.e_tot   # the carrier mf has no integrals
    eris = gccsd._PhysicistsERIs()
    eris.mol = mol
    eris.nocc = nocc
    eris.fock = fock
    eris.mo_energy = np.diag(fock).copy()
    eris.e_hf = ghf.e_tot
    eris.oooo = block(o, o, o, o)
    eris.ooov = block(o, o, o, v)
    eris.oovv = block(o, o, v, v)
    eris.ovov = block(o, v, o, v)
    eris.ovvo = block(o, v, v, o)
    eris.ovvv = block(o, v, v, v)
    eris.vvvv = block(v, v, v, v)
    mycc.kernel(eris=eris)
    out = {"ccsd": float(mycc.e_corr), "converged": bool(mycc.converged)}
    if with_t:
        out["ccsd(t)"] = out["ccsd"] + float(gccsd_t.kernel(mycc, eris, mycc.t1, mycc.t2, verbose=0))
    return out
