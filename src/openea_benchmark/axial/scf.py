"""ROHF of one axial determinant: a fixed (n_alpha, n_beta) per Lz block m.

Every Fock matrix is block diagonal in m, so the occupation of each block is a constant
of the iteration: the determinant keeps its M_L = sum_m m (n_alpha + n_beta) and cannot
fall to another Lambda.  Within a block the orbitals are filled by energy (doubly, then
singly occupied).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from .basis import AxialIntegrals


@dataclass
class AxialSCFResult:
    e_tot: float
    converged: bool
    cycles: int
    mo_coeff: np.ndarray      # axial AO basis, real; ordered doubly, singly, virtual
    mo_energy: np.ndarray     # Roothaan effective-Fock eigenvalues
    mo_occ: np.ndarray        # 2 / 1 / 0
    mo_m: np.ndarray
    fock_alpha: np.ndarray    # axial AO basis
    fock_beta: np.ndarray
    occupation: dict = field(default_factory=dict)


def coulomb_exchange(eri: np.ndarray, dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """J_pq = sum_rs (pq|sr) D_rs, K_pq = sum_rs (pr|sq) D_rs  (D_rs = sum_i c_ri c_si*)."""
    n = dm.shape[0]
    vj = (eri.reshape(n * n, n * n) @ dm.T.reshape(n * n)).reshape(n, n)
    vk = np.einsum("prsq,rs->pq", eri, dm, optimize=True)
    return vj, vk


def roothaan_fock(fa, fb, da, db, s):
    """ROHF effective Fock (Guest-Saunders canonicalization, as PySCF's ROHF)."""
    from pyscf.scf.rohf import get_roothaan_fock
    return np.asarray(get_roothaan_fock((fa, fb), (da, db), s))


def minao_density(mol, ints: AxialIntegrals) -> tuple[np.ndarray, np.ndarray]:
    """Superposition of atomic densities (cylindrically symmetric) in the axial basis."""
    from pyscf.scf.hf import init_guess_by_minao
    u = ints.transform
    d = u.conj().T @ init_guess_by_minao(mol) @ u
    return 0.5 * d.real, 0.5 * d.real


def axial_rohf(ints: AxialIntegrals, occupation: dict[int, tuple[int, int]], dm0, *,
               max_cycle: int = 300, conv_tol: float = 1e-10, conv_tol_grad: float = 1e-7,
               diis_space: int = 10, verbose: bool = False) -> AxialSCFResult:
    """``occupation``: {m: (n_alpha, n_beta)}, high spin per block (n_alpha >= n_beta)."""
    from pyscf.lib.diis import DIIS
    blocks = {int(k): np.flatnonzero(ints.m == k) for k in np.unique(ints.m)}
    occ = {int(k): (int(v[0]), int(v[1])) for k, v in occupation.items() if v[0] or v[1]}
    for k, (na, nb) in occ.items():
        if k not in blocks:
            raise ValueError(f"no basis function with m = {k}")
        if na < nb or na > len(blocks[k]):
            raise ValueError(f"m = {k}: occupation {(na, nb)} not high spin or above the block size")
    h, s, eri = ints.hcore, ints.ovlp, ints.eri
    n = h.shape[0]
    da, db = dm0
    diis = DIIS()
    diis.space = diis_space
    e_old, converged, e_tot = None, False, None
    for cycle in range(1, max_cycle + 1):
        ja, ka = coulomb_exchange(eri, da)
        jb, kb = coulomb_exchange(eri, db)
        fa, fb = h + ja + jb - ka, h + ja + jb - kb
        e_tot = float(np.sum(h * (da + db)) + 0.5 * (np.sum((ja + jb) * (da + db)) - np.sum(ka * da)
                                                     - np.sum(kb * db)) + ints.energy_nuc)
        f = roothaan_fock(fa, fb, da, db, s)
        err = f @ da @ s - s @ da @ f + f @ db @ s - s @ db @ f
        grad = float(np.abs(err).max())
        if verbose:
            print(f"cycle {cycle:3d}  E = {e_tot:.12f}  grad = {grad:.2e}", flush=True)
        if e_old is not None and abs(e_tot - e_old) < conv_tol and grad < conv_tol_grad:
            converged = True
            break
        e_old = e_tot
        if cycle > 1:
            f = diis.update(f, err)
        coeff, energy, occs, ms = np.zeros((n, n)), np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
        col = 0
        for k, idx in blocks.items():
            e_k, c_k = eigh(f[np.ix_(idx, idx)], s[np.ix_(idx, idx)])
            na, nb = occ.get(k, (0, 0))
            w = len(idx)
            coeff[idx, col:col + w], energy[col:col + w], ms[col:col + w] = c_k, e_k, k
            occs[col:col + nb], occs[col + nb:col + na] = 2.0, 1.0
            col += w
        da = coeff[:, occs > 0] @ coeff[:, occs > 0].T
        db = coeff[:, occs > 1] @ coeff[:, occs > 1].T
    # final orbitals: eigenvectors of the converged effective Fock
    coeff, energy, occs, ms = np.zeros((n, n)), np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
    f = roothaan_fock(fa, fb, da, db, s)
    col = 0
    for k, idx in blocks.items():
        e_k, c_k = eigh(f[np.ix_(idx, idx)], s[np.ix_(idx, idx)])
        na, nb = occ.get(k, (0, 0))
        w = len(idx)
        coeff[idx, col:col + w], energy[col:col + w], ms[col:col + w] = c_k, e_k, k
        occs[col:col + nb], occs[col + nb:col + na] = 2.0, 1.0
        col += w
    order = np.lexsort((energy, -occs))   # doubly, singly, virtual; by energy within
    return AxialSCFResult(e_tot=e_tot, converged=converged, cycles=cycle, mo_coeff=coeff[:, order],
                          mo_energy=energy[order], mo_occ=occs[order], mo_m=ms[order], fock_alpha=fa,
                          fock_beta=fb, occupation=occ)
