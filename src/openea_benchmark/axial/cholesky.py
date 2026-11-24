"""Cholesky-decomposed two-electron integrals in the axial basis (real arithmetic).

AO integrals: pivoted Cholesky with shell-pair column blocks, (mu nu|la si) ~ sum_P
B_P,mu nu B_P,la si with real symmetric B_P, error on every diagonal below ``tol``.
Axial basis: C_P = U^+ B_P U is Hermitian; R_P = Re C_P (symmetric) and I_P = Im C_P
(antisymmetric), and the real axial integrals are

    (pq|rs) = sum_P R_P,pq R_P,rs - I_P,pq I_P,rs

(the imaginary part of sum_P C_P,pq C_P,rs vanishes for the exact integrals and is dropped
for the truncated ones).  The sign makes the axial (pq|rs) indefinite as a matrix over
pairs, so CCpy's own Cholesky CC (one positive vector set) does not apply; CCpy gets its
integral blocks built from the vectors one block at a time instead, which removes the
N^4 AO and axial arrays (the axial aQZ oxides: N ~ 315, 78 GB per dense copy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ao_cholesky(mol, tol: float = 1.0e-6, block_selection: float = 1.0e-2, verbose: bool = False) -> np.ndarray:
    """Real AO Cholesky vectors (n_vec, nao, nao), symmetric in the AO pair.

    Pivots are AO pairs mu >= nu; one integral call gives the columns of the pivot's whole
    shell pair, of which every column with a residual diagonal above
    max(tol, block_selection * current maximum) becomes a vector (largest first)."""
    from pyscf import lib
    nao, nbas = mol.nao_nr(), mol.nbas
    loc = mol.ao_loc_nr()
    shell_of = np.repeat(np.arange(nbas), np.diff(loc))
    rows, cols = np.tril_indices(nao)
    npair = rows.size
    pair_index = np.full((nao, nao), -1)
    pair_index[rows, cols] = np.arange(npair)
    diag = np.empty(npair)
    for i in range(nbas):
        for j in range(i + 1):
            buf = mol.intor("int2e", shls_slice=(i, i + 1, j, j + 1, i, i + 1, j, j + 1))
            d = np.einsum("ijij->ij", buf)
            for a in range(loc[i], loc[i + 1]):
                for b in range(loc[j], min(loc[j + 1], a + 1)):
                    diag[pair_index[a, b]] = d[a - loc[i], b - loc[j]]
    residual = diag.copy()
    capacity = 8 * nao
    vec = np.empty((capacity, npair))
    nvec = 0
    while True:
        dmax = float(residual.max())
        if dmax < tol:
            break
        top = int(np.argmax(residual))
        si, sj = shell_of[rows[top]], shell_of[cols[top]]
        si, sj = max(si, sj), min(si, sj)
        block = mol.intor("int2e", aosym="s2ij", shls_slice=(0, nbas, 0, nbas, si, si + 1, sj, sj + 1))
        block = block.reshape(npair, -1)
        bpairs = np.array([[pair_index[max(a, b), min(a, b)] for b in range(loc[sj], loc[sj + 1])]
                           for a in range(loc[si], loc[si + 1])]).ravel()
        if nvec:
            block -= vec[:nvec].T @ vec[:nvec, bpairs]
        floor = max(tol, block_selection * dmax)
        taken = set()
        while True:
            d_block = residual[bpairs].copy()
            for k in taken:
                d_block[k] = -1.0
            k = int(np.argmax(d_block))
            if d_block[k] < floor or bpairs[k] in {bpairs[t] for t in taken}:
                break
            v = block[:, k] / np.sqrt(d_block[k])
            if nvec == capacity:
                capacity *= 2
                vec = np.concatenate([vec[:nvec], np.empty((capacity - nvec, npair))])
            vec[nvec] = v
            nvec += 1
            residual -= v * v
            np.maximum(residual, 0.0, out=residual)
            block -= np.outer(v, v[bpairs])
            # a diagonal shell pair lists (a, b) and (b, a): both map to one pair
            taken.update(t for t in range(bpairs.size) if bpairs[t] == bpairs[k])
        if verbose and nvec % 200 < 10:
            print(f"cholesky: {nvec} vectors, max residual {residual.max():.2e}", flush=True)
    return lib.unpack_tril(vec[:nvec])


@dataclass
class AxialCholesky:
    """(pq|rs) = sum_P R[P,p,q] R[P,r,s] - I[P,p,q] I[P,r,s] over axial AOs."""
    r: np.ndarray     # (n_vec, n, n) symmetric
    i: np.ndarray     # (n_vec, n, n) antisymmetric
    tol: float

    @property
    def nvec(self) -> int:
        return self.r.shape[0]

    def dense(self) -> np.ndarray:
        n = self.r.shape[1]
        r, i = self.r.reshape(self.nvec, n * n), self.i.reshape(self.nvec, n * n)
        return (r.T @ r - i.T @ i).reshape(n, n, n, n)

    def jk(self, dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """J_pq = sum_rs (pq|sr) D_rs, K_pq = sum_rs (pr|sq) D_rs for a real symmetric D."""
        dm = 0.5 * (dm + dm.T)
        vj = np.einsum("P,Ppq->pq", np.einsum("Psr,rs->P", self.r, dm, optimize=True), self.r, optimize=True)
        vk = np.zeros_like(dm)
        for start in range(0, self.nvec, 256):
            r, i = self.r[start:start + 256], self.i[start:start + 256]
            vk += (r @ dm @ r).sum(axis=0) - (i @ dm @ i).sum(axis=0)
        return vj, vk

    def mo(self, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectors over real axial MO coefficients: (C^T R_P C, C^T I_P C)."""
        return c.T @ self.r @ c, c.T @ self.i @ c


def axial_cholesky(mol, u: np.ndarray, tol: float = 1.0e-6, verbose: bool = False) -> AxialCholesky:
    """AO Cholesky vectors carried into the axial basis of ``axial_ao_transform``."""
    from .basis import _transform_index
    b = ao_cholesky(mol, tol=tol, verbose=verbose)
    c = _transform_index(b, u, 1, True)
    del b
    c = _transform_index(c, u, 2, False)
    return AxialCholesky(r=np.ascontiguousarray(c.real), i=np.ascontiguousarray(c.imag), tol=tol)


def mo_block(vec1, vec2, x1, x2, x3, x4, antisymmetrize: bool, chunk: int = 16) -> np.ndarray:
    """Physicists' <pq|rs> = (pr|qs) (minus (ps|qr) if ``antisymmetrize``) for p in x1, q in x2,
    r in x3, s in x4; vec1 = (R, I) of the spin of p, r and vec2 of q, s.  Fortran order
    (CCpy), filled in chunks of p so the only temporaries are chunk sized."""
    (r1, i1), (r2, i2) = vec1, vec2
    n1, n2, n3, n4 = (len(range(*x.indices(r1.shape[1] if k in (0, 2) else r2.shape[1])))
                      for k, x in enumerate((x1, x2, x3, x4)))
    out = np.empty((n1, n2, n3, n4), order="F")
    p0 = x1.start or 0
    nvec = r1.shape[0]
    b_r, b_i = r2[:, x2, x4].reshape(nvec, n2 * n4), i2[:, x2, x4].reshape(nvec, n2 * n4)
    if antisymmetrize:
        c_r, c_i = r2[:, x2, x3].reshape(nvec, n2 * n3), i2[:, x2, x3].reshape(nvec, n2 * n3)
    for a in range(0, n1, chunk):
        pc = slice(p0 + a, p0 + min(a + chunk, n1))
        npc = pc.stop - pc.start
        a_r, a_i = r1[:, pc, x3].reshape(nvec, npc * n3), i1[:, pc, x3].reshape(nvec, npc * n3)
        t = (a_r.T @ b_r - a_i.T @ b_i).reshape(npc, n3, n2, n4).transpose(0, 2, 1, 3)
        if antisymmetrize:
            e_r, e_i = r1[:, pc, x4].reshape(nvec, npc * n4), i1[:, pc, x4].reshape(nvec, npc * n4)
            t = t - (e_r.T @ c_r - e_i.T @ c_i).reshape(npc, n4, n2, n3).transpose(0, 2, 3, 1)
        out[a:a + npc] = t
    return out
