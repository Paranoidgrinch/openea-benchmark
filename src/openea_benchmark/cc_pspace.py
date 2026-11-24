"""Full CCSDT as CCpy CC(P) with every symmetry-allowed triple in P.

CCpy's dense CCSDT (``ccsdt``) keeps every T3 spin block as a full v^3 o^3 array
(each unique amplitude of the same-spin blocks 36 times, of the mixed blocks 4 times)
and spends its time in numpy reshapes and single-threaded array arithmetic; CC(P)
(``ccsdt_p``) keeps only the listed triples and runs OpenMP Fortran loops.  With the
list = all triples whose excitation conserves the reference symmetry, CC(P) solves
exactly the CCSDT equations (amplitudes of other symmetry are zero by symmetry).
CN 2Sigma+ aug-cc-pVDZ (C2v, 8 threads): dense 16.7 s/iteration and 3.4 GB, CC(P)
with all triples 53.3 s and 0.8 GB, CC(P) symmetry-allowed 7.3 s and 0.6 GB; the same
25 iterations and E_corr to 1e-10.

Symmetry labels: an abelian irrep number per orbital (combined with XOR) or an axial
m quantum number (combined by sum; complex m orbitals, Delta m = sum m_virt - sum m_occ).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def _tuples(labels: np.ndarray, k: int, combine: str) -> tuple[np.ndarray, np.ndarray]:
    idx = np.array(list(combinations(range(len(labels)), k)), dtype=np.int32).reshape(-1, k)
    if not len(idx):
        return idx, np.zeros(0, dtype=np.int64)
    lab = labels[idx]
    total = np.bitwise_xor.reduce(lab, axis=1) if combine == "xor" else lab.sum(axis=1)
    return idx, total


def _cartesian(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Rows (first[p], second[q]) for all p, q (first-major), as one int32 array."""
    p = np.repeat(np.arange(len(first)), len(second))
    q = np.tile(np.arange(len(second)), len(first))
    return np.hstack([first[p], second[q]])


def _match(occ_idx, occ_lab, vir_idx, vir_lab) -> np.ndarray:
    """[virt tuple, occ tuple] rows with equal symmetry label (0-based)."""
    blocks = []
    for value in np.intersect1d(np.unique(occ_lab), np.unique(vir_lab)):
        rows = _cartesian(occ_idx[occ_lab == value], vir_idx[vir_lab == value])
        no = occ_idx.shape[1]
        blocks.append(np.hstack([rows[:, no:], rows[:, :no]]))
    return np.vstack(blocks) if blocks else np.zeros((0, 6), dtype=np.int32)


def symmetric_triples(occ_a, vir_a, occ_b, vir_b, combine: str = "xor") -> dict[str, np.ndarray]:
    """t3_excitations for CCpy run_ccp: every triple (a, b, c, i, j, k), 1-based, a<b, i<j within
    a spin, whose excitation keeps the reference symmetry.  ``occ_a`` ... are the labels of the
    correlated occupied / virtual orbitals of each spin (CCpy order)."""
    occ_a, vir_a, occ_b, vir_b = (np.asarray(x, dtype=np.int64) for x in (occ_a, vir_a, occ_b, vir_b))
    out = {}
    # same spin
    for case, occ, vir in (("aaa", occ_a, vir_a), ("bbb", occ_b, vir_b)):
        oi, ol = _tuples(occ, 3, combine)
        vi, vl = _tuples(vir, 3, combine)
        out[case] = _match(oi, ol, vi, vl)
    # mixed: aab = (a<b alpha, c beta; i<j alpha, k beta), abb = (a alpha, b<c beta; i alpha, j<k beta)
    for case, o2, v2, o1, v1 in (("aab", occ_a, vir_a, occ_b, vir_b), ("abb", occ_b, vir_b, occ_a, vir_a)):
        pi, pl = _tuples(o2, 2, combine)
        qi, ql = _tuples(v2, 2, combine)
        si = np.arange(len(o1), dtype=np.int32)[:, None]
        ti = np.arange(len(v1), dtype=np.int32)[:, None]
        oc = _cartesian(pi, si) if case == "aab" else _cartesian(si, pi)
        vc = _cartesian(qi, ti) if case == "aab" else _cartesian(ti, qi)
        if combine == "xor":
            olab = (pl[np.repeat(np.arange(len(pi)), len(si))] ^ o1[np.tile(si[:, 0], len(pi))]) if case == "aab" \
                else (o1[np.repeat(si[:, 0], len(pi))] ^ pl[np.tile(np.arange(len(pi)), len(si))])
            vlab = (ql[np.repeat(np.arange(len(qi)), len(ti))] ^ v1[np.tile(ti[:, 0], len(qi))]) if case == "aab" \
                else (v1[np.repeat(ti[:, 0], len(qi))] ^ ql[np.tile(np.arange(len(qi)), len(ti))])
        else:
            olab = (pl[np.repeat(np.arange(len(pi)), len(si))] + o1[np.tile(si[:, 0], len(pi))]) if case == "aab" \
                else (o1[np.repeat(si[:, 0], len(pi))] + pl[np.tile(np.arange(len(pi)), len(si))])
            vlab = (ql[np.repeat(np.arange(len(qi)), len(ti))] + v1[np.tile(ti[:, 0], len(qi))]) if case == "aab" \
                else (v1[np.repeat(ti[:, 0], len(qi))] + ql[np.tile(np.arange(len(qi)), len(ti))])
        out[case] = _match(oc, olab, vc, vlab)
    for case, rows in out.items():
        out[case] = np.asfortranarray(rows + 1) if len(rows) else np.ones((1, 6), dtype=np.int32, order="F")
    return out


def run_symmetric_ccsdt(driver, labels_a, labels_b, noa: int, nob: int, combine: str = "xor") -> dict[str, int]:
    """driver.run_ccp('ccsdt_p') with every symmetry-allowed triple; ``labels_a``/``labels_b``:
    symmetry label of each correlated orbital of each spin (occupied first).  Returns the
    number of triples per spin block."""
    la, lb = np.asarray(labels_a), np.asarray(labels_b)
    ex = symmetric_triples(la[:noa], la[noa:], lb[:nob], lb[nob:], combine)
    driver.run_ccp(method="ccsdt_p", t3_excitations=ex)
    return {k: int(v.shape[0]) for k, v in ex.items()}
