"""ROHF of one axial determinant: a fixed (n_alpha, n_beta) per Lz block m.

Every Fock matrix is block diagonal in m, so the occupation of each block is a constant
of the iteration: the determinant keeps its M_L = sum_m m (n_alpha + n_beta) and cannot
fall to another Lambda.  Within a block the orbitals are filled by energy (doubly, then
singly occupied) unless ``open_target`` is given: a mask of axial AOs (e.g. the 4f shell of
a lanthanide) whose orbitals are to carry the open shell.  Then the first iteration takes
as singly occupied the orbitals of each block with the largest Mulliken weight on the mask,
and every later one the orbitals of maximum overlap with that first selection (initial
MOM, Barca/Gilbert/Gill 2018: overlap with the previous iteration lets the open orbital
drift, Pr m = 1 4f -> 82 % 6p in 123 cycles), separately for the doubly and singly
occupied sets.  At the HF level 4f^n lies 2-3 eV too high against 4f^(n-1) 5d (f
correlation), so the wanted determinant is an excited HF solution and needs this guard.  Pr 4f3 6s2 (M_L = 6): by energy alone
the m = 2 open orbital is 5d (4f2 5d 6s2), with the 4f mask it is 4f.
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


def _select(c_k, s_k, nd, ns, previous, weight_mask):
    """Column order of a block's eigenvectors: doubly, singly occupied, rest (see module doc)."""
    n = c_k.shape[1]
    if previous is not None:
        d_prev, s_prev = previous
        sc = s_k @ c_k
        pd = (d_prev.T @ sc) ** 2 if d_prev.size else np.zeros((0, n))
        ps = (s_prev.T @ sc) ** 2 if s_prev.size else np.zeros((0, n))
        dbl = list(np.argsort(-pd.sum(axis=0), kind="stable")[:nd])
        rest = [j for j in np.argsort(-ps.sum(axis=0), kind="stable") if j not in dbl]
        sgl = rest[:ns]
    elif weight_mask is not None and ns:
        # the lowest orbitals that are mostly on the mask (compact 4f, not the diffuse f
        # functions of the basis, which are 100 % f as well); eigenvalues are ascending
        w = np.einsum("pj,pj->j", c_k[weight_mask], (s_k @ c_k)[weight_mask])
        on = [j for j in range(n) if w[j] > 0.5]
        sgl = (on + [j for j in np.argsort(-w, kind="stable") if j not in on])[:ns]
        dbl = [j for j in range(n) if j not in sgl][:nd]
    else:
        dbl, sgl = list(range(nd)), list(range(nd, nd + ns))
    return dbl + sgl + [j for j in range(n) if j not in dbl and j not in sgl]


def axial_rohf(ints: AxialIntegrals, occupation: dict[int, tuple[int, int]], dm0, *,
               max_cycle: int = 300, conv_tol: float = 1e-10, conv_tol_grad: float = 1e-7,
               diis_space: int = 10, open_target: np.ndarray | None = None,
               level_shift: float = 0.0, reference: dict | None = None,
               verbose: bool = False) -> AxialSCFResult:
    """``occupation``: {m: (n_alpha, n_beta)}, high spin per block (n_alpha >= n_beta);
    ``open_target``: boolean mask of axial AOs for the open shell (switches MOM on);
    ``level_shift``: virtual orbitals raised by it, open ones by half, while iterating (the
    converged orbitals and energy do not depend on it); ``reference``: {m: (C_doubly,
    C_singly)} block orbitals the MOM follows from the first iteration (``core_guess``)."""
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
    mask = None if open_target is None else {k: np.asarray(open_target, bool)[idx] for k, idx in blocks.items()}

    def diagonalize(f, previous):
        coeff, energy, occs, ms = np.zeros((n, n)), np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
        kept, col = {}, 0
        for k, idx in blocks.items():
            s_k = s[np.ix_(idx, idx)]
            e_k, c_k = eigh(f[np.ix_(idx, idx)], s_k)
            na, nb = occ.get(k, (0, 0))
            if mask is not None:
                order_k = _select(c_k, s_k, nb, na - nb, None if previous is None else previous[k], mask[k])
                e_k, c_k = e_k[order_k], c_k[:, order_k]
            w = len(idx)
            coeff[idx, col:col + w], energy[col:col + w], ms[col:col + w] = c_k, e_k, k
            occs[col:col + nb], occs[col + nb:col + na] = 2.0, 1.0
            kept[k] = (c_k[:, :nb], c_k[:, nb:na])
            col += w
        # initial MOM: the reference stays the first selection
        return coeff, energy, occs, ms, (previous if previous is not None else kept) if mask is not None else None

    if reference is not None and mask is None:
        mask = {k: np.zeros(len(idx), bool) for k, idx in blocks.items()}
    previous = None if reference is None else {
        k: reference.get(k, (np.zeros((len(idx), 0)), np.zeros((len(idx), 0)))) for k, idx in blocks.items()}
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
        if level_shift:
            sda, sdb = s @ da @ s, s @ db @ s
            f = f + level_shift * (s - sda) + 0.5 * level_shift * (sda - sdb)
        coeff, energy, occs, ms, previous = diagonalize(f, previous)
        da = coeff[:, occs > 0] @ coeff[:, occs > 0].T
        db = coeff[:, occs > 1] @ coeff[:, occs > 1].T
    # final orbitals: eigenvectors of the converged effective Fock
    coeff, energy, occs, ms, _ = diagonalize(roothaan_fock(fa, fb, da, db, s), previous)
    order = np.lexsort((energy, -occs))   # doubly, singly, virtual; by energy within
    return AxialSCFResult(e_tot=e_tot, converged=converged, cycles=cycle, mo_coeff=coeff[:, order],
                          mo_energy=energy[order], mo_occ=occs[order], mo_m=ms[order], fock_alpha=fa,
                          fock_beta=fb, occupation=occ)


def core_guess(ints: AxialIntegrals, occupation: dict[int, tuple[int, int]], open_target: np.ndarray,
               core_occupation: dict[int, int], **kw) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    """Start of an open-shell axial SCF from a closed-shell core ion (UO pilot: UO4+).

    ``core_occupation``: {m: doubly occupied} of the closed core (same integrals, fewer
    electrons); its SCF gives well-defined empty open-shell orbitals.  Per block, the core
    orbitals stay doubly occupied, the remaining doubly occupied ones of ``occupation`` and
    the singly occupied ones are the lowest core virtuals with Mulliken weight > 0.5 on
    ``open_target`` (else the lowest virtuals).  Returns (dm0, MOM reference)."""
    core = axial_rohf(ints, {k: (v, v) for k, v in core_occupation.items()},
                      minao_like_zero(ints), **kw)
    if not core.converged:
        raise RuntimeError("closed-shell core SCF not converged")
    s = ints.ovlp
    w = np.einsum("pj,pj->j", core.mo_coeff * np.asarray(open_target, bool)[:, None], s @ core.mo_coeff)
    reference, cols_a, cols_b = {}, [], []
    for k in sorted(set(occupation) | set(core_occupation)):
        na, nb = occupation.get(k, (0, 0))
        ncore = core_occupation.get(k, 0)
        in_k = np.flatnonzero(core.mo_m == k)
        occ_core = [j for j in in_k if core.mo_occ[j] > 0][:ncore]
        virt = [j for j in in_k if core.mo_occ[j] == 0]           # energy order within the block
        on = [j for j in virt if w[j] > 0.5]
        extra = (on + [j for j in virt if j not in on])[:na - ncore]
        dbl, sgl = occ_core + extra[:nb - ncore], extra[nb - ncore:]
        if len(dbl) != nb or len(sgl) != na - nb:
            raise ValueError(f"m = {k}: core {ncore} cannot be completed to {(na, nb)}")
        idx = np.flatnonzero(ints.m == k)
        reference[k] = (core.mo_coeff[np.ix_(idx, dbl)], core.mo_coeff[np.ix_(idx, sgl)])
        cols_a += dbl + sgl
        cols_b += dbl
    ca, cb = core.mo_coeff[:, cols_a], core.mo_coeff[:, cols_b]
    return (ca @ ca.T, cb @ cb.T), reference


def minao_like_zero(ints: AxialIntegrals) -> tuple[np.ndarray, np.ndarray]:
    """Core-Hamiltonian start (zero density) for a closed-shell core."""
    n = ints.hcore.shape[0]
    return np.zeros((n, n)), np.zeros((n, n))


@dataclass
class AxialUHFResult:
    e_tot: float
    converged: bool
    cycles: int
    mo_coeff: tuple          # (C_alpha, C_beta), axial AO basis, occupied first (energy order)
    mo_energy: tuple
    nocc: tuple              # (n_alpha, n_beta)
    mo_m: tuple
    s2: float                # <S^2> of the determinant


def axial_uhf(ints: AxialIntegrals, occupation: dict[int, tuple[int, int]], dm0, *,
              reference: dict | None = None, max_cycle: int = 300, conv_tol: float = 1e-10,
              conv_tol_grad: float = 1e-7, diis_space: int = 12, level_shift: float = 0.0,
              follow: str = "update", block_of_ao: np.ndarray | None = None,
              verbose: bool = False) -> AxialUHFResult:
    """UHF of one axial determinant: (n_alpha, n_beta) fixed per m block (as the UO pilot GHF).

    ``reference``: {m: (C_alpha_occ, C_beta_occ)} block orbitals to follow by maximum overlap
    (``follow="update"``: the previous iteration, MOM; ``"initial"``: the reference, IMOM);
    without it each block is filled by energy.  ``block_of_ao``: finer block key per axial AO
    (``atomic_blocks``: (l, m) for one atom, where l is a symmetry of the restricted
    problem); ``occupation`` and ``reference`` are then keyed by it."""
    from pyscf.lib.diis import DIIS
    keys = [int(k) for k in ints.m] if block_of_ao is None else [tuple(k) for k in block_of_ao]
    blocks = {}
    for i, k in enumerate(keys):
        blocks.setdefault(k, []).append(i)
    blocks = {k: np.array(v) for k, v in sorted(blocks.items())}
    occ = {k: (int(v[0]), int(v[1])) for k, v in occupation.items() if v[0] or v[1]}
    for k in occ:
        if k not in blocks:
            raise ValueError(f"no basis function in block {k}")
    h, s, eri = ints.hcore, ints.ovlp, ints.eri
    n = h.shape[0]
    # Fock elements between blocks are zero by symmetry for m blocks; for (l, m) blocks the
    # restriction drops them, so gradient and DIIS use the block-diagonal Fock only
    same = np.zeros((n, n), bool)
    for idx in blocks.values():
        same[np.ix_(idx, idx)] = True
    da, db = dm0
    diis = DIIS()
    diis.space = diis_space
    ref = reference
    e_old, converged, e_tot = None, False, None

    def diagonalize(fa, fb, ref):
        out, new_ref = [], {}
        for spin, f in enumerate((fa, fb)):
            coeff, energy, ms, occs = np.zeros((n, n)), np.zeros(n), np.zeros(n, int), np.zeros(n)
            m_of = ints.m
            col = 0
            for k, idx in blocks.items():
                s_k = s[np.ix_(idx, idx)]
                e_k, c_k = eigh(f[np.ix_(idx, idx)], s_k)
                nk = occ.get(k, (0, 0))[spin]
                order = list(range(len(idx)))
                if ref is not None and ref.get(k) is not None and ref[k][spin].shape[1]:
                    # the followed orbitals by overlap, one each; the rest of the block by energy
                    target = ref[k][spin]
                    pick = []
                    for t in range(min(target.shape[1], nk)):
                        ov = (target[:, t] @ s_k @ c_k) ** 2
                        ov[pick] = -1.0
                        pick.append(int(np.argmax(ov)))
                    pick += [j for j in order if j not in pick][:nk - len(pick)]
                    order = sorted(pick) + [j for j in order if j not in pick]
                e_k, c_k = e_k[order], c_k[:, order]
                w = len(idx)
                coeff[idx, col:col + w], energy[col:col + w], ms[col:col + w] = c_k, e_k, m_of[idx[0]]
                occs[col:col + nk] = 1.0
                followed = 0 if ref is None or ref.get(k) is None else min(ref[k][spin].shape[1], nk)
                if followed:   # keep following the same orbitals (in pick order among the occupied)
                    new_ref.setdefault(k, [None, None])[spin] = c_k[:, [order.index(j) for j in pick[:followed]]]
                else:
                    new_ref.setdefault(k, [None, None])[spin] = c_k[:, :0]
                col += w
            out.append((coeff, energy, ms, occs))
        return out, {k: tuple(v) for k, v in new_ref.items()}

    for cycle in range(1, max_cycle + 1):
        ja, ka = coulomb_exchange(eri, da)
        jb, kb = coulomb_exchange(eri, db)
        fa, fb = (h + ja + jb - ka) * same, (h + ja + jb - kb) * same
        e_tot = float(np.sum(h * (da + db)) + 0.5 * (np.sum((ja + jb) * (da + db)) - np.sum(ka * da)
                                                     - np.sum(kb * db)) + ints.energy_nuc)
        err = np.concatenate([(fa @ da @ s - s @ da @ fa).ravel(), (fb @ db @ s - s @ db @ fb).ravel()])
        grad = float(np.abs(err).max())
        if verbose:
            print(f"cycle {cycle:3d}  E = {e_tot:.12f}  grad = {grad:.2e}", flush=True)
        if e_old is not None and abs(e_tot - e_old) < conv_tol and grad < conv_tol_grad:
            converged = True
            break
        e_old = e_tot
        if cycle > 1:
            fa, fb = diis.update(np.array([fa, fb]), err)
        if level_shift:
            fa = fa + level_shift * (s - s @ da @ s)
            fb = fb + level_shift * (s - s @ db @ s)
        res, new_ref = diagonalize(fa, fb, ref)
        if follow == "update" and ref is not None:
            ref = new_ref
        (ca, _, _, oa), (cb, _, _, ob) = res
        da, db = ca[:, oa > 0] @ ca[:, oa > 0].T, cb[:, ob > 0] @ cb[:, ob > 0].T
    res, _ = diagonalize((h + ja + jb - ka) * same, (h + ja + jb - kb) * same, ref)
    coeffs, energies, mss, noccs = [], [], [], []
    for coeff, energy, ms, occs in res:
        order = np.lexsort((energy, -occs))
        coeffs.append(coeff[:, order]); energies.append(energy[order]); mss.append(ms[order])
        noccs.append(int(occs.sum()))
    ca, cb = coeffs[0][:, :noccs[0]], coeffs[1][:, :noccs[1]]
    ov = ca.T @ s @ cb
    sz = 0.5 * (noccs[0] - noccs[1])
    s2 = float(sz * (sz + 1) + noccs[1] - np.sum(ov ** 2))
    return AxialUHFResult(e_tot=e_tot, converged=converged, cycles=cycle, mo_coeff=tuple(coeffs),
                          mo_energy=tuple(energies), nocc=tuple(noccs), mo_m=tuple(mss), s2=s2)


def atomic_blocks(mol, ints: AxialIntegrals) -> np.ndarray:
    """(l, m) of every axial AO of a single atom at the origin."""
    if mol.natm != 1:
        raise ValueError("(l, m) blocks are a symmetry of a single atom only")
    l = np.concatenate([[mol.bas_angular(ib)] * (2 * mol.bas_angular(ib) + 1) * mol.bas_nctr(ib)
                        for ib in range(mol.nbas)])
    return np.array(list(zip(l, ints.m)))


def atomic_occupation(closed: dict[int, int], open_alpha: dict[int, list[int]] | None = None,
                      open_beta: dict[int, list[int]] | None = None) -> dict:
    """{(l, m): (n_alpha, n_beta)} from closed shells per l ({l: number of closed shells})
    plus open electrons per l (lists of m)."""
    occ = {}
    for l, nshell in closed.items():
        for m in range(-l, l + 1):
            occ[(l, m)] = (nshell, nshell)
    for spin, extra in enumerate((open_alpha or {}, open_beta or {})):
        for l, ms in extra.items():
            for m in ms:
                a, b = occ.get((l, m), (0, 0))
                occ[(l, m)] = (a + 1, b) if spin == 0 else (a, b + 1)
    return occ


def embed_atomic_orbitals(mol, ints: AxialIntegrals, atom_index: int, atom_mol, atom_result: AxialUHFResult,
                          select) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    """Molecular start from one atom's converged (l, m)-blocked UHF (``atom_mol``, same basis).

    The atom's axial AOs are the molecule's axial AOs of that atom (the transform is per shell),
    so its orbitals embed by zero padding.  Returns the atom's density (alpha, beta) in the
    molecular axial basis and the MOM reference {m: (C_alpha, C_beta)} of the orbitals chosen
    by ``select(spin, j, m, l_weights) -> bool`` (e.g. the open 4f and 6s)."""
    offs = mol.aoslice_by_atom()[atom_index]
    lo, hi = int(offs[2]), int(offs[3])
    if hi - lo != atom_mol.nao_nr():
        raise ValueError("atom basis differs from the molecule's basis on that atom")
    n = ints.hcore.shape[0]
    lval = np.concatenate([[atom_mol.bas_angular(ib)] * (2 * atom_mol.bas_angular(ib) + 1)
                           * atom_mol.bas_nctr(ib) for ib in range(atom_mol.nbas)])
    s_at = atom_mol.intor("int1e_ovlp")
    dms, ref = [], {}
    for spin in (0, 1):
        c_at = atom_result.mo_coeff[spin][:, :atom_result.nocc[spin]]
        c = np.zeros((n, c_at.shape[1]))
        c[lo:hi] = c_at
        dms.append(c @ c.T)
    for spin in (0, 1):
        for j in range(atom_result.nocc[spin]):
            c_j = atom_result.mo_coeff[spin][:, j]
            m = int(atom_result.mo_m[spin][j])
            # Mulliken l weights in the atom's (real-valued axial) basis: overlap is the same
            weights = {l: float(np.sum((c_j * (ints_ovlp_block(ints, lo, hi) @ c_j))[lval == l])) for l in set(lval)}
            if select(spin, j, m, weights):
                idx = np.flatnonzero(ints.m == m)
                full = np.zeros(n)
                full[lo:hi] = c_j
                entry = ref.setdefault(m, [np.zeros((len(idx), 0)), np.zeros((len(idx), 0))])
                entry[spin] = np.hstack([entry[spin], full[idx][:, None]])
    return (dms[0], dms[1]), {m: tuple(v) for m, v in ref.items()}


def ints_ovlp_block(ints: AxialIntegrals, lo: int, hi: int) -> np.ndarray:
    return ints.ovlp[lo:hi, lo:hi]
