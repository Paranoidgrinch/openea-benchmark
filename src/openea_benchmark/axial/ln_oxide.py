"""Lanthanide monoxides LnO / LnO- as axial determinants: f-configuration screening.

Ionic picture Ln(2+ or +) O2-: closed [Xe] core of Ln plus O2- give per m block
{0: 14, +-1: 7, +-2: 2} doubly occupied orbitals; on top sit the 4f electrons (Hund high
spin, one axial determinant per choice of m values) and 0-2 electrons in the Ln 6s-like
sigma orbital (m = 0), spin parallel to the f shell.  Every candidate is solved as UHF in
m blocks, started from the l-pure (l, m)-blocked UHF of the free Ln ion with the same
f and s occupation (embedded) plus O minao, and the ion's open f and 6s orbitals are
followed by maximum overlap: without that the SCF slides 4f -> 5d in the same m block
(Pr: 3.6 eV too high).  ``f_purity`` reports whether the followed f orbitals kept
their f character.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

CLOSED_PER_M = {0: 14, 1: 7, -1: 7, 2: 2, -2: 2}   # Ln [Xe] + O2-
XE_CORE = 54
Z = {"La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
     "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71}


@dataclass(frozen=True)
class LnConfig:
    f_alpha: tuple[int, ...]
    f_beta: tuple[int, ...] = ()
    s_alpha: int = 0
    s_beta: int = 0

    @property
    def nf(self) -> int:
        return len(self.f_alpha) + len(self.f_beta)

    @property
    def ns(self) -> int:
        return self.s_alpha + self.s_beta

    @property
    def m_l(self) -> int:
        return sum(self.f_alpha) + sum(self.f_beta)

    @property
    def spin(self) -> int:   # 2S
        return len(self.f_alpha) - len(self.f_beta) + self.s_alpha - self.s_beta

    def label(self) -> str:
        f = f"4f{self.nf}[a{','.join(map(str, self.f_alpha))}" + (f";b{','.join(map(str, self.f_beta))}" if self.f_beta else "") + "]"
        return f + (f" 6s{self.ns}" if self.ns else "") + f" ML={self.m_l} 2S={self.spin}"


def f_configurations(nf: int, ns: int) -> list[LnConfig]:
    """Hund high-spin f^nf determinants (one per m choice, up to the mirror m -> -m, with
    M_L >= 0) and ns sigma electrons parallel to the f spin (alpha first)."""
    s_alpha, s_beta = (1, ns - 1) if ns else (0, 0)
    out, seen = [], set()
    ms = range(3, -4, -1)
    if nf <= 7:
        choices = [(c, ()) for c in itertools.combinations(ms, nf)]
    else:
        choices = [(tuple(ms), c) for c in itertools.combinations(ms, nf - 7)]
    for a, b in choices:
        mirror = (tuple(sorted((-m for m in a), reverse=True)), tuple(sorted((-m for m in b), reverse=True)))
        key = min((a, b), mirror)
        if key in seen:
            continue
        seen.add(key)
        cfg = LnConfig(a, b, s_alpha, s_beta)
        if cfg.m_l < 0:
            cfg = LnConfig(*mirror, s_alpha, s_beta)
        out.append(cfg)
    return out


def molecular_occupation(cfg: LnConfig) -> dict[int, tuple[int, int]]:
    occ = {k: [v, v] for k, v in CLOSED_PER_M.items()}
    for spin, ms in ((0, cfg.f_alpha), (1, cfg.f_beta)):
        for m in ms:
            occ.setdefault(m, [0, 0])[spin] += 1
    occ[0][0] += cfg.s_alpha
    occ[0][1] += cfg.s_beta
    return {k: tuple(v) for k, v in occ.items()}


def ion_charge(element: str, cfg: LnConfig) -> int:
    return Z[element] - XE_CORE - cfg.nf - cfg.ns


@dataclass
class OxidePoint:
    config: LnConfig
    charge: int
    r: float
    e_scf: float
    converged: bool
    s2: float
    f_purity: dict = field(default_factory=dict)   # m -> f weight of the followed occupied f orbitals
    correlation: dict = field(default_factory=dict)
    ion_e_scf: float | None = None
    scf_model: str = "free"


def _ln_basis(element, x):
    from openea_benchmark.heavy_elements import basis_name_for, resolve_named_basis
    return resolve_named_basis(basis_name_for(element, f"aug-cc-pV{x}Z"), element).shells


def ion_orbitals(element: str, cfg: LnConfig, ln_basis, cholesky_tol=1e-8):
    """(l, m)-blocked UHF of the free ion with the configuration's f and s electrons."""
    from pyscf import gto, scf
    from .basis import axial_integrals
    from .scf import atomic_blocks, atomic_occupation, axial_uhf, minao_density
    q = ion_charge(element, cfg)
    atom = gto.M(atom=f"{element} 0 0 0", basis={element: ln_basis}, charge=q, spin=cfg.spin, verbose=0)
    ints = axial_integrals(atom, hcore_ao=scf.ROHF(atom).sfx2c1e().get_hcore(), cholesky_tol=cholesky_tol)
    alpha, beta = {3: list(cfg.f_alpha)} if cfg.f_alpha else {}, {3: list(cfg.f_beta)} if cfg.f_beta else {}
    closed = {0: 5, 1: 4, 2: 2}
    if cfg.s_alpha and cfg.s_beta:
        closed[0] = 6
    elif cfg.s_alpha:
        alpha[0] = [0]
    occ = atomic_occupation(closed, alpha, beta)
    res = axial_uhf(ints, occ, minao_density(atom, ints), block_of_ao=atomic_blocks(atom, ints))
    return atom, res


def oxide_point(element: str, cfg: LnConfig, r: float, x: str = "D", methods=("ccsd(t)",),
                cholesky_tol: float = 1e-8, level_shift: float = 0.1, follow: str = "update",
                max_cycle: int = 300, diis_space: int = 12, diis_start: int = 2, freeze_f: bool = False,
                verbose: bool = False) -> OxidePoint:
    """SCF (+ CC) of one LnO^q determinant at bond length r (Angstrom).

    ``freeze_f``: the open 4f orbitals stay those of the free ion (``axial_uhf``
    freeze_reference), the 6s is not followed; for configurations whose free SCF slides
    4f -> 5d.  CC relaxes the f orbitals."""
    from pyscf import gto, scf
    from openea_benchmark.adaptive.stage3_execution import valence_frozen_core_orbitals
    from .basis import axial_integrals
    from .cc import axial_cc
    from .scf import axial_uhf, embed_atomic_orbitals, minao_density
    ln_basis = _ln_basis(element, x)
    atom, ion = ion_orbitals(element, cfg, ln_basis, cholesky_tol)
    charge = ion_charge(element, cfg) - 2
    mol = gto.M(atom=f"{element} 0 0 0; O 0 0 {r}", basis={element: ln_basis, "O": f"aug-cc-pv{x.lower()}z"},
                charge=charge, spin=cfg.spin, verbose=0)
    ints = axial_integrals(mol, hcore_ao=scf.UHF(mol).sfx2c1e().get_hcore(), cholesky_tol=cholesky_tol)
    s_index = {spin: (_outer_s(atom, ion, spin) if (cfg.s_alpha, cfg.s_beta)[spin] else None) for spin in (0, 1)}

    def follow(spin, j, m, w):
        # open f orbitals and (unless frozen f) the 6s (highest occupied s orbital of the ion)
        return w.get(3, 0.0) > 0.5 or (not freeze_f and j == s_index[spin])
    dm_ln, ref = embed_atomic_orbitals(mol, ints, 0, atom, ion, follow)
    on_ln = np.arange(mol.nao_nr()) < mol.aoslice_by_atom()[0][3]
    dm_o = minao_density(mol, ints)
    o = ~on_ln
    start = (dm_ln[0] + dm_o[0] * np.outer(o, o), dm_ln[1] + dm_o[1] * np.outer(o, o))
    res = axial_uhf(ints, molecular_occupation(cfg), start, reference=ref, level_shift=level_shift,
                    follow=follow, max_cycle=max_cycle, diis_space=diis_space, diis_start=diis_start,
                    freeze_reference=freeze_f, verbose=verbose)
    purity = f_purity(mol, ints, res, cfg)
    point = OxidePoint(cfg, charge, r, res.e_tot, res.converged, res.s2, purity, {}, ion.e_tot,
                       "frozen_f" if freeze_f else "free")
    if methods and res.converged:
        point.correlation = axial_cc(ints, res, mol.nelectron, valence_frozen_core_orbitals(mol), methods)
    return point


def _outer_s(atom, ion, spin: int) -> int:
    """Index of the highest occupied orbital of s character (> 0.5) of one spin of the ion."""
    lval = np.concatenate([[atom.bas_angular(ib)] * (2 * atom.bas_angular(ib) + 1) * atom.bas_nctr(ib)
                           for ib in range(atom.nbas)])
    s_ao = atom.intor("int1e_ovlp")
    n = ion.nocc[spin]
    c = ion.mo_coeff[spin][:, :n]
    w = np.einsum("pj,pj->j", c[lval == 0], (s_ao @ c)[lval == 0])   # axial and real AOs agree for l = 0
    cand = [j for j in range(n) if w[j] > 0.5]
    return max(cand, key=lambda j: ion.mo_energy[spin][j])


def f_purity(mol, ints, res, cfg: LnConfig) -> dict:
    """Ln-4f Mulliken population of the occupied orbitals of each spin and m block that
    should carry an f electron, minus the f population that block has without one (the
    same block of the closed-shell occupation is not available, so the background is the
    small f admixture of the closed orbitals, < 0.05 here); about 1 when the f electron
    stayed f, about 0 when it went to 5d."""
    lval = np.concatenate([[mol.bas_angular(ib)] * (2 * mol.bas_angular(ib) + 1) * mol.bas_nctr(ib)
                           for ib in range(mol.nbas)])
    on_ln = np.arange(mol.nao_nr()) < mol.aoslice_by_atom()[0][3]
    mask = on_ln & (lval == 3)
    out = {}
    for spin, ms in ((0, cfg.f_alpha), (1, cfg.f_beta)):
        c = res.mo_coeff[spin][:, :res.nocc[spin]]
        w = np.einsum("pj,pj->j", c[mask], (ints.ovlp @ c)[mask])
        m_occ = res.mo_m[spin][:res.nocc[spin]]
        for m in ms:
            out[f"{'ab'[spin]}{m}"] = round(float(w[m_occ == m].sum()), 3)
    return out
