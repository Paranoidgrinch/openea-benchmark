"""Route B: open f shells with f-in-core pseudopotentials.

An open 4f/5f shell hardly takes part in the bond of a lanthanide or actinide diatomic but
makes the single-reference chain fail (hundreds of near-degenerate f arrangements).  Route B
puts the f electrons into the core of an energy-consistent Stuttgart/Cologne pseudopotential
(ECP(46+n)MWB for Ce–Yb, ECP(78+n)MWB for Ac–Lr; ``data/f_in_core_pp.json``) and runs the
ordinary chain on the remaining valence electrons, once per f count ("variant").

Variants with different f counts have different energy zeros.  They are compared through
measured atomic levels (``data/f_shell_levels.json``, NIST ASD + Blaise-Wyart; user decision
2026-11-07: measured, not calculated, onsets):

    E_true(M, n) ~ E_PP(M, n) - E_PP(atom, n, c) + E_meas(atom, n, c)

with c the valence configuration of the lowest measured level of f count n (``reference_level``).
The f-shell multiplet and spin-orbit stabilisation of f^n is the same in atom and molecule and
cancels.

Core polarization.  The pseudopotentials come with a core-polarization potential (CPP;
Mueller, Meyer, Stoll 1984)

    V_cpp = -1/2 alpha f^2,   f = sum_i r_i/r_i^3 w(r_i) - sum_mu Q_mu R_mu/R_mu^3 w(R_mu),
    w(r) = (1 - exp(-delta r^2))^ncut,

r measured from the f-in-core atom, mu the other nuclei.  Expanded, it is a one-electron term
-alpha/2 (w^2/r^4 - 2 F_nuc . g), a separable two-electron term (pq|rs) += -alpha sum_k
G_k[pq] G_k[rs] with G_k = <p| x_k w / r^3 |q>, and a constant -alpha/2 F_nuc^2.  PySCF has no
CPP; ``apply_cpp`` adds the three terms to an in-core SCF (hcore, 8-fold ERIs, energy_nuc), from
which PySCF CCSD(T) builds its integrals.  The chain uses CPP as an additive increment at a
moderate basis (like the core-valence correction), so its large-basis stages stay unmodified.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

DATA = Path(__file__).parent / "data"
LANTHANIDES = "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb".split()
ACTINIDES = "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split()
OPEN_F_ELEMENTS = frozenset(LANTHANIDES) | frozenset(ACTINIDES) - {"Ac", "Th"}
# Run-wide choice of the f-in-core variant, e.g. "U=ECP81MWB" (several elements: comma
# separated); inherited by point-pool workers like the model Hamiltonian.
VARIANT_ENV = "OPENEA_F_IN_CORE"
CPP_GRID_LEVEL = 6


@lru_cache(maxsize=None)
def pp_library() -> dict:
    return json.loads((DATA / "f_in_core_pp.json").read_text())["elements"]


@lru_cache(maxsize=None)
def level_library() -> dict:
    return json.loads((DATA / "f_shell_levels.json").read_text())["elements"]


def variants(element: str) -> dict[str, dict]:
    """{ECP name: record} of the f-in-core pseudopotentials of ``element``."""
    return dict(pp_library().get(element, {}))


def selected_variants() -> dict[str, str]:
    """{element: ECP name} chosen for this run (``VARIANT_ENV``)."""
    raw = os.environ.get(VARIANT_ENV, "").strip()
    out = {}
    for item in filter(None, (s.strip() for s in raw.split(","))):
        element, name = item.split("=")
        if name not in variants(element):
            raise ValueError(f"{VARIANT_ENV}: no f-in-core pseudopotential {name} for {element}")
        out[element] = name
    return out


def reference_level(element: str, f_count: int) -> dict | None:
    """Lowest measured atomic level with ``f_count`` f electrons (any charge and valence
    configuration), or None when none is measured."""
    levels = level_library().get(element, {}).get("levels_by_f_count", {}).get(str(f_count), [])
    return min(levels, key=lambda e: e["energy_cm1"]) if levels else None


def comparable_f_counts(element: str) -> list[int]:
    """f counts of ``element`` that have both a pseudopotential and a measured level."""
    return sorted({rec["f_in_core"] for rec in variants(element).values()
                   if reference_level(element, rec["f_in_core"]) is not None})


# ------------------------------------------------------------------ core polarization

def _cpp_centers(mol: Any, cpp_by_atom: dict[int, dict]) -> list[tuple[int, dict]]:
    return [(i, cpp_by_atom[i]) for i in sorted(cpp_by_atom)]


def cpp_matrices(mol: Any, cpp_by_atom: dict[int, dict], grid_level: int = CPP_GRID_LEVEL):
    """(h1, G, e_const): one-electron CPP matrix, field integrals G[c, k] (c: CPP center,
    k: x, y, z) and the nuclear constant, for CPP centers {atom index: {alpha, delta, ncut}}."""
    from pyscf import dft
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.prune = None
    grids.build(with_non0tab=False)
    ao = dft.numint.eval_ao(mol, grids.coords)
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    nao = mol.nao_nr()
    h1 = np.zeros((nao, nao))
    centers = _cpp_centers(mol, cpp_by_atom)
    G = np.zeros((len(centers), 3, nao, nao))
    e_const = 0.0
    for c, (i, p) in enumerate(centers):
        alpha, delta, ncut = p["alpha"], p["delta"], p["ncut"]
        d = grids.coords - coords[i]
        r2 = np.einsum("gk,gk->g", d, d)
        r2s = np.where(r2 > 1e-24, r2, 1.0)
        w = np.where(r2 > 1e-24, (1.0 - np.exp(-delta * r2)) ** ncut, 0.0)
        g = d * (w / r2s ** 1.5)[:, None]                       # field of an electron
        f_nuc = np.zeros(3)
        for j in range(mol.natm):
            if j == i:
                continue
            R = coords[j] - coords[i]
            R2 = float(R @ R)
            f_nuc -= charges[j] * R / R2 ** 1.5 * (1.0 - np.exp(-delta * R2)) ** ncut
        weighted = ao * grids.weights[:, None]
        for k in range(3):
            G[c, k] = weighted.T @ (ao * g[:, k:k + 1])
        g2 = w ** 2 / r2s ** 2
        h1 += -0.5 * alpha * (weighted.T @ (ao * g2[:, None]))
        h1 += alpha * np.einsum("k,kpq->pq", f_nuc, G[c])
        e_const += -0.5 * alpha * float(f_nuc @ f_nuc)
        G[c] *= np.sqrt(alpha)  # (pq|rs) += -sum_k G[pq] G[rs] then needs no alpha
    h1 = 0.5 * (h1 + h1.T)
    return h1, G, e_const


def apply_cpp(mf: Any, cpp_by_atom: dict[int, dict], grid_level: int = CPP_GRID_LEVEL) -> Any:
    """``mf`` (not yet run) with the core-polarization potential of the given atoms added to
    its one-electron Hamiltonian, its in-core 8-fold ERIs and its nuclear energy."""
    from pyscf import ao2mo
    mol = mf.mol
    h1, G, e_const = cpp_matrices(mol, cpp_by_atom, grid_level)
    nao = mol.nao_nr()
    eri = ao2mo.restore(4, mol.intor("int2e", aosym="s8"), nao)
    rows, cols = np.tril_indices(nao)
    vec = G[:, :, rows, cols].reshape(-1, len(rows))
    eri -= vec.T @ vec
    mol.incore_anyway = True
    mf._eri = ao2mo.restore(8, eri, nao)
    base_hcore, base_enuc = mf.get_hcore, mf.energy_nuc
    mf.get_hcore = lambda *args, **kw: base_hcore(*args, **kw) + h1
    mf.energy_nuc = lambda *args, **kw: base_enuc(*args, **kw) + e_const
    mf.cpp_record = {"centers": {int(i): dict(p) for i, p in cpp_by_atom.items()}, "nuclear_constant": e_const,
                     "grid_level": grid_level}
    return mf
