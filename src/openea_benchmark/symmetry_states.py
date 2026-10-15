"""Symmetry-block SCF for diatomics: occupied orbitals <-> exact state labels.

Computations use the abelian subgroups C2v (heteronuclear) / D2h (homonuclear),
where the two components of every pi/delta shell are independent irreps.  (The
real E1x/E1y irreps of PySCF's Coov/Dooh couple both components, so a Pi state
with pi_x^2 pi_y^1 has no stationary point there - wave-0 OH X2Pi did not converge.)

The orbital angular momentum |m| of each molecular orbital is measured, not
assumed: <phi|Lz^2|phi> from the AO integrals int1e_cg_irxp (exactly 0, 1, 4, ...
for converged sigma/pi/delta orbitals).  For a determinant:

* 2S = n_alpha - n_beta (high-spin determinant, ROHF/UKS),
* each (|m|>0, g/u) family has two abelian irreps (its x- and y-like
  component); per spin, |n_comp1 - n_comp2| components are unpaired.  The
  determinant is an Lz^2 eigenfunction only with at most ONE unpaired
  component in total (Lambda = its |m|; none -> Lambda = 0).  Otherwise it mixes
  Lambda values (e.g. closed-shell (pi_x)^2 of O2 1Delta_g) and needs a
  multi-determinant / MR treatment,
* g/u = (-1)^(number of u electrons) for like nuclei,
* for Lambda = 0 all families are paired, and Sigma +/- = (-1)^(electrons in the
  sigma_v(xz)-odd component of every family), which is convention-independent.
"""
from __future__ import annotations

from .heavy_elements import apply_hamiltonian, mole_basis_kwargs

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from .state_candidates import MolecularSymmetry


@dataclass(frozen=True)
class OrbitalInfo:
    index: int
    energy: float
    irrep: str
    m: int            # measured |m|
    gu: str | None    # from the D2h irrep name, None for C2v


def point_group_for(atom_a: str, atom_b: str) -> str:
    return "D2h" if atom_a == atom_b else "C2v"


def build_symmetric_molecule(atoms: tuple[str, str], r_angstrom: float, *, basis, charge: int, two_s: int,
                             verbose: int = 0):
    from pyscf import gto
    return gto.M(atom=[(atoms[0], (0.0, 0.0, 0.0)), (atoms[1], (0.0, 0.0, float(r_angstrom)))],
                 **mole_basis_kwargs(atoms, basis), charge=charge, spin=two_s, unit="Angstrom",
                 symmetry=point_group_for(*atoms), verbose=verbose, dump_input=False)


def orbital_m_values(mol, mo_coeff) -> np.ndarray:
    """|m| of each orbital from <Lz^2> (the molecule lies on the z axis)."""
    s = mol.intor("int1e_ovlp")
    with mol.with_common_origin((0.0, 0.0, 0.0)):
        a = mol.intor("int1e_cg_irxp", comp=3)[2]   # real antisymmetric, Lz = -i a
    lz2 = -a @ np.linalg.solve(s, a)
    m2 = np.einsum("pi,pq,qi->i", mo_coeff, lz2, mo_coeff)
    return np.rint(np.sqrt(np.clip(m2, 0.0, None))).astype(int)


def orbital_table(mol, mo_coeff, mo_energy) -> list[OrbitalInfo]:
    from pyscf.scf import hf_symm
    names = dict(zip(mol.irrep_id, mol.irrep_name))
    irreps = [names[i] for i in hf_symm.get_orbsym(mol, mo_coeff)]
    ms = orbital_m_values(mol, mo_coeff)
    out = []
    for i, (e, irrep, m) in enumerate(zip(mo_energy, irreps, ms)):
        gu = irrep[-1] if mol.groupname == "D2h" else None
        out.append(OrbitalInfo(i, float(e), irrep, int(m), gu))
    return out


def label_from_orbitals(occ_alpha: Sequence[OrbitalInfo], occ_beta: Sequence[OrbitalInfo], *,
                        like_nuclei: bool) -> MolecularSymmetry | None:
    two_s = len(occ_alpha) - len(occ_beta)
    if two_s < 0:
        return None
    unpaired, n_u, n_comp2 = [], 0, 0
    for occ in (occ_alpha, occ_beta):
        families: dict[tuple[int, str | None], dict[str, int]] = {}
        for o in occ:
            if o.gu == "u":
                n_u += 1
            if o.m > 0:
                families.setdefault((o.m, o.gu), {}).setdefault(o.irrep, 0)
                families[(o.m, o.gu)][o.irrep] += 1
        for (m, _gu), comps in families.items():
            counts = sorted(comps.items())
            if len(counts) > 2:
                return None
            n1 = counts[0][1]
            n2 = counts[1][1] if len(counts) == 2 else 0
            unpaired += [m] * abs(n1 - n2)
            n_comp2 += min(n1, n2) if len(counts) == 2 else 0
    if len(unpaired) > 1:
        return None
    lam = unpaired[0] if unpaired else 0
    return MolecularSymmetry(
        two_s=two_s, lam=lam,
        gu=("u" if n_u % 2 else "g") if like_nuclei else None,
        reflection=("-" if n_comp2 % 2 else "+") if lam == 0 else None,
    )


@dataclass(frozen=True)
class BlockOccupation:
    symmetry: MolecularSymmetry
    irrep_nelec: tuple[tuple[str, tuple[int, int]], ...]
    orbital_energy_sum: float

    def as_dict(self) -> dict[str, tuple[int, int]]:
        return dict(self.irrep_nelec)


def enumerate_block_occupations(orbitals: Sequence[OrbitalInfo], *, n_alpha: int, n_beta: int,
                                like_nuclei: bool, below: int = 3, above: int = 5,
                                keep_per_label: int = 2) -> list[BlockOccupation]:
    """High-spin determinants (doubly-occupied set inside singly-occupied set) in a frontier window."""
    order = sorted(orbitals, key=lambda o: o.energy)
    lo = max(0, n_beta - below)
    hi = min(len(order), n_alpha + above)
    core, window = order[:lo], order[lo:hi]
    n_double_win, n_single = n_beta - lo, n_alpha - n_beta
    if n_double_win < 0 or n_double_win + n_single > len(window):
        return []
    found: dict[MolecularSymmetry, dict[tuple, float]] = {}
    for doubly in combinations(window, n_double_win):
        rest = [o for o in window if o not in doubly]
        for singly in combinations(rest, n_single):
            beta = list(core) + list(doubly)
            alpha = beta + list(singly)
            label = label_from_orbitals(alpha, beta, like_nuclei=like_nuclei)
            if label is None:
                continue
            counts: dict[str, list[int]] = {}
            for o in alpha:
                counts.setdefault(o.irrep, [0, 0])[0] += 1
            for o in beta:
                counts.setdefault(o.irrep, [0, 0])[1] += 1
            key = tuple(sorted((k, (v[0], v[1])) for k, v in counts.items()))
            energy = sum(o.energy for o in alpha) + sum(o.energy for o in beta)
            bucket = found.setdefault(label, {})
            if key not in bucket or energy < bucket[key]:
                bucket[key] = energy
    out = []
    for label, bucket in found.items():
        kept: list[tuple] = []
        for key, energy in sorted(bucket.items(), key=lambda kv: kv[1]):
            # x <-> y relabelled occupations are the degenerate partner component
            # of the same Lambda state, not a second state
            if any(swap_xy(key) == other for other in kept):
                continue
            kept.append(key)
            out.append(BlockOccupation(label, key, energy))
            if len(kept) >= keep_per_label:
                break
    return sorted(out, key=lambda o: (o.symmetry.two_s, o.orbital_energy_sum))


_XY_PARTNER = {"B1": "B2", "B2": "B1", "B2g": "B3g", "B3g": "B2g", "B2u": "B3u", "B3u": "B2u"}


def swap_xy(irrep_nelec_key: tuple) -> tuple:
    """Occupation with x- and y-type components exchanged (C2 rotation about z by 90 degrees)."""
    return tuple(sorted((_XY_PARTNER.get(k, k), v) for k, v in irrep_nelec_key))


@dataclass
class BlockSCFResult:
    symmetry: MolecularSymmetry | None
    target: MolecularSymmetry
    irrep_nelec: dict[str, tuple[int, int]]
    converged: bool
    energy_hartree: float | None
    stable_within_symmetry: bool | None
    following_steps: int = 0
    symmetry_breaking_instability: bool | None = None
    checkpoint: str | None = None
    warnings: list[str] = field(default_factory=list)


def _occupied_tables(mf) -> tuple[list[OrbitalInfo], list[OrbitalInfo]]:
    mol = mf.mol
    if getattr(mf.mo_occ, "ndim", 1) == 2:  # UHF/UKS
        ta = orbital_table(mol, mf.mo_coeff[0], mf.mo_energy[0])
        tb = orbital_table(mol, mf.mo_coeff[1], mf.mo_energy[1])
        return ([o for o, n in zip(ta, mf.mo_occ[0]) if n > 0.5], [o for o, n in zip(tb, mf.mo_occ[1]) if n > 0.5])
    t = orbital_table(mol, mf.mo_coeff, mf.mo_energy)
    return ([o for o, n in zip(t, mf.mo_occ) if n > 0.5], [o for o, n in zip(t, mf.mo_occ) if n > 1.5])


def label_of_mean_field(mf, *, like_nuclei: bool) -> MolecularSymmetry | None:
    alpha, beta = _occupied_tables(mf)
    return label_from_orbitals(alpha, beta, like_nuclei=like_nuclei)


def run_block_scf(mol, occupation: BlockOccupation, *, method: str = "HF", xc: str | None = None,
                  dm0=None, checkpoint: str | None = None, max_cycle: int = 200, conv_tol: float = 1e-10,
                  following_steps: int = 4, probe_symmetry_breaking: bool = True,
                  grid_level: int | None = None) -> BlockSCFResult:
    """Converge one symmetry block with the abelian irrep occupation imposed."""
    from pyscf import dft, scf

    target = occupation.symmetry
    if method == "HF":
        mf = scf.RHF(mol) if target.two_s == 0 else scf.ROHF(mol)
    else:
        mf = dft.RKS(mol) if target.two_s == 0 else dft.UKS(mol)
        mf.xc = xc
        if grid_level is not None:
            mf.grids.level = grid_level
    mf = apply_hamiltonian(mf)
    occ = occupation.as_dict()
    mf.irrep_nelec = {k: (v if target.two_s else v[0] + v[1]) for k, v in occ.items()}
    mf.max_cycle, mf.conv_tol = max_cycle, conv_tol
    if checkpoint:
        mf.chkfile = checkpoint
    mf.kernel(dm0=dm0)
    if not mf.converged:  # same problem, second-order solver from the DIIS end point
        try:
            mf = mf.newton()
            mf.kernel(mf.mo_coeff, mf.mo_occ)
        except Exception:
            pass
    steps, stable = 0, None
    if mf.converged:
        mo, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)
        while not stable and steps < following_steps:  # symmetry-preserving instability: follow
            mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ))
            steps += 1
            mo, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)
    like = mol.groupname == "D2h"
    got = label_of_mean_field(mf, like_nuclei=like) if mf.converged else None
    result = BlockSCFResult(symmetry=got, target=target, irrep_nelec=occ, converged=bool(mf.converged),
                            energy_hartree=float(mf.e_tot) if mf.converged else None,
                            stable_within_symmetry=None if stable is None else bool(stable),
                            following_steps=steps, checkpoint=checkpoint)
    if got is not None and got != target:
        result.warnings.append("CONVERGED_LABEL_DIFFERS_FROM_TARGET")
    if mf.converged and probe_symmetry_breaking:
        result.symmetry_breaking_instability = _symmetry_breaking_probe(mf)
        if result.symmetry_breaking_instability:
            result.warnings.append("SYMMETRY_BREAKING_INSTABILITY_RECORDED_NOT_FOLLOWED")
    return result


def _symmetry_breaking_probe(mf) -> bool | None:
    """Is the symmetric solution unstable once symmetry constraints are lifted? (evidence only)"""
    from pyscf.scf import stability as stab
    try:
        if getattr(mf.mo_occ, "ndim", 1) == 2:
            _, stable = stab.uhf_internal(mf, with_symmetry=False, return_status=True)
        elif mf.mol.spin:
            _, stable = stab.rohf_internal(mf, with_symmetry=False, return_status=True)
        else:
            _, stable = stab.rhf_internal(mf, with_symmetry=False, return_status=True)
        return not bool(stable)
    except Exception:
        return None
