"""Symmetry-block SCF for diatomics: irrep occupations <-> exact state labels.

PySCF builds linear molecules in Dooh / Coov with real irreps (A1g, E1gx, E1gy,
...; x-type even, y-type odd under sigma_v(xz)).  For a single determinant
with aufbau filling inside every irrep, the exact diatomic labels follow from
the electron counts per irrep alone:

* 2S  = n_alpha - n_beta (ROHF / high-spin determinant),
* Lambda^2 = sum over spins and over (|m|, g/u) families of m^2 |n_x - n_y|;
  a determinant is an Lz eigenstate only if this is a perfect square
  (otherwise it mixes Lambda values and needs a multi-determinant/MR
  treatment, e.g. the closed-shell (pi_x)^2 component of O2 1Delta_g),
* g/u = (-1)^(number of u electrons),
* Sigma +/- = (-1)^(number of electrons in sigma_v-odd irreps: E..y, A2..).

`enumerate_block_occupations` lists irrep occupations reachable from a
reference orbital ordering, labels them, and keeps the lowest few per label.
`run_block_scf` converges an SCF with that occupation imposed
(`irrep_nelec`) and checks stability *within* the symmetry; a
symmetry-breaking instability is not followed but reported as reference
evidence.  No state is chosen here: the energies decide downstream.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
import re
from typing import Any, Mapping, Sequence

from .state_candidates import MolecularSymmetry

_IRREP = re.compile(r"([AE])(\d+)([gu]?)([xy]?)$")


@dataclass(frozen=True)
class IrrepInfo:
    name: str
    m: int                 # |m| of the orbital (0 for A irreps)
    gu: str | None
    sigma_v_odd: bool      # y-type E and A2 irreps
    component: str | None  # 'x' / 'y' for E irreps


def parse_irrep(name: str) -> IrrepInfo:
    match = _IRREP.match(name)
    if not match:
        raise ValueError(f"not a linear-molecule irrep: {name}")
    kind, num, gu, xy = match.groups()
    if kind == "A":
        return IrrepInfo(name, 0, gu or None, num == "2", None)
    return IrrepInfo(name, int(num), gu or None, xy == "y", xy)


def label_from_irrep_nelec(irrep_nelec: Mapping[str, tuple[int, int]], *,
                           like_nuclei: bool) -> MolecularSymmetry | None:
    """Exact (2S, Lambda, g/u, +/-) of the aufbau-within-irrep determinant, or None if not an Lz eigenstate."""
    n_alpha = sum(a for a, _ in irrep_nelec.values())
    n_beta = sum(b for _, b in irrep_nelec.values())
    two_s = n_alpha - n_beta
    if two_s < 0:
        return None
    lam2 = 0
    families: dict[tuple[int, str | None], dict[str, list[int]]] = {}
    n_u = n_odd = 0
    for name, (na, nb) in irrep_nelec.items():
        info = parse_irrep(name)
        if info.gu == "u":
            n_u += na + nb
        if info.sigma_v_odd:
            n_odd += na + nb
        if info.m > 0:
            fam = families.setdefault((info.m, info.gu), {"x": [0, 0], "y": [0, 0]})
            fam[info.component][0] += na
            fam[info.component][1] += nb
    for (m, _gu), fam in families.items():
        for spin in (0, 1):
            lam2 += m * m * abs(fam["x"][spin] - fam["y"][spin])
    lam = int(round(lam2 ** 0.5))
    if lam * lam != lam2:
        return None
    return MolecularSymmetry(
        two_s=two_s, lam=lam,
        gu=("u" if n_u % 2 else "g") if like_nuclei else None,
        reflection=("-" if n_odd % 2 else "+") if lam == 0 else None,
    )


@dataclass(frozen=True)
class BlockOccupation:
    symmetry: MolecularSymmetry
    irrep_nelec: tuple[tuple[str, tuple[int, int]], ...]
    orbital_energy_sum: float

    def as_dict(self) -> dict[str, tuple[int, int]]:
        return dict(self.irrep_nelec)


def enumerate_block_occupations(orbital_irreps: Sequence[str], orbital_energies: Sequence[float], *,
                                n_alpha: int, n_beta: int, like_nuclei: bool,
                                below: int = 3, above: int = 5, keep_per_label: int = 2) -> list[BlockOccupation]:
    """High-spin determinants (doubly-occupied set inside singly-occupied set) in a frontier window."""
    order = sorted(range(len(orbital_energies)), key=lambda i: orbital_energies[i])
    lo = max(0, n_beta - below)
    hi = min(len(order), n_alpha + above)
    core, window = order[:lo], order[lo:hi]
    n_double_win, n_single = n_beta - lo, n_alpha - n_beta
    if n_double_win < 0 or n_double_win + n_single > len(window):
        return []
    found: dict[MolecularSymmetry, dict[tuple, float]] = {}
    for doubly in combinations(window, n_double_win):
        rest = [i for i in window if i not in doubly]
        for singly in combinations(rest, n_single):
            counts: dict[str, list[int]] = {}
            for i in core + list(doubly):
                c = counts.setdefault(orbital_irreps[i], [0, 0]); c[0] += 1; c[1] += 1
            for i in singly:
                counts.setdefault(orbital_irreps[i], [0, 0])[0] += 1
            key = tuple(sorted((k, (v[0], v[1])) for k, v in counts.items()))
            label = label_from_irrep_nelec(dict(key), like_nuclei=like_nuclei)
            if label is None:
                continue
            energy = sum(2 * orbital_energies[i] for i in core + list(doubly)) + sum(orbital_energies[i] for i in singly)
            bucket = found.setdefault(label, {})
            if key not in bucket or energy < bucket[key]:
                bucket[key] = energy
    out = []
    for label, bucket in found.items():
        for key, energy in sorted(bucket.items(), key=lambda kv: kv[1])[:keep_per_label]:
            out.append(BlockOccupation(label, key, energy))
    return sorted(out, key=lambda o: (o.symmetry.two_s, o.orbital_energy_sum))


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


def build_symmetric_molecule(atoms: tuple[str, str], r_angstrom: float, *, basis, charge: int, two_s: int,
                             verbose: int = 0):
    from pyscf import gto
    return gto.M(atom=[(atoms[0], (0.0, 0.0, 0.0)), (atoms[1], (0.0, 0.0, float(r_angstrom)))],
                 basis=basis, charge=charge, spin=two_s, unit="Angstrom", symmetry=True,
                 verbose=verbose, dump_input=False)


def run_block_scf(mol, occupation: BlockOccupation, *, method: str = "HF", xc: str | None = None,
                  dm0=None, checkpoint: str | None = None, max_cycle: int = 200, conv_tol: float = 1e-10,
                  following_steps: int = 4, probe_symmetry_breaking: bool = True) -> BlockSCFResult:
    """Converge one symmetry block with the irrep occupation imposed."""
    from pyscf import dft, scf

    target = occupation.symmetry
    if method == "HF":
        mf = scf.RHF(mol) if target.two_s == 0 else scf.ROHF(mol)
    else:
        mf = dft.RKS(mol) if target.two_s == 0 else dft.UKS(mol)
        mf.xc = xc
    occ = occupation.as_dict()
    mf.irrep_nelec = {k: (v if target.two_s else v[0] + v[1]) for k, v in occ.items()}
    mf.max_cycle, mf.conv_tol = max_cycle, conv_tol
    if checkpoint:
        mf.chkfile = checkpoint
    mf.kernel(dm0=dm0)
    steps, stable = 0, None
    if mf.converged:
        mo, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)
        while not stable and steps < following_steps:  # symmetry-preserving instability: follow
            mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ))
            steps += 1
            mo, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)
    got = None
    if mf.converged:
        irr = mf.get_irrep_nelec()
        irr = {k: (v if isinstance(v, tuple) else (v // 2, v // 2)) for k, v in irr.items()}
        got = label_from_irrep_nelec(irr, like_nuclei=target.gu is not None)
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


def _symmetry_breaking_probe(mf) -> bool:
    """Is the symmetric solution unstable once symmetry constraints are lifted? (evidence only)"""
    from pyscf.scf import stability as stab
    try:
        if hasattr(mf, "mo_occ") and getattr(mf.mo_occ, "ndim", 1) == 2:
            _, stable = stab.uhf_internal(mf, with_symmetry=False, return_status=True)
        elif mf.mol.spin:
            _, stable = stab.rohf_internal(mf, with_symmetry=False, return_status=True)
        else:
            _, stable = stab.rhf_internal(mf, with_symmetry=False, return_status=True)
        return not bool(stable)
    except Exception:
        return None
