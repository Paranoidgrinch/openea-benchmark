"""Symmetry-block DFT scout: from two element symbols to labelled local PECs.

For a charge q (0 or -1):

1. Wigner-Witmer candidates (``state_candidates``) define the target labels.
2. For each candidate spin 2S a symmetric reference SCF at the start geometry
   (sum of covalent radii unless given) provides orbital irreps/energies;
   ``enumerate_block_occupations`` lists high-spin irrep occupations and keeps
   those whose exact label is a candidate.  Candidate labels without a
   single-determinant occupation are reported as MR-only (e.g. 1Delta, open-shell
   singlets) instead of silently disappearing.
3. Every (label, occupation) block is scanned along R with the occupation
   imposed and each geometry seeded from its neighbour's checkpoint, until a
   strict sampled minimum is bracketed or the curve flattens at large R
   (dissociative) or a budget/SCF failure stops it.

Within a block, state identity is fixed by symmetry; the converged label is
checked at every point.  Symmetry-breaking instabilities are recorded, never
followed.  The result can be exported in the legacy scout-summary schema so
state selection, Stage-3 planning and the EA stages run unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .state_candidates import candidate_spin_sectors, enumerate_state_candidates
from .symmetry_states import (
    BlockOccupation, build_symmetric_molecule, enumerate_block_occupations, orbital_table, run_block_scf,
)

HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class SymmetryScoutPolicy:
    functional: str = "PBE0"
    basis: str = "def2-TZVPPD"
    grid_level: int = 5
    step_angstrom: float = 0.05
    fine_steps: int = 4             # constant steps near the start geometry
    step_growth: float = 1.5        # then geometric growth ...
    max_step_angstrom: float = 0.5  # ... up to this step
    max_points_per_side: int = 16
    min_r_angstrom: float = 0.5
    max_r_angstrom: float = 4.0
    plateau_energy_tolerance_ev: float = 0.01
    occupation_window_below: int = 3
    occupation_window_above: int = 5
    occupations_per_label: int = 2
    following_steps: int = 4


def covalent_start_r(atom_a: str, atom_b: str) -> float:
    from pyscf.data import elements, radii
    return round((radii.COVALENT[elements.charge(atom_a)] + radii.COVALENT[elements.charge(atom_b)])
                 * radii.BOHR, 2)


def _block_id(charge: int, occ: BlockOccupation) -> str:
    digest = sha256(json.dumps(occ.irrep_nelec).encode()).hexdigest()[:12]
    return f"q{charge:+d}__{occ.symmetry.label}__{digest}"


def _reference_orbitals(atoms, r, charge, two_s, policy):
    """Symmetric (abelian subgroup) reference SCF; returns alpha orbital table and (n_alpha, n_beta)."""
    from pyscf import dft
    mol = build_symmetric_molecule(atoms, r, basis=policy.basis, charge=charge, two_s=two_s)
    mf = dft.RKS(mol) if two_s == 0 else dft.UKS(mol)
    mf.xc, mf.verbose, mf.max_cycle = policy.functional, 0, 200
    mf.grids.level = policy.grid_level
    mf.kernel()
    mo = mf.mo_coeff if two_s == 0 else mf.mo_coeff[0]
    energies = mf.mo_energy if two_s == 0 else mf.mo_energy[0]
    return orbital_table(mol, mo, energies), mol.nelec


def plan_blocks(atom_a: str, atom_b: str, charge: int, r0: float, policy: SymmetryScoutPolicy,
                reference: Callable[..., tuple] | None = None) -> dict:
    """Occupations per candidate label; labels without a single-determinant occupation listed."""
    reference = reference or _reference_orbitals
    candidates = enumerate_state_candidates(atom_a, atom_b, charge)
    wanted = {c.symmetry for c in candidates}
    like = atom_a == atom_b
    blocks, covered = [], set()
    for two_s in candidate_spin_sectors(atom_a, atom_b, charge):
        orbitals, (n_a, n_b) = reference((atom_a, atom_b), r0, charge, two_s, policy)
        for occ in enumerate_block_occupations(
                orbitals, n_alpha=n_a, n_beta=n_b, like_nuclei=like,
                below=policy.occupation_window_below, above=policy.occupation_window_above,
                keep_per_label=policy.occupations_per_label):
            if occ.symmetry in wanted:
                blocks.append(occ)
                covered.add(occ.symmetry)
    missing = sorted((c for c in candidates if c.symmetry not in covered), key=lambda c: c.symmetry.label)
    return {
        "blocks": blocks,
        "mr_only_or_outside_window": [{"label": c.symmetry.label, "count": c.count,
                                       "channels": c.channels[:3]} for c in missing],
        "candidate_labels": sorted(s.label for s in wanted),
    }


def scan_block(atoms, charge: int, occ: BlockOccupation, r0: float, policy: SymmetryScoutPolicy,
               checkpoint_dir: Path, scf_runner: Callable[..., Any] | None = None) -> dict:
    """Seeded R scan inside one symmetry block until a bracketed minimum or a large-R plateau."""
    def default_runner(r, seed):
        from pyscf import lib
        mol = build_symmetric_molecule(atoms, r, basis=policy.basis, charge=charge, two_s=occ.symmetry.two_s)
        dm0 = None
        if seed:
            from pyscf import scf as _scf
            dm0 = _scf.hf.init_guess_by_chkfile(mol, seed, project=None)
        chk = str(checkpoint_dir / f"{_block_id(charge, occ)}__R{r:.4f}.chk")
        res = run_block_scf(mol, occ, method="DFT", xc=policy.functional, dm0=dm0, checkpoint=chk,
                            following_steps=policy.following_steps, grid_level=policy.grid_level)
        return res
    runner = scf_runner or default_runner
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    points: dict[float, dict] = {}

    def compute(r, seed):
        res = runner(round(r, 4), seed)
        points[round(r, 4)] = {
            "r_angstrom": round(r, 4), "energy_hartree": res.energy_hartree, "converged": res.converged,
            "label": None if res.symmetry is None else res.symmetry.label,
            "stable_within_symmetry": res.stable_within_symmetry,
            "symmetry_breaking_instability": res.symmetry_breaking_instability,
            "checkpoint": res.checkpoint, "warnings": list(res.warnings)}
        return points[round(r, 4)]

    center = compute(r0, None)
    if not center["converged"]:
        return {"status": "SCF_FAILED_AT_START", "points": sorted(points.values(), key=lambda p: p["r_angstrom"])}
    status = {}
    for direction in (-1, +1):
        prev, step, r = center, policy.step_angstrom, r0
        for k in range(1, policy.max_points_per_side + 1):
            if k > policy.fine_steps:
                step = min(step * policy.step_growth, policy.max_step_angstrom)
            r = round(r + direction * step, 4)
            if r < policy.min_r_angstrom or r > policy.max_r_angstrom:
                status[direction] = "R_LIMIT"
                break
            p = compute(r, prev["checkpoint"])
            if not p["converged"]:
                status[direction] = "SCF_FAILED"
                break
            if p["energy_hartree"] > prev["energy_hartree"]:
                status[direction] = "RISES"
                break
            if direction > 0 and abs(p["energy_hartree"] - prev["energy_hartree"]) * HARTREE_TO_EV \
                    < policy.plateau_energy_tolerance_ev and r > r0 + 1.0:
                status[direction] = "PLATEAU"
                break
            prev = p
        else:
            status[direction] = "BUDGET"
    ordered = sorted(points.values(), key=lambda p: p["r_angstrom"])
    ok = [p for p in ordered if p["converged"]]
    energies = [p["energy_hartree"] for p in ok]
    i_min = min(range(len(ok)), key=lambda i: energies[i])
    label_ok = all(p["label"] == occ.symmetry.label for p in ok)
    if not label_ok:
        result_status = "LABEL_CHANGED"
    elif 0 < i_min < len(ok) - 1 and energies[i_min - 1] > energies[i_min] < energies[i_min + 1]:
        result_status = "BRACKETED_SINGLE_MINIMUM"
    elif status.get(+1) == "PLATEAU" and i_min == len(ok) - 1:
        result_status = "DISSOCIATIVE_PLATEAU"
    else:
        result_status = "UNRESOLVED"
    return {"status": result_status, "side_status": {"lower_r": status.get(-1), "higher_r": status.get(+1)},
            "minimum": {"r_angstrom": ok[i_min]["r_angstrom"], "energy_hartree": energies[i_min],
                        "checkpoint": ok[i_min]["checkpoint"]},
            "symmetry_breaking_instability_points": sum(1 for p in ok if p["symmetry_breaking_instability"]),
            "points": ordered}


def _save(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2, default=str))
    os.replace(tmp, path)


def run_symmetry_scout(atom_a: str, atom_b: str, *, output_dir: Path, charges=(0, -1),
                       r0: float | None = None, policy: SymmetryScoutPolicy = SymmetryScoutPolicy(),
                       reference: Callable[..., tuple] | None = None,
                       scf_runner_factory: Callable[..., Any] | None = None) -> dict:
    """Resumable per block: completed blocks are reused from symmetry_scout.json."""
    output_dir = Path(output_dir)
    r0 = r0 if r0 is not None else covalent_start_r(atom_a, atom_b)
    path = output_dir / "symmetry_scout.json"
    signature = sha256(json.dumps({"policy": asdict(policy), "atoms": [atom_a, atom_b], "r0": r0},
                                  sort_keys=True).encode()).hexdigest()
    record = json.loads(path.read_text()) if path.exists() else {}
    if record.get("signature") != signature:
        record = {"signature": signature, "charges": {}}
    record.update(system=f"{atom_a}{atom_b}", atoms=[atom_a, atom_b], r0_angstrom=r0, policy=asdict(policy),
                  validation_metadata_used_in_computation=False)
    for charge in charges:
        entry = record["charges"].setdefault(str(charge), {"blocks": {}})
        if "plan" not in entry:
            plan = plan_blocks(atom_a, atom_b, charge, r0, policy, reference)
            entry["plan"] = {"candidate_labels": plan["candidate_labels"],
                             "mr_only_or_outside_window": plan["mr_only_or_outside_window"],
                             "blocks": [{"label": o.symmetry.label, "two_s": o.symmetry.two_s,
                                         "irrep_nelec": [[k, list(v)] for k, v in o.irrep_nelec],
                                         "orbital_energy_sum": o.orbital_energy_sum} for o in plan["blocks"]]}
            _save(path, record)
        for b in entry["plan"]["blocks"]:
            occ = BlockOccupation(_symmetry_from_label(b["label"]),
                                  tuple((k, tuple(v)) for k, v in b["irrep_nelec"]), b["orbital_energy_sum"])
            bid = _block_id(charge, occ)
            if bid in entry["blocks"]:
                continue
            print(f"[SYMMETRY SCOUT] {atom_a}{atom_b} q={charge:+d} {b['label']} {bid}", flush=True)
            runner = scf_runner_factory(charge, occ) if scf_runner_factory else None
            entry["blocks"][bid] = {"label": b["label"], "two_s": b["two_s"], "irrep_nelec": b["irrep_nelec"],
                                    **scan_block((atom_a, atom_b), charge, occ, r0, policy,
                                                 output_dir / "symmetry_checkpoints", runner)}
            _save(path, record)
    return record


def _symmetry_from_label(label: str):
    import re
    from .state_candidates import MolecularSymmetry, _LAMBDA_LETTER
    m = re.match(r"(\d+)([A-Za-z]+?)([gu]?)([+-]?)$", label)
    mult, name, gu, refl = m.groups()
    inverse = {v: k for k, v in _LAMBDA_LETTER.items()}
    return MolecularSymmetry(int(mult) - 1, inverse[name], gu or None, refl or None)
