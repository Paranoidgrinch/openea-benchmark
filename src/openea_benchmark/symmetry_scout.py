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
    max_bracket_refinements: int = 8
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
    # The grown steps can leave a coarse bracket around the discrete minimum;
    # bisect its wider side until both neighbours lie within one fine step, so
    # the high-level grid placed around it actually contains the minimum.
    for _ in range(policy.max_bracket_refinements):
        ok = [p for p in sorted(points.values(), key=lambda p: p["r_angstrom"]) if p["converged"]]
        i = min(range(len(ok)), key=lambda j: ok[j]["energy_hartree"])
        if not 0 < i < len(ok) - 1 or any(p["label"] != occ.symmetry.label for p in ok):
            break
        left, right = ok[i]["r_angstrom"] - ok[i - 1]["r_angstrom"], ok[i + 1]["r_angstrom"] - ok[i]["r_angstrom"]
        if max(left, right) <= policy.step_angstrom + 1e-9:
            break
        r = ok[i]["r_angstrom"] + (-left / 2 if left >= right else right / 2)
        if round(r, 4) in points or not compute(r, ok[i]["checkpoint"])["converged"]:
            break
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


# ------------------------------------------------------------ legacy export

_STATUS_MAP = {"BRACKETED_SINGLE_MINIMUM": "bracketed_single_minimum"}


def lowest_dissociation_limit(atomic: Mapping[str, Any], atom_a: str, atom_b: str, charge: int) -> dict | None:
    """Lowest fragment channel of this total charge, any spin coupling (computed DFT atoms)."""
    from .open_pec_extension import compatible_asymptote
    best = None
    for two_s in range(0, 13):
        asym = compatible_asymptote(atomic, atom_a, atom_b, charge, two_s)
        if asym and (best is None or asym["energy_hartree"] < best["energy_hartree"]):
            best = asym
    return best


def to_legacy_summary(record: Mapping[str, Any], *, atomic: Mapping[str, Any] | None = None,
                      asymptote_tolerance_ev: float = 0.25,
                      above_limit_margin_ev: float = 1.0) -> dict:
    """Symmetry-scout record -> scout-summary schema used by state selection and Stage-3 planning.

    Every block is one PEC component carrying ``symmetry_block``.  A large-R plateau is
    accepted as ``dissociative_to_asymptote`` only when it lands on the computed
    spin-compatible fragment asymptote (``atomic`` from the open-PEC executor);
    otherwise it stays open.  Labels without a single-determinant occupation are
    listed under ``mr_only_candidates`` (an explicit, unclosed coverage gap).
    """
    from .open_pec_extension import compatible_asymptote
    atom_a, atom_b = record["atoms"]
    pg = "D2h" if atom_a == atom_b else "C2v"
    method = {"functional": record["policy"]["functional"], "basis": record["policy"]["basis"]}
    sectors: dict[tuple[int, int], dict] = {}
    mr_only = []
    for q_str, entry in record["charges"].items():
        charge = int(q_str)
        mr_only += [{"charge": charge, **m} for m in entry["plan"]["mr_only_or_outside_window"]]
        # The ground state of a charge lies at or below its lowest dissociation
        # limit (separated ground-state fragments are an accessible state).  A
        # block whose lowest sampled energy is above that limit by more than a
        # DFT margin cannot be the ground state of this charge.
        limit = lowest_dissociation_limit(atomic, atom_a, atom_b, charge) if atomic else None
        for bid, block in entry["blocks"].items():
            two_s = int(block["two_s"])
            sector = sectors.setdefault((charge, two_s), {
                "charge": charge, "spin_2s": two_s, "tracking_mode": "manifold",
                "manifold_branch_graph_unambiguous": True,
                "manifold_branch_graph": {"discontinuous_edges": [], "bridged_edges": []},
                "sector_status": "dft_scout_complete",
                "adaptive_grid": {"stop_reason": "SYMMETRY_BLOCK_SCAN"},
                "method": method, "label": f"{record['system']} q={charge:+d} 2S={two_s}",
                "manifold_pec_scouts": [], "manifolds": [], "representative_roots": []})
            pts = []
            for p in block["points"]:
                if not p["converged"]:
                    continue
                root = f"{bid}__R{p['r_angstrom']:.4f}"
                pts.append({"r_angstrom": p["r_angstrom"], "energy_center_hartree": p["energy_hartree"],
                            "energy_spread_mev": 0.0, "manifold_id": root})
                sector["manifolds"].append({"manifold_id": root, "r_angstrom": p["r_angstrom"],
                                            "energy_center_hartree": p["energy_hartree"],
                                            "member_root_ids": [root], "unique_state_root_ids": [root]})
                sector["representative_roots"].append({
                    "root_id": root, "checkpoint_path": p["checkpoint"], "r_angstrom": p["r_angstrom"],
                    "energy_hartree": p["energy_hartree"], "s2": None,
                    "internal_stable": p["stable_within_symmetry"], "origin_guess": "SYMMETRY_BLOCK",
                    "status": "CANONICALIZED", "charge": charge, "spin_2s": two_s})
            status = _STATUS_MAP.get(block["status"], block["status"].lower())
            minimum_scout: dict[str, Any] = {"status": status, "component_id": bid, "candidates": []}
            lowest = min((p["energy_center_hartree"] for p in pts), default=None)
            reaches_large_r = block.get("side_status", {}).get("higher_r") in ("R_LIMIT", "PLATEAU", "RISES")
            if (limit is not None and lowest is not None and reaches_large_r
                    and (lowest - limit["energy_hartree"]) * HARTREE_TO_EV > above_limit_margin_ev):
                minimum_scout = {"status": "above_lowest_dissociation_limit", "component_id": bid, "candidates": [],
                                 "scout_status": status,
                                 "dissociation_limit_evidence": {
                                     "channel": limit["channel"], "limit_hartree": limit["energy_hartree"],
                                     "lowest_sampled_hartree": lowest, "margin_ev": above_limit_margin_ev,
                                     "lowest_minus_limit_ev": (lowest - limit["energy_hartree"]) * HARTREE_TO_EV}}
            elif status == "bracketed_single_minimum":
                m = block["minimum"]
                minimum_scout["candidates"] = [{"r_angstrom": m["r_angstrom"], "energy_hartree": m["energy_hartree"],
                                                "root_id": f"{bid}__R{m['r_angstrom']:.4f}"}]
            elif block["status"] == "DISSOCIATIVE_PLATEAU" and atomic:
                asym = compatible_asymptote(atomic, atom_a, atom_b, charge, two_s)
                last = max(pts, key=lambda p: p["r_angstrom"]) if pts else None
                if asym and last and abs(last["energy_center_hartree"] - asym["energy_hartree"]) * HARTREE_TO_EV \
                        <= asymptote_tolerance_ev:
                    minimum_scout = {"status": "dissociative_to_asymptote", "component_id": bid, "candidates": [],
                                     "asymptote_evidence": {
                                         "asymptote_channel": asym["channel"],
                                         "asymptote_hartree": asym["energy_hartree"],
                                         "plateau_minus_asymptote_ev":
                                             (last["energy_center_hartree"] - asym["energy_hartree"]) * HARTREE_TO_EV}}
                else:
                    minimum_scout["status"] = "plateau_not_on_computed_asymptote"
            sector["manifold_pec_scouts"].append({
                "component_id": bid, "pec": {"component_id": bid, "points": pts}, "minimum_scout": minimum_scout,
                "symmetry_block": {"label": block["label"], "point_group": pg, "two_s": two_s,
                                   "irrep_nelec": block["irrep_nelec"]}})
    spins = {q: sorted({s for (c, s) in sectors if c == q}) for q in (0, -1)}
    return {
        "system": record["system"], "atoms": [atom_a, atom_b],
        "computational_input": {"initial_R_angstrom": record["r0_angstrom"],
                                "candidate_neutral_2S": spins[0], "candidate_anion_2S": spins[-1],
                                "source": "WIGNER_WITMER_SYMMETRY_SCOUT"},
        "sectors": [sectors[k] for k in sorted(sectors)],
        "mr_only_candidates": mr_only,
        "validation_metadata_used_in_computation": False,
    }


def manifest_for_two_atoms(summary: Mapping[str, Any], *, basis: str = "def2-TZVPPD") -> dict:
    """Minimal manifest so the existing Stage-3 planner can run without a hand-written system entry."""
    ci = summary["computational_input"]
    return {
        "benchmark": {"name": "two_atom_input", "purpose": "automatic from element symbols"},
        "systems": {summary["system"]: {"atoms": list(summary["atoms"]),
                                        "initial_R_angstrom": ci["initial_R_angstrom"],
                                        "candidate_neutral_2S": ci["candidate_neutral_2S"],
                                        "candidate_anion_2S": ci["candidate_anion_2S"]}},
        "protocol": {"stage_3_high_level_local_pec": {
            "reference": "ROHF", "methods": ["CCSD", "CCSD(T)"], "basis": basis,
            "relative_grid_angstrom": [-0.05, -0.025, 0.0, 0.025, 0.05]}},
    }
