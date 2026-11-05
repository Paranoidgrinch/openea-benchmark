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

from .heavy_elements import apply_hamiltonian

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .point_pool import core_slot, run_parallel
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
    max_points_per_side: int = 24
    max_bracket_refinements: int = 8
    min_r_angstrom: float = 0.5
    # 4.0 A cut off weakly bound large-R states before their minimum (verify_v1: Li2/Na2
    # a3Sigmau+, Na2- 2Sigmag+ stopped at R_LIMIT and blocked the EA as open components).
    max_r_angstrom: float = 8.0
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
    mf = apply_hamiltonian(dft.RKS(mol) if two_s == 0 else dft.UKS(mol))
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


# Point labels from the determinant <Lz^2> (symmetry_states.label_of_determinant).
# Blocks scanned under the former per-orbital |m| rule are relabelled from their
# stored checkpoints on load (no new SCF).
LABEL_RULE = "DETERMINANT_LZ2_V1"
# Scan extent is not part of the scout signature: cached blocks stopped by a smaller
# extent are continued from their stored points (no rescan of existing geometries).
SCAN_EXTENT_FIELDS = ("max_r_angstrom", "max_points_per_side", "max_bracket_refinements")

# A charge whose blocks show no bracketed minimum at all (every curve flattens or falls
# toward its fragments) is scanned again with an empirical pair dispersion added to the
# DFT energy: the functional misses the long-range correlation that binds van der Waals
# pairs (He2, Ne2, Mg2, Zn2, Hg2, rare gas + anion), so their wells (meV to ~0.1 eV) are
# lost or flattened into a plateau.  The dispersion term only places the seed geometry;
# every energy that enters the EA comes from the coupled-cluster layers, which contain
# dispersion without any empirical term.  No plateau stop on this scan (the well is
# shallower than the plateau tolerance); a curve still falling at the R limit ends as
# ``DISPERSION_R_LIMIT`` and is checked against the fragment asymptote like a plateau.
DISPERSION_MODEL = "d3bj"
DISPERSION_R_LIMIT = "R_LIMIT_WITH_DISPERSION"
BOUND_SCAN_STATUSES = ("BRACKETED_SINGLE_MINIMUM", "BRACKETED_RELABELLED")


def _falls_to_large_r(block: Mapping[str, Any]) -> bool:
    ok = sorted((p for p in block["points"] if p["converged"]), key=lambda p: p["r_angstrom"])
    return len(ok) >= 2 and (block.get("side_status") or {}).get("higher_r") in ("PLATEAU", "R_LIMIT") \
        and min(range(len(ok)), key=lambda k: ok[k]["energy_hartree"]) == len(ok) - 1


def needs_dispersion_rescan(entry: Mapping[str, Any]) -> bool:
    """No block of this charge brackets a minimum and at least one falls toward its fragments
    (a block that fails or changes label at short R is not a van der Waals question)."""
    blocks = [b for b in (entry.get("blocks") or {}).values() if b.get("status") != "SCF_FAILED_AT_START"]
    return any(_falls_to_large_r(b) for b in blocks) and not any(
        classify_block_scan(b["points"], (b.get("side_status") or {}).get("higher_r"), b["label"])
        in BOUND_SCAN_STATUSES for b in blocks)


def dispersion_available() -> bool:
    try:
        import pyscf.dispersion  # noqa: F401  (package pyscf-dispersion)
    except ImportError:
        return False
    return True


def summary_blocks(entry: Mapping[str, Any]) -> tuple[dict, str | None]:
    """The block scans that represent a charge: the dispersion rescan when it exists."""
    disp = entry.get("dispersion") or {}
    if disp.get("blocks"):
        return disp["blocks"], disp.get("model")
    return entry.get("blocks") or {}, None


def _relabel_block_from_checkpoints(block: dict) -> bool:
    """Relabel the stored points of a block with the current rule; False if a checkpoint is missing."""
    from pyscf.lib import chkfile
    from .symmetry_states import label_of_determinant
    points = []
    for p in block["points"]:
        p = dict(p)
        if p.get("converged"):
            path = p.get("checkpoint")
            if not path or not Path(path).is_file():
                return False
            mol = chkfile.load_mol(path)
            scf = chkfile.load(path, "scf")
            sym = label_of_determinant(mol, scf["mo_coeff"], scf["mo_occ"], like_nuclei=mol.groupname == "D2h")
            p["label"] = None if sym is None else sym.label
        points.append(p)
    block["points"] = points
    block["status"] = classify_block_scan(points, block.get("side_status", {}).get("higher_r"), block["label"])
    block["label_rule"] = LABEL_RULE
    return True


def _bracket_label_consistent(ok: Sequence[Mapping[str, Any]], i: int, target: str) -> bool:
    """The minimum carries the block label and neither neighbour carries another one.

    A label of None means the determinant is not an Lz^2 eigenfunction at that
    geometry (e.g. sigma/delta mixing inside one abelian irrep): no evidence of
    another state.  Label changes outside the bracket (far edges of the scan)
    do not touch the minimum the high-level grid is placed around.
    """
    return ok[i]["label"] == target and all(ok[j]["label"] in (None, target) for j in (i - 1, i + 1))


def bracket_label(points: Sequence[Mapping[str, Any]]) -> str | None:
    """The one Lambda-S label shared by the minimum and both neighbours, else None.

    With determinant labels (an exact Lz^2 eigenvalue plus the imposed irreps) a block
    that converged onto another state along its whole bracket is a curve of that
    state with the same irrep occupation (verify_v1 LaO: X2Sigma+ found only inside the
    2Delta block; FeO-: 4Delta inside the 4Sigma- block)."""
    ok = [p for p in sorted(points, key=lambda p: p["r_angstrom"]) if p["converged"]]
    if len(ok) < 3:
        return None
    i = min(range(len(ok)), key=lambda k: ok[k]["energy_hartree"])
    if not 0 < i < len(ok) - 1:
        return None
    labels = {ok[j]["label"] for j in (i - 1, i, i + 1)}
    return labels.pop() if len(labels) == 1 else None


LAMBDA_MIXED_STATUS = "lambda_mixed_mr_route"


def _minimum_lambda_mixed(points: Sequence[Mapping[str, Any]]) -> bool:
    ok = [p for p in points if p["converged"]]
    return bool(ok) and min(ok, key=lambda p: p["energy_hartree"])["label"] is None


SCF_EDGE_STATUS = "scf_failed_edge_mr_route"


def _minimum_at_failed_edge(block: Mapping[str, Any]) -> bool:
    ok = sorted((p for p in block["points"] if p["converged"]), key=lambda p: p["r_angstrom"])
    if len(ok) < 2:
        return False
    i = min(range(len(ok)), key=lambda k: ok[k]["energy_hartree"])
    side = block.get("side_status") or {}
    return (i == 0 and side.get("lower_r") == "SCF_FAILED") or \
        (i == len(ok) - 1 and side.get("higher_r") == "SCF_FAILED")


def _symmetric_charge_delocalization(block: Mapping[str, Any], atomic: Mapping[str, Any], atom: str,
                                     charge: int, two_s: int, tolerance_ev: float) -> dict | None:
    """Like-atom ion still descending at the R limit below its charge-localized asymptote.

    D2h symmetry forces half of the extra charge onto each atom; at large R this
    delocalized state drops below X + X- (DFT delocalization error; verify_v1 Li2-,
    Na2-, Ag2- 2Sigmag+: 0.20-0.56 eV below at 7.5 A and still falling).  A repulsive
    curve toward X + X- carries no well of its own; the X + X- channel is handled by the
    electronic EA stage's anion dissociation channels.
    """
    from .open_pec_extension import compatible_asymptote
    ok = sorted((p for p in block["points"] if p["converged"]), key=lambda p: p["r_angstrom"])
    if len(ok) < 3 or (block.get("side_status") or {}).get("higher_r") != "R_LIMIT":
        return None
    if min(range(len(ok)), key=lambda k: ok[k]["energy_hartree"]) != len(ok) - 1:
        return None
    asym = compatible_asymptote(atomic, atom, atom, charge, two_s)
    if not asym:
        return None
    below_ev = (ok[-1]["energy_hartree"] - asym["energy_hartree"]) * HARTREE_TO_EV
    if below_ev > tolerance_ev:
        return None
    return {"asymptote_channel": asym["channel"], "asymptote_hartree": asym["energy_hartree"],
            "plateau_minus_asymptote_ev": below_ev, "r_last_angstrom": ok[-1]["r_angstrom"],
            "basis": "SYMMETRIC_CHARGE_DELOCALIZATION_BELOW_LOCALIZED_ASYMPTOTE"}


def classify_block_scan(points: Sequence[Mapping[str, Any]], higher_r_status: str | None, target: str) -> str:
    """Scan status from the stored points (also re-applied to cached scout records)."""
    ok = [p for p in sorted(points, key=lambda p: p["r_angstrom"]) if p["converged"]]
    if not ok:
        return "SCF_FAILED_AT_START"
    energies = [p["energy_hartree"] for p in ok]
    i_min = min(range(len(ok)), key=lambda i: energies[i])
    if 0 < i_min < len(ok) - 1 and energies[i_min - 1] > energies[i_min] < energies[i_min + 1]:
        if _bracket_label_consistent(ok, i_min, target):
            return "BRACKETED_SINGLE_MINIMUM"
        return "BRACKETED_RELABELLED" if bracket_label(points) not in (None, target) else "LABEL_CHANGED"
    if any(p["label"] != target for p in ok):
        return "LABEL_CHANGED"
    if higher_r_status in ("PLATEAU", DISPERSION_R_LIMIT) and i_min == len(ok) - 1:
        return "DISSOCIATIVE_PLATEAU"
    return "UNRESOLVED"


def scan_block(atoms, charge: int, occ: BlockOccupation, r0: float, policy: SymmetryScoutPolicy,
               checkpoint_dir: Path, scf_runner: Callable[..., Any] | None = None,
               existing: Mapping[str, Any] | None = None, dispersion: str | None = None) -> dict:
    """Seeded R scan inside one symmetry block until a bracketed minimum or a large-R plateau.

    ``existing`` (a stored block scan) is continued: its points are kept, sides that
    stopped at R_LIMIT/BUDGET are extended under the current policy, and the bracket
    refinement is applied again."""
    def default_runner(r, seed):
        from pyscf import lib
        mol = build_symmetric_molecule(atoms, r, basis=policy.basis, charge=charge, two_s=occ.symmetry.two_s)
        dm0 = None
        if seed:
            from pyscf import scf as _scf
            dm0 = _scf.hf.init_guess_by_chkfile(mol, seed, project=None)
        chk = str(checkpoint_dir / f"{_block_id(charge, occ)}__R{r:.4f}.chk")
        res = run_block_scf(mol, occ, method="DFT", xc=policy.functional, dm0=dm0, checkpoint=chk,
                            following_steps=policy.following_steps, grid_level=policy.grid_level,
                            dispersion=dispersion)
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

    status = {}
    starts = {}
    if existing is not None:
        for p in existing["points"]:
            points[round(p["r_angstrom"], 4)] = dict(p)
        center = points.get(round(r0, 4)) or min((p for p in points.values() if p["converged"]),
                                                 key=lambda p: abs(p["r_angstrom"] - r0))
        side = existing.get("side_status") or {}
        for direction, key in ((-1, "lower_r"), (+1, "higher_r")):
            if side.get(key) not in ("R_LIMIT", "BUDGET", "SCF_FAILED"):
                status[direction] = side.get(key)
                continue
            own = sorted((p for p in points.values() if p["converged"]
                          and direction * (p["r_angstrom"] - center["r_angstrom"]) > 0),
                         key=lambda p: direction * p["r_angstrom"])
            if side.get(key) == "SCF_FAILED":
                # continue from the last converged point, retrying the failed step halved
                failed = [p for p in points.values() if not p["converged"]
                          and direction * (p["r_angstrom"] - (own[-1] if own else center)["r_angstrom"]) > 0]
                if failed:
                    last = own[-1] if own else center
                    gap = min(abs(f["r_angstrom"] - last["r_angstrom"]) for f in failed)
                    starts[direction] = (last, gap, len(own) + 1, "RETRY_HALF")
                    continue
            if not own:
                starts[direction] = (center, policy.step_angstrom, 1)
                continue
            last = own[-1]
            step = abs(last["r_angstrom"] - (own[-2] if len(own) > 1 else center)["r_angstrom"])
            starts[direction] = (last, step, len(own) + 1)
    else:
        center = compute(r0, None)
        if not center["converged"]:
            return {"status": "SCF_FAILED_AT_START",
                    "points": sorted(points.values(), key=lambda p: p["r_angstrom"])}
        starts = {-1: (center, policy.step_angstrom, 1), +1: (center, policy.step_angstrom, 1)}
    for direction, start in starts.items():
        prev, step, k0 = start[:3]
        retry_half = len(start) > 3
        r = prev["r_angstrom"]
        for k in range(k0, policy.max_points_per_side + 1):
            if retry_half:
                step, retry_half = step / 2.0, False
            elif k > policy.fine_steps:
                step = min(step * policy.step_growth, policy.max_step_angstrom)
            r = round(r + direction * step, 4)
            if r < policy.min_r_angstrom or r > policy.max_r_angstrom:
                status[direction] = DISPERSION_R_LIMIT if dispersion and direction > 0 else "R_LIMIT"
                break
            p = compute(r, prev["checkpoint"])
            if not p["converged"]:
                # One retry at half the step from the last converged geometry: a closer
                # seed often converges (verify_v1 NiO- 2Delta failed at 1.80 A and the
                # minimum stayed unbracketed at 1.85 A).
                step = step / 2.0
                r = round(prev["r_angstrom"] + direction * step, 4)
                p = points.get(r) or compute(r, prev["checkpoint"])
                if not p["converged"] or r < policy.min_r_angstrom:
                    status[direction] = "SCF_FAILED"
                    break
            if p["energy_hartree"] > prev["energy_hartree"]:
                status[direction] = "RISES"
                break
            if not dispersion and direction > 0 and abs(p["energy_hartree"] - prev["energy_hartree"]) \
                    * HARTREE_TO_EV < policy.plateau_energy_tolerance_ev and r > r0 + 1.0:
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
        if not 0 < i < len(ok) - 1 or ok[i]["label"] is None:
            break  # refine any bracket whose minimum is an exact-Lambda determinant
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
    result_status = classify_block_scan(ordered, status.get(+1), occ.symmetry.label)
    return {"status": result_status, "label_rule": LABEL_RULE,
            "side_status": {"lower_r": status.get(-1), "higher_r": status.get(+1)},
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
    level = {k: v for k, v in asdict(policy).items() if k not in SCAN_EXTENT_FIELDS}
    signature = sha256(json.dumps({"policy": level, "atoms": [atom_a, atom_b], "r0": r0},
                                  sort_keys=True).encode()).hexdigest()
    record = json.loads(path.read_text()) if path.exists() else {}
    if record.get("signature") != signature:
        old_level = {k: v for k, v in (record.get("policy") or {}).items() if k not in SCAN_EXTENT_FIELDS}
        # Records from before the extent split: same level of theory -> keep, extend below.
        if not (old_level == level and record.get("atoms") == [atom_a, atom_b] and record.get("r0_angstrom") == r0):
            record = {"charges": {}}
        record["signature"] = signature
    record.update(system=f"{atom_a}{atom_b}", atoms=[atom_a, atom_b], r0_angstrom=r0, policy=asdict(policy),
                  validation_metadata_used_in_computation=False)
    relabelled = False
    for entry in record["charges"].values():
        for bid, block in list(entry.get("blocks", {}).items()):
            if block.get("label_rule") != LABEL_RULE and block.get("status") != "SCF_FAILED_AT_START":
                if _relabel_block_from_checkpoints(block):
                    relabelled = True
                else:
                    del entry["blocks"][bid]  # checkpoint gone: rescan this block
    if relabelled:
        _save(path, record)
    extent = {k: getattr(policy, k) for k in SCAN_EXTENT_FIELDS}
    for q_str, entry in record["charges"].items():
        cont = []
        for b in entry.get("plan", {}).get("blocks", ()):
            occ = BlockOccupation(_symmetry_from_label(b["label"]),
                                  tuple((k, tuple(v)) for k, v in b["irrep_nelec"]), b["orbital_energy_sum"])
            bid = _block_id(int(q_str), occ)
            block = entry["blocks"].get(bid)
            if block and block.get("extent") != extent and block.get("status") != "SCF_FAILED_AT_START":
                cont.append((b, bid, occ, block))

        def store_cont(index: int, scan: dict, entry=entry, cont=cont) -> None:
            b, bid, _, _ = cont[index]
            entry["blocks"][bid] = {"label": b["label"], "two_s": b["two_s"], "irrep_nelec": b["irrep_nelec"],
                                    **scan, "extent": extent}
            _save(path, record)

        args = [((atom_a, atom_b), int(q_str), occ, r0, policy, output_dir / "symmetry_checkpoints",
                 scf_runner_factory(int(q_str), occ) if scf_runner_factory else None, block)
                for _, _, occ, block in cont]
        if scf_runner_factory:
            for index, item in enumerate(args):
                store_cont(index, scan_block(*item))
        elif args:
            run_parallel(scan_block, args, on_result=store_cont)
    for charge in charges:
        entry = record["charges"].setdefault(str(charge), {"blocks": {}})
        if "plan" not in entry:
            with core_slot():  # reference DFT per spin sector: one machine-wide core slot
                plan = plan_blocks(atom_a, atom_b, charge, r0, policy, reference)
            entry["plan"] = {"candidate_labels": plan["candidate_labels"],
                             "mr_only_or_outside_window": plan["mr_only_or_outside_window"],
                             "blocks": [{"label": o.symmetry.label, "two_s": o.symmetry.two_s,
                                         "irrep_nelec": [[k, list(v)] for k, v in o.irrep_nelec],
                                         "orbital_energy_sum": o.orbital_energy_sum} for o in plan["blocks"]]}
            _save(path, record)
        todo = []
        for b in entry["plan"]["blocks"]:
            occ = BlockOccupation(_symmetry_from_label(b["label"]),
                                  tuple((k, tuple(v)) for k, v in b["irrep_nelec"]), b["orbital_energy_sum"])
            bid = _block_id(charge, occ)
            if bid in entry["blocks"]:
                continue
            print(f"[SYMMETRY SCOUT] {atom_a}{atom_b} q={charge:+d} {b['label']} {bid}", flush=True)
            todo.append((b, bid, occ))

        def store(index: int, scan: dict, entry=entry, todo=todo) -> None:
            b, bid, _ = todo[index]
            entry["blocks"][bid] = {"label": b["label"], "two_s": b["two_s"], "irrep_nelec": b["irrep_nelec"],
                                    **scan, "extent": {k: getattr(policy, k) for k in SCAN_EXTENT_FIELDS}}
            _save(path, record)

        # Blocks are independent scans (own seeds, own checkpoint names): they run
        # concurrently on the point pool; injected test runners stay serial.
        args = [((atom_a, atom_b), charge, occ, r0, policy, output_dir / "symmetry_checkpoints",
                 scf_runner_factory(charge, occ) if scf_runner_factory else None) for _, _, occ in todo]
        if scf_runner_factory:
            for index, item in enumerate(args):
                store(index, scan_block(*item))
        else:
            run_parallel(scan_block, args, on_result=store)
    for charge in charges:
        entry = record["charges"][str(charge)]
        if not needs_dispersion_rescan(entry):
            continue
        if scf_runner_factory is None and not dispersion_available():
            entry["dispersion"] = {"model": DISPERSION_MODEL, "blocks": {}, "status": "DISPERSION_UNAVAILABLE"}
            _save(path, record)
            continue
        disp = entry.setdefault("dispersion", {"model": DISPERSION_MODEL, "blocks": {}})
        disp.pop("status", None)
        todo = []
        for b in entry["plan"]["blocks"]:
            occ = BlockOccupation(_symmetry_from_label(b["label"]),
                                  tuple((k, tuple(v)) for k, v in b["irrep_nelec"]), b["orbital_energy_sum"])
            bid = _block_id(charge, occ)
            if bid not in disp["blocks"]:
                print(f"[SYMMETRY SCOUT] {atom_a}{atom_b} q={charge:+d} {b['label']} {bid} +{DISPERSION_MODEL}",
                      flush=True)
                todo.append((b, bid, occ))

        def store_disp(index: int, scan: dict, disp=disp, todo=todo) -> None:
            b, bid, _ = todo[index]
            disp["blocks"][bid] = {"label": b["label"], "two_s": b["two_s"], "irrep_nelec": b["irrep_nelec"],
                                   **scan, "extent": {k: getattr(policy, k) for k in SCAN_EXTENT_FIELDS}}
            _save(path, record)

        args = [((atom_a, atom_b), charge, occ, r0, policy, output_dir / f"symmetry_checkpoints_{DISPERSION_MODEL}",
                 scf_runner_factory(charge, occ, dispersion=DISPERSION_MODEL) if scf_runner_factory else None,
                 None, DISPERSION_MODEL) for _, _, occ in todo]
        if scf_runner_factory:
            for index, item in enumerate(args):
                store_disp(index, scan_block(*item))
        else:
            run_parallel(scan_block, args, on_result=store_disp)
    return record


def _symmetry_from_label(label: str):
    import re
    from .state_candidates import MolecularSymmetry, _LAMBDA_LETTER
    m = re.match(r"(\d+)([A-Za-z]+?)([gu]?)([+-]?)$", label)
    mult, name, gu, refl = m.groups()
    inverse = {v: k for k, v in _LAMBDA_LETTER.items()}
    return MolecularSymmetry(int(mult) - 1, inverse[name], gu or None, refl or None)


# ------------------------------------------------------------ legacy export

_STATUS_MAP = {"BRACKETED_SINGLE_MINIMUM": "bracketed_single_minimum",
               "BRACKETED_RELABELLED": "bracketed_single_minimum"}


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
    # Canonical key order: a freshly computed record (insertion order) and the same
    # record reloaded from symmetry_scout.json must give an identical summary, since
    # the summary digest fingerprints every Stage-3 point on resume.
    record = json.loads(json.dumps(record, sort_keys=True))
    atomic = None if atomic is None else json.loads(json.dumps(atomic, sort_keys=True))
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
        blocks, dispersion = summary_blocks(entry)
        charge_method = method if dispersion is None else {**method, "dispersion": dispersion}
        for bid, block in blocks.items():
            two_s = int(block["two_s"])
            sector = sectors.setdefault((charge, two_s), {
                "charge": charge, "spin_2s": two_s, "tracking_mode": "manifold",
                "manifold_branch_graph_unambiguous": True,
                "manifold_branch_graph": {"discontinuous_edges": [], "bridged_edges": []},
                "sector_status": "dft_scout_complete",
                "adaptive_grid": {"stop_reason": "SYMMETRY_BLOCK_SCAN"},
                "method": charge_method, "label": f"{record['system']} q={charge:+d} 2S={two_s}",
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
            if block["status"] != "SCF_FAILED_AT_START":  # cached records: current classification rule
                block = {**block, "status": classify_block_scan(
                    block["points"], block.get("side_status", {}).get("higher_r"), block["label"])}
            status = _STATUS_MAP.get(block["status"], block["status"].lower())
            minimum_scout: dict[str, Any] = {"status": status, "component_id": bid, "candidates": []}
            if block["status"] == "UNRESOLVED" and _minimum_at_failed_edge(block):
                # Still descending into a geometry where the block SCF does not converge (also
                # after half-step retries; verify_v1 NiO- 2Delta below 1.725 A): the single
                # determinant cannot resolve this state.  MR competitor layer bounds its block.
                minimum_scout = {**minimum_scout, "status": SCF_EDGE_STATUS}
            if block["status"] == "LABEL_CHANGED" and _minimum_lambda_mixed(block["points"]):
                # The block determinant at its lowest point is not an Lz^2 eigenfunction
                # (e.g. pi^1 delta^1 mixes Pi and Phi): no single-reference state of any
                # Lambda.  Route the target term to the MR competitor layer, which bounds
                # every state of this (charge, spin, irrep) block; the final verdict stays
                # blocked until that layer clears it.
                minimum_scout = {**minimum_scout, "status": LAMBDA_MIXED_STATUS}
            lowest = min((p["energy_center_hartree"] for p in pts), default=None)
            reaches_large_r = block.get("side_status", {}).get("higher_r") in ("R_LIMIT", "PLATEAU", "RISES",
                                                                              DISPERSION_R_LIMIT)
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
            if (atom_a == atom_b and charge != 0 and atomic and block["status"] == "UNRESOLVED"
                    and minimum_scout["status"] == "unresolved"):
                evidence = _symmetric_charge_delocalization(block, atomic, atom_a, charge, two_s,
                                                            asymptote_tolerance_ev)
                if evidence:
                    minimum_scout = {"status": "dissociative_to_asymptote", "component_id": bid,
                                     "candidates": [], "asymptote_evidence": evidence}
            if minimum_scout["status"] in (LAMBDA_MIXED_STATUS, SCF_EDGE_STATUS) and not any(
                    m["charge"] == charge and m["label"] == block["label"] for m in mr_only):
                mr_only.append({"charge": charge, "label": block["label"], "count": 1, "channels": [],
                                "reason": ("LAMBDA_MIXED_SINGLE_DETERMINANT"
                                           if minimum_scout["status"] == LAMBDA_MIXED_STATUS
                                           else "SCF_FAILS_ON_DESCENDING_EDGE"),
                                "component_id": bid})
            symmetry_block = {"label": block["label"], "point_group": pg, "two_s": two_s,
                              "irrep_nelec": block["irrep_nelec"]}
            if block["status"] == "BRACKETED_RELABELLED":
                # Same irrep occupation, converged state carries another exact label.
                symmetry_block = {**symmetry_block, "label": bracket_label(block["points"]),
                                  "block_target_label": block["label"]}
            sector["manifold_pec_scouts"].append({
                "component_id": bid, "pec": {"component_id": bid, "points": pts}, "minimum_scout": minimum_scout,
                "symmetry_block": symmetry_block})
    for (charge, _), sector in sectors.items():
        _merge_coincident_label_changed(sector, mr_only, charge)
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


# Same block-SCF state reached from two different irrep blocks: DFT grid and
# convergence noise of ~1e-6 Eh (verify_v1 FeO 5Sigma- vs 5Delta: 1.1e-6 Eh);
# distinct states at one geometry differ by orders of magnitude more.
SCOUT_MERGE_ENERGY_TOLERANCE_HARTREE = 1.0e-5


# Bracket refinements of two blocks can round the same midpoint differently (verify_v1
# FeO- 6Sigma- block on 6Delta: 1.6503 vs 1.6504 A, so the duplicate was not merged).
SCOUT_MERGE_R_TOLERANCE_ANGSTROM = 1.0e-3


def _energy_on_curve(points: Sequence[Mapping[str, Any]], r: float) -> float | None:
    """Energy of a scout curve at ``r``: its point there, or linear between the two points
    around ``r`` when its nearest point is within SCOUT_MERGE_R_TOLERANCE_ANGSTROM."""
    pts = sorted((q for q in points if q.get("energy_center_hartree") is not None), key=lambda q: q["r_angstrom"])
    if not pts:
        return None
    near = min(pts, key=lambda q: abs(q["r_angstrom"] - r))
    if abs(near["r_angstrom"] - r) < 1.0e-6:
        return near["energy_center_hartree"]
    if abs(near["r_angstrom"] - r) > SCOUT_MERGE_R_TOLERANCE_ANGSTROM:
        return None
    lo = [q for q in pts if q["r_angstrom"] < r]
    hi = [q for q in pts if q["r_angstrom"] > r]
    if not lo or not hi:
        return None
    a, b = lo[-1], hi[0]
    t = (r - a["r_angstrom"]) / (b["r_angstrom"] - a["r_angstrom"])
    return a["energy_center_hartree"] + t * (b["energy_center_hartree"] - a["energy_center_hartree"])


def _merge_coincident_label_changed(sector: dict, mr_only: list | None = None, charge: int | None = None) -> None:
    """A label-changed block whose minimum and both neighbours coincide (same R, same
    energy) with points of a bracketed component of the same sector converged onto
    that state (verify_v1 FeO 5Sigma- -> 5Delta, VO 4Delta -> 4Sigma-): it is
    covered by that component, not an open competitor."""
    from .open_pec_extension import MERGED_STATUS
    scouts = sector["manifold_pec_scouts"]
    # Blocks that kept their own label come first, so a relabelled duplicate merges into them.
    targets = (("bracketed_single_minimum", LAMBDA_MIXED_STATUS, SCF_EDGE_STATUS)
               if mr_only is not None else ("bracketed_single_minimum",))
    bracketed = sorted((s for s in scouts if s["minimum_scout"]["status"] in targets),
                       key=lambda s: (s["minimum_scout"]["status"] != "bracketed_single_minimum",
                                      "block_target_label" in s["symmetry_block"]))
    for scout in scouts:
        pts = sorted(scout["pec"]["points"], key=lambda p: p["r_angstrom"])
        relabelled = "block_target_label" in scout["symmetry_block"] \
            and scout["minimum_scout"]["status"] == "bracketed_single_minimum"
        if not (scout["minimum_scout"]["status"] == "label_changed" or relabelled) or len(pts) < 3:
            continue
        i = min(range(len(pts)), key=lambda k: pts[k]["energy_center_hartree"])
        probe = pts[max(0, i - 1):i + 2]
        for other in bracketed:
            if other is scout or other["minimum_scout"]["status"] not in targets:
                continue
            if relabelled and other["minimum_scout"]["status"] != "bracketed_single_minimum":
                continue
            if relabelled and other["symmetry_block"]["label"] != scout["symmetry_block"]["label"]:
                continue
            hits = []
            for p in probe:
                ref = _energy_on_curve(other["pec"]["points"], p["r_angstrom"])
                if ref is None or abs(p["energy_center_hartree"] - ref) >= SCOUT_MERGE_ENERGY_TOLERANCE_HARTREE:
                    break
                hits.append({"r_angstrom": p["r_angstrom"], "delta_hartree": p["energy_center_hartree"] - ref})
            if len(hits) == len(probe) >= 2 and other["minimum_scout"]["status"] != "bracketed_single_minimum":
                # Same SCF solution as a state routed to the MR layer: same route.
                scout["minimum_scout"] = {**scout["minimum_scout"], "status": other["minimum_scout"]["status"],
                                          "scout_status": "label_changed",
                                          "coincident_with_component_id": other["component_id"]}
                label = scout["symmetry_block"]["label"]
                if not any(m["charge"] == charge and m["label"] == label for m in mr_only):
                    mr_only.append({"charge": charge, "label": label, "count": 1, "channels": [],
                                    "reason": "COINCIDES_WITH_MR_ROUTED_COMPONENT",
                                    "component_id": scout["component_id"]})
                break
            if len(hits) == len(probe) >= 2:
                scout["minimum_scout"] = {**scout["minimum_scout"], "status": MERGED_STATUS,
                                          "scout_status": "bracketed_relabelled" if relabelled else "label_changed",
                                          "merged_into_component_id": other["component_id"],
                                          "coincident_points": hits}
                break


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
