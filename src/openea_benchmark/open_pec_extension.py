"""D09 executor: extend open scout PEC components and compute DFT fragment asymptotes.

The DFT scout stops after a bounded grid extension.  Components whose sampled
energy still decreases at a grid edge stay "open" and keep D04/D09 open forever.
This module performs the planner's ``EXTEND_OPEN_PEC`` action:

1. Each open component is continued along its decreasing direction at the scout
   DFT level.  Every new geometry is seeded from the previous geometry's
   checkpoint (adiabatic continuation of the same electronic state); internal
   instabilities are followed.  A jump in <S^2> is a continuity warning.
2. Atomic fragment energies are computed at the same DFT level for every atomic
   charge (0, -1) and every 2S up to a bound; no atomic ground-state term is
   assumed.  The lowest spin-coupling-compatible channel gives the component's
   asymptote.
3. Classification (sampled evidence only, never pruning):
   - ``MINIMUM_BRACKETED``: the continuation found a strict discrete minimum;
     the component becomes an ordinary Stage-3 seed candidate.
   - ``DISSOCIATIVE_TO_ASYMPTOTE``: the energy flattened at large R onto the
     computed compatible asymptote; no sampled well exists along the path.
   - ``UNRESOLVED``: anything else (no plateau, asymptote mismatch, SCF failure,
     continuity warning, short-R limit).

``apply_open_pec_extension`` merges the evidence into an in-memory copy of the
scout summary so the existing state-selection and Stage-3 planners are reused
unchanged.  No experimental data are read.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .local_pec import LocalPEC, LocalPECPoint
from .minimum_scout import MinimumScoutThresholds, scout_local_pec_minimum

HARTREE_TO_EV = 27.211386245988
OPEN_STATUSES = ("decreases_toward_higher_r", "decreases_toward_lower_r")
DISSOCIATIVE_STATUS = "dissociative_to_asymptote"
EXTENSION_ORIGIN = "OPEN_PEC_CONTINUATION"


@dataclass(frozen=True)
class OpenPECExtensionPolicy:
    step_schedule_angstrom: tuple[float, ...] = (
        0.05, 0.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.5, 0.75, 1.0,
    )
    min_r_angstrom: float = 0.5
    plateau_energy_tolerance_ev: float = 0.01
    plateau_min_r_angstrom: float = 2.5
    asymptote_match_tolerance_ev: float = 0.25
    s2_jump_warning: float = 0.3
    max_atomic_2s: int = 6
    stability_following_steps: int = 4
    minimum_energy_tolerance_hartree: float = 1.0e-6

    def __post_init__(self) -> None:
        if not self.step_schedule_angstrom or any(s <= 0 for s in self.step_schedule_angstrom):
            raise ValueError("step schedule must be non-empty and positive")
        for name in ("min_r_angstrom", "plateau_energy_tolerance_ev", "plateau_min_r_angstrom",
                     "asymptote_match_tolerance_ev", "s2_jump_warning", "minimum_energy_tolerance_hartree"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.max_atomic_2s < 0 or self.stability_following_steps < 0:
            raise ValueError("max_atomic_2s and stability_following_steps must be non-negative")


# ---------------------------------------------------------------- PySCF backend

def _dft_mean_field(mol, functional: str, grid_level: int, conv_tol: float, max_cycle: int):
    from pyscf import dft

    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    mf.xc = functional
    mf.grids.level = grid_level
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.verbose = 0
    return mf


def _converge(mf, dm0, following_steps: int):
    mf.kernel(dm0=dm0)
    for _ in range(following_steps):
        mo, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)
        if stable and mf.converged:
            break
        mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ))
    return mf


def pyscf_atom_energy(symbol: str, charge: int, spin_2s: int, *, functional: str, basis: str,
                      grid_level: int, policy: OpenPECExtensionPolicy) -> dict:
    from pyscf import gto

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, charge=charge, spin=spin_2s, verbose=0)
    best = None
    for guess in ("minao", "atom", "huckel"):
        try:
            mf = _dft_mean_field(mol, functional, grid_level, 1.0e-9, 200)
            mf.init_guess = guess
            mf = _converge(mf, None, policy.stability_following_steps)
        except Exception:  # an unavailable guess is not a failed state search
            continue
        if mf.converged and (best is None or mf.e_tot < best["energy_hartree"]):
            best = {"energy_hartree": float(mf.e_tot), "guess": guess,
                    "s2": float(mf.spin_square()[0]) if spin_2s else 0.0}
    return best or {"energy_hartree": None, "guess": None, "s2": None}


def pyscf_continuation_point(*, atoms: tuple[str, str], charge: int, spin_2s: int, r_angstrom: float,
                             seed_checkpoint: str, checkpoint: str, functional: str, basis: str,
                             grid_level: int, policy: OpenPECExtensionPolicy) -> dict:
    from pyscf import gto

    mol = gto.M(atom=[(atoms[0], (0.0, 0.0, 0.0)), (atoms[1], (0.0, 0.0, float(r_angstrom)))],
                basis=basis, charge=charge, spin=spin_2s, unit="Angstrom", verbose=0)
    mf = _dft_mean_field(mol, functional, grid_level, 1.0e-9, 200)
    dm0 = mf.from_chk(seed_checkpoint)
    mf.chkfile = checkpoint
    mf = _converge(mf, dm0, policy.stability_following_steps)
    mo, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)
    return {"r_angstrom": float(r_angstrom), "energy_hartree": float(mf.e_tot),
            "converged": bool(mf.converged), "internal_stable": bool(stable),
            "s2": float(mf.spin_square()[0]) if spin_2s else 0.0, "checkpoint_path": checkpoint}


# ------------------------------------------------------------------ evidence

def compatible_asymptote(atomic: Mapping[str, Mapping[str, Any]], atom_a: str, atom_b: str,
                         charge: int, spin_2s: int) -> dict | None:
    """Lowest fragment channel whose atomic spins can couple to the molecular 2S."""
    best = None
    for key_a, rec_a in atomic.items():
        sym_a, q_a, s_a = _parse_atom_key(key_a)
        if sym_a != atom_a or rec_a.get("energy_hartree") is None:
            continue
        for key_b, rec_b in atomic.items():
            sym_b, q_b, s_b = _parse_atom_key(key_b)
            if sym_b != atom_b or rec_b.get("energy_hartree") is None or q_a + q_b != charge:
                continue
            if not (abs(s_a - s_b) <= spin_2s <= s_a + s_b and (s_a + s_b - spin_2s) % 2 == 0):
                continue
            energy = float(rec_a["energy_hartree"]) + float(rec_b["energy_hartree"])
            if best is None or energy < best["energy_hartree"]:
                best = {"energy_hartree": energy, "channel": f"{key_a} + {key_b}"}
    return best


def atom_key(symbol: str, charge: int, spin_2s: int) -> str:
    return f"{symbol}|q{charge:+d}|2S{spin_2s}"


def _parse_atom_key(key: str) -> tuple[str, int, int]:
    symbol, q, s = key.split("|")
    return symbol, int(q[1:]), int(s[2:])


def _atomic_electrons(symbol: str, charge: int) -> int:
    from pyscf.data.elements import charge as nuclear_charge

    return int(nuclear_charge(symbol)) - int(charge)


def classify_continuation(scan: list[dict], *, direction: int, asymptote: dict | None,
                          policy: OpenPECExtensionPolicy, terminated: str) -> dict:
    """Pure classification of one continuation scan (scan[0] is the scout edge)."""
    energies = [p["energy_hartree"] for p in scan]
    lowest = min(range(len(scan)), key=lambda i: energies[i])
    reasons = []
    if any(not p.get("converged", True) or p.get("internal_stable") is False for p in scan[1:]):
        reasons.append("SCF_NOT_CONVERGED_OR_UNSTABLE")
    s2 = [p.get("s2") for p in scan if p.get("s2") is not None]
    if any(abs(b - a) > policy.s2_jump_warning for a, b in zip(s2, s2[1:])):
        reasons.append("S2_JUMP_POSSIBLE_STATE_SWITCH")
    result = {"lowest_sampled_hartree": energies[lowest],
              "lowest_sampled_r_angstrom": scan[lowest]["r_angstrom"],
              "terminated": terminated, "reasons": reasons}
    if asymptote is not None:
        result["asymptote_hartree"] = asymptote["energy_hartree"]
        result["asymptote_channel"] = asymptote["channel"]
        result["lowest_minus_asymptote_ev"] = (energies[lowest] - asymptote["energy_hartree"]) * HARTREE_TO_EV
    if reasons:
        return {**result, "classification": "UNRESOLVED"}
    if terminated == "MINIMUM_BRACKETED" and 0 < lowest < len(scan) - 1:
        return {**result, "classification": "MINIMUM_BRACKETED"}
    if terminated == "PLATEAU" and direction > 0:
        if asymptote is None:
            return {**result, "classification": "UNRESOLVED", "reasons": ["NO_COMPATIBLE_ASYMPTOTE"]}
        mismatch = abs(energies[-1] - asymptote["energy_hartree"]) * HARTREE_TO_EV
        result["plateau_minus_asymptote_ev"] = (energies[-1] - asymptote["energy_hartree"]) * HARTREE_TO_EV
        if mismatch <= policy.asymptote_match_tolerance_ev:
            return {**result, "classification": "DISSOCIATIVE_TO_ASYMPTOTE"}
        return {**result, "classification": "UNRESOLVED", "reasons": ["PLATEAU_DOES_NOT_MATCH_ASYMPTOTE"]}
    return {**result, "classification": "UNRESOLVED", "reasons": [f"TERMINATED_{terminated}"]}


def continue_component(*, edge: dict, direction: int, policy: OpenPECExtensionPolicy,
                       compute: Callable[[float, str], dict]) -> tuple[list[dict], str]:
    """Walk outward from the scout edge until a minimum, plateau or limit."""
    scan = [dict(edge)]
    r = float(edge["r_angstrom"])
    seed = edge["checkpoint_path"]
    for step in policy.step_schedule_angstrom:
        r_new = round(r + direction * step, 6)
        if r_new < policy.min_r_angstrom:
            return scan, "SHORT_R_LIMIT"
        point = compute(r_new, seed)
        scan.append(point)
        if not point["converged"]:
            return scan, "SCF_FAILURE"
        delta_ev = (point["energy_hartree"] - scan[-2]["energy_hartree"]) * HARTREE_TO_EV
        if point["energy_hartree"] > scan[-2]["energy_hartree"] + policy.minimum_energy_tolerance_hartree:
            return scan, "MINIMUM_BRACKETED"
        if direction > 0 and r_new >= policy.plateau_min_r_angstrom and abs(delta_ev) < policy.plateau_energy_tolerance_ev:
            return scan, "PLATEAU"
        r, seed = r_new, point["checkpoint_path"]
    return scan, "STEP_BUDGET_EXHAUSTED"


# ---------------------------------------------------------------- orchestration

def _open_components(summary: Mapping[str, Any]):
    for sector in summary["sectors"]:
        for scout in sector.get("manifold_pec_scouts", ()) if sector.get("tracking_mode") == "manifold" else ():
            status = (scout.get("minimum_scout") or {}).get("status")
            if status in OPEN_STATUSES:
                yield sector, scout, status


def _edge(sector: Mapping[str, Any], scout: Mapping[str, Any], status: str) -> tuple[dict, int]:
    points = sorted(scout["pec"]["points"], key=lambda p: float(p["r_angstrom"]))
    point, direction = (points[-1], +1) if status == "decreases_toward_higher_r" else (points[0], -1)
    manifolds = [m for m in sector.get("manifolds", ()) if m.get("manifold_id") == point["manifold_id"]]
    if len(manifolds) != 1 or not manifolds[0].get("unique_state_root_ids"):
        raise ValueError(f"{scout['component_id']}: edge manifold not uniquely linked to a root")
    roots = {r["root_id"]: r for r in sector.get("representative_roots", ())}
    root = roots[manifolds[0]["unique_state_root_ids"][0]]
    return {"r_angstrom": float(point["r_angstrom"]), "energy_hartree": float(point["energy_center_hartree"]),
            "checkpoint_path": root["checkpoint_path"], "s2": root.get("s2"), "converged": True,
            "internal_stable": root.get("internal_stable"), "root_id": root["root_id"],
            "source": "SCOUT_EDGE"}, direction


def _save_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        json.dump(data, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def run_open_pec_extension(*, summary: Mapping[str, Any], output_dir: Path,
                           policy: OpenPECExtensionPolicy = OpenPECExtensionPolicy(),
                           grid_level: int = 5,
                           atom_energy: Callable[..., dict] | None = None,
                           continuation_point: Callable[..., dict] | None = None) -> dict:
    """Execute EXTEND_OPEN_PEC for every open component; resumable per item."""
    atom_energy = atom_energy or pyscf_atom_energy
    continuation_point = continuation_point or pyscf_continuation_point
    output_dir = Path(output_dir)
    atom_a, atom_b = summary["atoms"]
    method = summary["sectors"][0]["method"]
    functional, basis = method["functional"], method["basis"]
    signature = sha256(json.dumps({"policy": asdict(policy), "method": method, "grid_level": grid_level,
                                   "atoms": [atom_a, atom_b]}, sort_keys=True).encode()).hexdigest()
    path = output_dir / "open_pec_extension.json"
    record: dict[str, Any] = {"signature": signature, "atomic": {}, "components": {}}
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("signature") == signature:
            record = previous
    record.update(system=summary.get("system"), atoms=[atom_a, atom_b], functional=functional,
                  basis=basis, policy=asdict(policy), validation_metadata_used_in_computation=False)

    for symbol in sorted({atom_a, atom_b}):
        for charge in (0, -1):
            n = _atomic_electrons(symbol, charge)
            for spin_2s in range(n % 2, min(n, policy.max_atomic_2s) + 1, 2):
                key = atom_key(symbol, charge, spin_2s)
                if key not in record["atomic"]:
                    record["atomic"][key] = atom_energy(symbol, charge, spin_2s, functional=functional,
                                                        basis=basis, grid_level=grid_level, policy=policy)
                    _save_json(path, record)

    checkpoint_dir = output_dir / "open_pec_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for sector, scout, status in _open_components(summary):
        cid = scout["component_id"]
        if cid in record["components"]:
            continue
        charge, spin_2s = int(sector["charge"]), int(sector["spin_2s"])
        edge, direction = _edge(sector, scout, status)
        print(f"[OPEN-PEC] {summary.get('system')} q={charge:+d} 2S={spin_2s} {cid} from R={edge['r_angstrom']:.3f} "
              f"{'outward' if direction > 0 else 'inward'}", flush=True)

        def compute(r_new: float, seed: str) -> dict:
            chk = checkpoint_dir / f"{cid}__R{r_new:.6f}.chk"
            try:
                return continuation_point(atoms=(atom_a, atom_b), charge=charge, spin_2s=spin_2s,
                                          r_angstrom=r_new, seed_checkpoint=seed, checkpoint=str(chk),
                                          functional=functional, basis=basis, grid_level=grid_level,
                                          policy=policy)
            except Exception as exc:  # recorded, never silently converted to a geometry
                return {"r_angstrom": r_new, "energy_hartree": float("nan"), "converged": False,
                        "error": f"{type(exc).__name__}: {exc}", "checkpoint_path": None}

        scan, terminated = continue_component(edge=edge, direction=direction, policy=policy, compute=compute)
        scan = [p for p in scan if p["energy_hartree"] == p["energy_hartree"]]  # drop NaN failure points
        asymptote = compatible_asymptote(record["atomic"], atom_a, atom_b, charge, spin_2s)
        record["components"][cid] = {
            "charge": charge, "spin_2s": spin_2s, "scout_status": status, "direction": direction,
            "scan": scan,
            **classify_continuation(scan, direction=direction, asymptote=asymptote, policy=policy,
                                    terminated=terminated),
        }
        _save_json(path, record)
    return record


# -------------------------------------------------------------- summary merge

def apply_open_pec_extension(summary: Mapping[str, Any], extension: Mapping[str, Any] | None) -> dict:
    """Return a summary copy with extension evidence merged into the open components."""
    merged = deepcopy(dict(summary))
    if not extension:
        return merged
    thresholds = MinimumScoutThresholds(
        energy_tolerance_hartree=float(extension["policy"]["minimum_energy_tolerance_hartree"]))
    components = extension.get("components", {})
    for sector in merged["sectors"]:
        for scout in sector.get("manifold_pec_scouts", ()) if sector.get("tracking_mode") == "manifold" else ():
            ext = components.get(scout.get("component_id"))
            if not ext or (scout.get("minimum_scout") or {}).get("status") not in OPEN_STATUSES:
                continue
            if ext["classification"] == "MINIMUM_BRACKETED":
                _merge_minimum(sector, scout, ext, thresholds)
            elif ext["classification"] == "DISSOCIATIVE_TO_ASYMPTOTE":
                scout["minimum_scout"] = {
                    **scout["minimum_scout"], "status": DISSOCIATIVE_STATUS, "candidates": [],
                    "scout_status_before_extension": scout["minimum_scout"]["status"],
                    "asymptote_evidence": {k: ext.get(k) for k in (
                        "asymptote_channel", "asymptote_hartree", "lowest_sampled_hartree",
                        "lowest_sampled_r_angstrom", "plateau_minus_asymptote_ev")},
                    "extended_r_max_angstrom": max(p["r_angstrom"] for p in ext["scan"]),
                }
    merged["open_pec_extension_applied"] = True
    return merged


def _merge_minimum(sector: dict, scout: dict, ext: Mapping[str, Any], thresholds) -> None:
    cid = scout["component_id"]
    for point in ext["scan"][1:]:
        digest = sha256(f"{cid}|{point['r_angstrom']:.6f}".encode()).hexdigest()[:16]
        manifold_id = f"open_pec_ext_{digest}"
        root_id = f"{cid}__ext__R{point['r_angstrom']:.6f}"
        scout["pec"]["points"].append({"r_angstrom": point["r_angstrom"], "energy_center_hartree": point["energy_hartree"],
                                       "energy_spread_mev": 0.0, "manifold_id": manifold_id})
        sector.setdefault("manifolds", []).append({
            "manifold_id": manifold_id, "r_angstrom": point["r_angstrom"],
            "energy_center_hartree": point["energy_hartree"], "member_root_ids": [root_id],
            "unique_state_root_ids": [root_id], "origin": EXTENSION_ORIGIN})
        sector.setdefault("representative_roots", []).append({
            "root_id": root_id, "checkpoint_path": point["checkpoint_path"], "r_angstrom": point["r_angstrom"],
            "energy_hartree": point["energy_hartree"], "s2": point.get("s2"),
            "internal_stable": point.get("internal_stable"), "origin_guess": EXTENSION_ORIGIN,
            "status": "CANONICALIZED", "charge": sector["charge"], "spin_2s": sector["spin_2s"]})
    scout["pec"]["points"].sort(key=lambda p: float(p["r_angstrom"]))
    pec = LocalPEC(component_id=cid, points=tuple(
        LocalPECPoint(root_id=p["manifold_id"], r_angstrom=float(p["r_angstrom"]),
                      energy_hartree=float(p["energy_center_hartree"])) for p in scout["pec"]["points"]))
    rescout = scout_local_pec_minimum(pec, thresholds=thresholds)
    scout["minimum_scout"] = {
        "status": rescout.status.value, "component_id": cid,
        "candidates": [asdict(c) for c in rescout.candidates],
        "adjacent_delta_e_hartree": list(rescout.adjacent_delta_e_hartree),
        "scout_status_before_extension": scout["minimum_scout"]["status"],
        "extended_by": EXTENSION_ORIGIN,
    }


# ------------------------------------------------------------- D09 reviewer

def asymptote_review_from_extension(merged_summary: Mapping[str, Any], extension: Mapping[str, Any] | None,
                                    selection_report: Mapping[str, Any]):
    """Automated scout-level D09 review built only from computed evidence.

    CLEARED requires: (1) the extension ran, (2) no state-selection component
    remains unresolved (every component is bracketed or dissociative onto its
    computed asymptote), and (3) a spin-compatible DFT fragment asymptote was
    computed for every scouted sector.  Otherwise UNRESOLVED with the reasons.
    The rationale names it automated and scout-level so it cannot be mistaken
    for a high-level asymptote study.
    """
    from .adaptive.model import Review, ReviewStatus

    if not extension:
        return Review(ReviewStatus.UNRESOLVED, (), "Open-PEC extension/asymptote evidence not computed")
    evidence_id = "open-pec-extension:" + sha256(
        json.dumps(extension, sort_keys=True, default=str).encode()).hexdigest()
    reasons = []
    for label, group in selection_report.get("charge_groups", {}).items():
        for item in group.get("unresolved_components", ()):
            reasons.append(f"{label} q={item.get('charge')} 2S={item.get('spin_2s')} "
                           f"{item.get('component_id')}: {item.get('minimum_status')}")
    atom_a, atom_b = merged_summary["atoms"]
    for sector in merged_summary["sectors"]:
        if compatible_asymptote(extension.get("atomic", {}), atom_a, atom_b,
                                int(sector["charge"]), int(sector["spin_2s"])) is None:
            reasons.append(f"no compatible fragment asymptote for q={sector['charge']} 2S={sector['spin_2s']}")
    if reasons:
        return Review(ReviewStatus.UNRESOLVED, (evidence_id,),
                      "AUTOMATED_SCOUT_LEVEL_ASYMPTOTE_REVIEW not cleared: " + "; ".join(reasons))
    return Review(ReviewStatus.CLEARED, (evidence_id,),
                  "AUTOMATED_SCOUT_LEVEL_ASYMPTOTE_REVIEW: every scouted component is bracketed or "
                  "dissociative onto its computed spin-compatible fragment asymptote; DFT scout level only")
