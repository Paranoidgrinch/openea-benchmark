"""Multireference term-gap route for MR-only states (Patch 54).

Stage 3 (ROHF -> CCSD(T)) cannot represent states that need several determinants
with equal weight (1Delta, 1Sigma-, open-shell singlets, ...).  The MR competitor
layer (Patch 51/53) only bounds them from below; a block it cannot exclude kept
the run UNRESOLVED.  This layer computes such a state relative to the selected
single-reference state of the same charge:

  E0(MR state) = E0(SR state, CCSD(T)/CBS + corrections) + gap0

* gap0 = adiabatic electronic gap + Delta ZPE, both from SC-NEVPT2 curves of the
  two states in their own (spin, C2v) blocks (``mr_competitors.block_energies``:
  state-averaged full-valence CASSCF, roots classified by Lambda and g/u);
* the state is followed as *the* root of its Lambda (and parity), never as the
  lowest root of its block; each minimum is found by a downhill walk on a
  0.025 A grid and a quartic through five points, ZPE harmonic at that minimum;
* basis: the gap at the TZ minima is recomputed with ``comparison_basis``; the
  larger basis gives the central value, the difference is the basis term;
* method term: ``method_uncertainty_ev`` (pilot value checked against measured
  1Delta/1Sigma+ term energies, scripts/openea_mr_term_gap_validation.py);
  ZPE term: ``zpe_relative_uncertainty`` of |Delta ZPE| plus, per state, the ZPE
  spread between the five-point quartic and a seven-point least-squares quartic.

The EA then uses whichever state is lower.  With gap0 in [lo, hi] the ground
state of that charge lies at offset min(0, gap0) in [min(0, lo), min(0, hi)]
relative to the SR state, so an overlapping interval widens the EA instead of
forcing a choice.  Corrections computed for the SR state (core-valence, scalar
relativity, post-CCSD(T)) are transferred with their full size as extra
uncertainty when the MR state can be the ground state.  First-order spin-orbit of
each state with S > 0 and Lambda > 0 (``spin_orbit_stage``, any Lambda since Patch
55) enters gap0, so E0 refers to the lowest spin-orbit level.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .electronic_ea_stage import HARTREE_TO_EV, _save_json
from .mr_competitors import (MRCompetitorPolicy, block_energies, candidate_block, candidate_lambda,
                             candidate_parity, selected_block)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
AMU_TO_ME = 1822.888486209


@dataclass(frozen=True)
class MRTermGapPolicy:
    """Explicit pilot policy, not a validated production threshold set."""
    basis: str = "def2-TZVPPD"
    comparison_basis: str = "def2-QZVPPD"
    step_angstrom: float = 0.025
    max_walk_steps: int = 40
    # 1.5 x the largest error of ten measured term energies (RMS 0.039, max 0.067 eV).
    method_uncertainty_ev: float = 0.10
    zpe_relative_uncertainty: float = 0.25
    max_memory_mb: int = 16000

    def block_policy(self, basis: str | None = None) -> MRCompetitorPolicy:
        return MRCompetitorPolicy(basis=basis or self.basis, max_memory_mb=self.max_memory_mb)


def matched_energy(record: Mapping[str, Any], lam: int | None, parity: int | None) -> float | None:
    """NEVPT2 energy of the root that is the state (Lambda and g/u assigned and equal)."""
    if record.get("status") != "COMPLETED":
        return None
    match = [x["e_hartree"] for x in record.get("roots") or ()
             if (lam is None or x.get("lambda") == lam) and (parity is None or x.get("parity") == parity)]
    return min(match) if match else None


def harmonic_zpe_ev(k_hartree_per_a2: float, masses: tuple[float, float]) -> float:
    mu = masses[0] * masses[1] / (masses[0] + masses[1]) * AMU_TO_ME
    k = k_hartree_per_a2 / ANGSTROM_TO_BOHR ** 2
    return 0.5 * float(np.sqrt(k / mu)) * HARTREE_TO_EV


def state_minimum(energy_at, r_start: float, policy: MRTermGapPolicy, grid_hint: Sequence[float] = ()) -> dict:
    """Downhill walk on the step grid from the lowest known point, quartic through five points.

    ``energy_at(r)`` returns the state's energy or None (state not found / failed)."""
    h = policy.step_angstrom
    cache: dict[float, float | None] = {}

    def e(r):
        r = round(r, 4)
        if r not in cache:
            cache[r] = energy_at(r)
        return cache[r]

    known = [(round(r, 4), e(r)) for r in grid_hint] or [(round(r_start, 4), e(r_start))]
    known = [(r, v) for r, v in known if v is not None]
    if not known:
        return {"status": "STATE_NOT_FOUND", "points": {str(k): v for k, v in cache.items()}}
    r0 = min(known, key=lambda x: x[1])[0]
    for _ in range(policy.max_walk_steps):
        left, mid, right = e(r0 - h), e(r0), e(r0 + h)
        if left is None or right is None:
            return {"status": "STATE_LOST_ALONG_CURVE", "r": r0,
                    "points": {str(k): v for k, v in sorted(cache.items())}}
        if left < mid:
            r0 = round(r0 - h, 4)
        elif right < mid:
            r0 = round(r0 + h, 4)
        else:
            break
    else:
        return {"status": "NO_MINIMUM_WITHIN_WALK", "points": {str(k): v for k, v in sorted(cache.items())}}
    rs = [round(r0 + i * h, 4) for i in (-2, -1, 0, 1, 2)]
    es = [e(r) for r in rs]
    if any(v is None for v in es):
        return {"status": "STATE_LOST_ALONG_CURVE", "r": r0,
                "points": {str(k): v for k, v in sorted(cache.items())}}
    p = np.polynomial.Polynomial.fit(np.array(rs) - r0, es, 4).convert()
    roots = [z.real for z in p.deriv().roots() if abs(z.imag) < 1e-10 and abs(z.real) <= h]
    if not roots:
        return {"status": "NO_MINIMUM_IN_FIT", "points": {str(k): v for k, v in sorted(cache.items())}}
    x = min(roots, key=lambda z: p(z))
    k = float(p.deriv(2)(x))
    if k <= 0:
        return {"status": "NO_MINIMUM_IN_FIT", "points": {str(k_): v for k_, v in sorted(cache.items())}}
    # Curvature check: least-squares quartic through seven points.  NEVPT2 curves
    # carry small kinks (~0.3-1 mEh, S2 1Delta) that a five-point interpolation turns
    # into a wrong force constant; the spread of the two fits enters the ZPE bound.
    rs7 = [round(r0 + i * h, 4) for i in (-3, -2, -1, 0, 1, 2, 3)]
    es7 = [e(r) for r in rs7]
    k7 = None
    if all(v is not None for v in es7):
        p7 = np.polynomial.Polynomial.fit(np.array(rs7) - r0, es7, 4).convert()
        k7 = float(p7.deriv(2)(x))
    return {"status": "COMPLETED", "r_e": r0 + float(x), "e_min_hartree": float(p(x)), "k_hartree_per_a2": k,
            "k7_hartree_per_a2": k7, "r_grid": r0, "points": {str(r): v for r, v in sorted(cache.items())}}


def term_gap(*, atoms: tuple[str, str], charge: int, reference: Mapping[str, Any], state: Mapping[str, Any],
             r_start: float, policy: MRTermGapPolicy, cache_dir: Path, grid_hint: Sequence[float] = (),
             block_fn=None, basis_cache_dirs: Mapping[str, Path] | None = None, soc_fn=None) -> dict:
    """Gap0 = E0(state) - E0(reference) of one charge, both as specific Lambda/(g/u) roots.

    ``reference``/``state``: {"label", "mult", "irrep", "lambda", "parity"}."""
    from .vibrational_stage import main_isotopologue_masses
    fn = block_fn or block_energies
    cache_dir = Path(cache_dir)
    m = main_isotopologue_masses(atoms)
    masses = (m.mass_a_u, m.mass_b_u)

    def curve(spec, basis):
        def energy_at(r):
            rec = fn(atoms, charge, spec["mult"], spec["irrep"], round(r, 4), policy.block_policy(basis),
                     Path((basis_cache_dirs or {}).get(basis, cache_dir / basis.lower()))
                     / f"q{charge}_m{spec['mult']}_{spec['irrep']}_r{r:.4f}.json")
            return matched_energy(rec, spec["lambda"], spec["parity"])
        return energy_at

    out: dict[str, Any] = {"charge": charge, "reference": dict(reference), "state": dict(state)}
    mins = {}
    for role, spec in (("reference", reference), ("state", state)):
        mins[role] = state_minimum(curve(spec, policy.basis), r_start, policy, grid_hint)
        out[f"{role}_minimum"] = mins[role]
        if mins[role]["status"] != "COMPLETED":
            return {**out, "status": f"{role.upper()}_{mins[role]['status']}"}
    gap_tz = (mins["state"]["e_min_hartree"] - mins["reference"]["e_min_hartree"]) * HARTREE_TO_EV
    # Larger basis at the TZ minima (grid point nearest r_e; same point for both bases).
    big = {}
    for role, spec in (("reference", reference), ("state", state)):
        r = mins[role]["r_grid"]
        big[role] = curve(spec, policy.comparison_basis)(r)
        tz_at = curve(spec, policy.basis)(r)
        if big[role] is None or tz_at is None:
            return {**out, "status": f"{role.upper()}_COMPARISON_BASIS_FAILED"}
        big[role] -= tz_at  # basis shift at that geometry
    basis_shift = (big["state"] - big["reference"]) * HARTREE_TO_EV
    gap_e = gap_tz + basis_shift
    zpe = {role: harmonic_zpe_ev(mins[role]["k_hartree_per_a2"], masses) for role in mins}
    fit_spread = {}
    for role in mins:
        k7 = mins[role].get("k7_hartree_per_a2")
        fit_spread[role] = (zpe[role] if not k7 or k7 <= 0
                            else abs(harmonic_zpe_ev(k7, masses) - zpe[role]))
    d_zpe = zpe["state"] - zpe["reference"]
    zpe_term = policy.zpe_relative_uncertainty * abs(d_zpe) + sum(fit_spread.values())
    half = abs(basis_shift) + policy.method_uncertainty_ev + zpe_term
    gap0 = gap_e + d_zpe
    # First-order spin-orbit (Patch 55): E0 means the lowest SO level of each state.
    so = {}
    for role, spec in (("reference", reference), ("state", state)):
        if spec["mult"] > 1 and (spec["lambda"] or 0) > 0:
            so[role] = (soc_fn or _state_soc)(atoms, mins[role]["r_e"], charge, spec["label"],
                                             cache_dir / "spin_orbit" / f"q{charge}_{spec['label']}")
            if so[role].get("delta_so_ev") is None:
                return {**out, "status": f"{role.upper()}_SPIN_ORBIT_{so[role].get('status')}", "spin_orbit": so}
        else:
            so[role] = {"status": "FIRST_ORDER_ZERO", "delta_so_ev": 0.0, "bound_ev": 0.0}
    d_so = so["state"]["delta_so_ev"] - so["reference"]["delta_so_ev"]
    gap0 += d_so
    half += so["state"]["bound_ev"] + so["reference"]["bound_ev"]
    return {**out, "status": "COMPLETED", "delta_so_ev": d_so,
            "spin_orbit": {k: {x: v.get(x) for x in ("status", "delta_so_ev", "bound_ev", "splitting_cm1")}
                           for k, v in so.items()},
            "gap_e_tz_ev": gap_tz, "basis_shift_ev": basis_shift,
            "gap_e_ev": gap_e, "zpe_ev": zpe, "zpe_fit_spread_ev": fit_spread, "zpe_term_ev": zpe_term,
            "delta_zpe_ev": d_zpe, "gap0_ev": gap0,
            "gap0_half_width_ev": half, "gap0_lower_ev": gap0 - half, "gap0_upper_ev": gap0 + half}


def _state_soc(atoms, r, charge, label, workdir) -> dict:
    from .spin_orbit_stage import SpinOrbitStagePolicy, state_spin_orbit_shift
    return state_spin_orbit_shift(atoms=atoms, r_angstrom=r, charge=charge, symmetry_label=label,
                                  irrep_nelec=None, workdir=Path(workdir), policy=SpinOrbitStagePolicy())


def _spec_from_label(label: str) -> dict | None:
    """'2Deltag' or 'q=-1 2Deltag' -> block and Lambda/(g/u) of the state."""
    text = label.split(" ", 1)[1] if label.startswith("q=") else label
    block = selected_block(text)
    if block is None:
        return None
    return {"label": text, "mult": block[0], "irrep": block[1], "lambda": candidate_lambda(text),
            "parity": candidate_parity(text)}


def _gap_task(atoms, charge, reference, state, r_start, policy, cache_dir, grid_hint, basis_cache_dirs=None):
    return term_gap(atoms=atoms, charge=charge, reference=reference, state=state, r_start=r_start,
                    policy=policy, cache_dir=Path(cache_dir), grid_hint=grid_hint,
                    basis_cache_dirs=basis_cache_dirs)


def run_mr_term_gap_stage(*, mr_competitors: Mapping[str, Any] | None, output_dir: Path,
                          policy: MRTermGapPolicy = MRTermGapPolicy(), task_fn=None,
                          competitor_blocks_dir: Path | None = None) -> dict:
    """Term gaps for every MR-only candidate the competitor layer did not exclude.

    ``competitor_blocks_dir``: the competitor layer's block caches (same level at
    ``policy.basis``), reused for the primary-basis points."""
    from .point_pool import run_parallel
    output_dir = Path(output_dir)
    record: dict[str, Any] = {"stage": "MR_TERM_GAP", "policy": asdict(policy),
                              "validation_metadata_used_in_computation": False}

    def done(**fields) -> dict:
        record.update(fields)
        _save_json(output_dir / "mr_term_gap.json", record)
        return record

    if not mr_competitors or mr_competitors.get("status") not in ("CLEARED", "UNRESOLVED_MR_COMPETITOR"):
        return done(status="NOT_ATTEMPTED",
                    reason=f"MR competitor layer {None if not mr_competitors else mr_competitors.get('status')}")
    open_labels = [(b["charge"], label) for b in mr_competitors.get("blocks", ()) if b["status"] != "EXCLUDED"
                   for label in b["candidates"]]
    if not open_labels:
        return done(status="CLEARED", gaps=[], reason="every MR-only candidate excluded by its lower bound")
    selected = mr_competitors.get("selected") or {}
    competitor_offsets = (mr_competitors.get("policy") or {}).get("r_offsets_angstrom") or ()
    jobs, meta = [], []
    for charge, label in open_labels:
        sel = selected.get(str(charge))
        ref, spec = (None if sel is None else _spec_from_label(sel["label"])), _spec_from_label(label)
        if ref is None or spec is None or candidate_block(label) is None:
            return done(status="UNRESOLVED_UNCLASSIFIED", label=label)
        r0 = float(sel["r"])
        hint = tuple(round(r0 + dr, 4) for dr in competitor_offsets)
        reuse = None if competitor_blocks_dir is None else {policy.basis: str(competitor_blocks_dir)}
        jobs.append((tuple(sel["atoms"]), charge, ref, spec, r0, policy, str(output_dir / "blocks"), hint, reuse))
        meta.append({"charge": charge, "label": label, "role": sel["role"]})
        print(f"[MR-GAP] q={charge} {label} vs {ref['label']}", flush=True)
    results = run_parallel(task_fn or _gap_task, jobs)
    gaps = [{**m, **res} for m, res in zip(meta, results)]
    failed = [g for g in gaps if g.get("status") != "COMPLETED"]
    return done(status="CLEARED" if not failed else "UNRESOLVED_GAP_NOT_COMPUTED", gaps=gaps)


def ground_state_offsets(gap_stage: Mapping[str, Any] | None) -> dict:
    """Per role: interval of E0(ground state) - E0(SR state) = min(0, lowest gap0)."""
    offsets = {"neutral": {"lower_ev": 0.0, "upper_ev": 0.0, "central_ev": 0.0, "mr_possible": [],
                           "mr_ground_state": None},
               "anion": {"lower_ev": 0.0, "upper_ev": 0.0, "central_ev": 0.0, "mr_possible": [],
                         "mr_ground_state": None}}
    for g in (gap_stage or {}).get("gaps", ()):
        if g.get("status") != "COMPLETED":
            continue
        o = offsets[g["role"]]
        o["lower_ev"] = min(o["lower_ev"], g["gap0_lower_ev"])
        o["upper_ev"] = min(o["upper_ev"], g["gap0_upper_ev"])
        if g["gap0_ev"] < o["central_ev"]:
            o["central_ev"], o["mr_ground_state"] = g["gap0_ev"], g["label"]
        if g["gap0_lower_ev"] < 0.0:
            o["mr_possible"].append(g["label"])
    return offsets


def apply_mr_term_gaps(adiabatic: Mapping[str, Any] | None, gap_stage: Mapping[str, Any] | None) -> dict | None:
    """EA0 with the ground state of each charge chosen among SR and MR states.

    EA = E0(neutral) - E0(anion): a lower neutral MR state lowers it, a lower anion
    MR state raises it.  Unchanged (no extra width) when no MR state can be lower."""
    if adiabatic is None or adiabatic.get("ea0_ev") is None:
        return None if adiabatic is None else dict(adiabatic)
    off = ground_state_offsets(gap_stage)
    n, a = off["neutral"], off["anion"]
    out = dict(adiabatic)
    out["mr_ground_state_offsets"] = off
    if not n["mr_possible"] and not a["mr_possible"]:
        return out
    # Corrections computed for the SR states, transferred to an MR state: full size as uncertainty.
    transfer = sum(abs(float(adiabatic.get(k) or 0.0)) for k in ("delta_cv_ev", "delta_sr_ev", "delta_post_ccsd_t_ev"))
    blocking = []
    with_soc = {g["label"] for g in (gap_stage or {}).get("gaps", ()) if g.get("spin_orbit")}
    for role in ("neutral", "anion"):
        for label in off[role]["mr_possible"]:
            spec = _spec_from_label(label)
            if spec and spec["mult"] > 1 and (spec["lambda"] or 0) > 0 and label not in with_soc:
                blocking.append(f"{role} MR state {label}: first-order spin-orbit not computed")
    ea0 = float(adiabatic["ea0_ev"]) + n["central_ev"] - a["central_ev"]
    lower = float(adiabatic["ea0_lower_ev"]) + n["lower_ev"] - a["upper_ev"] - transfer
    upper = float(adiabatic["ea0_upper_ev"]) + n["upper_ev"] - a["lower_ev"] + transfer
    out.update(ea0_ev=ea0, ea0_lower_ev=lower, ea0_upper_ev=upper, mr_transfer_uncertainty_ev=transfer,
               ea0_sr_states_ev=adiabatic["ea0_ev"])
    if blocking:
        out.update(status="NEED_MORE_EVIDENCE_MR_STATE_SPIN_ORBIT", mr_blocking=blocking)
    elif out.get("status") == "CLEARED" and lower <= 0.0:
        out.update(status="NEED_MORE_EVIDENCE_BINDING_UNDETERMINED" if upper >= 0.0 else "ANION_UNBOUND",
                   binding="UNDETERMINED" if upper >= 0.0 else "UNBOUND")
    return out
