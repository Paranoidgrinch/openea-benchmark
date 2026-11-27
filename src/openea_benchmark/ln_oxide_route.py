"""Route B in the workflow: lanthanide monoxides with an open 4f shell (axial determinants).

Which molecules (``route_b_element``): LnO with Ln = Ce ... Yb.  Every such pair has ionic
configurations Ln^(2-q)+ O^2- with an open 4f shell.  Real-orbital determinants represent a
Hund f^n term only for n in {0, 1, 6, 7, 8, 13, 14}, and the Stage-3 chain cannot rank the
f arrangements; complex (L_z-adapted) determinants represent the maximum-M_L component of
every f^n term as one determinant (``axial``).  La and Lu (no open f) stay on the standard
route.  Other Ln compounds and the actinides fail closed (ROUTE_B_NOT_AVAILABLE): the ionic
closed-shell model of ``axial.ln_oxide`` is specific to O^2-.

Per charge q (0, -1), all CCSD(T) frozen core, X2C-1e, axial m blocks:

1. Screen: every Hund high-spin f determinant of the ionic f counts (f^k-ns 6s^ns,
   ``screen_candidates``) at R_SCREEN_ANGSTROM in aug-cc-pVDZ; free SCF following the ion's
   4f and 6s, frozen-f SCF where that slides 4f -> 5d or fails.  CC that does not converge
   is retried with shifted denominators (``axial.cc``).
2. Candidates: converged configurations within SCREEN_WINDOW_EV of the lowest.  A
   configuration whose CC fails even then blocks the route unless its SCF energy lies more
   than UNCONVERGED_SCF_MARGIN_EV above the lowest SCF energy (TbO 4f9 vs 4f8 6s: SCF
   +6.3 eV, correlation recovers 2.4 eV of it).
3. Lowest Omega: every candidate gets Delta_SO of its lowest Omega component (X2C-1e two-
   component GHF + GCCSD(T), uncontracted Ln aDZ, at R_SCREEN); the lowest screen + SO
   energy is selected.  First-order SO of an f shell (Tb: zeta ~ 0.2 eV) can reorder the
   M_L configurations of one f count.
4. Selected configuration: CCSD(T)/aTZ walk downhill in WALK_STEP_ANGSTROM steps, quartic
   through five points: R_e, harmonic omega_e, ZPE.  Candidates within
   COMPETITOR_RECHECK_EV of it (after SO) are recomputed at R_e in aTZ (+ their SO); a lower
   one is selected instead and walked (once).
5. At R_e: CCSD(T)/aQZ (correlation T/Q-extrapolated X^-3, SCF from QZ), Delta_T = CCSDT -
   CCSD(T) in aDZ, Delta_CV (Ln 4s4p4d + O 1s correlated, aug-cc-pwCVTZ), Delta_SO.

EA_0 = E_0(neutral) - E_0(anion); bounds (worst case, added): CBS 0.5 |EA(CBS) - EA(QZ)|,
Delta_T 0.5 |Delta_T| plus the beyond-CCSDT allowance max(Q_ALLOWANCE_MIN_EV, |Delta_T|) of
the standard route without quadruples, Delta_CV 0.25 |Delta_CV| + 0.005 eV, Delta_SO
0.3 |Delta_SO| (spin-orbit layer), Delta_ZPE 0.2 |Delta_ZPE| + 0.001 eV (harmonic), and a
competitor closer than the bounds widens the interval by its gap.  Ln atomic IPs with the
same composite (CBS + Delta_T + Delta_CV): Pr -0.004, Yb +0.003 eV against NIST.

Measured molecular data are not used.  Validation molecule during development: TbO (no
tests on the v2 hold-out CeO, GdO).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

H2EV = 27.211386245988
AMU_TO_ME = 1822.888486209
ANG_TO_BOHR = 1.0 / 0.529177210903
CM1_PER_HARTREE = 219474.6313632

ROUTE_B_LANTHANIDES = ("Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb")
LANTHANIDES_ALL = ("La",) + ROUTE_B_LANTHANIDES + ("Lu",)
ACTINIDES = ("Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr")

R_SCREEN_ANGSTROM = 1.82          # LnO screen distance; the aTZ walk finds R_e from here
SCREEN_WINDOW_EV = 0.5
COMPETITOR_RECHECK_EV = 0.1
UNCONVERGED_SCF_MARGIN_EV = 4.0
WALK_STEP_ANGSTROM = 0.02
WALK_MAX_STEPS = 30
SO_VIRTUAL_CUTOFF_HARTREE = 5.0
Q_ALLOWANCE_MIN_EV = 0.02
# Memory reserved per point at the gate (measured RSS: TbO aTZ 9 GB, Nd awCVTZ 46 GB).
RESERVE_MB = {"D": 15000, "T": 30000, "Q": 100000, "CV": 60000, "SO": 40000}


def route_b_element(atom_a: str, atom_b: str) -> str | None:
    """The lanthanide of an LnO pair that Route B treats, else None."""
    pair = {atom_a, atom_b}
    for ln in ROUTE_B_LANTHANIDES:
        if pair == {ln, "O"}:
            return ln
    return None


def route_b_unavailable(atom_a: str, atom_b: str) -> str | None:
    """Reason why an open-f pair cannot run on any route yet (None: not such a pair)."""
    pair = {atom_a, atom_b}
    open_f = [el for el in pair if el in ROUTE_B_LANTHANIDES or (el in ACTINIDES and el not in ("Ac", "Th"))]
    if not open_f or route_b_element(atom_a, atom_b):
        return None
    return (f"ROUTE_B_NOT_AVAILABLE: open-f element {open_f[0]} only as a monoxide of a lanthanide "
            f"(axial ionic model Ln^(2-q)+ O^2-)")


def screen_candidates(element: str, charge: int):
    """Ln^(2-q)+: k valence electrons beyond [Xe]; f^(k-ns) 6s^ns, ns = 0..2 (6s^2 only for
    the anion, 6s^0 not for the anion)."""
    from .axial.ln_oxide import XE_CORE, Z, f_configurations
    k = Z[element] - XE_CORE - (2 + charge)
    out = []
    for ns in (0, 1, 2):
        nf = k - ns
        if 0 <= nf <= 14 and (ns < 2 or charge < 0) and not (ns == 0 and charge < 0):
            out += f_configurations(nf, ns)
    return out


@dataclass(frozen=True)
class PointSpec:
    element: str
    config: dict              # LnConfig fields
    r: float
    basis: str                # D, T, Q
    methods: tuple = ("ccsd(t)",)
    freeze_f: bool = False
    cv: bool = False
    so: bool = False

    def key(self) -> str:
        from .axial.ln_oxide import LnConfig
        label = LnConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in self.config.items()}).label()
        kind = "so" if self.so else ("cv" if self.cv else "")
        return (f"{self.element}O_{label.replace(' ', '_')}_{kind}{self.basis}_{self.r:.4f}_"
                f"{'+'.join(self.methods)}{'_frozenf' if self.freeze_f else ''}")

    def reserve_mb(self) -> float:
        return RESERVE_MB["SO" if self.so else "CV" if self.cv else self.basis]


def _config(spec: PointSpec):
    from .axial.ln_oxide import LnConfig
    return LnConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec.config.items()})


def compute_point(spec: PointSpec) -> dict:
    """One axial point (SCF + CC, or the SO increment) as a JSON record; failures recorded."""
    from .axial.ln_oxide import oxide_point, soc_increment
    t0 = time.time()
    cfg = _config(spec)
    if spec.so:
        so = soc_increment(spec.element, cfg, spec.r, spec.basis, freeze_f=spec.freeze_f,
                           virtual_cutoff=SO_VIRTUAL_CUTOFF_HARTREE)
        rec = {"status": "COMPLETED" if so.get("status") == "COMPLETED" else so.get("status", "FAILED"),
               "delta_so_hartree": (so.get("delta_so_hartree") or {}).get("ccsd(t)"),
               "delta_so_hf_hartree": (so.get("delta_so_hartree") or {}).get("hf"),
               "omega_twice": so.get("omega_twice")}
    else:
        p = oxide_point(spec.element, cfg, spec.r, spec.basis, spec.methods, freeze_f=spec.freeze_f, cv=spec.cv,
                        max_cycle=300)
        ok = p.converged and p.cc_status == "CONVERGED"
        rec = {"status": "COMPLETED" if ok else ("SCF_NOT_CONVERGED" if not p.converged else "CC_NOT_CONVERGED"),
               "e_scf": p.e_scf, "scf_converged": p.converged, "correlation": p.correlation,
               "cc_status": p.cc_status, "cc_diagnostics": p.cc_diagnostics, "s2": p.s2,
               "f_purity": p.f_purity, "scf_model": p.scf_model}
    return {"spec": {**asdict(spec), "methods": list(spec.methods)}, "seconds": time.time() - t0, **rec}


def cached_point(spec: PointSpec, cache: Path, fn: Callable[[PointSpec], dict] = compute_point) -> dict:
    path = Path(cache) / f"{spec.key()}.json"
    if path.is_file():
        return json.loads(path.read_text())
    rec = fn(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, default=float))
    return rec


def _run_points(specs: list[PointSpec], cache: Path, fn) -> list[dict]:
    """Points on the shared pool, each reserving its estimate at the memory gate."""
    from .point_pool import run_parallel
    if not specs:
        return []
    return run_parallel(cached_point, [(s, cache, fn) for s in specs],
                        reserve_mb=[s.reserve_mb() for s in specs], hold_reservations=True)


def total(rec: Mapping[str, Any], method: str = "ccsd(t)") -> float:
    return float(rec["e_scf"]) + float(rec["correlation"][method])


def screen_charge(element: str, charge: int, cache: Path, fn) -> dict:
    """Step 1-2 for one charge: screen records, candidates within the window, blocking
    unconverged configurations."""
    cfgs = screen_candidates(element, charge)
    free = _run_points([PointSpec(element, asdict(c), R_SCREEN_ANGSTROM, "D") for c in cfgs], cache, fn)
    recs = []
    retry = []
    for c, rec in zip(cfgs, free):
        purity = min((rec.get("f_purity") or {}).values(), default=1.0)
        if rec["status"] != "COMPLETED" or purity < 0.8:
            retry.append(c)
        else:
            recs.append((c, rec))
    frozen = _run_points([PointSpec(element, asdict(c), R_SCREEN_ANGSTROM, "D", freeze_f=True) for c in retry],
                         cache, fn)
    recs += list(zip(retry, frozen))
    ok = [(c, r) for c, r in recs if r["status"] == "COMPLETED"]
    if not ok:
        return {"status": "NO_CONVERGED_CONFIGURATION", "n_screened": len(cfgs)}
    e0 = min(total(r) for _, r in ok)
    scf_vals = [float(r["e_scf"]) for _, r in recs if r.get("e_scf") is not None and r.get("scf_converged")]
    s0 = min(scf_vals) if scf_vals else None
    rows = []
    for c, r in recs:
        row = {"label": c.label(), "config": asdict(c), "status": r["status"], "scf_model": r.get("scf_model"),
               "e_scf": r.get("e_scf"), "e_ccsd_t": total(r) if r["status"] == "COMPLETED" else None,
               "freeze_f": r["spec"]["freeze_f"]}
        if row["e_ccsd_t"] is not None:
            row["gap_ev"] = (row["e_ccsd_t"] - e0) * H2EV
        if s0 is not None and r.get("e_scf") is not None and r.get("scf_converged"):
            row["scf_gap_ev"] = (float(r["e_scf"]) - s0) * H2EV
        rows.append(row)
    blocking = [row["label"] for row in rows if row["status"] != "COMPLETED"
                and (row.get("scf_gap_ev") is None or row["scf_gap_ev"] < UNCONVERGED_SCF_MARGIN_EV)]
    candidates = sorted((row for row in rows if row.get("gap_ev") is not None and row["gap_ev"] <= SCREEN_WINDOW_EV),
                        key=lambda row: row["gap_ev"])
    return {"status": "BLOCKED_UNCONVERGED" if blocking else "SCREENED", "rows": rows,
            "candidates": candidates, "blocking": blocking, "n_screened": len(cfgs)}


def quartic_minimum(points: Mapping[float, float], masses: tuple[float, float]) -> dict:
    """R_e, E_min, omega_e and harmonic ZPE from a quartic through the five points around
    the lowest one (Angstrom, Hartree)."""
    rs = sorted(points)
    i = min(range(len(rs)), key=lambda j: points[rs[j]])
    five = rs[i - 2:i + 3]
    c = np.polyfit(np.array(five) - five[2], [points[x] for x in five], 4)
    step = max(abs(five[1] - five[0]), 1e-6)
    roots = [z.real for z in np.roots(np.polyder(c)) if abs(z.imag) < 1e-10 and abs(z.real) <= 2 * step]
    dr = min(roots, key=lambda z: np.polyval(c, z))
    k2 = np.polyval(np.polyder(c, 2), dr)
    mu = masses[0] * masses[1] / sum(masses) * AMU_TO_ME
    omega = np.sqrt(k2 / ANG_TO_BOHR ** 2 / mu)
    return {"r_e": float(five[2] + dr), "e_min": float(np.polyval(c, dr)), "omega_cm1": float(omega * CM1_PER_HARTREE),
            "zpe_ev": float(0.5 * omega * H2EV), "r_grid": rs}


def walk_minimum(element: str, row: Mapping[str, Any], cache: Path, fn, masses) -> dict:
    """Step 4: aTZ downhill walk from R_SCREEN_ANGSTROM (five points around the lowest)."""
    pts = {round(R_SCREEN_ANGSTROM + d * WALK_STEP_ANGSTROM, 4): None for d in (-2, -1, 0, 1, 2)}
    for _ in range(WALK_MAX_STEPS):
        missing = [r for r, e in pts.items() if e is None]
        for r, rec in zip(missing, _run_points([PointSpec(element, row["config"], r, "T", freeze_f=row["freeze_f"])
                                                for r in missing], cache, fn)):
            if rec["status"] != "COMPLETED":
                raise RuntimeError(f"{row['label']} aTZ R={r:.4f}: {rec['status']}")
            pts[r] = total(rec)
        rs = sorted(pts)
        i = min(range(len(rs)), key=lambda j: pts[rs[j]])
        if 2 <= i <= len(rs) - 3:
            break
        pts[round(rs[0] - WALK_STEP_ANGSTROM if i < 2 else rs[-1] + WALK_STEP_ANGSTROM, 4)] = None
    else:
        raise RuntimeError(f"{row['label']}: no aTZ minimum within {WALK_MAX_STEPS} steps")
    return quartic_minimum(pts, masses)


def so_spec(element: str, row: Mapping[str, Any], r: float) -> PointSpec:
    return PointSpec(element, row["config"], round(r, 4), "D", methods=(), freeze_f=row["freeze_f"], so=True)


def state_energies(element: str, row: Mapping[str, Any], r_e: float, cache: Path, fn) -> dict:
    """Step 5 at R_e for one selected configuration."""
    r = round(r_e, 4)
    specs = [PointSpec(element, row["config"], r, "T", freeze_f=row["freeze_f"]),
             PointSpec(element, row["config"], r, "Q", freeze_f=row["freeze_f"]),
             PointSpec(element, row["config"], r, "D", methods=("ccsd(t)", "ccsdt"), freeze_f=row["freeze_f"]),
             PointSpec(element, row["config"], r, "T", freeze_f=row["freeze_f"], cv=True),
             so_spec(element, row, r)]
    tz, qz, dz, cv, so = _run_points(specs, cache, fn)
    failed = [s.key() for s, rec in zip(specs, (tz, qz, dz, cv, so)) if rec["status"] != "COMPLETED"]
    if failed:
        return {"status": "POINTS_FAILED", "failed": failed}
    e_cbs = float(qz["e_scf"]) + (4 ** 3 * qz["correlation"]["ccsd(t)"] - 3 ** 3 * tz["correlation"]["ccsd(t)"]) / (4 ** 3 - 3 ** 3)
    return {"status": "COMPLETED", "r_e": r, "e_tz": total(tz), "e_qz": total(qz), "e_cbs": e_cbs,
            "delta_t": dz["correlation"]["ccsdt"] - dz["correlation"]["ccsd(t)"],
            "delta_cv": cv["correlation"]["cv_ccsd(t)"] - cv["correlation"]["val_ccsd(t)"],
            "delta_so": float(so["delta_so_hartree"]), "omega_twice": so.get("omega_twice")}


def select_charge(element: str, charge: int, screen: Mapping[str, Any], cache: Path, fn, masses) -> dict:
    """Steps 3-4: lowest Omega among the candidates, aTZ minimum, competitor recheck."""
    cands = list(screen["candidates"])
    sos = _run_points([so_spec(element, row, R_SCREEN_ANGSTROM) for row in cands], cache, fn)
    ranked = []
    for row, so in zip(cands, sos):
        if so["status"] != "COMPLETED":
            return {"status": "SO_FAILED", "config": row["label"], "reason": so["status"]}
        ranked.append({**row, "delta_so_screen": float(so["delta_so_hartree"]),
                       "e_screen_so": row["e_ccsd_t"] + float(so["delta_so_hartree"])})
    ranked.sort(key=lambda row: row["e_screen_so"])
    selected, history = ranked[0], []
    for _ in range(2):
        mn = walk_minimum(element, selected, cache, fn, masses)
        history.append({"label": selected["label"], **mn})
        e_sel = mn["e_min"] + selected["delta_so_screen"]
        rivals = [row for row in ranked if row is not selected
                  and (row["e_screen_so"] - selected["e_screen_so"]) * H2EV <= COMPETITOR_RECHECK_EV]
        checks = []
        for row, rec in zip(rivals, _run_points([PointSpec(element, row["config"], round(mn["r_e"], 4), "T",
                                                           freeze_f=row["freeze_f"]) for row in rivals], cache, fn)):
            if rec["status"] != "COMPLETED":
                return {"status": "COMPETITOR_FAILED", "config": row["label"], "reason": rec["status"]}
            checks.append({**row, "gap_tz_so_ev": (total(rec) + row["delta_so_screen"] - e_sel) * H2EV})
        lower = [c for c in checks if c["gap_tz_so_ev"] < 0.0]
        if not lower:
            return {"status": "SELECTED", "selected": selected, "minimum": mn, "competitors": checks,
                    "ranked": ranked, "history": history}
        selected = min(lower, key=lambda c: c["gap_tz_so_ev"])
    return {"status": "SELECTION_NOT_STABLE", "history": history}


def assemble(neutral: Mapping[str, Any], anion: Mapping[str, Any], sel_n: Mapping[str, Any],
             sel_a: Mapping[str, Any]) -> dict:
    """EA_0 and its worst-case interval from the two states (Hartree inputs)."""
    ea_cbs = (neutral["e_cbs"] - anion["e_cbs"]) * H2EV
    ea_qz = (neutral["e_qz"] - anion["e_qz"]) * H2EV
    terms = {"ea_cbs_ev": ea_cbs,
             "delta_t_ev": (neutral["delta_t"] - anion["delta_t"]) * H2EV,
             "delta_cv_ev": (neutral["delta_cv"] - anion["delta_cv"]) * H2EV,
             "delta_so_ev": (neutral["delta_so"] - anion["delta_so"]) * H2EV,
             "delta_zpe_ev": sel_n["minimum"]["zpe_ev"] - sel_a["minimum"]["zpe_ev"]}
    bounds = {"cbs": 0.5 * abs(ea_cbs - ea_qz),
              "delta_t": 0.5 * abs(terms["delta_t_ev"]),
              "beyond_ccsdt": max(Q_ALLOWANCE_MIN_EV, abs(terms["delta_t_ev"])),
              "delta_cv": 0.25 * abs(terms["delta_cv_ev"]) + 0.005,
              "delta_so": 0.3 * abs(terms["delta_so_ev"]),
              "delta_zpe": 0.2 * abs(terms["delta_zpe_ev"]) + 0.001}
    ea0 = ea_cbs + sum(terms[k] for k in ("delta_t_ev", "delta_cv_ev", "delta_so_ev", "delta_zpe_ev"))
    half = sum(bounds.values())
    lower, upper = ea0 - half, ea0 + half
    # A competitor of either charge closer than the bounds could be the lowest level instead:
    # EA = E0(neutral) - E0(anion), so a lower neutral lowers, a lower anion raises EA_0.
    widen = {}
    for role, sel, sign in (("neutral", sel_n, -1.0), ("anion", sel_a, +1.0)):
        gaps = [c["gap_tz_so_ev"] for c in sel.get("competitors", ()) if c["gap_tz_so_ev"] < half]
        if gaps:
            shift = half - min(gaps)
            widen[role] = shift
            if sign < 0:
                lower -= shift
            else:
                upper += shift
    return {"ea0_ev": ea0, "ea0_lower_ev": lower, "ea0_upper_ev": upper, "terms": terms, "bounds_ev": bounds,
            "competitor_widening_ev": widen}


def run_route_b(*, element: str, output_dir: Path, point_fn: Callable[[PointSpec], dict] = compute_point) -> dict:
    """The whole route for one LnO pair; writes route_b.json and returns the record."""
    from pyscf.data.elements import COMMON_ISOTOPE_MASSES, charge as z_of
    output_dir = Path(output_dir)
    cache = output_dir / "points"
    masses = (COMMON_ISOTOPE_MASSES[z_of(element)], COMMON_ISOTOPE_MASSES[z_of("O")])   # most abundant isotopes
    record: dict[str, Any] = {"stage": "ROUTE_B_AXIAL_LN_OXIDE", "element": element,
                              "policy": {"r_screen_angstrom": R_SCREEN_ANGSTROM, "screen_window_ev": SCREEN_WINDOW_EV,
                                         "competitor_recheck_ev": COMPETITOR_RECHECK_EV,
                                         "unconverged_scf_margin_ev": UNCONVERGED_SCF_MARGIN_EV},
                              "validation_metadata_used_in_computation": False}

    def done(**fields):
        record.update(fields)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "route_b.json").write_text(json.dumps(record, indent=1, default=float))
        return record

    def chain(role: str, q: int) -> dict:
        """Screen, selection and R_e energies of one charge (the two charges run side by side)."""
        print(f"[ROUTE B] {element}O {role}: f-configuration screen", flush=True)
        out = {"screen": screen_charge(element, q, cache, point_fn)}
        if out["screen"]["status"] != "SCREENED":
            return {**out, "status": "UNRESOLVED",
                    "reason": f"screen: {out['screen'].get('blocking') or out['screen']['status']}"}
        print(f"[ROUTE B] {element}O {role}: lowest Omega, aTZ minimum", flush=True)
        out["selection"] = select_charge(element, q, out["screen"], cache, point_fn, masses)
        if out["selection"]["status"] != "SELECTED":
            return {**out, "status": "UNRESOLVED", "reason": f"selection: {out['selection']['status']}"}
        sel = out["selection"]
        print(f"[ROUTE B] {element}O {role}: {sel['selected']['label']} at R_e {sel['minimum']['r_e']:.4f} A "
              f"(aQZ, Delta_T, Delta_CV, Delta_SO)", flush=True)
        out["state"] = state_energies(element, sel["selected"], sel["minimum"]["r_e"], cache, point_fn)
        if out["state"]["status"] != "COMPLETED":
            return {**out, "status": "UNRESOLVED", "reason": f"R_e points: {out['state']}"}
        return {**out, "status": "COMPLETED"}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {role: pool.submit(chain, role, q) for role, q in (("neutral", 0), ("anion", -1))}
        chains = {role: f.result() for role, f in futures.items()}
    record["charges"] = chains
    open_ = {role: c["reason"] for role, c in chains.items() if c["status"] != "COMPLETED"}
    if open_:
        return done(status="UNRESOLVED", reason="; ".join(f"{role} {r}" for role, r in open_.items()))
    result = assemble(chains["neutral"]["state"], chains["anion"]["state"],
                      chains["neutral"]["selection"], chains["anion"]["selection"])
    return done(status="CLEARED", **result)


def final_verdict(route: Mapping[str, Any]) -> dict:
    """Outcome in the format of ``final_resolver``."""
    if route.get("status") != "CLEARED":
        return {"outcome": "UNRESOLVED", "blocking": [f"route B: {route.get('reason')}"], "warnings": []}
    lo, hi = route["ea0_lower_ev"], route["ea0_upper_ev"]
    warnings = [f"route B (axial determinants, lanthanide monoxide); competitor widening "
                f"{route['competitor_widening_ev']}"] if route.get("competitor_widening_ev") else \
        ["route B (axial determinants, lanthanide monoxide)"]
    if hi < 0.0:
        return {"outcome": "UNBOUND", "blocking": [], "warnings": warnings,
                "reason": "EA_0 interval below zero (route B)"}
    if lo <= 0.0:
        return {"outcome": "EA_NEAR_ZERO", "blocking": [], "warnings": warnings, "ea_ev": route["ea0_ev"],
                "ea_lower_ev": lo, "ea_upper_ev": hi,
                "reason": "EA_0 interval contains zero (route B)",
                "scope": "adiabatic EA_0, lowest spin-orbit levels, main isotopologue"}
    return {"outcome": "EA", "blocking": [], "warnings": warnings, "ea_ev": route["ea0_ev"],
            "ea_lower_ev": lo, "ea_upper_ev": hi,
            "scope": "adiabatic EA_0, lowest spin-orbit levels, main isotopologue"}
