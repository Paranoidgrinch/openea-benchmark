"""Multireference nuclear curve for the zero-point layer (Patch 50).

When no single ROHF reference carries a state over the bond-length range the
v=0 wavefunction needs (AlO X2Sigma+: the seed branch ends at 1.69 A, the
followed branch changes character at 1.46 A), the CCSD(T) curve cannot be
extended.  The zero-point correction then comes from a multireference curve:

* full-valence CASSCF (ns, np of each atom; 1s for H/He; noble-gas cores
  inactive; the rule of ``spin_orbit_stage.full_valence_space``) in the state's
  spin and C2v symmetry block (homonuclear D2h labels are mapped to C2v), lowest
  root; strongly contracted NEVPT2 on top;
* the same primary basis as the Stage-3 curve (def2-TZVPPD), the same method for
  neutral and anion (the nuclear solver requires it);
* a walk outward from the TZ equilibrium in 0.025 A steps, every point seeded
  from its converged neighbour (projected CASSCF orbitals), until the curve lies
  ``clearance_ev`` above its minimum on both sides;
* continuity: every interior NEVPT2 energy within 0.5 mEh of the cubic through
  its four nearest neighbours, active natural occupations changing by < 0.1
  between neighbours.  A failed check stops the layer (fail closed).

Model uncertainty: the relative harmonic-frequency difference between the MR
curve and the selected CCSD(T) curve (quadratic fits around their minima) times
that role's ZPE, added to the Delta_ZPE half-width.  The electronic EA stays the
CCSD(T)/CBS value; only Delta_ZPE comes from this curve.

Vibrational binding of the anion (``mr_vibrational_binding``): CCSD(T) electronic
dissociation margin minus twice the MR anion ZPE must stay positive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .electronic_ea_stage import HARTREE_TO_EV, _save_json
from .spin_orbit_stage import D2H_TO_C2V, full_valence_space, parse_term_symbol

C2V_IRREPS = ("A1", "B1", "B2", "A2")  # order of full_valence_space tuples (Molcas a1 b1 b2 a2)


@dataclass(frozen=True)
class MRCurvePolicy:
    """Explicit pilot policy, not a validated production threshold set."""
    basis: str = "def2-TZVPPD"
    step_angstrom: float = 0.025
    clearance_ev: float = 0.6
    min_points_per_side: int = 4
    max_points_per_side: int = 24
    max_kink_hartree: float = 5.0e-4
    max_occupation_jump: float = 0.1
    max_memory_mb: int = 16000
    # Larger clearances tried in turn while the nuclear solver asks for a wider curve.
    extension_clearances_ev: tuple[float, ...] = (0.9, 1.2, 1.5)


def target_c2v_irrep(label: str) -> str | None:
    """C2v irrep of one component of a Sigma+/Sigma-/Pi state (lowest root is used)."""
    term = parse_term_symbol(label)
    if term is None:
        return None
    _, lam = term
    if lam == 0:
        return "A2" if label.endswith("-") else "A1"
    return "B1" if lam == 1 else None


def _c2v_counts(irrep_nelec: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    out = {k: [0, 0] for k in C2V_IRREPS}
    for irrep, (na, nb) in irrep_nelec.items():
        key = D2H_TO_C2V.get(irrep, irrep)
        out[key][0] += int(na)
        out[key][1] += int(nb)
    return {k: (a, b) for k, (a, b) in out.items()}


def _mol(atoms, r, charge, spin_2s, basis, policy):
    from pyscf import gto
    return gto.M(atom=[(atoms[0], (0, 0, 0)), (atoms[1], (0, 0, float(r)))], basis=basis, charge=charge,
                 spin=spin_2s, unit="Angstrom", symmetry="C2v", verbose=0, max_memory=policy.max_memory_mb)


def _point(atoms, r, charge, spin_2s, irrep_nelec, wfnsym, space, basis, mo_guess, mol_guess, chk, policy,
           projected=None):
    from pyscf import mcscf, mrpt, scf
    mol = _mol(atoms, r, charge, spin_2s, basis, policy)
    ncas = sum(space["ras2"])
    nelecas = space["active_electrons"]
    ncore = {k: v for k, v in zip(C2V_IRREPS, space["inactive"]) if v}
    nact = {k: v for k, v in zip(C2V_IRREPS, space["ras2"]) if v}
    if mo_guess is not None:
        try:
            return _point(atoms, r, charge, spin_2s, irrep_nelec, wfnsym, space, basis, None, None, chk, policy,
                          projected=(mo_guess, mol_guess))
        except RuntimeError:
            pass  # the neighbour's orbitals did not converge here: restart from ROHF (continuity is checked)
    # A converged SCF object at every point: NEVPT2 reads it (an unconverged
    # placeholder gave Hartree-sized errors).  CASSCF start orbitals come from the
    # neighbouring point when given, else from this SCF.
    mf = scf.ROHF(mol) if spin_2s else scf.RHF(mol)
    counts = _c2v_counts(irrep_nelec)
    mf.irrep_nelec = ({k: v for k, v in counts.items() if sum(v)} if spin_2s
                      else {k: sum(v) for k, v in counts.items() if sum(v)})
    guess = scf.rohf.init_guess_by_chkfile if spin_2s else scf.hf.init_guess_by_chkfile
    mf.kernel(dm0=guess(mol, chk, project=True))
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    if projected is None:
        mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, nact, ncore)
    else:
        mo = mcscf.project_init_guess(mc, projected[0], projected[1])
    mc.fcisolver.wfnsym = wfnsym
    mc.fix_spin_(ss=0.5 * spin_2s * (0.5 * spin_2s + 1))
    mc.max_cycle_macro = 200
    mc.kernel(mo)
    if not mc.converged:
        raise RuntimeError(f"CASSCF not converged at R={r}")
    e_pt2 = float(mrpt.NEVPT(mc).kernel())
    occ = np.sort(np.asarray(mc.mo_occ[mc.ncore:mc.ncore + ncas]))[::-1]
    return mol, mc.mo_coeff, float(mc.e_tot) + e_pt2, occ


def mr_state_curve(*, atoms, charge, spin_2s, irrep_nelec, label, r_center, chk, policy: MRCurvePolicy,
                   cache: Path, clearance_ev: float | None = None) -> dict:
    """Walk outward from r_center; returns {"r": [...], "e": [...]} or a failure record.

    A cached completed curve that does not yet reach ``clearance_ev`` (default
    ``policy.clearance_ev``) is extended outward from its ends, every new point
    seeded from its neighbour's CASSCF orbitals (stored next to the cache, file
    ``<cache>.ends.npz``).  Without stored orbitals the curve is walked again from
    the centre: a cold-started end point can land on a nearby CASSCF solution
    (AlO- 1.450 A, 0.57 mEh off the cubic through its neighbours).
    """
    space = full_valence_space(atoms, charge)
    wfnsym = target_c2v_irrep(label)
    if space is None or wfnsym is None or irrep_nelec is None:
        return {"status": "MR_ACTIVE_SPACE_NOT_DERIVED"}
    target = float(policy.clearance_ev if clearance_ev is None else clearance_ev)
    ends_file = cache.with_name(cache.name + ".ends.npz")
    cached = json.loads(cache.read_text()) if cache.is_file() else None
    if cached is not None:
        reached = float(cached.get("clearance_ev", policy.clearance_ev)) >= target - 1e-9
        if cached.get("status") == "COMPLETED" and reached:
            return cached
        # Before Patch 50c an extension started cold (no "seeded" flag); such a failed
        # curve is walked again, a failure of a seeded walk is final.
        cold = ("seeded" not in cached
                and float(cached.get("clearance_ev", policy.clearance_ev)) > policy.clearance_ev + 1e-9)
        if cached.get("status") != "COMPLETED" and not cold:
            return cached
        if cached.get("status") != "COMPLETED" or not ends_file.is_file():
            cached = None  # walk again from the centre
    clearance = target / HARTREE_TO_EV
    pts: dict[float, tuple[float, list]] = {}
    ends: dict[int, tuple[Any, Any]] = {}
    if cached is None:
        mol0, mo0, e0, occ0 = _point(atoms, r_center, charge, spin_2s, irrep_nelec, wfnsym, space, policy.basis,
                                     None, None, chk, policy)
        pts[round(r_center, 6)] = (e0, occ0.tolist())
        ends = {-1: (mol0, mo0), +1: (mol0, mo0)}
    else:
        r_center = float(cached.get("r_center", r_center))
        pts = {round(float(x), 6): (float(y), list(o))
               for x, y, o in zip(cached["r"], cached["e"], cached["active_occupations"])}
        with np.load(ends_file) as f:
            for direction, tag in ((-1, "lo"), (+1, "hi")):
                ends[direction] = (_mol(atoms, float(f[f"r_{tag}"]), charge, spin_2s, policy.basis, policy),
                                   np.array(f[f"mo_{tag}"]))
    end_orbitals: dict[int, tuple[float, Any]] = {}
    for direction in (-1, +1):
        mol_prev, mo_prev = ends[direction]
        side = [x for x in pts if direction * (x - r_center) > 1e-6]
        k0 = len(side)
        end_orbitals[direction] = (r_center + direction * k0 * policy.step_angstrom, mo_prev)
        if k0:
            emin = min(v[0] for v in pts.values())
            r_end = max(side, key=lambda x: direction * x)
            if k0 >= policy.min_points_per_side and pts[r_end][0] - emin > clearance:
                continue
        for k in range(k0 + 1, policy.max_points_per_side + 1):
            r = r_center + direction * k * policy.step_angstrom
            mol_prev, mo_prev, e, occ = _point(atoms, r, charge, spin_2s, irrep_nelec, wfnsym, space,
                                               policy.basis, mo_prev, mol_prev, chk, policy)
            pts[round(r, 6)] = (e, occ.tolist())
            end_orbitals[direction] = (r, mo_prev)
            emin = min(v[0] for v in pts.values())
            if k >= policy.min_points_per_side and e - emin > clearance:
                break
        else:
            record = {"status": "MR_CURVE_RANGE_NOT_REACHED", "clearance_ev": target}
            if cached is not None:  # keep the completed shorter curve usable
                return cached
            _save_json(cache, record)
            return record
    r = sorted(pts)
    e = [pts[x][0] for x in r]
    occ = [pts[x][1] for x in r]
    record = {"status": "COMPLETED", "r": r, "e": e, "active_occupations": occ, "active_space": space,
              "wfnsym_c2v": wfnsym, "method": "SC-NEVPT2/CASSCF(full valence)", "basis": policy.basis,
              "r_center": r_center, "clearance_ev": target, "seeded": True}
    problems = continuity_problems(r, e, occ, policy)
    if problems:
        record.update(status="MR_CURVE_CONTINUITY_FAILED", problems=problems)
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ends_file, r_lo=end_orbitals[-1][0], mo_lo=end_orbitals[-1][1],
                 r_hi=end_orbitals[+1][0], mo_hi=end_orbitals[+1][1])
    _save_json(cache, record)
    return record


def continuity_problems(r, e, occ, policy: MRCurvePolicy) -> list[str]:
    problems = []
    for i in range(len(r)):
        idx = sorted(range(len(r)), key=lambda j: abs(j - i))[1:5]
        if len(idx) < 4:
            continue
        coeff = np.polyfit([r[j] for j in idx], [e[j] for j in idx], 3)
        if abs(np.polyval(coeff, r[i]) - e[i]) > policy.max_kink_hartree:
            problems.append(f"energy kink at R={r[i]:.4f}")
    for i in range(1, len(r)):
        if max(abs(a - b) for a, b in zip(occ[i], occ[i - 1])) > policy.max_occupation_jump:
            problems.append(f"active occupation jump between R={r[i - 1]:.4f} and {r[i]:.4f}")
    return problems


def _harmonic_k(r, e, half_window: float = 0.08) -> float | None:
    """Second derivative at the minimum of a quartic fit within +-half_window of the lowest point.

    Evaluated at the fitted minimum, so curves sampled on different grids compare
    without the anharmonic bias of a parabola through off-centre points.
    """
    r, e = np.asarray(r, float), np.asarray(e, float)
    i0 = int(np.argmin(e))
    mask = np.abs(r - r[i0]) <= half_window + 1e-9
    if mask.sum() < 5:
        return None
    x0 = r[i0]
    p = np.polynomial.Polynomial.fit(r[mask] - x0, e[mask], 4).convert()
    roots = [z.real for z in p.deriv().roots() if abs(z.imag) < 1e-12 and abs(z.real) <= half_window]
    if not roots:
        return None
    xm = min(roots, key=lambda z: p(z))
    k = float(p.deriv(2)(xm))
    return k if k > 0 else None


def mr_vibrational_binding(anion_binding: Mapping[str, Any] | None, zpe_anion_ev: float,
                           numerical_bound_ev: float) -> dict:
    """Anion v=0 below the lowest dissociation channel, from the electronic margin and the MR ZPE.

    The MR curve (NEVPT2) and the atomic asymptotes (CCSD(T)) are different methods,
    so their absolute energies are not compared.  Instead: the conservative CCSD(T)
    electronic margin (lowest channel minus anion minimum, same basis) minus the anion
    ZPE, with the whole ZPE again as model uncertainty (the v=0 level lies at most
    2 ZPE above the electronic minimum).  Deep wells clear this easily; shallow ones
    stay UNRESOLVED.
    """
    status = None if not anion_binding else anion_binding.get("status")
    margin = None if not anion_binding else anion_binding.get("margin_hartree")
    if status == "UNBOUND":
        return {"status": "UNBOUND", "rule": "ELECTRONIC_MINIMUM_NOT_BELOW_CHANNEL"}
    if status != "BOUND" or margin is None:
        return {"status": "UNRESOLVED", "rule": "NO_ELECTRONIC_BINDING_MARGIN"}
    lower_ev = float(margin) * HARTREE_TO_EV - 2.0 * float(zpe_anion_ev) - float(numerical_bound_ev)
    return {"status": "BOUND" if lower_ev > 0.0 else "UNRESOLVED", "rule": "ELECTRONIC_MARGIN_MINUS_2ZPE",
            "electronic_margin_ev": float(margin) * HARTREE_TO_EV, "v0_margin_lower_ev": lower_ev}


def mr_zpe_fallback(*, electronic_ea: Mapping[str, Any], loops: Mapping[str, Any], masses, channels,
                    nuclear_settings, output_dir: Path, policy: MRCurvePolicy = MRCurvePolicy(),
                    curve_fn=None) -> dict:
    """Delta_ZPE from MR curves of the selected neutral and anion (same method for both)."""
    from .adaptive.nuclear_motion import NuclearMotionPEC, NuclearMotionStatus, run_diatomic_nuclear_motion
    curve_fn = curve_fn or mr_state_curve
    record: dict[str, Any] = {"policy": asdict(policy), "curves": {}}
    inputs = {}
    for role in ("neutral", "anion"):
        job = electronic_ea[f"selected_{role}_job"]
        loop = loops.get(job)
        cand = next(c for c in electronic_ea["candidates"][role] if c["job_id"] == job)
        if loop is None:
            return {**record, "status": "MR_NOT_ATTEMPTED", "reason": f"no refinement loop for {job}"}
        r_c = float(cand["r_central_angstrom"])
        done = [(req, res) for req, res in zip(loop.requests, loop.results)
                if getattr(res, "high_level_checkpoint_path", None)]
        if not done:
            return {**record, "status": "MR_NOT_ATTEMPTED", "reason": f"no checkpoint for {job}"}
        req, res = min(done, key=lambda item: abs(float(item[0].r_angstrom) - r_c))
        inputs[role] = (job, loop, req, res, r_c)
    # The nuclear solver may ask for a wider curve (boundary clearance or
    # finite-domain sensitivity): extend that role's curve to the next clearance.
    schedule = (policy.clearance_ev,) + tuple(c for c in policy.extension_clearances_ev if c > policy.clearance_ev)
    level = {"neutral": 0, "anion": 0}
    curves: dict[str, dict] = {}
    record["extension_rounds"] = []
    while True:
        pecs, model_terms = {}, {}
        for role, (job, loop, req, res, r_c) in inputs.items():
            curve = curve_fn(atoms=tuple(req.atoms), charge=req.charge, spin_2s=req.spin_2s,
                             irrep_nelec=req.irrep_nelec, label=req.symmetry_label, r_center=r_c,
                             chk=res.high_level_checkpoint_path, policy=policy,
                             cache=Path(output_dir) / "mr_curves" / f"{role}.json",
                             clearance_ev=schedule[level[role]])
            curves[role] = curve
            record["curves"][role] = {k: v for k, v in curve.items() if k != "active_occupations"}
            if curve.get("status") != "COMPLETED":
                return {**record, "status": f"MR_{role.upper()}_{curve.get('status')}"}
            pecs[role] = NuclearMotionPEC(role=role, state_id=job, r_angstrom=tuple(curve["r"]),
                                          energy_hartree=tuple(curve["e"]), method=curve["method"],
                                          basis=curve["basis"], evidence_ids=("OPENEA_MR_NUCLEAR_CURVE_FALLBACK",))
            sr = sorted((float(p.r_angstrom), float(p.energy_hartree)) for p in loop.final_pec.points)
            k_mr, k_sr = _harmonic_k(curve["r"], curve["e"]), _harmonic_k(*zip(*sr)) if len(sr) >= 3 else None
            model_terms[role] = None if not k_mr or not k_sr else abs(float(np.sqrt(k_mr / k_sr)) - 1.0)
        assessment = run_diatomic_nuclear_motion(neutral_pec=pecs["neutral"], anion_pec=pecs["anion"],
                                                 masses=masses, settings=nuclear_settings,
                                                 anion_dissociation_channels=channels)
        record["assessment_status"] = assessment.status.value
        if assessment.status is not NuclearMotionStatus.PEC_REFINEMENT_REQUIRED:
            break
        wider = [role for role in inputs
                 if (getattr(assessment, role).requires_lower_r_extension
                     or getattr(assessment, role).requires_upper_r_extension)
                 and level[role] + 1 < len(schedule)]
        if not wider:
            break
        for role in wider:
            level[role] += 1
        record["extension_rounds"].append({role: {"clearance_ev": schedule[level[role]],
                                                  "domain_difference_ev": getattr(assessment, role).domain_difference_ev,
                                                  "points_before": len(curves[role]["r"])} for role in wider})
        print(f"[ZPE] MR curves {', '.join(wider)}: {assessment.status.value}; extending to clearance "
              f"{', '.join(f'{schedule[level[r]]:.2f}' for r in wider)} eV", flush=True)
    if assessment.status is not NuclearMotionStatus.MODEL_CONVERGENCE_REQUIRED:
        return {**record, "status": f"MR_NUCLEAR_{assessment.status.value}", "reason": assessment.rationale}
    if any(v is None for v in model_terms.values()):
        return {**record, "status": "MR_MODEL_TERM_UNAVAILABLE", "model_terms": model_terms}
    zpe = {"neutral": float(assessment.neutral.zpe_ev), "anion": float(assessment.anion.zpe_ev)}
    half = (float(assessment.neutral.numerical_bound_ev) + float(assessment.anion.numerical_bound_ev)
            + sum(zpe[role] * model_terms[role] for role in zpe))
    # The solver compares absolute NEVPT2 levels with CCSD(T) asymptotes and has no
    # model bound, so it can only say UNRESOLVED; decide from the electronic margin.
    binding = mr_vibrational_binding(electronic_ea.get("anion_binding"), zpe["anion"],
                                     float(assessment.anion.numerical_bound_ev))
    return {**record, "status": "CLEARED", "delta_zpe_ev": float(assessment.correction_ev),
            "delta_zpe_half_width_ev": half, "zpe_neutral_ev": zpe["neutral"], "zpe_anion_ev": zpe["anion"],
            "model_terms_relative_frequency": model_terms,
            "anion_vibrational_binding": binding["status"], "vibrational_binding": binding,
            "solver_vibrational_binding": assessment.anion_vibrational_binding.status.value}
