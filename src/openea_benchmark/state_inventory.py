"""Inventory of every state the workflow found, reported next to the EA.

Only the selected neutral/anion pair carries the full correction chain (CBS, CV,
SR, post-CC, SOC, ZPE).  All other states keep their single-basis CCSD(T)
minimum from the electronic-EA stage; this module lists them with term energies
relative to the ground state of their charge and, for neutral states, the
detachment threshold from the anion ground state.  Purely a report: nothing here
enters the EA or the final verdict.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .electronic_ea_stage import HARTREE_TO_EV

SCOPE_NOTE = ("Single-basis CCSD(T) minima of the electronic-EA stage (no CBS, core-valence, relativistic, "
              "post-CCSD(T), spin-orbit or zero-point correction). Not part of the EA verdict; term energies of "
              "higher states of the same symmetry are less reliable than those of the lowest state per block.")


def _term_label(job_id: str) -> str | None:
    parts = job_id.split("__")
    return parts[4] if len(parts) > 4 else None


def _role(charge: int) -> str:
    return "neutral" if charge == 0 else "anion" if charge == -1 else f"q={charge:+d}"


def build_state_inventory(electronic_ea: Mapping[str, Any] | None, *,
                          final_verdict: Mapping[str, Any] | None = None,
                          mr_term_gap: Mapping[str, Any] | None = None,
                          mr_competitors: Mapping[str, Any] | None = None,
                          mr_only_labels: Sequence[str] = ()) -> dict | None:
    """``electronic_ea``: record of the electronic-EA stage (``candidates`` per role,
    ``selected_neutral_job``/``selected_anion_job``).  ``final_verdict``: if it issued an
    EA, thresholds are also given on that scale (final EA + single-basis term energy)."""
    if not electronic_ea or not electronic_ea.get("candidates"):
        return None
    selected = {"neutral": electronic_ea.get("selected_neutral_job"),
                "anion": electronic_ea.get("selected_anion_job")}
    rows = []
    for group in electronic_ea["candidates"].values():
        for c in group:
            rows.append({
                "job_id": c["job_id"], "charge": c.get("charge"), "role": _role(c.get("charge")),
                "term": _term_label(c["job_id"]), "spin_2s": c.get("spin_2s"),
                "branch_of": c.get("branch_of"), "status": c.get("equilibrium_status"),
                "r_e_angstrom": c.get("r_central_angstrom"),
                "energy_hartree": c.get("energy_central_hartree"),
                "energy_lower_hartree": c.get("energy_lower_hartree"),
                "energy_upper_hartree": c.get("energy_upper_hartree"),
            })
    ground = {}
    for role, job in selected.items():
        row = next((r for r in rows if r["job_id"] == job and r["energy_hartree"] is not None), None)
        if row is not None:
            ground[role] = row
    final_ea = (final_verdict or {}).get("ea_ev") if (final_verdict or {}).get("outcome") in ("EA", "EA_NEAR_ZERO") else None
    for r in rows:
        r["is_ground_state"] = r["job_id"] == selected.get(r["role"])
        e = r["energy_hartree"]
        ref = ground.get(r["role"])
        r["t_e_ev"] = None if e is None or ref is None else (e - ref["energy_hartree"]) * HARTREE_TO_EV
        r["t_e_bound_ev"] = None
        if r["t_e_ev"] is not None and r["energy_lower_hartree"] is not None and ref["energy_lower_hartree"] is not None:
            r["t_e_bound_ev"] = (abs(r["energy_upper_hartree"] - r["energy_lower_hartree"])
                                 + abs(ref["energy_upper_hartree"] - ref["energy_lower_hartree"])) * HARTREE_TO_EV
        if r["role"] == "neutral":
            # adiabatic detachment threshold anion ground state -> this neutral state
            a = ground.get("anion")
            r["detachment_threshold_ev"] = None if e is None or a is None else (e - a["energy_hartree"]) * HARTREE_TO_EV
            r["detachment_threshold_final_scale_ev"] = (None if final_ea is None or r["t_e_ev"] is None
                                                        else final_ea + r["t_e_ev"])
        elif r["role"] == "anion":
            # > 0: this anion state lies below the neutral ground state (bound at this level)
            n = ground.get("neutral")
            r["binding_vs_neutral_ground_ev"] = None if e is None or n is None else (n["energy_hartree"] - e) * HARTREE_TO_EV
            r["binding_final_scale_ev"] = (None if final_ea is None or r["t_e_ev"] is None
                                           else final_ea - r["t_e_ev"])
    rows.sort(key=lambda r: (-(r["charge"] or 0), r["energy_hartree"] is None, r["energy_hartree"] or 0.0, r["job_id"]))
    mr_rows = []
    gaps = {(g.get("charge"), g.get("label")): g for g in (mr_term_gap or {}).get("gaps", ())}
    for (charge, label), g in gaps.items():
        mr_rows.append({"charge": charge, "role": _role(charge), "term": label, "status": g.get("status"),
                        "t_0_ev": g.get("gap0_ev"), "t_0_lower_ev": g.get("gap0_lower_ev"),
                        "t_0_upper_ev": g.get("gap0_upper_ev"),
                        "method": "MR term gap relative to the selected state of this charge (incl. ZPE, SOC)"})
    covered = {lab for _, lab in gaps}
    for b in (mr_competitors or {}).get("blocks", ()):
        if b.get("status") == "EXCLUDED":
            for label in b.get("candidates", ()):
                covered.add(label)
                mr_rows.append({"label": label, "status": "EXCLUDED",
                                "lower_bound_margin_ev": b.get("lower_bound_margin_ev"),
                                "method": "MR-only: variational lower bound lies above the selected state of this "
                                          "charge by at least lower_bound_margin_ev"})
    for label in mr_only_labels:
        if label not in covered:
            mr_rows.append({"label": label, "status": "NOT_COMPUTED",
                            "method": "MR-only: no single-determinant CCSD(T) representation"})
    return {"scope": SCOPE_NOTE, "method": "CCSD(T)", "basis": electronic_ea.get("basis"),
            "ground_states": {role: (row["job_id"] if row else None) for role, row in
                              ((k, ground.get(k)) for k in ("neutral", "anion"))},
            "final_ea_ev_used_for_scale": final_ea,
            "states": rows, "mr_only_states": mr_rows,
            "counts": {"total": len(rows), "with_minimum": sum(r["energy_hartree"] is not None for r in rows),
                       "mr_only": len(mr_rows)},
            "validation_metadata_used_in_computation": False}


def format_state_table(inventory: Mapping[str, Any] | None) -> str:
    """Plain-text table for logs and reports."""
    if not inventory:
        return "(no state inventory)"
    lines = [f"Further states ({inventory['method']}/{inventory['basis']}; {inventory['scope']})",
             f"{'q':>3} {'term':<12} {'r_e/A':>7} {'T_e/eV':>8} {'thr/eV':>8} {'thrEA/eV':>8}  status"]
    for r in inventory["states"]:
        thr = r.get("detachment_threshold_ev", r.get("binding_vs_neutral_ground_ev"))
        thr_ea = r.get("detachment_threshold_final_scale_ev", r.get("binding_final_scale_ev"))
        f = lambda v, w, p: f"{v:{w}.{p}f}" if v is not None else f"{'-':>{w}}"
        lines.append(f"{r['charge']:>3} {(r['term'] or '?') + ('*' if r['is_ground_state'] else ''):<12} "
                     f"{f(r['r_e_angstrom'], 7, 4)} {f(r['t_e_ev'], 8, 3)} {f(thr, 8, 3)} {f(thr_ea, 8, 3)}  "
                     f"{r['status']}")
    for m in inventory["mr_only_states"]:
        extra = ""
        if m.get("t_0_ev") is not None:
            extra = f" T_0 {m['t_0_ev']:.3f} eV [{m['t_0_lower_ev']:.3f}, {m['t_0_upper_ev']:.3f}]"
        elif m.get("lower_bound_margin_ev") is not None:
            extra = f" (at least {m['lower_bound_margin_ev']:.2f} eV above)"
        lines.append(f"    MR-only {m.get('term') or m.get('label')}: {m['status']}{extra}")
    lines.append("* ground state of its charge; thr = detachment threshold from the anion ground state (neutral) "
                 "or binding below the neutral ground state (anion), both single-basis; thrEA = the same on the "
                 "scale of the final EA (final EA +/- T_e)")
    return "\n".join(lines)
