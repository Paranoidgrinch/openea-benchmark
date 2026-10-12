"""Final verdict of an automated two-atom run (Patch 52).

Collects the layer results and issues exactly one outcome:

* ``EA``         - adiabatic EA_0 with an interval, when every layer cleared;
* ``UNBOUND``    - the anion is not bound at the CBS level (continuum rule) and
                   no state competition is open;
* ``UNRESOLVED`` - otherwise, with every blocking reason listed.

Requirements for ``EA`` (each missing item is one blocking reason):

1. single-basis electronic EA evaluated, high-level state competition CLEARED
   for neutral and anion;
2. CBS layer CLEARED (state order at CBS, binding);
3. adiabatic assembly CLEARED (zero-point layer cleared, anion vibrationally
   bound, interval entirely above zero);
4. core-valence, scalar-relativity, spin-orbit and post-CCSD(T) corrections all
   included (a first-order-zero spin-orbit result counts as included);
5. MR-only competitor states excluded (or none listed).

Multireference screening values above the usual thresholds are reported as a
warning, not a blocker: the MR layer decides state questions, and the post-CCSD(T)
increments measure the correlation error directly.

The verdict never uses experimental reference data.
"""
from __future__ import annotations

from typing import Any, Mapping

CORRECTION_FLAGS = (("includes_core_valence", "core_valence_status", "core-valence"),
                    ("includes_scalar_relativity", "scalar_relativity_status", "scalar relativity"),
                    ("includes_soc", "spin_orbit_status", "spin-orbit"),
                    ("includes_post_ccsd_t", "post_ccsd_t_status", "post-CCSD(T)"))


def resolve_final_verdict(*, electronic_ea: Mapping[str, Any] | None, cbs: Mapping[str, Any] | None,
                          adiabatic: Mapping[str, Any] | None,
                          mr_competitors: Mapping[str, Any] | None,
                          mr_only_labels: list[str] | tuple[str, ...] = ()) -> dict:
    blocking: list[str] = []
    warnings: list[str] = []

    if not electronic_ea or electronic_ea.get("status") != "ELECTRONIC_EA_EVALUATED":
        blocking.append(f"electronic EA: {None if not electronic_ea else electronic_ea.get('status')}")
    else:
        for role, comp in (electronic_ea.get("high_level_competition") or {}).items():
            if comp.get("status") != "CLEARED":
                blocking.append(f"{role} state competition: {comp.get('status')}")

    if mr_only_labels:
        status = None if not mr_competitors else mr_competitors.get("status")
        if status != "CLEARED":
            blocking.append(f"MR-only competitor states: {status or 'NOT_COMPUTED'}")

    if cbs and cbs.get("multireference_exceeded"):
        warnings.append("multireference screening above thresholds: " + ", ".join(cbs["multireference_exceeded"]))

    cbs_status = None if not cbs else cbs.get("status")
    if cbs_status == "ANION_UNBOUND_AT_CBS":
        if blocking:
            return {"outcome": "UNRESOLVED", "blocking": blocking, "warnings": warnings,
                    "note": "anion unbound at CBS, but the state assignment is open"}
        return {"outcome": "UNBOUND", "blocking": [], "warnings": warnings,
                "reason": (cbs or {}).get("binding_reason")}
    if cbs_status != "CLEARED":
        blocking.append(f"CBS layer: {cbs_status}")

    if not adiabatic or adiabatic.get("status") != "CLEARED":
        blocking.append(f"adiabatic EA: {None if not adiabatic else adiabatic.get('status')}"
                        + ("" if not adiabatic or not adiabatic.get("reason") else f" ({adiabatic['reason']})"))
    else:
        for flag, status_key, name in CORRECTION_FLAGS:
            if not adiabatic.get(flag):
                blocking.append(f"{name} correction not included: {adiabatic.get(status_key)}")

    if blocking:
        return {"outcome": "UNRESOLVED", "blocking": blocking, "warnings": warnings}
    return {"outcome": "EA", "blocking": [], "warnings": warnings,
            "ea_ev": adiabatic["ea0_ev"], "ea_lower_ev": adiabatic["ea0_lower_ev"],
            "ea_upper_ev": adiabatic["ea0_upper_ev"],
            "scope": "adiabatic EA_0, lowest spin-orbit levels, main isotopologue"}
