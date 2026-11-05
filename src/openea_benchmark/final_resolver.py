"""Final verdict of an automated two-atom run (Patch 52).

Collects the layer results and issues exactly one outcome:

* ``EA``         - adiabatic EA_0 with an interval, when every layer cleared;
* ``UNBOUND``    - the anion is not bound at the CBS level (continuum rule) and
                   no state competition is open;
* ``NO_BOUND_MOLECULE`` - neither charge has a bound state: every scouted curve of
                   both charges (with the dispersion rescan) falls to its fragments;
* ``UNRESOLVED`` - otherwise, with every blocking reason listed.

A charge without any bound state is a result, not a gap: an anion whose every curve
falls to its fragments does not exist as a molecule (``UNBOUND``, reason
``ANION_HAS_NO_BOUND_STATE``); a neutral without one has the separated atoms as its
ground state (EA against that limit is not implemented: listed as a blocking reason).

Requirements for ``EA`` (each missing item is one blocking reason):

1. single-basis electronic EA evaluated, high-level state competition CLEARED
   for neutral and anion (or PENDING_MR_COMPETITORS: its collapsed / SCF-only
   candidates are then MR-only candidates under item 5);
2. CBS layer CLEARED (state order at CBS, binding);
3. adiabatic assembly CLEARED (zero-point layer cleared, anion vibrationally
   bound, interval entirely above zero);
4. core-valence, scalar-relativity, spin-orbit and post-CCSD(T) corrections all
   included (a first-order-zero spin-orbit result counts as included);
5. MR-only competitor states excluded (or none listed), or every one that the
   lower bound could not exclude computed by the term-gap route (Patch 54); the
   adiabatic EA then already takes the lower of SR and MR state per charge.

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
                          mr_only_labels: list[str] | tuple[str, ...] = (),
                          mr_term_gap: Mapping[str, Any] | None = None,
                          no_bound_state: tuple[str, ...] | list[str] = ()) -> dict:
    blocking: list[str] = []
    warnings: list[str] = []

    if "anion" in no_bound_state:
        reason = "ANION_HAS_NO_BOUND_STATE: every scouted anion curve falls to its fragments"
        if "neutral" in no_bound_state:
            return {"outcome": "NO_BOUND_MOLECULE", "blocking": [], "warnings": warnings,
                    "reason": reason + "; NEUTRAL_HAS_NO_BOUND_STATE as well"}
        return {"outcome": "UNBOUND", "blocking": [], "warnings": warnings, "reason": reason}
    if "neutral" in no_bound_state:
        blocking.append("neutral has no bound state (separated atoms are its ground state): "
                        "EA against the atom limit not implemented")

    if not electronic_ea or electronic_ea.get("status") != "ELECTRONIC_EA_EVALUATED":
        blocking.append(f"electronic EA: {None if not electronic_ea else electronic_ea.get('status')}")
    else:
        for role, comp in (electronic_ea.get("high_level_competition") or {}).items():
            # PENDING_MR_COMPETITORS: only candidates that need a multireference method are
            # open; their labels are MR-only candidates and are checked below.
            if comp.get("status") == "PENDING_MR_COMPETITORS" and not mr_only_labels:
                blocking.append(f"{role} state competition: PENDING_MR_COMPETITORS (MR layer not run)")
            elif comp.get("status") not in ("CLEARED", "PENDING_MR_COMPETITORS"):
                blocking.append(f"{role} state competition: {comp.get('status')}")
    for role, comp in ((electronic_ea or {}).get("high_level_competition") or {}).items():
        if comp.get("blocking_open_scout_components"):
            blocking.append(f"{role}: {len(comp['blocking_open_scout_components'])} open scout component(s) "
                            "at or below the selected state (no bracketed minimum)")
        if comp.get("open_scout_components_without_energy"):
            warnings.append(f"{role}: {len(comp['open_scout_components_without_energy'])} scout component(s) "
                            "without any converged energy (not compared)")

    mr_blocking = None
    if mr_only_labels:
        status = None if not mr_competitors else mr_competitors.get("status")
        gap_status = None if not mr_term_gap else mr_term_gap.get("status")
        if status != "CLEARED" and gap_status != "CLEARED":
            mr_blocking = (f"MR-only competitor states: {status or 'NOT_COMPUTED'}"
                           + (f", term gap {gap_status}" if gap_status else ""))
            blocking.append(mr_blocking)

    if cbs and cbs.get("multireference_exceeded"):
        warnings.append("multireference screening above thresholds: " + ", ".join(cbs["multireference_exceeded"]))

    cbs_status = None if not cbs else cbs.get("status")
    if cbs_status == "ANION_UNBOUND_AT_CBS":
        # A lower neutral state only lowers E0(neutral) and makes the EA more negative:
        # open MR-only candidates of the neutral cannot change UNBOUND, anion ones can.
        if mr_blocking and not any(str(label).startswith("q=-1") for label in mr_only_labels):
            blocking.remove(mr_blocking)
            warnings.append(mr_blocking + " (neutral only: cannot bind the anion)")
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
        if adiabatic.get("post_ccsd_t_triples_computed") is False:
            warnings.append("post-CCSD(T): no CCSDT/CCSDTQ (exceeds RAM/scratch); "
                            f"+-{adiabatic['post_ccsd_t_allowance_ev']:.3f} eV allowance in the interval")
        elif adiabatic.get("post_ccsd_t_quadruples_allowance_ev") is not None:
            warnings.append("post-CCSD(T): connected quadruples not computed (CCSDTQ exceeds RAM/scratch); "
                            f"+-{adiabatic['post_ccsd_t_quadruples_allowance_ev']:.3f} eV allowance in the interval")

    if blocking:
        return {"outcome": "UNRESOLVED", "blocking": blocking, "warnings": warnings}
    return {"outcome": "EA", "blocking": [], "warnings": warnings,
            "ea_ev": adiabatic["ea0_ev"], "ea_lower_ev": adiabatic["ea0_lower_ev"],
            "ea_upper_ev": adiabatic["ea0_upper_ev"],
            "scope": "adiabatic EA_0, lowest spin-orbit levels, main isotopologue"}
