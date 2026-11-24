"""Final verdict of an automated two-atom run (Patch 52).

Collects the layer results and issues exactly one outcome:

* ``EA``         - adiabatic EA_0 with an interval, when every layer cleared;
* ``UNBOUND``    - the anion is not bound: either decisively at the CBS level (continuum
                   rule, or the CBS interval below -``unbound_decision_margin_ev``, so no
                   later correction can bind it), or by the complete EA_0 interval (all
                   corrections included) lying below zero; no state competition is open;
* ``NO_BOUND_MOLECULE`` - neither charge has a bound state: every scouted curve of
                   both charges (with the dispersion rescan) falls to its fragments;
* ``NO_BOUND_NEUTRAL`` - the neutral has no bound state but the anion has one: there
                   is no neutral molecule, so a molecular EA is not defined (the anion
                   detaches onto a repulsive neutral curve);
* ``EA_NEAR_ZERO`` - every layer cleared except the sign: the final EA_0 interval
                   (all corrections included) contains zero and its half-width is at most
                   ``NEAR_ZERO_MAX_HALF_WIDTH_EV``; the EA is reported with its interval, the
                   sign (bound / unbound anion) is not determined within that accuracy;
* ``UNRESOLVED`` - otherwise, with every blocking reason listed.

A charge without any bound state is a result, not a gap: an anion whose every curve
falls to its fragments does not exist as a molecule (``UNBOUND``, reason
``ANION_HAS_NO_BOUND_STATE``); a neutral without one has the separated atoms as its
ground state, and the molecular EA is not defined (``NO_BOUND_NEUTRAL``).

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

# Largest half-width of a zero-straddling EA_0 interval that is still reported as a result.
NEAR_ZERO_MAX_HALF_WIDTH_EV = 0.1
# CBS binding reason of a negative CBS interval that later corrections could still move
# above zero (``cbs_stage.CBSStagePolicy.unbound_decision_margin_ev``): the layers run and
# the EA_0 interval decides the sign.
CBS_NEGATIVE_WITHIN_CORRECTION_MARGIN = "CBS_INTERVAL_NEGATIVE_WITHIN_CORRECTION_MARGIN"
# Reasons the binding is open that are a matter of precision only (a diffuse-basis
# continuum warning is not: the anion description itself is in question there).
_PRECISION_ONLY_CBS_REASONS = ("CBS_INTERVAL_CONTAINS_ZERO", CBS_NEGATIVE_WITHIN_CORRECTION_MARGIN)


def cbs_open_only_in_sign(cbs: Mapping[str, Any] | None) -> bool:
    """The CBS layer cleared everything except the sign of the EA: the EA_0 interval (all
    corrections included) then decides BOUND / UNBOUND / near zero."""
    if not cbs:
        return False
    if cbs.get("status") == "CLEARED":
        return True
    if cbs.get("status") != "NEED_MORE_EVIDENCE_BINDING" or cbs.get("binding_reason") not in _PRECISION_ONLY_CBS_REASONS:
        return False
    # the binding status hides the CBS-model status: it must have cleared on its own
    return cbs.get("cbs_model_status") == "CLEARED" or bool(cbs.get("cbs_model_bound_above_target_in_interval"))


def _near_zero(cbs: Mapping[str, Any] | None, adiabatic: Mapping[str, Any] | None) -> bool:
    """The only open item is the sign of an otherwise complete, narrow EA_0 interval."""
    if not adiabatic or not cbs_open_only_in_sign(cbs):
        return False
    if adiabatic.get("binding") != "UNDETERMINED" or adiabatic.get("anion_vibrational_binding") != "BOUND":
        return False
    if not all(adiabatic.get(flag) for flag, _, _ in CORRECTION_FLAGS):
        return False
    lo, hi = adiabatic.get("ea0_lower_ev"), adiabatic.get("ea0_upper_ev")
    if lo is None or hi is None or adiabatic.get("ea0_ev") is None:
        return False
    return lo <= 0.0 <= hi and (hi - lo) / 2.0 <= NEAR_ZERO_MAX_HALF_WIDTH_EV


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
        return {"outcome": "NO_BOUND_NEUTRAL", "blocking": [], "warnings": warnings,
                "reason": ("NEUTRAL_HAS_NO_BOUND_STATE: every scouted neutral curve falls to its fragments; "
                           "no neutral molecule, molecular EA not defined")}

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

    def unbound(reason: str | None, note: str) -> dict:
        # A lower neutral state only lowers E0(neutral) and makes the EA more negative:
        # open MR-only candidates of the neutral cannot change UNBOUND, anion ones can.
        if mr_blocking and not any(str(label).startswith("q=-1") for label in mr_only_labels):
            blocking.remove(mr_blocking)
            warnings.append(mr_blocking + " (neutral only: cannot bind the anion)")
        if blocking:
            return {"outcome": "UNRESOLVED", "blocking": blocking, "warnings": warnings, "note": note}
        return {"outcome": "UNBOUND", "blocking": [], "warnings": warnings, "reason": reason}

    cbs_status = None if not cbs else cbs.get("status")
    if cbs_status == "ANION_UNBOUND_AT_CBS":
        return unbound((cbs or {}).get("binding_reason"), "anion unbound at CBS, but the state assignment is open")
    cbs_sign_only = cbs_open_only_in_sign(cbs)
    if not cbs_sign_only:
        blocking.append(f"CBS layer: {cbs_status}")
    elif (adiabatic and adiabatic.get("status") == "ANION_UNBOUND" and adiabatic.get("ea0_upper_ev") is not None
          and all(adiabatic.get(flag) for flag, _, _ in CORRECTION_FLAGS)):
        # the complete EA_0 interval lies below zero (corrections included, e.g. a CBS EA just
        # above zero that the zero-point term pulls below it)
        return unbound(f"EA0_INTERVAL_NEGATIVE: EA_0 <= {adiabatic['ea0_upper_ev']:.3f} eV with all corrections",
                       "EA_0 interval below zero, but the state assignment is open")
    near_zero = _near_zero(cbs, adiabatic)

    if near_zero:
        if adiabatic.get("post_ccsd_t_triples_computed") is False:
            warnings.append("post-CCSD(T): no CCSDT/CCSDTQ (exceeds RAM/scratch); "
                            f"+-{adiabatic['post_ccsd_t_allowance_ev']:.3f} eV allowance in the interval")
        elif adiabatic.get("post_ccsd_t_quadruples_allowance_ev") is not None:
            warnings.append("post-CCSD(T): connected quadruples not computed (CCSDTQ exceeds RAM/scratch); "
                            f"+-{adiabatic['post_ccsd_t_quadruples_allowance_ev']:.3f} eV allowance in the interval")
    elif not adiabatic or adiabatic.get("status") != "CLEARED":
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
    if near_zero:
        return {"outcome": "EA_NEAR_ZERO", "blocking": [], "warnings": warnings,
                "ea_ev": adiabatic["ea0_ev"], "ea_lower_ev": adiabatic["ea0_lower_ev"],
                "ea_upper_ev": adiabatic["ea0_upper_ev"],
                "reason": ("EA_0 interval contains zero: sign (bound or unbound anion) not determined "
                           f"within the accuracy (half-width <= {NEAR_ZERO_MAX_HALF_WIDTH_EV} eV)"),
                "scope": "adiabatic EA_0, lowest spin-orbit levels, main isotopologue"}
    return {"outcome": "EA", "blocking": [], "warnings": warnings,
            "ea_ev": adiabatic["ea0_ev"], "ea_lower_ev": adiabatic["ea0_lower_ev"],
            "ea_upper_ev": adiabatic["ea0_upper_ev"],
            "scope": "adiabatic EA_0, lowest spin-orbit levels, main isotopologue"}
