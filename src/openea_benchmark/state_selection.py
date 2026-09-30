"""
Conservative Stage-2 electronic-state candidate selection.

This module does not run calculations, use experimental reference EAs,
establish electron binding, or permanently prune electronic states.

Energy comparisons are restricted to the same charge sector.
"""

from __future__ import annotations

from math import isfinite
from typing import Mapping


HARTREE_TO_EV = 27.211386245988


def _scouts(sector: Mapping) -> tuple:
    mode = sector.get("tracking_mode")

    if mode == "manifold":
        return tuple(
            sector.get("manifold_pec_scouts", ())
        )

    if mode == "root":
        return tuple(
            sector.get("pec_scouts", ())
        )

    return ()


def _graph(sector: Mapping) -> tuple[dict, bool]:
    mode = sector.get("tracking_mode")

    if mode == "manifold":
        graph = sector.get("manifold_branch_graph") or {}

        resolved = (
            sector.get("manifold_branch_graph_unambiguous")
            is True
        )

    elif mode == "root":
        graph = sector.get("branch_graph") or {}

        resolved = (
            sector.get("branch_graph_unambiguous")
            is True
        )

    else:
        graph = {}
        resolved = False

    return graph, resolved


def _observed_minimum(pec: Mapping) -> float | None:
    """
    Lowest sampled PEC energy.

    This is NOT a rigorous lower bound on the continuous PEC minimum.
    """

    energies = []

    for point in pec.get("points", ()):
        if "energy_center_hartree" in point:
            value = point["energy_center_hartree"]

        elif "energy_hartree" in point:
            value = point["energy_hartree"]

        else:
            continue

        energy = float(value)

        if not isfinite(energy):
            raise ValueError("non-finite sampled PEC energy")

        energies.append(energy)

    return min(energies) if energies else None


def build_state_selection_report(
    system_summary: Mapping,
) -> dict:
    """
    Produce an auditable list of Stage-2 candidate states.

    A bracketed PEC provides a high-level calculation seed, not proof of
    the global electronic ground state.

    Unbracketed components remain open, regardless of their sampled
    DFT energies. No experimental EA enters any decision.
    """

    groups = {
        "neutral": {
            "qualified": [],
            "unresolved": [],
            "flags": set(),
        },
        "anion": {
            "qualified": [],
            "unresolved": [],
            "flags": set(),
        },
    }

    for sector in system_summary["sectors"]:
        charge = int(sector["charge"])
        spin = int(sector["spin_2s"])

        if charge not in (0, -1):
            raise ValueError(
                f"Unexpected EA charge sector: {charge}"
            )

        label = "neutral" if charge == 0 else "anion"
        group = groups[label]

        graph, graph_resolved = _graph(sector)

        n_discontinuous = len(
            graph.get("discontinuous_edges", ())
        )

        n_bridged = len(
            graph.get("bridged_edges", ())
        )

        adaptive = sector.get("adaptive_grid") or {}
        adaptive_stop = adaptive.get("stop_reason")
        sector_status = sector.get("sector_status")

        if n_discontinuous:
            group["flags"].add(
                "DISCONTINUOUS_PEC_GRAPH"
            )

        if not graph_resolved:
            group["flags"].add(
                "CONTINUITY_REVIEW_REQUIRED"
            )

        if adaptive_stop == "EXTENSION_LIMIT_REACHED":
            group["flags"].add(
                "EXTENSION_LIMIT_REACHED"
            )

        if adaptive_stop == "CONTINUITY_REVIEW_REQUIRED":
            group["flags"].add(
                "CONTINUITY_REVIEW_REQUIRED"
            )

        entries = _scouts(sector)

        qualified_in_sector = 0
        unresolved_in_sector = 0

        if not entries:
            group["unresolved"].append({
                "charge": charge,
                "spin_2s": spin,
                "component_id": None,
                "minimum_status": "NO_RESOLVED_PEC",
                "sector_status": sector_status,
                "adaptive_stop": adaptive_stop,
                "observed_lowest_energy_hartree": None,
                "n_discontinuous_edges_in_sector": n_discontinuous,
            })

            unresolved_in_sector += 1

        for entry in entries:
            component_id = entry.get("component_id")

            pec = entry.get("pec") or {}
            scout = entry.get("minimum_scout") or {}

            status = scout.get("status")
            candidates = tuple(
                scout.get("candidates", ())
            )

            observed_min = _observed_minimum(pec)

            qualified = (
                status == "bracketed_single_minimum"
                and len(candidates) == 1
                and graph_resolved
                and sector_status == "dft_scout_complete"
            )

            if qualified:
                candidate = candidates[0]

                energy = float(
                    candidate["energy_hartree"]
                )

                r_min = float(
                    candidate["r_angstrom"]
                )

                if not (
                    isfinite(energy)
                    and isfinite(r_min)
                    and r_min > 0.0
                ):
                    raise ValueError(
                        "Invalid bracketed minimum candidate"
                    )

                group["qualified"].append({
                    "charge": charge,
                    "spin_2s": spin,
                    "component_id": component_id,
                    "r_candidate_angstrom": r_min,
                    "energy_hartree": energy,
                    "minimum_status": status,
                    "adaptive_stop": adaptive_stop,
                    "sector_bridged_edges": n_bridged,
                    "sector_discontinuous_edges": n_discontinuous,
                    "stage": "DFT_SCOUT_ONLY",
                })

                qualified_in_sector += 1

            else:
                group["unresolved"].append({
                    "charge": charge,
                    "spin_2s": spin,
                    "component_id": component_id,
                    "minimum_status": status,
                    "sector_status": sector_status,
                    "adaptive_stop": adaptive_stop,
                    "observed_lowest_energy_hartree": observed_min,
                    "n_discontinuous_edges_in_sector": n_discontinuous,
                })

                unresolved_in_sector += 1

        # A sector may contain both a bracketed PEC and an open PEC.
        # The sector-level adaptive STOP must not hide the open one.
        if qualified_in_sector and unresolved_in_sector:
            group["flags"].add(
                "MIXED_BRACKETED_AND_OPEN_COMPONENTS"
            )

    finalized = {}

    for label, group in groups.items():
        qualified = sorted(
            group["qualified"],
            key=lambda item: (
                item["energy_hartree"],
                item["spin_2s"],
                str(item["component_id"]),
            ),
        )

        unresolved = group["unresolved"]
        flags = set(group["flags"])

        if qualified:
            reference_energy = qualified[0]["energy_hartree"]

            for candidate in qualified:
                candidate["delta_from_lowest_bracketed_ev"] = (
                    candidate["energy_hartree"]
                    - reference_energy
                ) * HARTREE_TO_EV

            provisional_seed = dict(qualified[0])

        else:
            flags.add("NO_BRACKETED_SEED")
            provisional_seed = None

        if unresolved:
            competition_status = (
                "OPEN_COMPONENTS_REMAIN"
            )

        elif "DISCONTINUOUS_PEC_GRAPH" in flags:
            competition_status = (
                "SCOUTED_COMPONENTS_BRACKETED_WITH_BRANCH_WARNINGS"
            )

        else:
            competition_status = (
                "SCOUTED_COMPONENTS_BRACKETED"
            )

        finalized[label] = {
            "competition_status": competition_status,
            "provisional_high_level_seed": provisional_seed,
            "bracketed_candidates": qualified,
            "unresolved_components": unresolved,
            "flags": sorted(flags),
            "all_scouted_components_bracketed": (
                bool(qualified)
                and not unresolved
            ),
        }

    return {
        "system": system_summary.get("system"),
        "stage": "STAGE_2_STATE_SELECTION",
        "scope": (
            "ONLY_SAMPLED_SPIN_SECTORS_AND_IDENTIFIED_PECS"
        ),
        "charge_groups": finalized,
        "automatic_pruning_performed": False,
        "validation_metadata_used_in_selection": False,
        "ea_or_unbound_conclusion": "NOT_EVALUATED",
    }
