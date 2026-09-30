"""Auditable Stage-3 ROHF/CC job planning; no electronic calculations."""

from __future__ import annotations

from math import isclose
from typing import Mapping


def _matching_sector(summary, charge, spin):
    matches = [
        s for s in summary["sectors"]
        if int(s["charge"]) == charge
        and int(s["spin_2s"]) == spin
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one sector q={charge} 2S={spin}; "
            f"found {len(matches)}"
        )

    return matches[0]


def _matching_scout(sector, component_id):
    mode = sector.get("tracking_mode")

    if mode == "manifold":
        entries = sector.get("manifold_pec_scouts", ())
    elif mode == "root":
        entries = sector.get("pec_scouts", ())
    else:
        raise ValueError(f"Unsupported tracking mode: {mode}")

    matches = [
        entry for entry in entries
        if entry.get("component_id") == component_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Component {component_id}: expected one scout, "
            f"found {len(matches)}"
        )

    return matches[0]


def _checkpoint_provenance(sector, scout, center_r):
    """
    Link a PEC center to its actual Stage-2 electronic state.

    Never select an arbitrary DFT root when multiple members exist.
    Missing identity metadata produces an unresolved binding.
    """

    empty = {
        "link_status": "UNRESOLVED",
        "manifold_id": None,
        "member_root_ids": [],
        "unique_state_root_ids": [],
        "checkpoint_candidates": [],
        "diagnostic": None,
    }

    points = [
        point
        for point in scout.get("pec", {}).get("points", ())
        if isclose(
            float(point["r_angstrom"]),
            center_r,
            abs_tol=1e-8,
            rel_tol=0.0,
        )
    ]

    if len(points) != 1:
        return {
            **empty,
            "diagnostic": "CENTER_PEC_POINT_NOT_UNIQUE",
        }

    point = points[0]
    mode = sector.get("tracking_mode")

    if mode == "manifold":
        manifold_id = point.get("manifold_id")

        manifolds = [
            m for m in sector.get("manifolds", ())
            if m.get("manifold_id") == manifold_id
            and isclose(
                float(m["r_angstrom"]),
                center_r,
                abs_tol=1e-8,
                rel_tol=0.0,
            )
        ]

        if len(manifolds) != 1:
            return {
                **empty,
                "manifold_id": manifold_id,
                "diagnostic": "MANIFOLD_METADATA_NOT_UNIQUE",
            }

        manifold = manifolds[0]

        member_ids = tuple(
            manifold.get("member_root_ids") or ()
        )

        unique_ids = tuple(
            manifold.get("unique_state_root_ids") or ()
        )

        if unique_ids and not set(unique_ids).issubset(
            set(member_ids)
        ):
            raise ValueError(
                "Unique-state roots are not manifold members"
            )

    elif mode == "root":
        root_id = point.get("root_id")

        if not root_id:
            return {
                **empty,
                "diagnostic": "CENTER_ROOT_ID_MISSING",
            }

        manifold_id = None
        member_ids = (root_id,)
        unique_ids = (root_id,)

    else:
        return {
            **empty,
            "diagnostic": "TRACKING_MODE_UNSUPPORTED",
        }

    roots_by_id = {
        root["root_id"]: root
        for root in sector.get("representative_roots", ())
    }

    candidates = []
    missing = []

    for root_id in sorted(set(member_ids)):
        root = roots_by_id.get(root_id)

        if root is None:
            missing.append(root_id)
            continue

        if not isclose(
            float(root["r_angstrom"]),
            center_r,
            abs_tol=1e-8,
            rel_tol=0.0,
        ):
            raise ValueError(
                f"Root {root_id} has inconsistent geometry"
            )

        candidates.append({
            "root_id": root_id,
            "checkpoint_path": root.get("checkpoint_path"),
            "origin_guess": root.get("origin_guess"),
            "energy_hartree": root.get("energy_hartree"),
            "s2": root.get("s2"),
            "internal_stable": root.get("internal_stable"),
        })

    if missing or not candidates:
        status = "UNRESOLVED"
        diagnostic = (
            f"MISSING_ROOT_METADATA: {missing}"
            if missing else "NO_CHECKPOINT_CANDIDATES"
        )

    elif any(not c["checkpoint_path"] for c in candidates):
        status = "UNRESOLVED"
        diagnostic = "CHECKPOINT_PATH_MISSING"

    elif len(candidates) == 1:
        status = "SINGLE_DFT_INITIALIZATION"
        diagnostic = None

    else:
        status = "MULTIPLE_DFT_INITIALIZATIONS"
        diagnostic = (
            "ROHF reference must check electronic identity "
            "across available initializations"
        )

    return {
        "link_status": status,
        "manifold_id": manifold_id,
        "member_root_ids": list(member_ids),
        "unique_state_root_ids": list(unique_ids),
        "checkpoint_candidates": candidates,
        "diagnostic": diagnostic,
    }


def build_stage3_plan(
    *,
    system_summary: Mapping,
    selection_report: Mapping,
    manifest: Mapping,
) -> dict:
    system = str(system_summary["system"])

    if selection_report.get("system") != system:
        raise ValueError("Selection report belongs to another system")

    if system not in manifest["systems"]:
        raise ValueError(f"System absent from manifest: {system}")

    config = manifest["protocol"]["stage_3_high_level_local_pec"]

    offsets = tuple(
        float(x)
        for x in config["relative_grid_angstrom"]
    )

    if not offsets or 0.0 not in offsets:
        raise ValueError("Stage-3 grid must contain its center")

    jobs = []
    open_components = []

    for label, expected_charge in (
        ("neutral", 0),
        ("anion", -1),
    ):
        group = selection_report["charge_groups"][label]
        provisional = group.get("provisional_high_level_seed")

        open_components.extend(
            {
                **dict(item),
                "charge_group": label,
                "stage3_action": "COMPETITION_DIAGNOSTIC_REQUIRED",
            }
            for item in group["unresolved_components"]
        )

        for candidate in group["bracketed_candidates"]:
            charge = int(candidate["charge"])
            spin = int(candidate["spin_2s"])
            component_id = candidate["component_id"]

            if charge != expected_charge:
                raise ValueError("Inconsistent charge grouping")

            sector = _matching_sector(
                system_summary, charge, spin
            )

            scout = _matching_scout(
                sector, component_id
            )

            minimum = scout["minimum_scout"]

            if (
                minimum.get("status") != "bracketed_single_minimum"
                or len(minimum.get("candidates", ())) != 1
            ):
                raise ValueError(
                    f"{component_id}: no unique bracketed minimum"
                )

            source_minimum = minimum["candidates"][0]

            center = float(candidate["r_candidate_angstrom"])
            energy = float(candidate["energy_hartree"])

            if not isclose(
                center,
                float(source_minimum["r_angstrom"]),
                abs_tol=1e-8,
                rel_tol=0.0,
            ):
                raise ValueError("Selection/PEC center mismatch")

            if not isclose(
                energy,
                float(source_minimum["energy_hartree"]),
                abs_tol=1e-9,
                rel_tol=0.0,
            ):
                raise ValueError("Selection/PEC energy mismatch")

            grid = tuple(
                round(center + offset, 12)
                for offset in offsets
            )

            if (
                any(r <= 0.0 for r in grid)
                or len(grid) != len(set(grid))
                or grid != tuple(sorted(grid))
            ):
                raise ValueError("Invalid local Stage-3 geometry grid")

            source = _checkpoint_provenance(
                sector,
                scout,
                center,
            )

            is_primary = bool(
                provisional is not None
                and provisional["component_id"] == component_id
                and int(provisional["spin_2s"]) == spin
                and int(provisional["charge"]) == charge
            )

            jobs.append({
                "job_id": (
                    f"{system}__q{charge:+d}__2S{spin}"
                    f"__{component_id}"
                ),
                "system": system,
                "charge_group": label,
                "charge": charge,
                "spin_2s": spin,
                "component_id": component_id,
                "provisional_primary_seed": is_primary,
                "dft_center_r_angstrom": center,
                "dft_center_energy_hartree": energy,
                "dft_gap_from_group_seed_ev": candidate.get(
                    "delta_from_lowest_bracketed_ev"
                ),
                "reference": config["reference"],
                "methods": list(config["methods"]),
                "basis": config["basis"],
                "local_grid_angstrom": list(grid),
                "source_provenance": source,
                "requires_independent_state_identity_validation": True,
                "sector_bridged_edges": candidate.get(
                    "sector_bridged_edges"
                ),
                "sector_discontinuous_edges": candidate.get(
                    "sector_discontinuous_edges"
                ),
                "execution_status": "PLANNED_NOT_RUN",
            })

    job_ids = [job["job_id"] for job in jobs]

    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Duplicate Stage-3 job identifiers")

    return {
        "system": system,
        "stage": "STAGE_3_HIGH_LEVEL_PLANNING",
        "jobs": jobs,
        "n_jobs": len(jobs),
        "open_components": open_components,
        "n_open_components": len(open_components),
        "automatic_pruning_performed": False,
        "validation_metadata_used_in_planning": False,
        "high_level_calculations_performed": False,
        "ea_or_unbound_conclusion": "NOT_EVALUATED",
    }
