from copy import deepcopy

import pytest

from openea_benchmark.stage3_plan import (
    build_stage3_plan,
)
from openea_benchmark.state_selection import (
    build_state_selection_report,
)


def fixture():
    points = [
        {
            "manifold_id": f"m{i}",
            "r_angstrom": r,
            "energy_center_hartree": e,
        }
        for i, (r, e) in enumerate((
            (1.0, -9.9),
            (1.1, -10.0),
            (1.2, -9.9),
        ))
    ]

    sector = {
        "charge": 0,
        "spin_2s": 1,
        "tracking_mode": "manifold",
        "sector_status": "dft_scout_complete",
        "manifold_branch_graph_unambiguous": True,
        "manifold_branch_graph": {
            "bridged_edges": [],
            "discontinuous_edges": [],
        },
        "adaptive_grid": {
            "stop_reason": "MINIMUM_BRACKETED",
        },
        "manifold_pec_scouts": [{
            "component_id": "component_A",
            "pec": {"points": points},
            "minimum_scout": {
                "status": "bracketed_single_minimum",
                "candidates": [{
                    "r_angstrom": 1.1,
                    "energy_hartree": -10.0,
                }],
            },
        }],
        "manifolds": [{
            "manifold_id": "m1",
            "r_angstrom": 1.1,
            "member_root_ids": ["root_A", "root_B"],
            "unique_state_root_ids": ["root_A", "root_B"],
        }],
        "representative_roots": [
            {
                "root_id": root_id,
                "r_angstrom": 1.1,
                "checkpoint_path": path,
            }
            for root_id, path in (
                ("root_A", "a.chk"),
                ("root_B", "b.chk"),
            )
        ],
    }

    summary = {
        "system": "X",
        "sectors": [sector],
    }

    manifest = {
        "systems": {"X": {"atoms": ["H", "H"]}},
        "protocol": {
            "stage_3_high_level_local_pec": {
                "reference": "ROHF",
                "methods": ["CCSD", "CCSD(T)"],
                "basis": "def2-TZVPPD",
                "relative_grid_angstrom": [
                    -0.05, -0.025, 0.0, 0.025, 0.05
                ],
            },
        },
    }

    selection = build_state_selection_report(summary)

    return summary, selection, manifest


def test_local_grid_and_all_initializations_preserved():
    summary, selection, manifest = fixture()

    plan = build_stage3_plan(
        system_summary=summary,
        selection_report=selection,
        manifest=manifest,
    )

    assert plan["n_jobs"] == 1

    job = plan["jobs"][0]

    assert job["local_grid_angstrom"] == [
        1.05, 1.075, 1.1, 1.125, 1.15
    ]

    assert job["provisional_primary_seed"] is True

    source = job["source_provenance"]

    assert (
        source["link_status"]
        == "MULTIPLE_DFT_INITIALIZATIONS"
    )

    assert len(source["checkpoint_candidates"]) == 2

    assert plan["automatic_pruning_performed"] is False
    assert plan["high_level_calculations_performed"] is False


def test_missing_manifold_metadata_does_not_guess_root():
    summary, selection, manifest = fixture()

    summary["sectors"][0]["manifolds"] = []

    plan = build_stage3_plan(
        system_summary=summary,
        selection_report=selection,
        manifest=manifest,
    )

    source = plan["jobs"][0]["source_provenance"]

    assert source["link_status"] == "UNRESOLVED"
    assert source["checkpoint_candidates"] == []


def test_selection_source_energy_mismatch_fails_closed():
    summary, selection, manifest = fixture()

    selection["charge_groups"]["neutral"][
        "bracketed_candidates"
    ][0]["energy_hartree"] = -12.0

    with pytest.raises(ValueError):
        build_stage3_plan(
            system_summary=summary,
            selection_report=selection,
            manifest=manifest,
        )


def test_experimental_metadata_does_not_affect_jobs():
    summary, selection, manifest = fixture()

    first = build_stage3_plan(
        system_summary=summary,
        selection_report=selection,
        manifest=manifest,
    )

    changed = deepcopy(summary)
    changed["validation"] = {
        "electron_affinity_ev": 999.0,
    }

    second = build_stage3_plan(
        system_summary=changed,
        selection_report=selection,
        manifest=manifest,
    )

    assert first == second


def test_wrong_system_report_is_rejected():
    summary, selection, manifest = fixture()

    selection["system"] = "OTHER"

    with pytest.raises(ValueError):
        build_stage3_plan(
            system_summary=summary,
            selection_report=selection,
            manifest=manifest,
        )
