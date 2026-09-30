from copy import deepcopy

from openea_benchmark.state_selection import (
    build_state_selection_report,
)


def pec(
    energy,
    *,
    status="bracketed_single_minimum",
    component="c1",
):
    return {
        "component_id": component,
        "pec": {
            "points": [
                {
                    "r_angstrom": 1.0,
                    "energy_center_hartree": energy + 0.1,
                },
                {
                    "r_angstrom": 1.1,
                    "energy_center_hartree": energy,
                },
                {
                    "r_angstrom": 1.2,
                    "energy_center_hartree": energy + 0.1,
                },
            ]
        },
        "minimum_scout": {
            "status": status,
            "candidates": (
                [{
                    "r_angstrom": 1.1,
                    "energy_hartree": energy,
                }]
                if status == "bracketed_single_minimum"
                else []
            ),
        },
    }


def sector(
    charge,
    spin,
    entries,
    *,
    discontinuities=0,
    stop="MINIMUM_BRACKETED",
):
    return {
        "charge": charge,
        "spin_2s": spin,
        "tracking_mode": "manifold",
        "sector_status": "dft_scout_complete",
        "manifold_branch_graph_unambiguous": True,
        "manifold_branch_graph": {
            "bridged_edges": [],
            "discontinuous_edges": [
                ["left", "right"]
                for _ in range(discontinuities)
            ],
        },
        "manifold_pec_scouts": entries,
        "adaptive_grid": {
            "stop_reason": stop,
        },
    }


def test_lowest_bracketed_candidate_selected_within_same_charge():
    summary = {
        "system": "X",
        "sectors": [
            sector(0, 1, [pec(-10.0)]),
            sector(0, 3, [pec(-9.0)]),
            sector(-1, 0, [pec(-10.5)]),
            sector(-1, 2, [pec(-10.1)]),
        ],
    }

    report = build_state_selection_report(summary)

    assert (
        report["charge_groups"]["neutral"]
        ["provisional_high_level_seed"]["spin_2s"]
        == 1
    )

    assert (
        report["charge_groups"]["anion"]
        ["provisional_high_level_seed"]["spin_2s"]
        == 0
    )

    assert (
        report["automatic_pruning_performed"]
        is False
    )


def test_open_high_energy_sector_is_not_pruned():
    summary = {
        "system": "X",
        "sectors": [
            sector(0, 1, [pec(-10.0)]),
            sector(
                0,
                3,
                [pec(
                    -8.0,
                    status="decreases_toward_higher_r",
                )],
                stop="EXTENSION_LIMIT_REACHED",
            ),
        ],
    }

    group = (
        build_state_selection_report(summary)
        ["charge_groups"]["neutral"]
    )

    assert (
        group["competition_status"]
        == "OPEN_COMPONENTS_REMAIN"
    )

    assert len(group["unresolved_components"]) == 1

    assert (
        "EXTENSION_LIMIT_REACHED"
        in group["flags"]
    )


def test_bracketed_stop_cannot_hide_an_open_component():
    summary = {
        "system": "X",
        "sectors": [
            sector(
                -1,
                1,
                [
                    pec(
                        -10.0,
                        component="bracketed",
                    ),
                    pec(
                        -9.0,
                        status="decreases_toward_higher_r",
                        component="open",
                    ),
                ],
                discontinuities=1,
                stop="MINIMUM_BRACKETED",
            ),
        ],
    }

    group = (
        build_state_selection_report(summary)
        ["charge_groups"]["anion"]
    )

    assert len(group["bracketed_candidates"]) == 1
    assert len(group["unresolved_components"]) == 1

    assert (
        "MIXED_BRACKETED_AND_OPEN_COMPONENTS"
        in group["flags"]
    )

    assert (
        "DISCONTINUOUS_PEC_GRAPH"
        in group["flags"]
    )


def test_negative_dft_attachment_does_not_assign_unbound():
    summary = {
        "system": "X",
        "sectors": [
            sector(0, 0, [pec(-10.0)]),
            sector(-1, 1, [pec(-9.0)]),
        ],
    }

    report = build_state_selection_report(summary)

    assert (
        report["ea_or_unbound_conclusion"]
        == "NOT_EVALUATED"
    )


def test_experimental_metadata_cannot_change_selection():
    summary = {
        "system": "X",
        "sectors": [
            sector(0, 1, [pec(-10.0)]),
            sector(-1, 0, [pec(-10.5)]),
        ],
        "validation": {
            "electron_affinity_ev": 1.0,
        },
    }

    changed = deepcopy(summary)

    changed["validation"]["electron_affinity_ev"] = 9.0

    first = build_state_selection_report(summary)
    second = build_state_selection_report(changed)

    assert first == second

    assert (
        first["validation_metadata_used_in_selection"]
        is False
    )
