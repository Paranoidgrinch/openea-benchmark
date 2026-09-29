from openea_benchmark.adaptive_grid import (
    AdaptiveGridPolicy,
    GridDecisionAction,
    GridExtensionDirection,
    GridStopReason,
    decide_adaptive_grid_extension,
)


POLICY = AdaptiveGridPolicy(
    max_extra_points_per_side=4,
    max_extra_span_angstrom=0.20,
)


INITIAL = (
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
    1.30,
)


def sector(
    *,
    scouts,
    unambiguous=True,
):
    return {
        "tracking_mode": (
            "manifold"
        ),
        "manifold_branch_graph_unambiguous": (
            unambiguous
        ),
        "manifold_pec_scouts": (
            scouts
        ),
    }


def scout(
    status,
    r_values,
    *,
    component_id="component",
):
    return {
        "component_id": (
            component_id
        ),
        "pec": {
            "points": [
                {
                    "r_angstrom": r,
                    "energy_center_hartree": (
                        -1.0
                    ),
                    "energy_spread_mev": (
                        0.0
                    ),
                }
                for r in r_values
            ]
        },
        "minimum_scout": {
            "status": status,
            "candidates": [],
        },
    }


def decide(
    result,
    current=INITIAL,
):
    return (
        decide_adaptive_grid_extension(
            sector_result=result,
            current_geometries=(
                current
            ),
            initial_geometries=(
                INITIAL
            ),
            policy=POLICY,
        )
    )


def test_bracketed_minimum_stops():
    result = sector(
        scouts=(
            scout(
                "bracketed_single_minimum",
                INITIAL,
            ),
        )
    )

    decision = decide(
        result
    )

    assert (
        decision.action
        == GridDecisionAction.STOP
    )

    assert (
        decision.stop_reason
        == GridStopReason
        .MINIMUM_BRACKETED
    )

    assert not (
        decision.new_geometries_angstrom
    )


def test_right_edge_seeking_pec_extends_one_step():
    result = sector(
        scouts=(
            scout(
                "decreases_toward_higher_r",
                INITIAL,
            ),
        )
    )

    decision = decide(
        result
    )

    assert (
        decision.action
        == GridDecisionAction.EXTEND
    )

    assert (
        decision.directions
        == (
            GridExtensionDirection
            .HIGHER_R,
        )
    )

    assert (
        decision.new_geometries_angstrom
        == (
            1.35,
        )
    )


def test_left_edge_seeking_pec_extends_one_step():
    result = sector(
        scouts=(
            scout(
                "decreases_toward_lower_r",
                INITIAL,
            ),
        )
    )

    decision = decide(
        result
    )

    assert (
        decision.action
        == GridDecisionAction.EXTEND
    )

    assert (
        decision.directions
        == (
            GridExtensionDirection
            .LOWER_R,
        )
    )

    assert (
        decision.new_geometries_angstrom
        == (
            0.95,
        )
    )


def test_both_edges_can_request_extension():
    result = sector(
        scouts=(
            scout(
                "decreases_toward_lower_r",
                (
                    1.00,
                    1.05,
                    1.10,
                ),
                component_id="left",
            ),
            scout(
                "decreases_toward_higher_r",
                (
                    1.20,
                    1.25,
                    1.30,
                ),
                component_id="right",
            ),
        )
    )

    decision = decide(
        result
    )

    assert (
        decision.action
        == GridDecisionAction.EXTEND
    )

    assert (
        decision.directions
        == (
            GridExtensionDirection
            .LOWER_R,
            GridExtensionDirection
            .HIGHER_R,
        )
    )

    assert (
        decision.new_geometries_angstrom
        == (
            0.95,
            1.35,
        )
    )


def test_interior_edge_seeking_component_does_not_cross_discontinuity():
    result = sector(
        scouts=(
            scout(
                "decreases_toward_higher_r",
                (
                    1.00,
                    1.05,
                    1.10,
                ),
                component_id="interior",
            ),
            scout(
                "bracketed_single_minimum",
                (
                    1.20,
                    1.25,
                    1.30,
                ),
                component_id="right",
            ),
        )
    )

    decision = decide(
        result
    )

    assert (
        decision.action
        == GridDecisionAction.STOP
    )

    assert (
        decision.stop_reason
        == GridStopReason
        .MINIMUM_BRACKETED
    )

    assert not (
        decision.triggers
    )


def test_ambiguous_graph_requires_review():
    result = sector(
        scouts=(
            scout(
                "decreases_toward_higher_r",
                INITIAL,
            ),
        ),
        unambiguous=False,
    )

    decision = decide(
        result
    )

    assert (
        decision.action
        == GridDecisionAction.STOP
    )

    assert (
        decision.stop_reason
        == GridStopReason
        .CONTINUITY_REVIEW_REQUIRED
    )


def test_extension_limit_stops_run():
    current = (
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
        1.25,
        1.30,
        1.35,
        1.40,
        1.45,
        1.50,
    )

    result = sector(
        scouts=(
            scout(
                "decreases_toward_higher_r",
                current,
            ),
        )
    )

    decision = decide(
        result,
        current=current,
    )

    assert (
        decision.action
        == GridDecisionAction.STOP
    )

    assert (
        decision.stop_reason
        == GridStopReason
        .EXTENSION_LIMIT_REACHED
    )

    assert (
        decision.blocked_directions
        == (
            GridExtensionDirection
            .HIGHER_R,
        )
    )


def test_non_edge_flat_curve_does_not_extend():
    result = sector(
        scouts=(
            scout(
                "flat_or_unresolved",
                INITIAL,
            ),
        )
    )

    decision = decide(
        result
    )

    assert (
        decision.action
        == GridDecisionAction.STOP
    )

    assert (
        decision.stop_reason
        == GridStopReason
        .NO_EDGE_SEEKING_PEC
    )


def test_root_tracking_uses_root_graph_and_scouts():
    result = {
        "tracking_mode": "root",
        "branch_graph_unambiguous": (
            True
        ),
        "pec_scouts": (
            scout(
                "decreases_toward_higher_r",
                INITIAL,
            ),
        ),
    }

    decision = decide(
        result
    )

    assert (
        decision.action
        == GridDecisionAction.EXTEND
    )

    assert (
        decision.new_geometries_angstrom
        == (
            1.35,
        )
    )
