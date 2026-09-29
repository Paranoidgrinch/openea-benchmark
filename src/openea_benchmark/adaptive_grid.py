from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose
from typing import Mapping, Sequence


class GridExtensionDirection(
    str,
    Enum,
):
    LOWER_R = "LOWER_R"
    HIGHER_R = "HIGHER_R"


class GridDecisionAction(
    str,
    Enum,
):
    EXTEND = "EXTEND"
    STOP = "STOP"


class GridStopReason(
    str,
    Enum,
):
    MINIMUM_BRACKETED = (
        "MINIMUM_BRACKETED"
    )

    NO_EDGE_SEEKING_PEC = (
        "NO_EDGE_SEEKING_PEC"
    )

    EXTENSION_LIMIT_REACHED = (
        "EXTENSION_LIMIT_REACHED"
    )

    CONTINUITY_REVIEW_REQUIRED = (
        "CONTINUITY_REVIEW_REQUIRED"
    )


@dataclass(frozen=True)
class AdaptiveGridPolicy:
    """
    Explicit Stage-2 PEC extension limits.

    No universal defaults are provided deliberately.

    max_extra_points_per_side
        Hard cap on the number of geometries added below or above the
        original grid.

    max_extra_span_angstrom
        Hard cap on how far either edge may move beyond the original grid.
    """

    max_extra_points_per_side: int

    max_extra_span_angstrom: float

    geometry_tolerance_angstrom: float = (
        1.0e-9
    )

    def __post_init__(
        self,
    ) -> None:
        if (
            self.max_extra_points_per_side
            < 1
        ):
            raise ValueError(
                "max_extra_points_per_side "
                "must be >= 1"
            )

        if (
            self.max_extra_span_angstrom
            <= 0.0
        ):
            raise ValueError(
                "max_extra_span_angstrom "
                "must be > 0"
            )

        if (
            self.geometry_tolerance_angstrom
            <= 0.0
        ):
            raise ValueError(
                "geometry_tolerance_angstrom "
                "must be > 0"
            )


@dataclass(frozen=True)
class GridExtensionTrigger:
    component_id: str

    direction: GridExtensionDirection

    scout_status: str

    pec_min_r_angstrom: float

    pec_max_r_angstrom: float


@dataclass(frozen=True)
class AdaptiveGridDecision:
    action: GridDecisionAction

    stop_reason: GridStopReason | None

    directions: tuple[
        GridExtensionDirection,
        ...,
    ]

    new_geometries_angstrom: tuple[
        float,
        ...,
    ]

    triggers: tuple[
        GridExtensionTrigger,
        ...,
    ]

    blocked_directions: tuple[
        GridExtensionDirection,
        ...,
    ]

    current_grid_angstrom: tuple[
        float,
        ...,
    ]

    initial_grid_angstrom: tuple[
        float,
        ...,
    ]


def _sorted_grid(
    values: Sequence[
        float
    ],
    *,
    name: str,
) -> tuple[
    float,
    ...,
]:
    grid = tuple(
        float(
            value
        )
        for value in values
    )

    if len(
        grid
    ) < 2:
        raise ValueError(
            f"{name} requires at least two geometries"
        )

    if tuple(
        sorted(
            grid
        )
    ) != grid:
        raise ValueError(
            f"{name} must be sorted"
        )

    if len(
        set(
            grid
        )
    ) != len(
        grid
    ):
        raise ValueError(
            f"{name} contains duplicate geometries"
        )

    if any(
        value <= 0.0
        for value in grid
    ):
        raise ValueError(
            f"{name} geometries must be > 0"
        )

    return grid


def _sector_scouts(
    sector_result: Mapping,
) -> tuple:
    mode = sector_result.get(
        "tracking_mode"
    )

    if mode == "manifold":
        return tuple(
            sector_result.get(
                "manifold_pec_scouts",
                (),
            )
        )

    if mode == "root":
        return tuple(
            sector_result.get(
                "pec_scouts",
                (),
            )
        )

    return ()


def _continuity_is_resolved(
    sector_result: Mapping,
) -> bool:
    mode = sector_result.get(
        "tracking_mode"
    )

    if mode == "manifold":
        return (
            sector_result.get(
                "manifold_branch_graph_unambiguous"
            )
            is True
        )

    if mode == "root":
        return (
            sector_result.get(
                "branch_graph_unambiguous"
            )
            is True
        )

    return False


def _pec_geometry_range(
    scout_entry: Mapping,
) -> tuple[
    float,
    float,
] | None:
    pec = scout_entry.get(
        "pec"
    )

    if not isinstance(
        pec,
        Mapping,
    ):
        return None

    points = pec.get(
        "points",
        ()
    )

    values = []

    for point in points:
        if not isinstance(
            point,
            Mapping,
        ):
            continue

        if (
            "r_angstrom"
            not in point
        ):
            continue

        values.append(
            float(
                point[
                    "r_angstrom"
                ]
            )
        )

    if not values:
        return None

    return (
        min(
            values
        ),
        max(
            values
        ),
    )


def _edge_step(
    grid: tuple[
        float,
        ...,
    ],
    direction: GridExtensionDirection,
) -> float:
    if (
        direction
        == GridExtensionDirection.LOWER_R
    ):
        step = (
            grid[1]
            - grid[0]
        )

    else:
        step = (
            grid[-1]
            - grid[-2]
        )

    if step <= 0.0:
        raise ValueError(
            "grid edge step must be > 0"
        )

    return step


def _extra_points_on_side(
    current: tuple[
        float,
        ...,
    ],
    initial: tuple[
        float,
        ...,
    ],
    direction: GridExtensionDirection,
    *,
    tolerance: float,
) -> int:
    if (
        direction
        == GridExtensionDirection.LOWER_R
    ):
        return sum(
            1
            for value in current
            if (
                value
                < initial[0]
                - tolerance
            )
        )

    return sum(
        1
        for value in current
        if (
            value
            > initial[-1]
            + tolerance
        )
    )


def _extension_span(
    current: tuple[
        float,
        ...,
    ],
    initial: tuple[
        float,
        ...,
    ],
    direction: GridExtensionDirection,
) -> float:
    if (
        direction
        == GridExtensionDirection.LOWER_R
    ):
        return max(
            0.0,
            initial[0]
            - current[0],
        )

    return max(
        0.0,
        current[-1]
        - initial[-1],
    )


def decide_adaptive_grid_extension(
    *,
    sector_result: Mapping,
    current_geometries: Sequence[
        float
    ],
    initial_geometries: Sequence[
        float
    ],
    policy: AdaptiveGridPolicy,
) -> AdaptiveGridDecision:
    """
    Decide whether a completed Stage-2 sector grid should be extended.

    Only a resolved PEC component that actually touches the corresponding
    current grid boundary may request an extension.

    An edge-seeking PEC that terminates at an interior discontinuity does
    not trigger extension across that discontinuity.
    """
    current = _sorted_grid(
        current_geometries,
        name="current_geometries",
    )

    initial = _sorted_grid(
        initial_geometries,
        name="initial_geometries",
    )

    if not _continuity_is_resolved(
        sector_result
    ):
        return AdaptiveGridDecision(
            action=(
                GridDecisionAction.STOP
            ),
            stop_reason=(
                GridStopReason
                .CONTINUITY_REVIEW_REQUIRED
            ),
            directions=(),
            new_geometries_angstrom=(),
            triggers=(),
            blocked_directions=(),
            current_grid_angstrom=(
                current
            ),
            initial_grid_angstrom=(
                initial
            ),
        )

    scouts = _sector_scouts(
        sector_result
    )

    triggers = []

    has_bracketed_minimum = False

    tolerance = (
        policy
        .geometry_tolerance_angstrom
    )

    for entry in scouts:
        minimum = entry.get(
            "minimum_scout",
            {}
        )

        status = minimum.get(
            "status"
        )

        if (
            status
            == "bracketed_single_minimum"
        ):
            has_bracketed_minimum = (
                True
            )

        geometry_range = (
            _pec_geometry_range(
                entry
            )
        )

        if (
            geometry_range
            is None
        ):
            continue

        pec_min_r, pec_max_r = (
            geometry_range
        )

        component_id = str(
            entry.get(
                "component_id",
                "",
            )
        )

        if (
            status
            == "decreases_toward_higher_r"
            and isclose(
                pec_max_r,
                current[-1],
                abs_tol=tolerance,
                rel_tol=0.0,
            )
        ):
            triggers.append(
                GridExtensionTrigger(
                    component_id=(
                        component_id
                    ),
                    direction=(
                        GridExtensionDirection
                        .HIGHER_R
                    ),
                    scout_status=(
                        status
                    ),
                    pec_min_r_angstrom=(
                        pec_min_r
                    ),
                    pec_max_r_angstrom=(
                        pec_max_r
                    ),
                )
            )

        if (
            status
            == "decreases_toward_lower_r"
            and isclose(
                pec_min_r,
                current[0],
                abs_tol=tolerance,
                rel_tol=0.0,
            )
        ):
            triggers.append(
                GridExtensionTrigger(
                    component_id=(
                        component_id
                    ),
                    direction=(
                        GridExtensionDirection
                        .LOWER_R
                    ),
                    scout_status=(
                        status
                    ),
                    pec_min_r_angstrom=(
                        pec_min_r
                    ),
                    pec_max_r_angstrom=(
                        pec_max_r
                    ),
                )
            )

    requested_directions = tuple(
        direction
        for direction in (
            GridExtensionDirection
            .LOWER_R,
            GridExtensionDirection
            .HIGHER_R,
        )
        if any(
            trigger.direction
            == direction
            for trigger in triggers
        )
    )

    if not requested_directions:
        reason = (
            GridStopReason
            .MINIMUM_BRACKETED
            if has_bracketed_minimum
            else GridStopReason
            .NO_EDGE_SEEKING_PEC
        )

        return AdaptiveGridDecision(
            action=(
                GridDecisionAction.STOP
            ),
            stop_reason=reason,
            directions=(),
            new_geometries_angstrom=(),
            triggers=tuple(
                triggers
            ),
            blocked_directions=(),
            current_grid_angstrom=(
                current
            ),
            initial_grid_angstrom=(
                initial
            ),
        )

    allowed = []

    blocked = []

    new_values = []

    for direction in (
        requested_directions
    ):
        n_extra = (
            _extra_points_on_side(
                current,
                initial,
                direction,
                tolerance=tolerance,
            )
        )

        span = _extension_span(
            current,
            initial,
            direction,
        )

        step = _edge_step(
            current,
            direction,
        )

        prospective_span = (
            span
            + step
        )

        if (
            n_extra
            >= (
                policy
                .max_extra_points_per_side
            )
            or prospective_span
            > (
                policy
                .max_extra_span_angstrom
                + tolerance
            )
        ):
            blocked.append(
                direction
            )

            continue

        if (
            direction
            == GridExtensionDirection
            .LOWER_R
        ):
            new_r = (
                current[0]
                - step
            )

        else:
            new_r = (
                current[-1]
                + step
            )

        new_r = round(
            new_r,
            12,
        )

        if new_r <= 0.0:
            blocked.append(
                direction
            )

            continue

        allowed.append(
            direction
        )

        new_values.append(
            new_r
        )

    if not allowed:
        return AdaptiveGridDecision(
            action=(
                GridDecisionAction.STOP
            ),
            stop_reason=(
                GridStopReason
                .EXTENSION_LIMIT_REACHED
            ),
            directions=(),
            new_geometries_angstrom=(),
            triggers=tuple(
                triggers
            ),
            blocked_directions=tuple(
                blocked
            ),
            current_grid_angstrom=(
                current
            ),
            initial_grid_angstrom=(
                initial
            ),
        )

    return AdaptiveGridDecision(
        action=(
            GridDecisionAction.EXTEND
        ),
        stop_reason=None,
        directions=tuple(
            allowed
        ),
        new_geometries_angstrom=tuple(
            sorted(
                new_values
            )
        ),
        triggers=tuple(
            triggers
        ),
        blocked_directions=tuple(
            blocked
        ),
        current_grid_angstrom=(
            current
        ),
        initial_grid_angstrom=(
            initial
        ),
    )
