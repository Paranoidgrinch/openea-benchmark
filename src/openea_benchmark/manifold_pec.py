from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite
from typing import Sequence

from .electronic_manifold import (
    ElectronicManifoldPoint,
    ManifoldContinuity,
    ManifoldContinuityRelation,
    ManifoldThresholds,
    compare_electronic_manifolds,
)
from .local_pec import (
    LocalPEC,
    LocalPECPoint,
)
from .minimum_scout import (
    MinimumScoutResult,
    MinimumScoutThresholds,
    scout_local_pec_minimum,
)


@dataclass(frozen=True)
class ManifoldBranchComponent:
    component_id: str

    member_manifold_ids: tuple[
        str,
        ...,
    ]

    geometry_values: tuple[
        float,
        ...,
    ]

    has_branching: bool

    has_gauge_unresolved_boundary: bool

    is_unambiguous: bool


@dataclass(frozen=True)
class ManifoldBranchGraph:
    """
    Layered continuity graph for ElectronicManifoldPoint objects.

    CONTINUOUS manifold comparisons define graph connectivity.

    GAUGE_UNRESOLVED comparisons are retained explicitly as unresolved
    boundaries and are never silently promoted to continuity.

    DISCONTINUOUS comparisons separate electronic branches.
    """

    geometry_values: tuple[
        float,
        ...,
    ]

    manifold_ids_by_geometry: tuple[
        tuple[
            float,
            tuple[
                str,
                ...,
            ],
        ],
        ...,
    ]

    comparisons: tuple[
        ManifoldContinuity,
        ...,
    ]

    continuous_edges: tuple[
        tuple[
            str,
            str,
        ],
        ...,
    ]

    gauge_unresolved_edges: tuple[
        tuple[
            str,
            str,
        ],
        ...,
    ]

    discontinuous_edges: tuple[
        tuple[
            str,
            str,
        ],
        ...,
    ]

    components: tuple[
        ManifoldBranchComponent,
        ...,
    ]

    topology_ambiguous_manifold_ids: tuple[
        str,
        ...,
    ]

    gauge_unresolved_manifold_ids: tuple[
        str,
        ...,
    ]

    @property
    def is_fully_unambiguous(
        self,
    ) -> bool:
        return (
            not self.gauge_unresolved_edges
            and not self.topology_ambiguous_manifold_ids
            and all(
                component.is_unambiguous
                for component
                in self.components
            )
        )


@dataclass(frozen=True)
class ManifoldPECPoint:
    manifold_id: str

    r_angstrom: float

    energy_center_hartree: float

    energy_spread_mev: float

    def __post_init__(
        self,
    ) -> None:
        if not self.manifold_id.strip():
            raise ValueError(
                "manifold_id must be non-empty"
            )

        if (
            not isfinite(
                self.r_angstrom
            )
            or self.r_angstrom <= 0.0
        ):
            raise ValueError(
                "r_angstrom must be finite and > 0"
            )

        if not isfinite(
            self.energy_center_hartree
        ):
            raise ValueError(
                "energy_center_hartree must be finite"
            )

        if (
            not isfinite(
                self.energy_spread_mev
            )
            or self.energy_spread_mev < 0.0
        ):
            raise ValueError(
                "energy_spread_mev must be finite and >= 0"
            )


@dataclass(frozen=True)
class ManifoldPEC:
    component_id: str

    points: tuple[
        ManifoldPECPoint,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:
        if not self.component_id.strip():
            raise ValueError(
                "component_id must be non-empty"
            )

        if len(
            self.points
        ) < 2:
            raise ValueError(
                "manifold PEC requires at least two points"
            )

        for left, right in zip(
            self.points,
            self.points[1:],
        ):
            if (
                right.r_angstrom
                <= left.r_angstrom
            ):
                raise ValueError(
                    "manifold PEC geometries must be strictly increasing"
                )


class ManifoldPECRejectionReason(
    str,
    Enum,
):
    BRANCHING_TOPOLOGY = (
        "BRANCHING_TOPOLOGY"
    )

    GAUGE_UNRESOLVED_BOUNDARY = (
        "GAUGE_UNRESOLVED_BOUNDARY"
    )

    INSUFFICIENT_POINTS = (
        "INSUFFICIENT_POINTS"
    )


@dataclass(frozen=True)
class RejectedManifoldPECComponent:
    component_id: str

    member_manifold_ids: tuple[
        str,
        ...,
    ]

    reason: (
        ManifoldPECRejectionReason
    )


@dataclass(frozen=True)
class ManifoldPECConstructionResult:
    pecs: tuple[
        ManifoldPEC,
        ...,
    ]

    rejected_components: tuple[
        RejectedManifoldPECComponent,
        ...,
    ]


def _context(
    manifold: ElectronicManifoldPoint,
) -> tuple:
    return (
        manifold.molecule,
        manifold.atom_a,
        manifold.atom_b,
        manifold.charge,
        manifold.spin_2s,
        manifold.functional,
        manifold.basis,
        manifold.reference,
        manifold.ecp_assignments,
    )


def _validate_manifolds(
    manifolds: Sequence[
        ElectronicManifoldPoint
    ],
) -> dict[
    str,
    ElectronicManifoldPoint,
]:
    if not manifolds:
        raise ValueError(
            "manifold graph requires at least one manifold"
        )

    by_id = {}

    for manifold in manifolds:
        if (
            manifold.manifold_id
            in by_id
        ):
            raise ValueError(
                "duplicate manifold_id: "
                f"{manifold.manifold_id}"
            )

        by_id[
            manifold.manifold_id
        ] = manifold

    contexts = {
        _context(
            manifold
        )
        for manifold
        in manifolds
    }

    if len(
        contexts
    ) != 1:
        raise ValueError(
            "all manifolds in one graph must share "
            "molecule, atoms, charge, spin sector, "
            "functional, basis, reference, and ECP context"
        )

    return by_id


def _layers(
    manifolds: Sequence[
        ElectronicManifoldPoint
    ],
) -> tuple[
    tuple[
        float,
        tuple[
            ElectronicManifoldPoint,
            ...,
        ],
    ],
    ...,
]:
    by_r = {}

    for manifold in manifolds:
        by_r.setdefault(
            manifold.r_angstrom,
            [],
        ).append(
            manifold
        )

    return tuple(
        (
            r,
            tuple(
                sorted(
                    by_r[r],
                    key=lambda item:
                        item.manifold_id,
                )
            ),
        )
        for r in sorted(
            by_r
        )
    )


def _component_id(
    manifold_ids: Sequence[
        str
    ],
) -> str:
    payload = "|".join(
        sorted(
            manifold_ids
        )
    )

    digest = sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    return (
        "manifold_component_"
        + digest
    )


def build_manifold_branch_graph(
    manifolds: Sequence[
        ElectronicManifoldPoint
    ],
    *,
    thresholds: ManifoldThresholds,
) -> ManifoldBranchGraph:
    """
    Construct a conservative layered graph of electronic manifolds.

    No gauge-unresolved comparison is promoted automatically. Explicit
    bridge logic remains a separate workflow layer.
    """
    by_id = (
        _validate_manifolds(
            manifolds
        )
    )

    layers = _layers(
        manifolds
    )

    comparisons = []

    continuous_edges = []
    gauge_edges = []
    discontinuous_edges = []

    for (
        _left_r,
        left_layer,
    ), (
        _right_r,
        right_layer,
    ) in zip(
        layers,
        layers[1:],
    ):
        for left in left_layer:
            for right in right_layer:
                comparison = (
                    compare_electronic_manifolds(
                        left,
                        right,
                        thresholds=(
                            thresholds
                        ),
                    )
                )

                comparisons.append(
                    comparison
                )

                edge = (
                    left.manifold_id,
                    right.manifold_id,
                )

                if (
                    comparison.relation
                    == ManifoldContinuityRelation.CONTINUOUS
                ):
                    continuous_edges.append(
                        edge
                    )

                elif (
                    comparison.relation
                    == ManifoldContinuityRelation.GAUGE_UNRESOLVED
                ):
                    gauge_edges.append(
                        edge
                    )

                elif (
                    comparison.relation
                    == ManifoldContinuityRelation.DISCONTINUOUS
                ):
                    discontinuous_edges.append(
                        edge
                    )

                else:
                    raise RuntimeError(
                        "unknown manifold continuity relation: "
                        f"{comparison.relation}"
                    )

    adjacency = {
        manifold_id: set()
        for manifold_id
        in by_id
    }

    for left_id, right_id in (
        continuous_edges
    ):
        adjacency[
            left_id
        ].add(
            right_id
        )

        adjacency[
            right_id
        ].add(
            left_id
        )

    gauge_ids = {
        manifold_id
        for edge in gauge_edges
        for manifold_id in edge
    }

    visited = set()
    components = []

    topology_ambiguous_ids = set()

    ordered_ids = sorted(
        by_id,
        key=lambda manifold_id: (
            by_id[
                manifold_id
            ].r_angstrom,
            manifold_id,
        ),
    )

    for start_id in ordered_ids:
        if start_id in visited:
            continue

        stack = [
            start_id
        ]

        member_ids = []

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(
                current
            )

            member_ids.append(
                current
            )

            stack.extend(
                sorted(
                    adjacency[
                        current
                    ],
                    reverse=True,
                )
            )

        member_ids = sorted(
            member_ids,
            key=lambda manifold_id: (
                by_id[
                    manifold_id
                ].r_angstrom,
                manifold_id,
            ),
        )

        geometry_values = tuple(
            sorted(
                {
                    by_id[
                        manifold_id
                    ].r_angstrom
                    for manifold_id
                    in member_ids
                }
            )
        )

        has_branching = (
            len(
                geometry_values
            )
            != len(
                member_ids
            )
        )

        if has_branching:
            topology_ambiguous_ids.update(
                member_ids
            )

        has_gauge_boundary = any(
            manifold_id
            in gauge_ids
            for manifold_id
            in member_ids
        )

        is_unambiguous = not (
            has_branching
            or has_gauge_boundary
        )

        components.append(
            ManifoldBranchComponent(
                component_id=(
                    _component_id(
                        member_ids
                    )
                ),
                member_manifold_ids=tuple(
                    member_ids
                ),
                geometry_values=(
                    geometry_values
                ),
                has_branching=(
                    has_branching
                ),
                has_gauge_unresolved_boundary=(
                    has_gauge_boundary
                ),
                is_unambiguous=(
                    is_unambiguous
                ),
            )
        )

    return ManifoldBranchGraph(
        geometry_values=tuple(
            r
            for r, _layer
            in layers
        ),
        manifold_ids_by_geometry=tuple(
            (
                r,
                tuple(
                    manifold.manifold_id
                    for manifold
                    in layer
                ),
            )
            for r, layer
            in layers
        ),
        comparisons=tuple(
            comparisons
        ),
        continuous_edges=tuple(
            continuous_edges
        ),
        gauge_unresolved_edges=tuple(
            gauge_edges
        ),
        discontinuous_edges=tuple(
            discontinuous_edges
        ),
        components=tuple(
            components
        ),
        topology_ambiguous_manifold_ids=tuple(
            sorted(
                topology_ambiguous_ids
            )
        ),
        gauge_unresolved_manifold_ids=tuple(
            sorted(
                gauge_ids
            )
        ),
    )


def construct_manifold_pecs(
    manifolds: Sequence[
        ElectronicManifoldPoint
    ],
    graph: ManifoldBranchGraph,
) -> ManifoldPECConstructionResult:
    by_id = {
        manifold.manifold_id:
            manifold
        for manifold
        in manifolds
    }

    graph_ids = {
        manifold_id
        for _r, manifold_ids
        in graph.manifold_ids_by_geometry
        for manifold_id
        in manifold_ids
    }

    if (
        set(
            by_id
        )
        != graph_ids
    ):
        raise ValueError(
            "manifold set does not match graph manifold set"
        )

    pecs = []
    rejected = []

    for component in (
        graph.components
    ):
        if component.has_branching:
            rejected.append(
                RejectedManifoldPECComponent(
                    component_id=(
                        component.component_id
                    ),
                    member_manifold_ids=(
                        component.member_manifold_ids
                    ),
                    reason=(
                        ManifoldPECRejectionReason
                        .BRANCHING_TOPOLOGY
                    ),
                )
            )

            continue

        if (
            component
            .has_gauge_unresolved_boundary
        ):
            rejected.append(
                RejectedManifoldPECComponent(
                    component_id=(
                        component.component_id
                    ),
                    member_manifold_ids=(
                        component.member_manifold_ids
                    ),
                    reason=(
                        ManifoldPECRejectionReason
                        .GAUGE_UNRESOLVED_BOUNDARY
                    ),
                )
            )

            continue

        if len(
            component.member_manifold_ids
        ) < 2:
            rejected.append(
                RejectedManifoldPECComponent(
                    component_id=(
                        component.component_id
                    ),
                    member_manifold_ids=(
                        component.member_manifold_ids
                    ),
                    reason=(
                        ManifoldPECRejectionReason
                        .INSUFFICIENT_POINTS
                    ),
                )
            )

            continue

        points = tuple(
            ManifoldPECPoint(
                manifold_id=(
                    manifold.manifold_id
                ),
                r_angstrom=(
                    manifold.r_angstrom
                ),
                energy_center_hartree=(
                    manifold.energy_center_hartree
                ),
                energy_spread_mev=(
                    manifold.energy_spread_mev
                ),
            )
            for manifold
            in sorted(
                (
                    by_id[
                        manifold_id
                    ]
                    for manifold_id
                    in component.member_manifold_ids
                ),
                key=lambda item:
                    item.r_angstrom,
            )
        )

        pecs.append(
            ManifoldPEC(
                component_id=(
                    component.component_id
                ),
                points=points,
            )
        )

    return ManifoldPECConstructionResult(
        pecs=tuple(
            pecs
        ),
        rejected_components=tuple(
            rejected
        ),
    )


def scout_manifold_pec_minimum(
    pec: ManifoldPEC,
    *,
    thresholds: MinimumScoutThresholds,
) -> MinimumScoutResult:
    """
    Apply the existing conservative discrete minimum policy to a
    manifold-energy curve.

    LocalPECPoint.root_id carries the manifold_id in this adapter.
    """
    proxy = LocalPEC(
        component_id=(
            pec.component_id
        ),
        points=tuple(
            LocalPECPoint(
                root_id=(
                    point.manifold_id
                ),
                r_angstrom=(
                    point.r_angstrom
                ),
                energy_hartree=(
                    point.energy_center_hartree
                ),
            )
            for point
            in pec.points
        ),
    )

    return scout_local_pec_minimum(
        proxy,
        thresholds=thresholds,
    )
