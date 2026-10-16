from __future__ import annotations

from dataclasses import dataclass
from math import isfinite



@dataclass(frozen=True)
class LocalPECPoint:
    """
    One geometry/energy point on a uniquely traced local electronic branch.
    """

    root_id: str
    r_angstrom: float
    energy_hartree: float

    def __post_init__(self) -> None:
        if not self.root_id.strip():
            raise ValueError(
                "root_id must be non-empty"
            )

        if (
            not isfinite(self.r_angstrom)
            or self.r_angstrom <= 0.0
        ):
            raise ValueError(
                "r_angstrom must be finite and > 0"
            )

        if not isfinite(
            self.energy_hartree
        ):
            raise ValueError(
                "energy_hartree must be finite"
            )


@dataclass(frozen=True)
class LocalPEC:
    """
    Ordered local potential-energy curve segment.

    A LocalPEC is only a uniquely traced electronic branch over at least two
    geometries. It does not identify a minimum, equilibrium geometry, ground
    state, or final physical electronic state.
    """

    component_id: str
    points: tuple[
        LocalPECPoint,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError(
                "component_id must be non-empty"
            )

        if len(self.points) < 2:
            raise ValueError(
                "local PEC requires at least two points"
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
                    "local PEC points must be strictly ordered by geometry"
                )


