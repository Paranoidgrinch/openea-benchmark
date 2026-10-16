from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np



@dataclass(frozen=True)
class CheckpointAuditSettings:
    """
    Numerical consistency tolerances for checkpoint reconstruction.

    These are file/numerical integrity tolerances, not electronic-state
    identity thresholds and not physical accuracy criteria.
    """

    energy_tol_hartree: float = 1.0e-8
    geometry_tol_angstrom: float = 1.0e-10
    electron_trace_tol: float = 1.0e-7
    spin_trace_tol: float = 1.0e-7
    overlap_eigenvalue_floor: float = 1.0e-10

    def __post_init__(self) -> None:
        values = (
            self.energy_tol_hartree,
            self.geometry_tol_angstrom,
            self.electron_trace_tol,
            self.spin_trace_tol,
            self.overlap_eigenvalue_floor,
        )

        if any(
            not isfinite(float(x))
            or float(x) <= 0.0
            for x in values
        ):
            raise ValueError(
                "checkpoint audit tolerances must be finite and > 0"
            )


