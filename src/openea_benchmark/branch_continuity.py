from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np



HARTREE_TO_EV = 27.211386245988


class BranchRelation(str, Enum):
    """
    Pairwise continuity relation between roots at different R.

    CONTINUOUS:
        Available occupied-subspace and spin evidence supports following
        the same electronic branch.

    DISCONTINUOUS:
        Available evidence clearly indicates a branch change.

    AMBIGUOUS:
        Evidence is intermediate or the geometry spacing is too large for
        the configured continuity policy.

    AMBIGUOUS is a scientific result and must not silently be converted
    into either of the other relations.
    """

    CONTINUOUS = "CONTINUOUS"
    DISCONTINUOUS = "DISCONTINUOUS"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class BranchThresholds:
    """
    Explicit branch-continuity thresholds.

    No production defaults are provided.

    The workflow must validate these values before prediction calculations.

    continuous_occ_min
        Minimum singular value of every occupied spin subspace required for
        a positive continuity assignment.

    discontinuous_occ_min
        If any occupied-subspace minimum singular value falls at or below
        this value, the pair may be classified as discontinuous.

    continuous_delta_s2 / discontinuous_delta_s2
        Corresponding bounds for changes in <S^2>.

    max_step_angstrom
        Maximum R spacing at which an automatic continuity assignment is
        allowed. Larger gaps remain AMBIGUOUS even if orbital overlap happens
        to be high.
    """

    continuous_occ_min: float
    discontinuous_occ_min: float

    continuous_delta_s2: float
    discontinuous_delta_s2: float

    max_step_angstrom: float

    def __post_init__(self) -> None:
        values = (
            self.continuous_occ_min,
            self.discontinuous_occ_min,
            self.continuous_delta_s2,
            self.discontinuous_delta_s2,
            self.max_step_angstrom,
        )

        if any(
            not isfinite(float(x))
            for x in values
        ):
            raise ValueError(
                "branch thresholds must be finite"
            )

        if not (
            0.0
            <= self.discontinuous_occ_min
            < self.continuous_occ_min
            <= 1.0
        ):
            raise ValueError(
                "occupied-overlap thresholds must satisfy "
                "0 <= discontinuous < continuous <= 1"
            )

        if not (
            0.0
            <= self.continuous_delta_s2
            < self.discontinuous_delta_s2
        ):
            raise ValueError(
                "delta-S2 thresholds must satisfy "
                "0 <= continuous < discontinuous"
            )

        if self.max_step_angstrom <= 0.0:
            raise ValueError(
                "max_step_angstrom must be > 0"
            )


@dataclass(frozen=True)
class BranchComparison:
    root_a: str
    root_b: str

    r_a_angstrom: float
    r_b_angstrom: float
    delta_r_angstrom: float

    delta_energy_mev: float
    delta_s2: float

    alpha_singular_values: tuple[
        float,
        ...,
    ]
    beta_singular_values: tuple[
        float,
        ...,
    ]

    alpha_occ_overlap_min: float
    alpha_occ_overlap_mean: float

    beta_occ_overlap_min: float
    beta_occ_overlap_mean: float

    relation: BranchRelation


def classify_branch_metrics(
    *,
    delta_r_angstrom: float,
    delta_s2: float,
    alpha_occ_overlap_min: float,
    beta_occ_overlap_min: float,
    thresholds: BranchThresholds,
) -> BranchRelation:
    """
    Conservative branch-continuity classification.

    Energy is deliberately absent from this decision.

    Along a physical PEC a branch energy can move strongly with R, so energy
    proximity is stored as a diagnostic but is not used as primary branch
    identity evidence.
    """
    values = (
        delta_r_angstrom,
        delta_s2,
        alpha_occ_overlap_min,
        beta_occ_overlap_min,
    )

    if any(
        not isfinite(float(x))
        for x in values
    ):
        raise ValueError(
            "branch metrics must be finite"
        )

    dr = abs(
        float(delta_r_angstrom)
    )

    ds2 = abs(
        float(delta_s2)
    )

    alpha = float(
        alpha_occ_overlap_min
    )

    beta = float(
        beta_occ_overlap_min
    )

    if not (
        0.0 <= alpha <= 1.0
        and 0.0 <= beta <= 1.0
    ):
        raise ValueError(
            "occupied-subspace overlaps must lie in [0, 1]"
        )

    if dr <= 0.0:
        raise ValueError(
            "branch comparison requires different geometries"
        )

    if dr > thresholds.max_step_angstrom:
        return BranchRelation.AMBIGUOUS

    worst_overlap = min(
        alpha,
        beta,
    )

    if (
        worst_overlap
        >= thresholds.continuous_occ_min
        and ds2
        <= thresholds.continuous_delta_s2
    ):
        return BranchRelation.CONTINUOUS

    if (
        worst_overlap
        <= thresholds.discontinuous_occ_min
        or ds2
        >= thresholds.discontinuous_delta_s2
    ):
        return BranchRelation.DISCONTINUOUS

    return BranchRelation.AMBIGUOUS


