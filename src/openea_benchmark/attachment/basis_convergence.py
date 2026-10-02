"""Electronic-EA basis/diffuse convergence evidence.

This layer does not average basis-set results and does not call a finite-basis
EA a CBS value.  It independently audits:

1. cardinal convergence at fixed augmentation level;
2. diffuse convergence at fixed cardinal number.

The output is convergence evidence plus a requested next action.  Residual
estimates are conservative convergence estimates, not statistical confidence
intervals and not a complete production EA uncertainty budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class BasisConvergenceStatus(str, Enum):
    CLEARED = "CLEARED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"


class BasisConvergenceAction(str, Enum):
    NONE = "NONE"
    COMPUTE_NEXT_CARDINAL = "COMPUTE_NEXT_CARDINAL"
    COMPUTE_DOUBLE_AUGMENTED = "COMPUTE_DOUBLE_AUGMENTED"
    COMPUTE_MORE_DIFFUSE = "COMPUTE_MORE_DIFFUSE"
    REPAIR_INPUT_EVIDENCE = "REPAIR_INPUT_EVIDENCE"


@dataclass(frozen=True)
class EAIntervalEV:
    lower_ev: float
    central_ev: float
    upper_ev: float

    def __post_init__(self) -> None:
        lo = float(self.lower_ev)
        mid = float(self.central_ev)
        hi = float(self.upper_ev)
        if not all(isfinite(x) for x in (lo, mid, hi)):
            raise ValueError("EA interval values must be finite")
        if lo > mid or mid > hi:
            raise ValueError("EA interval must satisfy lower <= central <= upper")

    @property
    def half_width_ev(self) -> float:
        return max(
            self.central_ev - self.lower_ev,
            self.upper_ev - self.central_ev,
        )


@dataclass(frozen=True)
class ElectronicEABasisPoint:
    basis_name: str
    cardinal_number: int
    augmentation_level: int
    method_signature: str
    ea: EAIntervalEV
    decision_status: str = "BOUND"
    evidence_quality: str = "CONVERGENCE_ESTIMATED"
    is_production_ea: bool = False

    def __post_init__(self) -> None:
        if not self.basis_name.strip():
            raise ValueError("basis_name must be non-empty")
        if int(self.cardinal_number) < 2:
            raise ValueError("cardinal_number must be >= 2")
        if int(self.augmentation_level) < 0:
            raise ValueError("augmentation_level must be >= 0")
        if not self.method_signature.strip():
            raise ValueError("method_signature must be non-empty")
        if self.is_production_ea:
            raise ValueError(
                "finite-basis convergence points cannot be promoted to production EA here"
            )


@dataclass(frozen=True)
class BasisConvergenceSettings:
    cardinal_increment_target_ev: float
    cardinal_contraction_ratio_max: float
    diffuse_increment_target_ev: float
    diffuse_contraction_ratio_max: float
    force_double_augmentation: bool

    def __post_init__(self) -> None:
        for name in (
            "cardinal_increment_target_ev",
            "diffuse_increment_target_ev",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")

        for name in (
            "cardinal_contraction_ratio_max",
            "diffuse_contraction_ratio_max",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not (0.0 < value < 1.0):
                raise ValueError(f"{name} must be finite and between 0 and 1")


@dataclass(frozen=True)
class CardinalConvergenceAssessment:
    status: BasisConvergenceStatus
    action: BasisConvergenceAction
    augmentation_level: int
    highest_cardinal: int | None
    latest_increment_bound_ev: float | None
    contraction_ratio: float | None
    residual_estimate_ev: float | None
    expanded_highest_interval: EAIntervalEV | None
    evidence_quality: str
    evidence: tuple[str, ...]
    rationale: str
    is_production_ea: bool = False


@dataclass(frozen=True)
class DiffuseConvergenceAssessment:
    status: BasisConvergenceStatus
    action: BasisConvergenceAction
    cardinal_number: int
    highest_augmentation_level: int | None
    latest_increment_bound_ev: float | None
    contraction_ratio: float | None
    residual_estimate_ev: float | None
    expanded_best_interval: EAIntervalEV | None
    evidence_quality: str
    evidence: tuple[str, ...]
    rationale: str
    is_production_ea: bool = False


def _difference_interval(
    newer: ElectronicEABasisPoint,
    older: ElectronicEABasisPoint,
) -> tuple[float, float]:
    return (
        newer.ea.lower_ev - older.ea.upper_ev,
        newer.ea.upper_ev - older.ea.lower_ev,
    )


def _absolute_change_bound(
    newer: ElectronicEABasisPoint,
    older: ElectronicEABasisPoint,
) -> float:
    lo, hi = _difference_interval(newer, older)
    return max(abs(lo), abs(hi))


def _central_change(
    newer: ElectronicEABasisPoint,
    older: ElectronicEABasisPoint,
) -> float:
    return newer.ea.central_ev - older.ea.central_ev


def _expanded_interval(
    point: ElectronicEABasisPoint,
    residual_ev: float,
) -> EAIntervalEV:
    return EAIntervalEV(
        lower_ev=point.ea.lower_ev - residual_ev,
        central_ev=point.ea.central_ev,
        upper_ev=point.ea.upper_ev + residual_ev,
    )


def _validate_compatible(points: tuple[ElectronicEABasisPoint, ...]) -> str:
    methods = {p.method_signature for p in points}
    if len(methods) != 1:
        raise ValueError("basis convergence points must share one method_signature")

    bad = [p.basis_name for p in points if p.decision_status != "BOUND"]
    if bad:
        raise ValueError(
            "basis convergence requires BOUND electronic-EA points: "
            + ", ".join(bad)
        )

    return next(iter(methods))


def assess_cardinal_convergence(
    points: tuple[ElectronicEABasisPoint, ...],
    *,
    augmentation_level: int,
    settings: BasisConvergenceSettings,
) -> CardinalConvergenceAssessment:
    series = tuple(
        sorted(
            (p for p in points if p.augmentation_level == augmentation_level),
            key=lambda p: p.cardinal_number,
        )
    )

    if len(series) < 3:
        return CardinalConvergenceAssessment(
            status=BasisConvergenceStatus.NEED_MORE_EVIDENCE,
            action=BasisConvergenceAction.COMPUTE_NEXT_CARDINAL,
            augmentation_level=augmentation_level,
            highest_cardinal=(
                max((p.cardinal_number for p in series), default=None)
            ),
            latest_increment_bound_ev=None,
            contraction_ratio=None,
            residual_estimate_ev=None,
            expanded_highest_interval=None,
            evidence_quality="UNKNOWN",
            evidence=("CARDINAL_SERIES_REQUIRES_AT_LEAST_THREE_POINTS",),
            rationale="At least three same-augmentation cardinal points are required",
        )

    _validate_compatible(series)

    cardinals = [p.cardinal_number for p in series]
    if len(cardinals) != len(set(cardinals)):
        return CardinalConvergenceAssessment(
            status=BasisConvergenceStatus.UNRESOLVED,
            action=BasisConvergenceAction.REPAIR_INPUT_EVIDENCE,
            augmentation_level=augmentation_level,
            highest_cardinal=max(cardinals),
            latest_increment_bound_ev=None,
            contraction_ratio=None,
            residual_estimate_ev=None,
            expanded_highest_interval=None,
            evidence_quality="UNKNOWN",
            evidence=("DUPLICATE_CARDINAL_EVIDENCE",),
            rationale="More than one point exists for the same cardinal number",
        )

    high3 = series[-3:]
    x0, x1, x2 = (p.cardinal_number for p in high3)
    if not (x1 == x0 + 1 and x2 == x1 + 1):
        return CardinalConvergenceAssessment(
            status=BasisConvergenceStatus.NEED_MORE_EVIDENCE,
            action=BasisConvergenceAction.COMPUTE_NEXT_CARDINAL,
            augmentation_level=augmentation_level,
            highest_cardinal=x2,
            latest_increment_bound_ev=None,
            contraction_ratio=None,
            residual_estimate_ev=None,
            expanded_highest_interval=None,
            evidence_quality="UNKNOWN",
            evidence=("HIGHEST_THREE_CARDINALS_NOT_CONSECUTIVE",),
            rationale="The highest three cardinal points must be consecutive",
        )

    p0, p1, p2 = high3
    previous_change = _central_change(p1, p0)
    latest_change = _central_change(p2, p1)
    latest_bound = _absolute_change_bound(p2, p1)

    tiny = 1.0e-15
    if abs(previous_change) <= tiny:
        contraction = 0.0 if abs(latest_change) <= tiny else None
    else:
        contraction = abs(latest_change) / abs(previous_change)

    same_direction = (
        abs(previous_change) <= tiny
        or abs(latest_change) <= tiny
        or previous_change * latest_change > 0.0
    )

    if (
        contraction is not None
        and same_direction
        and contraction < 1.0
        and contraction <= settings.cardinal_contraction_ratio_max
        and latest_bound <= settings.cardinal_increment_target_ev
    ):
        residual = latest_bound / max(1.0 - contraction, 1.0e-12)
        return CardinalConvergenceAssessment(
            status=BasisConvergenceStatus.CLEARED,
            action=BasisConvergenceAction.NONE,
            augmentation_level=augmentation_level,
            highest_cardinal=x2,
            latest_increment_bound_ev=latest_bound,
            contraction_ratio=contraction,
            residual_estimate_ev=residual,
            expanded_highest_interval=_expanded_interval(p2, residual),
            evidence_quality="CONVERGENCE_ESTIMATED",
            evidence=(
                "THREE_CONSECUTIVE_CARDINAL_POINTS",
                "LATEST_CARDINAL_INCREMENT_WITHIN_TARGET",
                "CARDINAL_INCREMENT_CONTRACTING",
                "NO_BASIS_AVERAGING",
            ),
            rationale=(
                "The same-augmentation cardinal series is contracting and the "
                "latest interval-aware increment is within the explicit target"
            ),
        )

    evidence = ["CARDINAL_CONVERGENCE_NOT_CLEARED"]
    if not same_direction:
        evidence.append("CARDINAL_SERIES_OSCILLATORY")
    if contraction is None or contraction >= 1.0:
        evidence.append("CARDINAL_INCREMENT_NOT_CONTRACTING")
    elif contraction > settings.cardinal_contraction_ratio_max:
        evidence.append("CARDINAL_CONTRACTION_TOO_SLOW")
    if latest_bound > settings.cardinal_increment_target_ev:
        evidence.append("LATEST_CARDINAL_INCREMENT_ABOVE_TARGET")

    return CardinalConvergenceAssessment(
        status=BasisConvergenceStatus.NEED_MORE_EVIDENCE,
        action=BasisConvergenceAction.COMPUTE_NEXT_CARDINAL,
        augmentation_level=augmentation_level,
        highest_cardinal=x2,
        latest_increment_bound_ev=latest_bound,
        contraction_ratio=contraction,
        residual_estimate_ev=None,
        expanded_highest_interval=None,
        evidence_quality="UNKNOWN",
        evidence=tuple(evidence),
        rationale="A higher cardinal point is required before basis convergence can be cleared",
    )


def assess_diffuse_convergence(
    points: tuple[ElectronicEABasisPoint, ...],
    *,
    cardinal_number: int,
    settings: BasisConvergenceSettings,
) -> DiffuseConvergenceAssessment:
    series = tuple(
        sorted(
            (p for p in points if p.cardinal_number == cardinal_number),
            key=lambda p: p.augmentation_level,
        )
    )

    if len(series) < 2:
        return DiffuseConvergenceAssessment(
            status=BasisConvergenceStatus.NEED_MORE_EVIDENCE,
            action=BasisConvergenceAction.COMPUTE_MORE_DIFFUSE,
            cardinal_number=cardinal_number,
            highest_augmentation_level=(
                max((p.augmentation_level for p in series), default=None)
            ),
            latest_increment_bound_ev=None,
            contraction_ratio=None,
            residual_estimate_ev=None,
            expanded_best_interval=None,
            evidence_quality="UNKNOWN",
            evidence=("DIFFUSE_SERIES_REQUIRES_AT_LEAST_TWO_POINTS",),
            rationale="At least two augmentation levels are required",
        )

    _validate_compatible(series)

    levels = [p.augmentation_level for p in series]
    if len(levels) != len(set(levels)):
        return DiffuseConvergenceAssessment(
            status=BasisConvergenceStatus.UNRESOLVED,
            action=BasisConvergenceAction.REPAIR_INPUT_EVIDENCE,
            cardinal_number=cardinal_number,
            highest_augmentation_level=max(levels),
            latest_increment_bound_ev=None,
            contraction_ratio=None,
            residual_estimate_ev=None,
            expanded_best_interval=None,
            evidence_quality="UNKNOWN",
            evidence=("DUPLICATE_AUGMENTATION_EVIDENCE",),
            rationale="More than one point exists for the same augmentation level",
        )

    by_level = {p.augmentation_level: p for p in series}
    if 0 not in by_level or 1 not in by_level:
        return DiffuseConvergenceAssessment(
            status=BasisConvergenceStatus.NEED_MORE_EVIDENCE,
            action=BasisConvergenceAction.COMPUTE_MORE_DIFFUSE,
            cardinal_number=cardinal_number,
            highest_augmentation_level=max(levels),
            latest_increment_bound_ev=None,
            contraction_ratio=None,
            residual_estimate_ev=None,
            expanded_best_interval=None,
            evidence_quality="UNKNOWN",
            evidence=("NONAUG_AND_AUG_PAIR_REQUIRED",),
            rationale="Diffuse convergence requires matched non-augmented and augmented evidence",
        )

    p0 = by_level[0]
    p1 = by_level[1]
    first_bound = _absolute_change_bound(p1, p0)
    first_change = _central_change(p1, p0)

    need_double = (
        settings.force_double_augmentation
        or first_bound > settings.diffuse_increment_target_ev
    )

    if not need_double:
        residual = first_bound
        return DiffuseConvergenceAssessment(
            status=BasisConvergenceStatus.CLEARED,
            action=BasisConvergenceAction.NONE,
            cardinal_number=cardinal_number,
            highest_augmentation_level=1,
            latest_increment_bound_ev=first_bound,
            contraction_ratio=None,
            residual_estimate_ev=residual,
            expanded_best_interval=_expanded_interval(p1, residual),
            evidence_quality="CONVERGENCE_ESTIMATED",
            evidence=(
                "NONAUG_TO_AUG_INCREMENT_WITHIN_TARGET",
                "DOUBLE_AUGMENTATION_NOT_TRIGGERED_BY_POLICY",
                "NO_BASIS_AVERAGING",
            ),
            rationale=(
                "The interval-aware non-augmented to augmented shift is within "
                "the explicit diffuse target"
            ),
        )

    if 2 not in by_level:
        return DiffuseConvergenceAssessment(
            status=BasisConvergenceStatus.NEED_MORE_EVIDENCE,
            action=BasisConvergenceAction.COMPUTE_DOUBLE_AUGMENTED,
            cardinal_number=cardinal_number,
            highest_augmentation_level=1,
            latest_increment_bound_ev=first_bound,
            contraction_ratio=None,
            residual_estimate_ev=None,
            expanded_best_interval=None,
            evidence_quality="UNKNOWN",
            evidence=(
                "DOUBLE_AUGMENTATION_REQUIRED",
                "DOUBLE_AUGMENTED_POINT_MISSING",
            ),
            rationale="A double-augmented point is required by the diffuse-convergence policy",
        )

    p2 = by_level[2]
    latest_bound = _absolute_change_bound(p2, p1)
    latest_change = _central_change(p2, p1)

    tiny = 1.0e-15
    if abs(first_change) <= tiny:
        contraction = 0.0 if abs(latest_change) <= tiny else None
    else:
        contraction = abs(latest_change) / abs(first_change)

    same_direction = (
        abs(first_change) <= tiny
        or abs(latest_change) <= tiny
        or first_change * latest_change > 0.0
    )

    if (
        contraction is not None
        and same_direction
        and contraction < 1.0
        and contraction <= settings.diffuse_contraction_ratio_max
        and latest_bound <= settings.diffuse_increment_target_ev
    ):
        residual = latest_bound / max(1.0 - contraction, 1.0e-12)
        return DiffuseConvergenceAssessment(
            status=BasisConvergenceStatus.CLEARED,
            action=BasisConvergenceAction.NONE,
            cardinal_number=cardinal_number,
            highest_augmentation_level=2,
            latest_increment_bound_ev=latest_bound,
            contraction_ratio=contraction,
            residual_estimate_ev=residual,
            expanded_best_interval=_expanded_interval(p2, residual),
            evidence_quality="CONVERGENCE_ESTIMATED",
            evidence=(
                "NONAUG_AUG_DOUBLEAUG_SERIES_AVAILABLE",
                "LATEST_DIFFUSE_INCREMENT_WITHIN_TARGET",
                "DIFFUSE_INCREMENT_CONTRACTING",
                "NO_BASIS_AVERAGING",
            ),
            rationale=(
                "The augmentation series is contracting and the aug->d-aug "
                "interval-aware increment is within the explicit target"
            ),
        )

    evidence = ["DIFFUSE_CONVERGENCE_NOT_CLEARED"]
    if not same_direction:
        evidence.append("DIFFUSE_SERIES_OSCILLATORY")
    if contraction is None or contraction >= 1.0:
        evidence.append("DIFFUSE_INCREMENT_NOT_CONTRACTING")
    elif contraction > settings.diffuse_contraction_ratio_max:
        evidence.append("DIFFUSE_CONTRACTION_TOO_SLOW")
    if latest_bound > settings.diffuse_increment_target_ev:
        evidence.append("LATEST_DIFFUSE_INCREMENT_ABOVE_TARGET")

    return DiffuseConvergenceAssessment(
        status=BasisConvergenceStatus.NEED_MORE_EVIDENCE,
        action=BasisConvergenceAction.COMPUTE_MORE_DIFFUSE,
        cardinal_number=cardinal_number,
        highest_augmentation_level=2,
        latest_increment_bound_ev=latest_bound,
        contraction_ratio=contraction,
        residual_estimate_ev=None,
        expanded_best_interval=None,
        evidence_quality="UNKNOWN",
        evidence=tuple(evidence),
        rationale="Additional diffuse evidence is required before convergence can be cleared",
    )
