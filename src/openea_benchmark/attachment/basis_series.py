"""Helpers for assembling real finite-basis electronic-EA series.

This module is intentionally thin: it translates already-completed electronic
EA decisions into `ElectronicEABasisPoint` records.  It does not recompute,
average, extrapolate, or reinterpret the underlying EA evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from .basis_convergence import (
    BasisConvergenceSettings,
    CardinalConvergenceAssessment,
    EAIntervalEV,
    ElectronicEABasisPoint,
    assess_cardinal_convergence,
)
from .decision import EADecision, EADecisionStatus


@dataclass(frozen=True)
class BasisSeriesDescriptor:
    basis_name: str
    cardinal_number: int
    augmentation_level: int

    def __post_init__(self) -> None:
        if not self.basis_name.strip():
            raise ValueError("basis_name must be non-empty")
        if self.cardinal_number < 2:
            raise ValueError("cardinal_number must be >= 2")
        if self.augmentation_level < 0:
            raise ValueError("augmentation_level must be >= 0")


def basis_point_from_ea_decision(
    *,
    descriptor: BasisSeriesDescriptor,
    method_signature: str,
    decision: EADecision,
    evidence_quality: str = "CONVERGENCE_ESTIMATED",
) -> ElectronicEABasisPoint:
    """Convert one cleared electronic-EA decision into basis evidence."""
    if decision.status is not EADecisionStatus.BOUND:
        raise ValueError(
            f"{descriptor.basis_name}: electronic EA decision is not BOUND"
        )
    if (
        decision.ea_lower_ev is None
        or decision.ea_central_ev is None
        or decision.ea_upper_ev is None
    ):
        raise ValueError(
            f"{descriptor.basis_name}: numeric EA interval is incomplete"
        )
    if not method_signature.strip():
        raise ValueError("method_signature must be non-empty")

    return ElectronicEABasisPoint(
        basis_name=descriptor.basis_name,
        cardinal_number=descriptor.cardinal_number,
        augmentation_level=descriptor.augmentation_level,
        method_signature=method_signature,
        ea=EAIntervalEV(
            lower_ev=decision.ea_lower_ev,
            central_ev=decision.ea_central_ev,
            upper_ev=decision.ea_upper_ev,
        ),
        decision_status=decision.status.value,
        evidence_quality=evidence_quality,
        is_production_ea=False,
    )


def assess_cardinal_ea_series(
    points: tuple[ElectronicEABasisPoint, ...],
    *,
    augmentation_level: int,
    settings: BasisConvergenceSettings,
) -> CardinalConvergenceAssessment:
    """Delegate a real EA series to the generic cardinal-convergence policy."""
    return assess_cardinal_convergence(
        points,
        augmentation_level=augmentation_level,
        settings=settings,
    )
