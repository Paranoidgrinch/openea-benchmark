"""Small immutable evidence and interval contracts for the adaptive engine.

Intervals represent documented ESTIMATED ranges, not guaranteed mathematical
error bounds or calibrated statistical confidence intervals. The final sign
classifier refuses to decide in the absence of required scientific evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class DiagnosticID(str, Enum):
    D01_SPIN = 'D01'
    D02_SCF_STABILITY = 'D02'
    D03_MULTIREFERENCE = 'D03'
    D04_STATE_COMPETITION = 'D04'
    D05_REFERENCE_SENSITIVITY = 'D05'
    D06_TRIPLES_RELIABILITY = 'D06'
    D07_DIFFUSE_BASIS = 'D07'
    D08_ATTACHMENT = 'D08'
    D09_PEC_ASYMPTOTES = 'D09'
    D10_SCALAR_RELATIVITY = 'D10'
    D11_SOC = 'D11'
    D12_NUCLEAR_MOTION = 'D12'


class ReviewStatus(str, Enum):
    PENDING = 'PENDING'
    CLEARED = 'CLEARED'
    CONFIRMED = 'CONFIRMED'
    UNRESOLVED = 'UNRESOLVED'
    NOT_APPLICABLE = 'NOT_APPLICABLE'


class MethodRole(str, Enum):
    """Canonical scientific role of a method or calculation."""

    PRODUCTION = 'PRODUCTION'
    DIAGNOSTIC = 'DIAGNOSTIC'
    REFINEMENT = 'REFINEMENT'
    VALIDATION = 'VALIDATION'


@dataclass(frozen=True)
class Review:
    """An epistemic review, not merely a software exit status.

    For gates, CLEARED means the corresponding requirement has sufficient
    scientific evidence; never infer it just because an SCF calculation ran.
    """
    status: ReviewStatus
    evidence_ids: tuple[str, ...] = ()
    rationale: str = ''

    def __post_init__(self) -> None:
        if self.status in (ReviewStatus.CLEARED, ReviewStatus.CONFIRMED):
            if not self.evidence_ids or any(not x.strip() for x in self.evidence_ids):
                raise ValueError('CLEARED/CONFIRMED requires concrete evidence IDs')
        if self.status == ReviewStatus.NOT_APPLICABLE and not self.rationale.strip():
            raise ValueError('NOT_APPLICABLE requires a physical rationale')


@dataclass(frozen=True)
class DiagnosticRecord:
    identifier: DiagnosticID
    review: Review
    state_refs: tuple[str, ...] = ()
    recommended_branch: str | None = None


@dataclass(frozen=True, order=True)
class Interval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (isfinite(self.lower) and isfinite(self.upper)):
            raise ValueError('Interval endpoints must be finite')
        if self.lower > self.upper:
            raise ValueError('Interval lower must not exceed upper')

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2.0

    def add(self, other: 'Interval') -> 'Interval':
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def subtract(self, other: 'Interval') -> 'Interval':
        return Interval(self.lower - other.upper, self.upper - other.lower)


