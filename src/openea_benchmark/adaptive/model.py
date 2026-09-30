"""Small immutable evidence and interval contracts for the adaptive engine.

Intervals represent documented ESTIMATED ranges, not guaranteed mathematical
error bounds or calibrated statistical confidence intervals. The final sign
classifier refuses to decide in the absence of required scientific evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Iterable


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


class EvidenceQuality(str, Enum):
    DIRECTLY_TESTED = 'DIRECTLY_TESTED'
    CONVERGENCE_ESTIMATED = 'CONVERGENCE_ESTIMATED'
    INDIRECTLY_ESTIMATED = 'INDIRECTLY_ESTIMATED'
    UNKNOWN = 'UNKNOWN'


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


@dataclass(frozen=True)
class UncertaintyComponent:
    """An additive EA correction or uncertainty OFFSET, in electronvolts.

    A missing critical component cannot be silently substituted by zero.
    The component's interval is an estimated range for the actual correction.
    """
    name: str
    correction_ev: Interval | None
    quality: EvidenceQuality
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError('Component name must be nonempty')
        if self.quality is EvidenceQuality.UNKNOWN:
            if self.correction_ev is not None:
                raise ValueError('UNKNOWN requires an explicitly missing interval')
        elif self.correction_ev is None or not self.evidence_ids:
            raise ValueError('Estimated correction requires interval and evidence IDs')


@dataclass(frozen=True)
class EAEstimate:
    """Central EA baseline plus separately bounded corrections.

    `baseline_offset_ev` describes baseline uncertainty around `baseline_ev`.
    Unknown components block generation of a closed EA interval.
    Avoid adding corrections already included in the baseline.
    """
    baseline_ev: float
    baseline_offset_ev: Interval
    baseline_evidence_ids: tuple[str, ...]
    corrections: tuple[UncertaintyComponent, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.baseline_ev):
            raise ValueError('Baseline must be finite')
        if not self.baseline_evidence_ids:
            raise ValueError('Baseline requires evidence/provenance')
        names = [term.name for term in self.corrections]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate correction names; double counting risk')

    @property
    def missing_components(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.corrections if c.correction_ev is None)

    def estimated_interval(self) -> Interval | None:
        if self.missing_components:
            return None
        total = Interval(self.baseline_ev, self.baseline_ev).add(self.baseline_offset_ev)
        for term in self.corrections:
            assert term.correction_ev is not None
            total = total.add(term.correction_ev)
        return total


@dataclass(frozen=True)
class GateSet:
    """G1: state coverage, G2: attachment/continuum, G3: EA sign reliability.

    `CLEARED` on G3 does not imply that the *precision target* is achieved.
    It means the supplied EA interval is defensible for sign classification.
    """
    state_completeness: Review = field(default_factory=lambda: Review(ReviewStatus.PENDING))
    attachment_resolution: Review = field(default_factory=lambda: Review(ReviewStatus.PENDING))
    energy_reliability: Review = field(default_factory=lambda: Review(ReviewStatus.PENDING))

    def open_gates(self) -> tuple[str, ...]:
        return tuple(
            name for name, item in (
                ('G1_STATE_COMPLETENESS', self.state_completeness),
                ('G2_ATTACHMENT_RESOLUTION', self.attachment_resolution),
                ('G3_ENERGY_RELIABILITY', self.energy_reliability),
            ) if item.status != ReviewStatus.CLEARED
        )


def ground_state_interval(candidate_intervals: Iterable[Interval]) -> Interval:
    """Conservative min-envelope of INCLUDED candidates only.

    Does not itself certify that all physically relevant states were found;
    G1 must separately certify state-search completeness.
    """
    items = tuple(candidate_intervals)
    if not items:
        raise ValueError('At least one candidate energy interval is required')
    return Interval(min(x.lower for x in items), min(x.upper for x in items))


def ea_from_state_intervals(
    neutral_candidates: Iterable[Interval],
    anion_candidates: Iterable[Interval],
) -> Interval:
    """EA = minimum neutral E(v=0) minus minimum anion E(v=0).

    If any anion candidate is a discretized continuum artifact, the caller
    must withhold G2 and NOT use the sign to declare BOUND/UNBOUND.
    """
    return ground_state_interval(neutral_candidates).subtract(
        ground_state_interval(anion_candidates)
    )
