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


class MethodRole(str, Enum):
    """Canonical scientific role of a method or calculation."""

    PRODUCTION = 'PRODUCTION'
    DIAGNOSTIC = 'DIAGNOSTIC'
    REFINEMENT = 'REFINEMENT'
    VALIDATION = 'VALIDATION'


class ScientificResolutionStatus(str, Enum):
    """Canonical terminal scientific outcome of OpenEA."""

    BOUND = 'BOUND'
    UNBOUND = 'UNBOUND'
    UNRESOLVED = 'UNRESOLVED'


class ReferenceCharacterStatus(str, Enum):
    """Outcome of the mandatory reference-character gate.

    The two aliases preserve source compatibility with the pre-refactor advisor
    while serializing to the new canonical vocabulary.
    """

    SAFE_SINGLE_REFERENCE = 'SAFE_SINGLE_REFERENCE'
    BORDERLINE = 'BORDERLINE'
    MULTIREFERENCE_RISK = 'MULTIREFERENCE_RISK'
    UNRESOLVED = 'UNRESOLVED'

    # Transitional source-level aliases; do not use in new provenance.
    SINGLE_REFERENCE = SAFE_SINGLE_REFERENCE
    MULTIREFERENCE = MULTIREFERENCE_RISK


@dataclass(frozen=True)
class ReferenceCharacterAssessment:
    """Reviewed gate result with explicit evidence and consequences.

    This object records a scientific assessment; it does not infer reference
    character from one scalar diagnostic.  `MULTIREFERENCE_RISK` routes away
    from the single-reference production path.  `BORDERLINE` may use that path
    only with expanded diagnostics and uncertainty.
    """

    status: ReferenceCharacterStatus
    evidence_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    diagnostic_ids: tuple[DiagnosticID, ...] = ()

    def __post_init__(self) -> None:
        if self.status is not ReferenceCharacterStatus.UNRESOLVED:
            if not self.evidence_ids or any(not x.strip() for x in self.evidence_ids):
                raise ValueError('Resolved reference-character assessment requires evidence IDs')
        if not self.reasons or any(not x.strip() for x in self.reasons):
            raise ValueError('Reference-character assessment requires explicit reasons')

    @property
    def authorizes_single_reference(self) -> bool:
        return self.status in (
            ReferenceCharacterStatus.SAFE_SINGLE_REFERENCE,
            ReferenceCharacterStatus.BORDERLINE,
        )

    @property
    def requires_expanded_diagnostics(self) -> bool:
        return self.status is ReferenceCharacterStatus.BORDERLINE

    @property
    def requires_multireference_branch(self) -> bool:
        return self.status is ReferenceCharacterStatus.MULTIREFERENCE_RISK


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
class ErrorBudget:
    """Canonical component-resolved uncertainty budget for an EA.

    Component intervals are additive correction ranges.  Their half-widths are
    used only to identify the currently dominant *documented* uncertainty.
    Unknown components keep the budget open and therefore block automatic
    precision-refinement recommendations.
    """

    baseline_offset_ev: Interval
    baseline_evidence_ids: tuple[str, ...]
    components: tuple[UncertaintyComponent, ...] = ()
    baseline_name: str = 'BASELINE'

    def __post_init__(self) -> None:
        if not self.baseline_name.strip():
            raise ValueError('Baseline name must be nonempty')
        if not self.baseline_evidence_ids or any(not x.strip() for x in self.baseline_evidence_ids):
            raise ValueError('Error-budget baseline requires evidence IDs')
        names = [self.baseline_name] + [term.name for term in self.components]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate error-budget component names')

    @property
    def missing_components(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components if c.correction_ev is None)

    @property
    def is_closed(self) -> bool:
        return not self.missing_components

    def uncertainty_half_widths(self) -> tuple[tuple[str, float], ...] | None:
        if not self.is_closed:
            return None
        items = [(self.baseline_name, self.baseline_offset_ev.half_width)]
        for component in self.components:
            assert component.correction_ev is not None
            items.append((component.name, component.correction_ev.half_width))
        return tuple(items)

    def dominant_uncertainty_source(self) -> str | None:
        items = self.uncertainty_half_widths()
        if items is None or not items:
            return None
        # Stable tie-break by component name makes planning reproducible.
        return max(items, key=lambda item: (item[1], item[0]))[0]


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


def _review_closes_gate(review: Review) -> bool:
    """Return whether a reviewed requirement is scientifically closed.

    NOT_APPLICABLE closes a gate only because Review itself requires an
    explicit physical rationale for that status.
    """

    return review.status in (ReviewStatus.CLEARED, ReviewStatus.NOT_APPLICABLE)


@dataclass(frozen=True)
class EnergyReliabilityGateSet:
    """Canonical G3 decomposition from the OpenEA-v1 production contract.

    G3a reference-method validity
    G3b basis/diffuse convergence
    G3c correlation reliability
    G3d missing physical corrections
    G3e uncertainty closure
    """

    reference_method_validity: Review = field(default_factory=lambda: Review(ReviewStatus.PENDING))
    basis_diffuse_convergence: Review = field(default_factory=lambda: Review(ReviewStatus.PENDING))
    correlation_reliability: Review = field(default_factory=lambda: Review(ReviewStatus.PENDING))
    physical_corrections: Review = field(default_factory=lambda: Review(ReviewStatus.PENDING))
    uncertainty_closure: Review = field(default_factory=lambda: Review(ReviewStatus.PENDING))

    def open_gates(self) -> tuple[str, ...]:
        return tuple(
            name for name, item in (
                ('G3A_REFERENCE_METHOD_VALIDITY', self.reference_method_validity),
                ('G3B_BASIS_DIFFUSE_CONVERGENCE', self.basis_diffuse_convergence),
                ('G3C_CORRELATION_RELIABILITY', self.correlation_reliability),
                ('G3D_PHYSICAL_CORRECTIONS', self.physical_corrections),
                ('G3E_UNCERTAINTY_CLOSURE', self.uncertainty_closure),
            ) if not _review_closes_gate(item)
        )

    @property
    def is_closed(self) -> bool:
        return not self.open_gates()

    def aggregate_review(self) -> Review:
        """Legacy-compatible composite G3 review derived from the subgates."""

        evidence = tuple(dict.fromkeys(
            evidence_id
            for review in (
                self.reference_method_validity,
                self.basis_diffuse_convergence,
                self.correlation_reliability,
                self.physical_corrections,
                self.uncertainty_closure,
            )
            for evidence_id in review.evidence_ids
        ))
        if self.is_closed:
            if evidence:
                return Review(
                    ReviewStatus.CLEARED,
                    evidence,
                    'G3a-G3e are individually closed.',
                )
            return Review(
                ReviewStatus.NOT_APPLICABLE,
                (),
                'All G3 subgates are physically not applicable.',
            )
        return Review(
            ReviewStatus.UNRESOLVED,
            evidence,
            'Open G3 subgates: ' + ', '.join(self.open_gates()),
        )


@dataclass(frozen=True)
class GateSet:
    """G1/G2 plus legacy or component-resolved G3 scientific gates.

    Existing callers may continue to provide ``energy_reliability`` only.
    New production code should provide ``energy_subgates``; when present,
    G3a-G3e are authoritative and the legacy composite cannot hide an open
    subgate.  Precision-target achievement remains a separate assessment.
    """

    state_completeness: Review = field(default_factory=lambda: Review(ReviewStatus.PENDING))
    attachment_resolution: Review = field(default_factory=lambda: Review(ReviewStatus.PENDING))
    energy_reliability: Review = field(default_factory=lambda: Review(ReviewStatus.PENDING))
    energy_subgates: EnergyReliabilityGateSet | None = None

    @property
    def effective_energy_reliability(self) -> Review:
        if self.energy_subgates is not None:
            return self.energy_subgates.aggregate_review()
        return self.energy_reliability

    def open_gates(self) -> tuple[str, ...]:
        open_items: list[str] = []
        if not _review_closes_gate(self.state_completeness):
            open_items.append('G1_STATE_COMPLETENESS')
        if not _review_closes_gate(self.attachment_resolution):
            open_items.append('G2_ATTACHMENT_RESOLUTION')
        if self.energy_subgates is not None:
            open_items.extend(self.energy_subgates.open_gates())
        elif not _review_closes_gate(self.energy_reliability):
            open_items.append('G3_ENERGY_RELIABILITY')
        return tuple(open_items)


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
