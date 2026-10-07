"""Canonical final scientific-resolution orchestration for OpenEA v1.

This layer is intentionally small. It does not run electronic-structure jobs
and it does not reinterpret local execution success as scientific evidence.
It joins the independently reviewed OpenEA gates into exactly one terminal
scientific decision path.

Scientific invariants
---------------------
* G1 state completeness is mandatory for every terminal claim.
* A robust G2 result that establishes that no physically bound anion exists
  may terminate early as UNBOUND; a precise negative EA is not required.
* A BOUND result requires G2 physical validity plus closed G3a--G3e evidence
  and a closed adiabatic EA0 interval.
* Nuclear motion is mandatory for a final molecular EA0. It may be bounded,
  but it may not be silently treated as zero or NOT_APPLICABLE.
* A conflict between G2 (physically bound anion) and a non-positive final EA0
  interval is UNRESOLVED, not silently converted into UNBOUND.
* Requested numerical precision is reported separately from scientific status.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from openea_benchmark.attachment.asymptote import BindingAssessment, BindingStatus

from .decision import PrecisionStatus, evaluate_estimate
from .model import (
    GateSet,
    Interval,
    Review,
    ReviewStatus,
    ScientificResolutionStatus,
)
from .production_evidence import NUCLEAR_MOTION, ProductionEvidenceBundle


class PhysicalValidityStatus(str, Enum):
    """Canonical G2 physical-validity outcome before final EA arithmetic."""

    PHYSICALLY_BOUND_ANION = "PHYSICALLY_BOUND_ANION"
    NO_PHYSICALLY_BOUND_ANION = "NO_PHYSICALLY_BOUND_ANION"
    UNRESOLVED = "UNRESOLVED"


class ScientificResolutionPath(str, Enum):
    EARLY_PHYSICAL_UNBOUND = "EARLY_PHYSICAL_UNBOUND"
    ADIABATIC_EA0 = "ADIABATIC_EA0"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class PhysicalValidityAssessment:
    """Explicit G2 outcome with provenance.

    This contract is deliberately stronger than a fragmentation-only binding
    test. ``PHYSICALLY_BOUND_ANION`` means attachment/continuum validity and
    relevant molecular binding have been reviewed sufficiently for G2.
    """

    status: PhysicalValidityStatus
    evidence_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reasons or any(not x.strip() for x in self.reasons):
            raise ValueError("Physical-validity assessment requires explicit reasons")
        if self.status is not PhysicalValidityStatus.UNRESOLVED:
            if not self.evidence_ids or any(not x.strip() for x in self.evidence_ids):
                raise ValueError("Resolved physical-validity assessment requires evidence IDs")

    def as_g2_review(self) -> Review:
        if self.status is PhysicalValidityStatus.UNRESOLVED:
            return Review(
                ReviewStatus.UNRESOLVED,
                self.evidence_ids,
                "; ".join(self.reasons),
            )
        return Review(
            ReviewStatus.CLEARED,
            self.evidence_ids,
            "; ".join(self.reasons),
        )


@dataclass(frozen=True)
class ScientificResolutionResult:
    molecule: str
    status: ScientificResolutionStatus
    precision_status: PrecisionStatus
    path: ScientificResolutionPath
    ea0_interval_ev: Interval | None
    gates: GateSet
    physical_validity: PhysicalValidityAssessment
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    estimated_not_certified: bool = True


def _review_closed(review: Review) -> bool:
    return review.status in (ReviewStatus.CLEARED, ReviewStatus.NOT_APPLICABLE)


def physical_validity_from_binding(
    binding: BindingAssessment,
    *,
    electron_attachment_review: Review | None = None,
) -> PhysicalValidityAssessment:
    """Conservatively bridge existing binding evidence into canonical G2.

    Fragmentation unboundness is by itself sufficient to exclude a stable
    molecular anion. Fragmentation *boundness* is not sufficient to prove
    electron attachment: a separately CLEARED attachment/continuum review is
    required before G2 is marked physically bound.
    """

    binding_evidence = tuple(binding.evidence)
    if binding.status is BindingStatus.UNBOUND:
        return PhysicalValidityAssessment(
            PhysicalValidityStatus.NO_PHYSICALLY_BOUND_ANION,
            binding_evidence,
            ("Anion is not bound below the reviewed molecular dissociation limit.",),
        )

    if binding.status is BindingStatus.UNRESOLVED:
        return PhysicalValidityAssessment(
            PhysicalValidityStatus.UNRESOLVED,
            binding_evidence,
            ("Molecular binding/asymptote evidence is unresolved.",),
        )

    # A molecular minimum below dissociation does not by itself exclude a
    # finite-basis electron-continuum artifact. D08/attachment evidence must
    # therefore be explicitly reviewed.
    if electron_attachment_review is None:
        return PhysicalValidityAssessment(
            PhysicalValidityStatus.UNRESOLVED,
            binding_evidence,
            ("Molecular binding is resolved, but attachment/continuum validity has not been reviewed.",),
        )

    evidence = tuple(dict.fromkeys(binding_evidence + electron_attachment_review.evidence_ids))
    if electron_attachment_review.status is ReviewStatus.CLEARED:
        return PhysicalValidityAssessment(
            PhysicalValidityStatus.PHYSICALLY_BOUND_ANION,
            evidence,
            ("Molecular binding and electron attachment/continuum validity are both cleared.",),
        )

    return PhysicalValidityAssessment(
        PhysicalValidityStatus.UNRESOLVED,
        evidence,
        ("Molecular binding is resolved, but attachment/continuum validity is not cleared.",),
    )


def _empty_or_bundle_gates(
    state_completeness: Review,
    physical_validity: PhysicalValidityAssessment,
    production_evidence: ProductionEvidenceBundle | None,
) -> GateSet:
    return GateSet(
        state_completeness=state_completeness,
        attachment_resolution=physical_validity.as_g2_review(),
        energy_reliability=Review(ReviewStatus.PENDING),
        energy_subgates=None if production_evidence is None else production_evidence.energy_gates,
    )


def _bundle_evidence(bundle: ProductionEvidenceBundle | None) -> tuple[str, ...]:
    if bundle is None:
        return ()
    items: list[str] = list(bundle.estimate.baseline_evidence_ids)
    for component in bundle.estimate.corrections:
        items.extend(component.evidence_ids)
    for review in (
        bundle.energy_gates.reference_method_validity,
        bundle.energy_gates.basis_diffuse_convergence,
        bundle.energy_gates.correlation_reliability,
        bundle.energy_gates.physical_corrections,
        bundle.energy_gates.uncertainty_closure,
    ):
        items.extend(review.evidence_ids)
    return tuple(dict.fromkeys(items))


def _nuclear_motion_ready(bundle: ProductionEvidenceBundle) -> bool:
    item = bundle.physical_corrections.get(NUCLEAR_MOTION)
    return (
        item is not None
        and item.review.status is ReviewStatus.CLEARED
        and item.component is not None
        and item.component.correction_ev is not None
    )


def resolve_scientific_outcome(
    *,
    molecule: str,
    state_completeness: Review,
    physical_validity: PhysicalValidityAssessment,
    production_evidence: ProductionEvidenceBundle | None = None,
    requested_half_width_ev: float = 0.020,
    critical_open_questions: tuple[str, ...] = (),
) -> ScientificResolutionResult:
    """Return the single canonical OpenEA scientific outcome.

    The early-unbound path implements the production contract's rule that a
    robust absence of a physically bound anion terminates before expensive
    single-/multireference production refinement. The bound path is stricter:
    it requires a closed, explicitly adiabatic EA0 evidence bundle.
    """

    if not molecule.strip():
        raise ValueError("Molecule must be specified")
    if not isfinite(float(requested_half_width_ev)) or requested_half_width_ev <= 0.0:
        raise ValueError("requested_half_width_ev must be finite and positive")
    if any(not item.strip() for item in critical_open_questions):
        raise ValueError("critical_open_questions must not contain empty values")

    gates = _empty_or_bundle_gates(state_completeness, physical_validity, production_evidence)
    evidence = tuple(dict.fromkeys(
        state_completeness.evidence_ids
        + physical_validity.evidence_ids
        + _bundle_evidence(production_evidence)
    ))

    if state_completeness.status is not ReviewStatus.CLEARED:
        return ScientificResolutionResult(
            molecule,
            ScientificResolutionStatus.UNRESOLVED,
            PrecisionStatus.UNDETERMINED,
            ScientificResolutionPath.UNRESOLVED,
            None,
            gates,
            physical_validity,
            ("G1_STATE_COMPLETENESS_NOT_CLEARED",),
            evidence,
        )

    if critical_open_questions:
        return ScientificResolutionResult(
            molecule,
            ScientificResolutionStatus.UNRESOLVED,
            PrecisionStatus.UNDETERMINED,
            ScientificResolutionPath.UNRESOLVED,
            None,
            gates,
            physical_validity,
            tuple("CRITICAL_OPEN_QUESTION:" + item for item in critical_open_questions),
            evidence,
        )

    if physical_validity.status is PhysicalValidityStatus.UNRESOLVED:
        return ScientificResolutionResult(
            molecule,
            ScientificResolutionStatus.UNRESOLVED,
            PrecisionStatus.UNDETERMINED,
            ScientificResolutionPath.UNRESOLVED,
            None,
            gates,
            physical_validity,
            ("G2_PHYSICAL_VALIDITY_NOT_RESOLVED",),
            evidence,
        )

    if physical_validity.status is PhysicalValidityStatus.NO_PHYSICALLY_BOUND_ANION:
        return ScientificResolutionResult(
            molecule,
            ScientificResolutionStatus.UNBOUND,
            PrecisionStatus.NOT_APPLICABLE,
            ScientificResolutionPath.EARLY_PHYSICAL_UNBOUND,
            None,
            gates,
            physical_validity,
            ("NO_PHYSICALLY_BOUND_ANION_WITH_G1_CLEARED",),
            evidence,
        )

    if production_evidence is None:
        return ScientificResolutionResult(
            molecule,
            ScientificResolutionStatus.UNRESOLVED,
            PrecisionStatus.UNDETERMINED,
            ScientificResolutionPath.UNRESOLVED,
            None,
            gates,
            physical_validity,
            ("G3_PRODUCTION_EVIDENCE_MISSING",),
            evidence,
        )

    if production_evidence.closure_actions:
        return ScientificResolutionResult(
            molecule,
            ScientificResolutionStatus.UNRESOLVED,
            PrecisionStatus.UNDETERMINED,
            ScientificResolutionPath.UNRESOLVED,
            production_evidence.interval,
            gates,
            physical_validity,
            tuple(
                "OPEN_PRODUCTION_CLOSURE_ACTION:" + action.action_id
                for action in production_evidence.closure_actions
            ),
            evidence,
        )

    # For a molecular BOUND result the final quantity is EA0, not a purely
    # electronic EA. NOT_APPLICABLE cannot close this particular obligation.
    if not _nuclear_motion_ready(production_evidence):
        return ScientificResolutionResult(
            molecule,
            ScientificResolutionStatus.UNRESOLVED,
            PrecisionStatus.UNDETERMINED,
            ScientificResolutionPath.UNRESOLVED,
            production_evidence.interval,
            gates,
            physical_validity,
            ("NUCLEAR_MOTION_NOT_RESOLVED_FOR_ADIABATIC_EA0",),
            evidence,
        )

    decision = evaluate_estimate(
        molecule,
        production_evidence.estimate,
        gates,
        target_half_width_ev=requested_half_width_ev,
    )

    if decision.category is ScientificResolutionStatus.UNBOUND:
        # G2 says a physically bound anion exists while the closed EA0 model
        # says electron detachment is non-endoergic. Do not silently choose one
        # side of this scientific conflict.
        return ScientificResolutionResult(
            molecule,
            ScientificResolutionStatus.UNRESOLVED,
            PrecisionStatus.UNDETERMINED,
            ScientificResolutionPath.UNRESOLVED,
            decision.ea0_ev,
            gates,
            physical_validity,
            ("G2_G3_CONFLICT_PHYSICALLY_BOUND_BUT_EA0_NONPOSITIVE",),
            evidence,
        )

    path = (
        ScientificResolutionPath.ADIABATIC_EA0
        if decision.category is ScientificResolutionStatus.BOUND
        else ScientificResolutionPath.UNRESOLVED
    )
    return ScientificResolutionResult(
        molecule,
        decision.category,
        decision.precision,
        path,
        decision.ea0_ev,
        gates,
        physical_validity,
        decision.reasons,
        evidence,
    )
