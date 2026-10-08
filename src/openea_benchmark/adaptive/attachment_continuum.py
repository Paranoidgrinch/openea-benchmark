"""Canonical G2 electron-attachment / continuum evidence for OpenEA v1.

This module answers a narrower question than molecular dissociation stability:

    Is the candidate anion a physically bound electron-attached state rather
    than a finite-Gaussian-basis discretization of the electron continuum?

The module is deliberately evidence-first.  It does not run PySCF and it does
not allow a positive finite-basis Delta-CC electron affinity to close G2 by
itself.  A future execution layer may populate the contracts here with
EA-EOM-CCSD and stabilization calculations.

Policy summary
--------------
* direct Delta-CC diffuse convergence is necessary but not sufficient;
* a state-resolved EA-EOM-CCSD series provides an independent attachment
  channel and must itself remain bound under diffuse enlargement;
* clearly VALENCE_BOUND states may close D08 from reviewed attachment
  character + direct diffuse convergence + converged EA-EOM evidence;
* DIFFUSE_BOUND and NEAR_THRESHOLD states additionally require an explicit
  stabilization/continuum review;
* conflicting bound/unbound evidence is UNRESOLVED;
* no uncomputed diagnostic is represented as zero evidence.

All tolerances are explicit policy inputs.  The defaults are intentionally
conservative engineering defaults for the evidence *assessment*, not claims
of universal chemical accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable

from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceStatus,
    DiffuseConvergenceAssessment,
)

from .method_basis_advisor import AttachmentCharacter
from .model import Review, ReviewStatus


HARTREE_TO_EV = 27.211386245988


def pyscf_eom_eigenvalue_to_attachment_ea_ev(omega_hartree: float) -> float:
    """Convert PySCF EA-EOM charged-excitation energy to OpenEA EA sign.

    PySCF reports the EOM-EA eigenvalue as E(N+1)-E(N). OpenEA records
    electron affinity as E(N)-E(N+1), so the sign is reversed.
    """
    omega = float(omega_hartree)
    if not isfinite(omega):
        raise ValueError("omega_hartree must be finite")
    return -omega * HARTREE_TO_EV


class AttachmentContinuumStatus(str, Enum):
    BOUND_ATTACHMENT_CLEARED = "BOUND_ATTACHMENT_CLEARED"
    NO_BOUND_ATTACHMENT = "NO_BOUND_ATTACHMENT"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"


class EOMEAAssessmentStatus(str, Enum):
    BOUND_CONVERGED = "BOUND_CONVERGED"
    UNBOUND_CONVERGED = "UNBOUND_CONVERGED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"


class StabilizationStatus(str, Enum):
    STABLE_BOUND = "STABLE_BOUND"
    STABLE_UNBOUND = "STABLE_UNBOUND"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class AttachmentContinuumSettings:
    """Explicit numerical policy for G2 evidence convergence.

    Energies use the electron-affinity sign convention: positive values mean
    E(neutral) - E(anion) > 0.  The settings do not decide whether a state is
    chemically valence or diffuse bound; that classification must have its own
    reviewed evidence.
    """

    eom_increment_target_ev: float = 0.01
    eom_contraction_ratio_max: float = 0.8
    stabilization_span_target_ev: float = 0.01
    minimum_bound_margin_ev: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "eom_increment_target_ev",
            "stabilization_span_target_ev",
            "minimum_bound_margin_ev",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        ratio = float(self.eom_contraction_ratio_max)
        if not isfinite(ratio) or not (0.0 < ratio < 1.0):
            raise ValueError("eom_contraction_ratio_max must lie strictly between 0 and 1")


@dataclass(frozen=True)
class EOMEAAttachmentPoint:
    """One state-resolved EA-EOM-CCSD attachment point.

    ``attachment_ea_ev`` uses OpenEA's EA sign convention, i.e.
    ``E_neutral - E_anion``.  PySCF's EOM-EA eigenvalue is normally the
    charged excitation energy ``E(N+1)-E(N)`` and therefore must be negated
    before constructing this record.
    """

    augmentation_level: int
    attachment_ea_ev: float
    state_identity: Review
    evidence_ids: tuple[str, ...]
    one_particle_weight: float | None = None
    basis_label: str | None = None

    def __post_init__(self) -> None:
        if self.augmentation_level < 0:
            raise ValueError("augmentation_level must be >= 0")
        if not isfinite(float(self.attachment_ea_ev)):
            raise ValueError("attachment_ea_ev must be finite")
        if not self.evidence_ids or any(not x.strip() for x in self.evidence_ids):
            raise ValueError("EOM-EA point requires evidence IDs")
        if self.one_particle_weight is not None:
            value = float(self.one_particle_weight)
            if not isfinite(value) or not (0.0 <= value <= 1.0 + 1.0e-12):
                raise ValueError("one_particle_weight must be in [0,1]")


@dataclass(frozen=True)
class EOMEAAttachmentAssessment:
    status: EOMEAAssessmentStatus
    review: Review
    highest_augmentation_level: int | None
    latest_increment_ev: float | None
    contraction_ratio: float | None
    residual_estimate_ev: float | None
    conservative_ea_lower_ev: float | None
    conservative_ea_upper_ev: float | None


@dataclass(frozen=True)
class StabilizationPoint:
    """Attachment energy at one explicit diffuse-exponent scaling factor."""

    scale_factor: float
    attachment_ea_ev: float
    state_identity: Review
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isfinite(float(self.scale_factor)) or self.scale_factor <= 0.0:
            raise ValueError("scale_factor must be finite and positive")
        if not isfinite(float(self.attachment_ea_ev)):
            raise ValueError("attachment_ea_ev must be finite")
        if not self.evidence_ids or any(not x.strip() for x in self.evidence_ids):
            raise ValueError("stabilization point requires evidence IDs")


@dataclass(frozen=True)
class StabilizationAssessment:
    status: StabilizationStatus
    review: Review
    energy_span_ev: float | None
    conservative_ea_lower_ev: float | None
    conservative_ea_upper_ev: float | None


@dataclass(frozen=True)
class AttachmentContinuumAssessment:
    status: AttachmentContinuumStatus
    attachment_character: AttachmentCharacter
    review: Review
    direct_diffuse_review: Review
    eom_ea: EOMEAAttachmentAssessment | None
    stabilization: StabilizationAssessment | None
    evidence_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reasons or any(not x.strip() for x in self.reasons):
            raise ValueError("Attachment-continuum assessment requires explicit reasons")
        if self.status in (
            AttachmentContinuumStatus.BOUND_ATTACHMENT_CLEARED,
            AttachmentContinuumStatus.NO_BOUND_ATTACHMENT,
        ):
            if not self.evidence_ids:
                raise ValueError("Resolved attachment-continuum assessment requires evidence")
            if self.review.status is not ReviewStatus.CLEARED:
                raise ValueError("Resolved attachment-continuum assessment requires CLEARED review")

    @property
    def bound_attachment_cleared(self) -> bool:
        return self.status is AttachmentContinuumStatus.BOUND_ATTACHMENT_CLEARED

    @property
    def no_bound_attachment(self) -> bool:
        return self.status is AttachmentContinuumStatus.NO_BOUND_ATTACHMENT


def _closed(review: Review | None) -> bool:
    return review is not None and review.status in (ReviewStatus.CLEARED, ReviewStatus.NOT_APPLICABLE)


def _evidence(*groups: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(x for group in groups for x in group))


def direct_diffuse_review_from_assessment(
    assessment: DiffuseConvergenceAssessment | None,
) -> Review:
    """Translate direct Delta-CC diffuse convergence into a G2 evidence review.

    This review is deliberately *not* an attachment/continuum decision.
    """
    if assessment is None:
        return Review(
            ReviewStatus.PENDING,
            (),
            "Direct Delta-CC diffuse convergence has not been assessed.",
        )
    if assessment.status is BasisConvergenceStatus.CLEARED:
        return Review(
            ReviewStatus.CLEARED,
            tuple(assessment.evidence),
            "Direct Delta-CC electron affinity is converged with respect to the reviewed diffuse series; this is necessary but not sufficient for continuum exclusion.",
        )
    return Review(
        ReviewStatus.UNRESOLVED,
        tuple(assessment.evidence),
        "Direct Delta-CC diffuse convergence is not cleared.",
    )


def assess_eom_ea_diffuse_series(
    points: tuple[EOMEAAttachmentPoint, ...],
    *,
    settings: AttachmentContinuumSettings,
) -> EOMEAAttachmentAssessment:
    """Assess state-resolved EA-EOM evidence across diffuse augmentation.

    At least three consecutive augmentation levels are required so the latest
    increment can be checked for contraction.  The residual estimate is the
    latest absolute increment divided by ``1-r`` when the sequence contracts.
    This is convergence evidence, not a calibrated statistical confidence
    interval.
    """
    if len(points) < 3:
        return EOMEAAttachmentAssessment(
            EOMEAAssessmentStatus.NEED_MORE_EVIDENCE,
            Review(ReviewStatus.UNRESOLVED, (), "EA-EOM diffuse series requires at least three state-resolved augmentation levels."),
            max((p.augmentation_level for p in points), default=None),
            None,
            None,
            None,
            None,
            None,
        )

    ordered = tuple(sorted(points, key=lambda p: p.augmentation_level))
    levels = tuple(p.augmentation_level for p in ordered)
    if len(levels) != len(set(levels)):
        raise ValueError("duplicate EA-EOM augmentation levels")
    hi = ordered[-3:]
    if not (
        hi[1].augmentation_level == hi[0].augmentation_level + 1
        and hi[2].augmentation_level == hi[1].augmentation_level + 1
    ):
        return EOMEAAttachmentAssessment(
            EOMEAAssessmentStatus.NEED_MORE_EVIDENCE,
            Review(ReviewStatus.UNRESOLVED, (), "Highest three EA-EOM augmentation levels are not consecutive."),
            ordered[-1].augmentation_level,
            None,
            None,
            None,
            None,
            None,
        )

    all_evidence = _evidence(*(p.evidence_ids + p.state_identity.evidence_ids for p in ordered))
    if any(p.state_identity.status is not ReviewStatus.CLEARED for p in ordered):
        return EOMEAAttachmentAssessment(
            EOMEAAssessmentStatus.UNRESOLVED,
            Review(ReviewStatus.UNRESOLVED, all_evidence, "EA-EOM root/state identity is not cleared across the diffuse series."),
            ordered[-1].augmentation_level,
            None,
            None,
            None,
            None,
            None,
        )

    e0, e1, e2 = (p.attachment_ea_ev for p in hi)
    d1 = e1 - e0
    d2 = e2 - e1
    latest = abs(d2)
    tiny = 1.0e-15
    if abs(d1) <= tiny:
        ratio = 0.0 if abs(d2) <= tiny else None
    else:
        ratio = abs(d2) / abs(d1)
    same_direction = abs(d1) <= tiny or abs(d2) <= tiny or d1 * d2 > 0.0

    converged = (
        ratio is not None
        and same_direction
        and ratio <= settings.eom_contraction_ratio_max
        and latest <= settings.eom_increment_target_ev
    )
    if not converged:
        return EOMEAAttachmentAssessment(
            EOMEAAssessmentStatus.NEED_MORE_EVIDENCE,
            Review(ReviewStatus.UNRESOLVED, all_evidence, "EA-EOM attachment energy is not yet diffusely converged under the explicit convergence policy."),
            ordered[-1].augmentation_level,
            latest,
            ratio,
            None,
            None,
            None,
        )

    residual = latest / max(1.0 - ratio, 1.0e-12)
    lower = e2 - residual
    upper = e2 + residual
    margin = settings.minimum_bound_margin_ev

    if lower > margin:
        status = EOMEAAssessmentStatus.BOUND_CONVERGED
        rationale = "State-resolved EA-EOM attachment remains strictly bound after diffuse residual expansion."
    elif upper <= -margin:
        status = EOMEAAssessmentStatus.UNBOUND_CONVERGED
        rationale = "State-resolved EA-EOM attachment remains non-binding after diffuse residual expansion."
    else:
        status = EOMEAAssessmentStatus.UNRESOLVED
        rationale = "Converged EA-EOM attachment interval overlaps the electron-detachment threshold."

    return EOMEAAttachmentAssessment(
        status,
        Review(ReviewStatus.CLEARED if status is not EOMEAAssessmentStatus.UNRESOLVED else ReviewStatus.UNRESOLVED, all_evidence, rationale),
        ordered[-1].augmentation_level,
        latest,
        ratio,
        residual,
        lower,
        upper,
    )


def assess_stabilization_series(
    points: tuple[StabilizationPoint, ...],
    *,
    settings: AttachmentContinuumSettings,
    continuum_discrimination_review: Review | None = None,
) -> StabilizationAssessment:
    """Assess a precomputed diffuse-exponent stabilization series.

    This function cannot promote stationary finite-basis energies to continuum
    exclusion without a *separate* scientifically cleared continuum review.
    A smooth pseudocontinuum eigenvalue can be stationary in a finite scan.
    Root continuity and stable energies are necessary diagnostics, not proof.
    """
    if len(points) < 3:
        return StabilizationAssessment(
            StabilizationStatus.NEED_MORE_EVIDENCE,
            Review(ReviewStatus.UNRESOLVED, (), "Stabilization review requires at least three scale points."),
            None,
            None,
            None,
        )
    ordered = tuple(sorted(points, key=lambda p: p.scale_factor))
    scales = tuple(p.scale_factor for p in ordered)
    if len(scales) != len(set(scales)):
        raise ValueError("duplicate stabilization scale factors")
    evidence = _evidence(*(p.evidence_ids + p.state_identity.evidence_ids for p in ordered))
    if any(p.state_identity.status is not ReviewStatus.CLEARED for p in ordered):
        return StabilizationAssessment(
            StabilizationStatus.UNRESOLVED,
            Review(ReviewStatus.UNRESOLVED, evidence, "Attachment root identity is not cleared across the stabilization scan."),
            None,
            None,
            None,
        )

    if not (min(scales) < 1.0 < max(scales)) or 1.0 not in scales:
        return StabilizationAssessment(
            StabilizationStatus.NEED_MORE_EVIDENCE,
            Review(ReviewStatus.UNRESOLVED, evidence,
                   "Stabilization review requires the unscaled baseline and factors on both sides of unity."),
            None, None, None,
        )
    energies = tuple(p.attachment_ea_ev for p in ordered)
    span = max(energies) - min(energies)
    lower = min(energies) - span
    upper = max(energies) + span
    if span > settings.stabilization_span_target_ev:
        return StabilizationAssessment(
            StabilizationStatus.NEED_MORE_EVIDENCE,
            Review(ReviewStatus.UNRESOLVED, evidence, "Attachment energy is too sensitive to diffuse-exponent scaling to clear the stabilization diagnostic."),
            span,
            lower,
            upper,
        )

    # A flat finite-Gaussian basis EOM spectrum can be a pseudostate plateau.
    # Separately reviewed continuum discrimination is mandatory for a G2-
    # effective STABLE_BOUND or STABLE_UNBOUND classification.
    extra = continuum_discrimination_review
    external_ids = () if extra is None else extra.evidence_ids
    forbidden_prefixes = ("G2_EOM:", "G2_ROOT_PROPOSAL:", "G2_STABILIZATION_PROFILE:")
    independent = (extra is not None and extra.status is ReviewStatus.CLEARED and
                   bool(external_ids) and not set(external_ids).intersection(evidence) and
                   all(not x.startswith(forbidden_prefixes) for x in external_ids))
    if not independent:
        return StabilizationAssessment(
            StabilizationStatus.UNRESOLVED,
            Review(ReviewStatus.UNRESOLVED, _evidence(evidence, external_ids),
                   "Stationary finite-basis EOM energies do not exclude continuum pseudostates; an independent CLEARED continuum-discrimination review is required."),
            span, lower, upper,
        )
    evidence = _evidence(evidence, external_ids)
    margin = settings.minimum_bound_margin_ev
    if lower > margin:
        status = StabilizationStatus.STABLE_BOUND
        rationale = "The state-resolved attachment energy is stable and strictly bound across the explicit diffuse-exponent scan."
    elif upper <= -margin:
        status = StabilizationStatus.STABLE_UNBOUND
        rationale = "The state-resolved attachment energy is stable and non-binding across the explicit diffuse-exponent scan."
    else:
        status = StabilizationStatus.UNRESOLVED
        rationale = "The stabilization envelope overlaps the electron-detachment threshold."
    return StabilizationAssessment(
        status,
        Review(ReviewStatus.CLEARED if status is not StabilizationStatus.UNRESOLVED else ReviewStatus.UNRESOLVED, evidence, rationale),
        span,
        lower,
        upper,
    )


def assess_attachment_continuum(
    *,
    attachment_character: AttachmentCharacter,
    attachment_character_review: Review,
    direct_diffuse_review: Review,
    eom_ea: EOMEAAttachmentAssessment | None,
    stabilization: StabilizationAssessment | None = None,
) -> AttachmentContinuumAssessment:
    """Build the authoritative typed D08 attachment/continuum assessment."""
    evidence = _evidence(
        attachment_character_review.evidence_ids,
        direct_diffuse_review.evidence_ids,
        () if eom_ea is None else eom_ea.review.evidence_ids,
        () if stabilization is None else stabilization.review.evidence_ids,
    )

    if attachment_character is AttachmentCharacter.UNRESOLVED or not _closed(attachment_character_review):
        return AttachmentContinuumAssessment(
            AttachmentContinuumStatus.NEED_MORE_EVIDENCE,
            attachment_character,
            Review(ReviewStatus.UNRESOLVED, evidence, "Attachment character is not scientifically cleared."),
            direct_diffuse_review,
            eom_ea,
            stabilization,
            evidence,
            ("ATTACHMENT_CHARACTER_NOT_CLEARED",),
        )
    if not _closed(direct_diffuse_review):
        return AttachmentContinuumAssessment(
            AttachmentContinuumStatus.NEED_MORE_EVIDENCE,
            attachment_character,
            Review(ReviewStatus.UNRESOLVED, evidence, "Direct Delta-CC diffuse convergence is required before D08 can close."),
            direct_diffuse_review,
            eom_ea,
            stabilization,
            evidence,
            ("DIRECT_DIFFUSE_CONVERGENCE_NOT_CLEARED",),
        )
    if eom_ea is None or eom_ea.status in (EOMEAAssessmentStatus.NEED_MORE_EVIDENCE, EOMEAAssessmentStatus.UNRESOLVED):
        return AttachmentContinuumAssessment(
            AttachmentContinuumStatus.NEED_MORE_EVIDENCE,
            attachment_character,
            Review(ReviewStatus.UNRESOLVED, evidence, "Independent state-resolved EA-EOM attachment evidence is not cleared."),
            direct_diffuse_review,
            eom_ea,
            stabilization,
            evidence,
            ("EOM_EA_ATTACHMENT_NOT_CLEARED",),
        )

    require_stabilization = attachment_character in (
        AttachmentCharacter.DIFFUSE_BOUND,
        AttachmentCharacter.NEAR_THRESHOLD,
        AttachmentCharacter.CONTINUUM_LIKE,
    )
    if require_stabilization:
        if stabilization is None or stabilization.status in (
            StabilizationStatus.NEED_MORE_EVIDENCE,
            StabilizationStatus.UNRESOLVED,
        ):
            return AttachmentContinuumAssessment(
                AttachmentContinuumStatus.NEED_MORE_EVIDENCE,
                attachment_character,
                Review(ReviewStatus.UNRESOLVED, evidence, "Diffuse/near-threshold attachment requires explicit stabilization/continuum evidence."),
                direct_diffuse_review,
                eom_ea,
                stabilization,
                evidence,
                ("STABILIZATION_REQUIRED_FOR_DIFFUSE_OR_NEAR_THRESHOLD_ATTACHMENT",),
            )

    eom_bound = eom_ea.status is EOMEAAssessmentStatus.BOUND_CONVERGED
    eom_unbound = eom_ea.status is EOMEAAssessmentStatus.UNBOUND_CONVERGED
    stab_bound = stabilization is not None and stabilization.status is StabilizationStatus.STABLE_BOUND
    stab_unbound = stabilization is not None and stabilization.status is StabilizationStatus.STABLE_UNBOUND

    if (eom_bound and stab_unbound) or (eom_unbound and stab_bound):
        return AttachmentContinuumAssessment(
            AttachmentContinuumStatus.UNRESOLVED,
            attachment_character,
            Review(ReviewStatus.UNRESOLVED, evidence, "EA-EOM and stabilization evidence disagree on electron binding."),
            direct_diffuse_review,
            eom_ea,
            stabilization,
            evidence,
            ("CONFLICTING_ATTACHMENT_CONTINUUM_EVIDENCE",),
        )

    if eom_unbound and attachment_character in (
        AttachmentCharacter.NEAR_THRESHOLD,
        AttachmentCharacter.CONTINUUM_LIKE,
    ) and stab_unbound:
        return AttachmentContinuumAssessment(
            AttachmentContinuumStatus.NO_BOUND_ATTACHMENT,
            attachment_character,
            Review(ReviewStatus.CLEARED, evidence, "Independent state-resolved EA-EOM and stabilization evidence exclude a bound electron-attached state under the reviewed policy."),
            direct_diffuse_review,
            eom_ea,
            stabilization,
            evidence,
            ("NO_BOUND_ELECTRON_ATTACHMENT_CLEARED",),
        )

    if eom_unbound and attachment_character in (
        AttachmentCharacter.VALENCE_BOUND,
        AttachmentCharacter.DIFFUSE_BOUND,
    ):
        return AttachmentContinuumAssessment(
            AttachmentContinuumStatus.UNRESOLVED,
            attachment_character,
            Review(ReviewStatus.UNRESOLVED, evidence, "Reviewed bound-state character conflicts with non-binding EA-EOM evidence."),
            direct_diffuse_review,
            eom_ea,
            stabilization,
            evidence,
            ("BOUND_CHARACTER_CONFLICTS_WITH_EOM_ATTACHMENT",),
        )

    if eom_bound and attachment_character is AttachmentCharacter.CONTINUUM_LIKE:
        return AttachmentContinuumAssessment(
            AttachmentContinuumStatus.UNRESOLVED,
            attachment_character,
            Review(ReviewStatus.UNRESOLVED, evidence, "Reviewed continuum-like character conflicts with bound EA-EOM evidence."),
            direct_diffuse_review,
            eom_ea,
            stabilization,
            evidence,
            ("CONTINUUM_CHARACTER_CONFLICTS_WITH_EOM_ATTACHMENT",),
        )

    if eom_bound and (not require_stabilization or stab_bound):
        return AttachmentContinuumAssessment(
            AttachmentContinuumStatus.BOUND_ATTACHMENT_CLEARED,
            attachment_character,
            Review(ReviewStatus.CLEARED, evidence, "State-resolved attachment evidence clears the candidate against finite-basis continuum risk under the reviewed policy."),
            direct_diffuse_review,
            eom_ea,
            stabilization,
            evidence,
            ("BOUND_ELECTRON_ATTACHMENT_CLEARED",),
        )

    return AttachmentContinuumAssessment(
        AttachmentContinuumStatus.UNRESOLVED,
        attachment_character,
        Review(ReviewStatus.UNRESOLVED, evidence, "Attachment/continuum evidence does not support a consistent terminal D08 result."),
        direct_diffuse_review,
        eom_ea,
        stabilization,
        evidence,
        ("ATTACHMENT_CONTINUUM_INCONSISTENT",),
    )
