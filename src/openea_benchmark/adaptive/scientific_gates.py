"""Canonical OpenEA-v1 scientific gate construction.

This module translates already-reviewed evidence into the refined G3a--G3e
gates.  It does not run electronic-structure calculations and deliberately
fails closed when required evidence is missing.
"""
from __future__ import annotations

from collections.abc import Mapping

from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceStatus,
    CardinalConvergenceAssessment,
    DiffuseConvergenceAssessment,
)
from openea_benchmark.attachment.component_resolved_cbs import CBSResolvedResult, CBSStatus

from .multireference import (
    MRBranchStatus,
    MRProductionCapability,
    resolve_multireference_branch,
)
from .model import (
    EnergyReliabilityGateSet,
    ErrorBudget,
    ReferenceCharacterAssessment,
    ReferenceCharacterStatus,
    Review,
    ReviewStatus,
)


def _closed(review: Review) -> bool:
    return review.status in (ReviewStatus.CLEARED, ReviewStatus.NOT_APPLICABLE)


def _merge_evidence(*reviews: Review) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        evidence_id
        for review in reviews
        for evidence_id in review.evidence_ids
    ))


def reference_method_validity_review(
    assessment: ReferenceCharacterAssessment,
    *,
    expanded_diagnostics: Review | None = None,
    mr_capability: MRProductionCapability | None = None,
    reassessment_triggers: Mapping[str, Review] | None = None,
) -> Review:
    """Build G3a from the mandatory Reference Character Gate.

    SAFE_SINGLE_REFERENCE closes G3a directly. BORDERLINE remains open until
    an explicit expanded-diagnostics review is closed. MULTIREFERENCE_RISK
    never clears the single-reference method-validity gate.  A later confirmed
    method-validity warning (for example unstable DeltaT3) reopens G3a.
    """

    triggers = {} if reassessment_triggers is None else dict(reassessment_triggers)
    open_triggers = {
        name: review for name, review in triggers.items()
        if not _closed(review)
    }
    if open_triggers:
        evidence = tuple(dict.fromkeys(
            assessment.evidence_ids
            + tuple(eid for review in open_triggers.values() for eid in review.evidence_ids)
        ))
        return Review(
            ReviewStatus.UNRESOLVED,
            evidence,
            'G3a reopened by method-validity reassessment trigger(s): '
            + ', '.join(sorted(open_triggers)),
        )

    if assessment.status is ReferenceCharacterStatus.SAFE_SINGLE_REFERENCE:
        return Review(
            ReviewStatus.CLEARED,
            assessment.evidence_ids,
            'G3a cleared: reference character supports the single-reference production path.',
        )

    if assessment.status is ReferenceCharacterStatus.BORDERLINE:
        if expanded_diagnostics is not None and _closed(expanded_diagnostics):
            evidence = tuple(dict.fromkeys(
                assessment.evidence_ids + expanded_diagnostics.evidence_ids
            ))
            if expanded_diagnostics.status is ReviewStatus.NOT_APPLICABLE:
                # A borderline assessment cannot be made acceptable by simply
                # declaring the requested expanded diagnostics N/A.
                return Review(
                    ReviewStatus.UNRESOLVED,
                    evidence,
                    'G3a remains open: BORDERLINE reference character requires completed expanded diagnostics.',
                )
            return Review(
                ReviewStatus.CLEARED,
                evidence,
                'G3a conditionally cleared: BORDERLINE reference character was followed by reviewed expanded diagnostics; uncertainty must remain enlarged.',
            )
        evidence = assessment.evidence_ids + (() if expanded_diagnostics is None else expanded_diagnostics.evidence_ids)
        return Review(
            ReviewStatus.UNRESOLVED,
            tuple(dict.fromkeys(evidence)),
            'G3a remains open: BORDERLINE reference character requires expanded diagnostics before production is authorized.',
        )

    if assessment.status is ReferenceCharacterStatus.MULTIREFERENCE_RISK:
        mr = resolve_multireference_branch(mr_capability)
        if mr.status is MRBranchStatus.PRODUCTION_AUTHORIZED:
            evidence = tuple(dict.fromkeys(assessment.evidence_ids + mr.evidence_ids))
            return Review(
                ReviewStatus.CLEARED,
                evidence,
                'G3a cleared on the validated multireference production branch.',
            )
        return Review(
            ReviewStatus.UNRESOLVED,
            tuple(dict.fromkeys(assessment.evidence_ids + mr.evidence_ids)),
            'G3a remains open: multireference risk requires a validated MR production method.',
        )

    return Review(
        ReviewStatus.UNRESOLVED,
        assessment.evidence_ids,
        'G3a remains open because reference character is unresolved.',
    )


def cbs_resolution_review(cbs: CBSResolvedResult | None) -> Review:
    """Review component-resolved CBS model closure without promoting it to EA0."""

    if cbs is None:
        return Review(
            ReviewStatus.PENDING,
            (),
            'Component-resolved CBS assessment has not been supplied.',
        )
    evidence = tuple(cbs.evidence)
    if cbs.status is CBSStatus.CLEARED:
        return Review(
            ReviewStatus.CLEARED,
            evidence,
            'Component-resolved CBS model sensitivity is cleared.',
        )
    return Review(
        ReviewStatus.UNRESOLVED,
        evidence,
        'Component-resolved CBS model sensitivity is not cleared.',
    )


def basis_diffuse_convergence_review(
    cardinal: CardinalConvergenceAssessment | None,
    diffuse: DiffuseConvergenceAssessment | None,
    cbs: CBSResolvedResult | None = None,
) -> Review:
    """Build G3b from cardinal, diffuse, and (when supplied) CBS evidence."""

    cbs_review = None if cbs is None else cbs_resolution_review(cbs)
    evidence = tuple(dict.fromkeys(
        tuple(() if cardinal is None else cardinal.evidence)
        + tuple(() if diffuse is None else diffuse.evidence)
        + tuple(() if cbs_review is None else cbs_review.evidence_ids)
    ))
    if cardinal is None or diffuse is None:
        return Review(
            ReviewStatus.UNRESOLVED,
            evidence,
            'G3b requires both cardinal and diffuse convergence assessments.',
        )
    if (
        cardinal.status is BasisConvergenceStatus.CLEARED
        and diffuse.status is BasisConvergenceStatus.CLEARED
        and (cbs_review is None or _closed(cbs_review))
    ):
        return Review(
            ReviewStatus.CLEARED,
            evidence,
            'G3b cleared: cardinal and diffuse convergence are resolved'
            + (' and component-resolved CBS sensitivity is cleared.' if cbs_review is not None else '.'),
        )
    return Review(
        ReviewStatus.UNRESOLVED,
        evidence,
        'G3b remains open: cardinal, diffuse, and/or supplied CBS convergence is not cleared.',
    )


def correlation_reliability_review(review: Review | None) -> Review:
    """Build G3c from reviewed correlation-reliability evidence.

    A CONFIRMED diagnostic means a problem was confirmed, not that the gate is
    satisfied.  Such evidence therefore keeps G3c open.
    """

    if review is None:
        return Review(
            ReviewStatus.PENDING,
            (),
            'G3c correlation reliability has not yet been reviewed.',
        )
    if _closed(review):
        return review
    return Review(
        ReviewStatus.UNRESOLVED,
        review.evidence_ids,
        'G3c remains open: ' + (review.rationale or review.status.value),
    )


def physical_corrections_review(reviews: Mapping[str, Review]) -> Review:
    """Build G3d for the set of physical corrections judged relevant.

    The caller decides which corrections are relevant (e.g. CV, scalar
    relativity, SOC, nuclear motion). An empty mapping is deliberately not
    interpreted as zero missing physics.
    """

    if not reviews:
        return Review(
            ReviewStatus.UNRESOLVED,
            (),
            'G3d has no reviewed physical-correction inventory.',
        )
    evidence = _merge_evidence(*reviews.values())
    open_names = tuple(name for name, review in reviews.items() if not _closed(review))
    if open_names:
        return Review(
            ReviewStatus.UNRESOLVED,
            evidence,
            'G3d open physical corrections: ' + ', '.join(sorted(open_names)),
        )
    if evidence:
        return Review(
            ReviewStatus.CLEARED,
            evidence,
            'G3d cleared: every declared relevant physical correction is resolved or physically not applicable.',
        )
    return Review(
        ReviewStatus.NOT_APPLICABLE,
        (),
        'All declared physical corrections are physically not applicable.',
    )


def uncertainty_closure_review(
    error_budget: ErrorBudget,
    *,
    additional_requirements: Mapping[str, Review] | None = None,
) -> Review:
    """Build G3e from the error budget plus explicit closure requirements.

    ``additional_requirements`` is used for epistemic obligations that are not
    themselves additive corrections, such as confirming that a BORDERLINE
    reference-character uncertainty enlargement has actually been represented.
    """

    requirements = {} if additional_requirements is None else dict(additional_requirements)
    evidence = tuple(dict.fromkeys(
        error_budget.baseline_evidence_ids
        + tuple(
            evidence_id
            for component in error_budget.components
            for evidence_id in component.evidence_ids
        )
        + tuple(
            evidence_id
            for review in requirements.values()
            for evidence_id in review.evidence_ids
        )
    ))
    open_requirements = tuple(
        name for name, review in requirements.items() if not _closed(review)
    )
    if error_budget.is_closed and not open_requirements:
        return Review(
            ReviewStatus.CLEARED,
            evidence,
            'G3e cleared: error-budget components and additional uncertainty obligations are closed.',
        )

    reasons: list[str] = []
    if error_budget.missing_components:
        reasons.append('unbounded components: ' + ', '.join(error_budget.missing_components))
    if open_requirements:
        reasons.append('open uncertainty obligations: ' + ', '.join(sorted(open_requirements)))
    return Review(
        ReviewStatus.UNRESOLVED,
        evidence,
        'G3e remains open; ' + '; '.join(reasons),
    )


def build_energy_reliability_gates(
    *,
    reference_character: ReferenceCharacterAssessment,
    expanded_reference_diagnostics: Review | None,
    cardinal: CardinalConvergenceAssessment | None,
    diffuse: DiffuseConvergenceAssessment | None,
    correlation_reliability: Review | None,
    cbs: CBSResolvedResult | None = None,
    physical_corrections: Mapping[str, Review],
    error_budget: ErrorBudget,
    mr_capability: MRProductionCapability | None = None,
    borderline_uncertainty_review: Review | None = None,
    reference_reassessment_triggers: Mapping[str, Review] | None = None,
) -> EnergyReliabilityGateSet:
    """Construct the authoritative G3a--G3e gate bundle."""

    uncertainty_requirements: dict[str, Review] = {}
    if reference_character.status is ReferenceCharacterStatus.BORDERLINE:
        uncertainty_requirements['BORDERLINE_REFERENCE_CHARACTER'] = (
            borderline_uncertainty_review
            if borderline_uncertainty_review is not None
            else Review(
                ReviewStatus.PENDING,
                (),
                'BORDERLINE reference character requires explicit uncertainty enlargement.',
            )
        )

    return EnergyReliabilityGateSet(
        reference_method_validity=reference_method_validity_review(
            reference_character,
            expanded_diagnostics=expanded_reference_diagnostics,
            mr_capability=mr_capability,
            reassessment_triggers=reference_reassessment_triggers,
        ),
        basis_diffuse_convergence=basis_diffuse_convergence_review(cardinal, diffuse, cbs),
        correlation_reliability=correlation_reliability_review(correlation_reliability),
        physical_corrections=physical_corrections_review(physical_corrections),
        uncertainty_closure=uncertainty_closure_review(
            error_budget,
            additional_requirements=uncertainty_requirements,
        ),
    )
