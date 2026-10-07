"""Bridge existing OpenEA production evidence into the v1 gate/error contracts.

This module is deliberately *not* an electronic-structure engine.  It consumes
already-produced Stage-3 / CBS / correction assessments and translates them
into:

* reviewed G3a--G3e evidence;
* a component-resolved :class:`EAEstimate` / :class:`ErrorBudget`;
* explicit closure actions for unresolved evidence;
* an optional precision-refinement recommendation.

Scientific invariants
---------------------
* A missing correction is UNKNOWN, never ``0 +/- 0``.
* A one-electron SFX2C1E correction does not silently close the missing
  two-electron relativistic remainder.
* CCSDT is diagnostic.  ``POST_CC_WARNING`` reopens reference-method validity;
  it never requests CCSDTQ here.
* No numerical result produced by the legacy validation modules is promoted to
  a final adiabatic EA merely by passing through this bridge.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from math import isfinite
from typing import Mapping

from openea_benchmark.attachment.basis_convergence import (
    CardinalConvergenceAssessment,
    DiffuseConvergenceAssessment,
)
from openea_benchmark.attachment.component_resolved_cbs import (
    CBSResolvedResult,
    CBSStatus,
)
from openea_benchmark.attachment.core_valence_correction import (
    CoreValenceAssessment,
    CoreValenceStatus,
)
from openea_benchmark.attachment.post_ccsd_t import PostCCAssessment
from openea_benchmark.attachment.scalar_relativity import ScalarRelativityAssessment

from .model import (
    EAEstimate,
    EnergyReliabilityGateSet,
    ErrorBudget,
    EvidenceQuality,
    Interval,
    MethodRole,
    ReferenceCharacterAssessment,
    ReferenceCharacterStatus,
    Review,
    ReviewStatus,
    UncertaintyComponent,
)
from .multireference import MRProductionCapability
from .nuclear_motion import NuclearMotionAssessment, NuclearMotionStatus
from .planner import ProductionRoutePlan, ProductionRouteStatus, plan_production_route
from .precision_controller import (
    PrecisionActionCandidate,
    PrecisionPlan,
    PrecisionPlanningStatus,
    plan_precision_refinement,
)
from .scientific_gates import build_energy_reliability_gates


CORE_VALENCE = 'CORE_VALENCE'
SCALAR_RELATIVITY = 'SCALAR_RELATIVITY'
SCALAR_RELATIVITY_REMAINDER = 'SCALAR_RELATIVITY_REMAINDER'
SOC = 'SOC'
NUCLEAR_MOTION = 'NUCLEAR_MOTION'
ADIABATIC_NUCLEAR_REMAINDER = 'ADIABATIC_NUCLEAR_REMAINDER'
POST_CC = 'POST_CC'
CBS_DIFFUSE_RESIDUAL = 'CBS_DIFFUSE_RESIDUAL'
CBS_GEOMETRY_TRANSFER = 'CBS_GEOMETRY_TRANSFER'
REFERENCE_CHARACTER = 'REFERENCE_CHARACTER'


@dataclass(frozen=True)
class CorrectionEvidence:
    """One correction's reviewed status plus its additive EA interval.

    A CLEARED correction must provide an uncertainty component.  A physically
    NOT_APPLICABLE correction must not provide one.  Open evidence may carry
    an action but no fabricated numerical interval.
    """

    name: str
    review: Review
    component: UncertaintyComponent | None
    action: str | None = None
    method_role: MethodRole = MethodRole.PRODUCTION

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError('Correction evidence requires a non-empty name')
        if self.review.status is ReviewStatus.CLEARED:
            if self.component is None or self.component.correction_ev is None:
                raise ValueError('CLEARED correction requires a bounded component')
            if self.component.name != self.name:
                raise ValueError('Correction evidence/component names must match')
        if self.review.status is ReviewStatus.NOT_APPLICABLE and self.component is not None:
            raise ValueError('NOT_APPLICABLE correction must not add an EA component')
        if self.action is not None and not self.action.strip():
            raise ValueError('Correction action must be non-empty when supplied')


class ClosurePriority(IntEnum):
    METHOD_VALIDITY = 0
    BASIS_AND_CBS = 10
    CORRELATION = 20
    PHYSICAL_CORRECTION = 30
    UNCERTAINTY = 40


@dataclass(frozen=True)
class ProductionClosureAction:
    action_id: str
    priority: ClosurePriority
    gate_id: str
    addresses_component: str
    method_role: MethodRole
    evidence_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.action_id.strip() or not self.gate_id.strip():
            raise ValueError('Closure action requires action and gate IDs')
        if not self.addresses_component.strip() or not self.reason.strip():
            raise ValueError('Closure action requires component and rationale')


class ProductionEvidencePlanningStatus(str, Enum):
    GATE_CLOSURE_REQUIRED = 'GATE_CLOSURE_REQUIRED'
    PRECISION_REFINEMENT_RECOMMENDED = 'PRECISION_REFINEMENT_RECOMMENDED'
    READY_FOR_SCIENTIFIC_DECISION = 'READY_FOR_SCIENTIFIC_DECISION'


@dataclass(frozen=True)
class ProductionEvidenceBundle:
    route: ProductionRoutePlan
    estimate: EAEstimate
    error_budget: ErrorBudget
    energy_gates: EnergyReliabilityGateSet
    physical_corrections: Mapping[str, CorrectionEvidence]
    correlation_review: Review
    closure_actions: tuple[ProductionClosureAction, ...]

    @property
    def interval(self) -> Interval | None:
        return self.estimate.estimated_interval()

    @property
    def next_closure_action(self) -> ProductionClosureAction | None:
        return self.closure_actions[0] if self.closure_actions else None


@dataclass(frozen=True)
class ProductionEvidencePlan:
    status: ProductionEvidencePlanningStatus
    bundle: ProductionEvidenceBundle
    precision_plan: PrecisionPlan | None
    reason: str


def _symmetric_interval(center: float, half_width: float) -> Interval:
    center = float(center)
    half_width = abs(float(half_width))
    if not isfinite(center) or not isfinite(half_width):
        raise ValueError('Correction center/bound must be finite')
    return Interval(center - half_width, center + half_width)


def _unknown_component(name: str, evidence: tuple[str, ...] = ()) -> UncertaintyComponent:
    return UncertaintyComponent(name, None, EvidenceQuality.UNKNOWN, evidence)


def _bounded_component(
    name: str,
    center: float,
    half_width: float,
    evidence: tuple[str, ...],
    *,
    quality: EvidenceQuality = EvidenceQuality.CONVERGENCE_ESTIMATED,
) -> UncertaintyComponent:
    return UncertaintyComponent(
        name,
        _symmetric_interval(center, half_width),
        quality,
        tuple(evidence),
    )


def core_valence_evidence(assessment: CoreValenceAssessment | None) -> CorrectionEvidence:
    if assessment is None:
        return CorrectionEvidence(
            CORE_VALENCE,
            Review(ReviewStatus.PENDING, (), 'Core-valence relevance/correction has not been reviewed.'),
            None,
            'ASSESS_CORE_VALENCE',
            MethodRole.PRODUCTION,
        )
    evidence = tuple(assessment.evidence)
    if assessment.status is CoreValenceStatus.CLEARED:
        if assessment.central_correction_ev is None or assessment.convergence_bound_ev is None:
            return CorrectionEvidence(
                CORE_VALENCE,
                Review(ReviewStatus.UNRESOLVED, evidence, 'Core-valence assessment is marked CLEARED but lacks a numerical correction bound.'),
                None,
                'REPAIR_CORE_VALENCE_EVIDENCE',
                MethodRole.REFINEMENT,
            )
        return CorrectionEvidence(
            CORE_VALENCE,
            Review(ReviewStatus.CLEARED, evidence, 'Core-valence correction is cardinal-converged within its active policy.'),
            _bounded_component(
                CORE_VALENCE,
                assessment.central_correction_ev,
                assessment.convergence_bound_ev,
                evidence,
            ),
        )
    return CorrectionEvidence(
        CORE_VALENCE,
        Review(ReviewStatus.UNRESOLVED, evidence, 'Core-valence correction is not yet converged/usable.'),
        None,
        assessment.action,
        MethodRole.REFINEMENT,
    )


def scalar_relativity_evidence(assessment: ScalarRelativityAssessment | None) -> CorrectionEvidence:
    if assessment is None:
        return CorrectionEvidence(
            SCALAR_RELATIVITY,
            Review(ReviewStatus.PENDING, (), 'Scalar-relativistic correction has not been reviewed.'),
            None,
            'ASSESS_SCALAR_RELATIVITY',
            MethodRole.PRODUCTION,
        )
    evidence = tuple(assessment.evidence)
    if assessment.status == 'CLEARED' and assessment.convergence_bound_ev is not None:
        return CorrectionEvidence(
            SCALAR_RELATIVITY,
            Review(ReviewStatus.CLEARED, evidence, 'The explicit SFX2C1E scalar correction is cardinal-converged within its active policy.'),
            _bounded_component(
                SCALAR_RELATIVITY,
                assessment.central_correction_ev,
                assessment.convergence_bound_ev,
                evidence,
            ),
        )
    action = assessment.action or 'REVIEW_SCALAR_RELATIVITY_MODEL'
    return CorrectionEvidence(
        SCALAR_RELATIVITY,
        Review(ReviewStatus.UNRESOLVED, evidence, 'Scalar-relativistic correction is not yet converged/usable.'),
        None,
        action,
        MethodRole.REFINEMENT,
    )


def post_cc_evidence(assessment: PostCCAssessment | None) -> CorrectionEvidence | None:
    """Translate the optional CCSDT diagnostic.

    ``None`` means CCSDT was not authorized/run.  It does not create a fake
    zero correction.  G3c must then be supplied by other reviewed correlation
    reliability evidence before it can close.
    """

    if assessment is None:
        return None
    evidence = tuple(assessment.evidence)
    if assessment.status == 'CLEARED':
        if assessment.central_correction_ev is None or assessment.combined_bound_ev is None:
            return CorrectionEvidence(
                POST_CC,
                Review(ReviewStatus.UNRESOLVED, evidence, 'CCSDT diagnostic is marked CLEARED but lacks an authorized DeltaT3 correction bound.'),
                None,
                'REPAIR_POST_CC_EVIDENCE',
                MethodRole.DIAGNOSTIC,
            )
        return CorrectionEvidence(
            POST_CC,
            Review(ReviewStatus.CLEARED, evidence, 'CCSDT indicates a small, basis-stable DeltaT3; optional correction is authorized.'),
            _bounded_component(
                POST_CC,
                assessment.central_correction_ev,
                assessment.combined_bound_ev,
                evidence,
            ),
            None,
            MethodRole.DIAGNOSTIC,
        )
    if assessment.status == 'POST_CC_WARNING':
        return CorrectionEvidence(
            POST_CC,
            Review(ReviewStatus.CONFIRMED, evidence, 'CCSDT DeltaT3 is large and/or basis-unstable; single-reference method validity must be reassessed.'),
            None,
            'REASSESS_REFERENCE_CHARACTER',
            MethodRole.DIAGNOSTIC,
        )
    return CorrectionEvidence(
        POST_CC,
        Review(ReviewStatus.UNRESOLVED, evidence, 'CCSDT triples-reliability diagnostic needs additional evidence.'),
        None,
        assessment.action,
        MethodRole.DIAGNOSTIC,
    )



def nuclear_motion_evidence(assessment: NuclearMotionAssessment | None) -> CorrectionEvidence:
    """Translate a reviewed diatomic v=0 solve into the additive EA correction.

    The correction itself is ``ZPE(neutral)-ZPE(anion)``.  Whether the anion
    v=0 level remains below dissociation is a G2 physical-validity question and
    is intentionally retained separately on ``assessment.anion_vibrational_binding``.
    """
    if assessment is None:
        return pending_correction(
            NUCLEAR_MOTION,
            'SOLVE_NUCLEAR_MOTION',
            'Final adiabatic EA requires reviewed J=0 nuclear-motion evidence.',
        )
    evidence = tuple(assessment.evidence_ids)
    if assessment.status is NuclearMotionStatus.CLEARED:
        if assessment.correction_ev is None or assessment.correction_half_width_ev is None:
            return CorrectionEvidence(
                NUCLEAR_MOTION,
                Review(ReviewStatus.UNRESOLVED, evidence, 'Nuclear-motion result is marked CLEARED but lacks a bounded DeltaZPE correction.'),
                None,
                'REPAIR_NUCLEAR_MOTION_EVIDENCE',
                MethodRole.PRODUCTION,
            )
        return CorrectionEvidence(
            NUCLEAR_MOTION,
            Review(ReviewStatus.CLEARED, evidence, 'Anharmonic J=0 v=0 levels are converged and DeltaZPE is bounded.'),
            _bounded_component(
                NUCLEAR_MOTION,
                assessment.correction_ev,
                assessment.correction_half_width_ev,
                evidence,
                quality=EvidenceQuality.CONVERGENCE_ESTIMATED,
            ),
            None,
            MethodRole.PRODUCTION,
        )
    return CorrectionEvidence(
        NUCLEAR_MOTION,
        Review(ReviewStatus.UNRESOLVED, evidence, assessment.rationale),
        None,
        assessment.action or 'REVIEW_NUCLEAR_MOTION',
        MethodRole.REFINEMENT,
    )

def not_applicable_correction(name: str, rationale: str) -> CorrectionEvidence:
    return CorrectionEvidence(
        name,
        Review(ReviewStatus.NOT_APPLICABLE, (), rationale),
        None,
        None,
        MethodRole.PRODUCTION,
    )


def pending_correction(name: str, action: str, rationale: str) -> CorrectionEvidence:
    return CorrectionEvidence(
        name,
        Review(ReviewStatus.PENDING, (), rationale),
        None,
        action,
        MethodRole.PRODUCTION,
    )


def bounded_external_correction(
    name: str,
    *,
    central_ev: float,
    half_width_ev: float,
    evidence_ids: tuple[str, ...],
    rationale: str,
    quality: EvidenceQuality = EvidenceQuality.CONVERGENCE_ESTIMATED,
) -> CorrectionEvidence:
    if not evidence_ids:
        raise ValueError('Bounded external correction requires evidence IDs')
    return CorrectionEvidence(
        name,
        Review(ReviewStatus.CLEARED, evidence_ids, rationale),
        _bounded_component(name, central_ev, half_width_ev, evidence_ids, quality=quality),
    )


def _cbs_baseline(cbs: CBSResolvedResult) -> tuple[float, Interval, tuple[UncertaintyComponent, ...]]:
    evidence = tuple(cbs.evidence)
    model_bound = abs(float(cbs.cbs_model_sensitivity_bound_ev))
    baseline_offset = Interval(-model_bound, model_bound)
    extra: list[UncertaintyComponent] = []

    if cbs.diffuse_residual_estimate_ev is None:
        extra.append(_unknown_component(CBS_DIFFUSE_RESIDUAL, evidence))
    else:
        extra.append(_bounded_component(
            CBS_DIFFUSE_RESIDUAL, 0.0, abs(float(cbs.diffuse_residual_estimate_ev)), evidence,
        ))

    if cbs.fixed_geometry_transfer_check_ev is None:
        extra.append(_unknown_component(CBS_GEOMETRY_TRANSFER, evidence))
    else:
        extra.append(_bounded_component(
            CBS_GEOMETRY_TRANSFER, 0.0, abs(float(cbs.fixed_geometry_transfer_check_ev)), evidence,
            quality=EvidenceQuality.DIRECTLY_TESTED,
        ))

    return float(cbs.ea_cbs_plus_diffuse_primary_ev), baseline_offset, tuple(extra)


def _combine_correlation_reviews(
    override: Review | None,
    post_cc: CorrectionEvidence | None,
) -> Review:
    reviews = tuple(x for x in (
        override,
        None if post_cc is None else post_cc.review,
    ) if x is not None)
    if not reviews:
        return Review(
            ReviewStatus.PENDING,
            (),
            'Correlation reliability has not been reviewed; CCSDT is optional and is not assumed to be required.',
        )
    evidence = tuple(dict.fromkeys(eid for review in reviews for eid in review.evidence_ids))
    if any(review.status is ReviewStatus.CONFIRMED for review in reviews):
        return Review(
            ReviewStatus.CONFIRMED,
            evidence,
            'Correlation-reliability warning confirmed; method/reference reassessment is required.',
        )
    if any(review.status not in (ReviewStatus.CLEARED, ReviewStatus.NOT_APPLICABLE) for review in reviews):
        return Review(
            ReviewStatus.UNRESOLVED,
            evidence,
            'Correlation reliability remains open.',
        )
    if evidence:
        return Review(ReviewStatus.CLEARED, evidence, 'Available correlation-reliability evidence is cleared.')
    return Review(
        ReviewStatus.NOT_APPLICABLE,
        (),
        'All supplied correlation-reliability requirements are physically not applicable.',
    )


def _default_physical_corrections(
    cv: CoreValenceAssessment | None,
    sr: ScalarRelativityAssessment | None,
    nuclear: NuclearMotionAssessment | None,
    overrides: Mapping[str, CorrectionEvidence],
) -> dict[str, CorrectionEvidence]:
    sr_evidence = scalar_relativity_evidence(sr)
    sr_remainder_action = (
        'BOUND_SCALAR_RELATIVITY_REMAINDER'
        if sr_evidence.review.status is ReviewStatus.CLEARED
        else None
    )
    items = {
        CORE_VALENCE: core_valence_evidence(cv),
        SCALAR_RELATIVITY: sr_evidence,
        SCALAR_RELATIVITY_REMAINDER: CorrectionEvidence(
            SCALAR_RELATIVITY_REMAINDER,
            Review(
                ReviewStatus.PENDING,
                (),
                'SFX2C1E is one-electron scalar relativity; missing two-electron picture-change/related effects require an explicit bound or N/A rationale.',
            ),
            None,
            sr_remainder_action,
            MethodRole.PRODUCTION,
        ),
        SOC: pending_correction(
            SOC,
            'ASSESS_SOC',
            'SOC relevance/correction has not been reviewed.',
        ),
        NUCLEAR_MOTION: nuclear_motion_evidence(nuclear),
        ADIABATIC_NUCLEAR_REMAINDER: pending_correction(
            ADIABATIC_NUCLEAR_REMAINDER,
            'BOUND_ADIABATIC_NUCLEAR_REMAINDER',
            'The J=0 Born-Oppenheimer vibrational correction excludes DBOC and non-adiabatic nuclear-motion effects; their relevance/residual must be bounded explicitly.',
        ),
    }
    for name, value in overrides.items():
        if name != value.name:
            raise ValueError('Physical-correction override key must match evidence.name')
        items[name] = value
    return items


def _closure_actions(
    *,
    route: ProductionRoutePlan,
    cardinal: CardinalConvergenceAssessment | None,
    diffuse: DiffuseConvergenceAssessment | None,
    cbs: CBSResolvedResult,
    post_cc: CorrectionEvidence | None,
    correlation_review: Review,
    physical: Mapping[str, CorrectionEvidence],
    reference_character: ReferenceCharacterAssessment,
    borderline_reference_uncertainty: UncertaintyComponent | None,
) -> tuple[ProductionClosureAction, ...]:
    actions: list[ProductionClosureAction] = []

    for action in route.required_actions:
        if action == 'ENLARGE_REFERENCE_CHARACTER_UNCERTAINTY' and borderline_reference_uncertainty is not None:
            continue
        actions.append(ProductionClosureAction(
            action,
            ClosurePriority.METHOD_VALIDITY,
            'G3A_REFERENCE_METHOD_VALIDITY',
            REFERENCE_CHARACTER,
            MethodRole.DIAGNOSTIC,
            route.evidence_ids,
            route.rationale,
        ))

    if cardinal is None:
        actions.append(ProductionClosureAction(
            'ASSESS_CARDINAL_CONVERGENCE', ClosurePriority.BASIS_AND_CBS,
            'G3B_BASIS_DIFFUSE_CONVERGENCE', 'CARDINAL_CONVERGENCE', MethodRole.REFINEMENT,
            (), 'Cardinal convergence assessment is missing.',
        ))
    elif getattr(cardinal.action, 'value', cardinal.action) != 'NONE':
        actions.append(ProductionClosureAction(
            str(getattr(cardinal.action, 'value', cardinal.action)), ClosurePriority.BASIS_AND_CBS,
            'G3B_BASIS_DIFFUSE_CONVERGENCE', 'CARDINAL_CONVERGENCE', MethodRole.REFINEMENT,
            tuple(cardinal.evidence), cardinal.rationale,
        ))

    if diffuse is None:
        actions.append(ProductionClosureAction(
            'ASSESS_DIFFUSE_CONVERGENCE', ClosurePriority.BASIS_AND_CBS,
            'G3B_BASIS_DIFFUSE_CONVERGENCE', 'DIFFUSE_CONVERGENCE', MethodRole.REFINEMENT,
            (), 'Diffuse convergence assessment is missing.',
        ))
    elif getattr(diffuse.action, 'value', diffuse.action) != 'NONE':
        actions.append(ProductionClosureAction(
            str(getattr(diffuse.action, 'value', diffuse.action)), ClosurePriority.BASIS_AND_CBS,
            'G3B_BASIS_DIFFUSE_CONVERGENCE', 'DIFFUSE_CONVERGENCE', MethodRole.REFINEMENT,
            tuple(diffuse.evidence), diffuse.rationale,
        ))

    if cbs.status is not CBSStatus.CLEARED:
        for action in cbs.next_actions or ('REVIEW_CBS_MODEL',):
            if action.startswith('DO_NOT_'):
                continue
            actions.append(ProductionClosureAction(
                action, ClosurePriority.BASIS_AND_CBS,
                'G3B_BASIS_DIFFUSE_CONVERGENCE', 'CBS_MODEL', MethodRole.REFINEMENT,
                tuple(cbs.evidence), 'Component-resolved CBS model sensitivity is not cleared.',
            ))

    if cbs.diffuse_residual_estimate_ev is None:
        actions.append(ProductionClosureAction(
            'BOUND_DIFFUSE_RESIDUAL', ClosurePriority.UNCERTAINTY,
            'G3E_UNCERTAINTY_CLOSURE', CBS_DIFFUSE_RESIDUAL, MethodRole.REFINEMENT,
            tuple(cbs.evidence), 'CBS diffuse residual is unknown and must be bounded explicitly.',
        ))
    if cbs.fixed_geometry_transfer_check_ev is None:
        actions.append(ProductionClosureAction(
            'COMPUTE_FIXED_GEOMETRY_TRANSFER_CHECK', ClosurePriority.UNCERTAINTY,
            'G3E_UNCERTAINTY_CLOSURE', CBS_GEOMETRY_TRANSFER, MethodRole.REFINEMENT,
            tuple(cbs.evidence), 'Fixed-geometry to PEC transfer uncertainty is unknown.',
        ))

    if post_cc is not None and post_cc.review.status not in (ReviewStatus.CLEARED, ReviewStatus.NOT_APPLICABLE):
        if post_cc.action:
            actions.append(ProductionClosureAction(
                post_cc.action,
                ClosurePriority.METHOD_VALIDITY if post_cc.review.status is ReviewStatus.CONFIRMED else ClosurePriority.CORRELATION,
                'G3A_REFERENCE_METHOD_VALIDITY' if post_cc.review.status is ReviewStatus.CONFIRMED else 'G3C_CORRELATION_RELIABILITY',
                POST_CC,
                post_cc.method_role,
                post_cc.review.evidence_ids,
                post_cc.review.rationale,
            ))

    if (
        correlation_review.status not in (ReviewStatus.CLEARED, ReviewStatus.NOT_APPLICABLE)
        and post_cc is None
    ):
        actions.append(ProductionClosureAction(
            'ASSESS_CORRELATION_RELIABILITY', ClosurePriority.CORRELATION,
            'G3C_CORRELATION_RELIABILITY', 'CORRELATION_RELIABILITY', MethodRole.DIAGNOSTIC,
            correlation_review.evidence_ids, correlation_review.rationale,
        ))

    for name, item in physical.items():
        if item.review.status in (ReviewStatus.CLEARED, ReviewStatus.NOT_APPLICABLE):
            continue
        if item.action:
            actions.append(ProductionClosureAction(
                item.action, ClosurePriority.PHYSICAL_CORRECTION,
                'G3D_PHYSICAL_CORRECTIONS', name, item.method_role,
                item.review.evidence_ids, item.review.rationale,
            ))

    if (
        reference_character.status is ReferenceCharacterStatus.BORDERLINE
        and borderline_reference_uncertainty is None
    ):
        actions.append(ProductionClosureAction(
            'ENLARGE_REFERENCE_CHARACTER_UNCERTAINTY', ClosurePriority.UNCERTAINTY,
            'G3E_UNCERTAINTY_CLOSURE', REFERENCE_CHARACTER, MethodRole.DIAGNOSTIC,
            reference_character.evidence_ids,
            'BORDERLINE reference character requires an explicit uncertainty contribution/bound.',
        ))

    unique: dict[tuple[str, str], ProductionClosureAction] = {}
    for item in actions:
        unique[(item.action_id, item.addresses_component)] = item
    return tuple(sorted(unique.values(), key=lambda x: (int(x.priority), x.gate_id, x.action_id, x.addresses_component)))


def build_single_reference_production_evidence_bundle(
    *,
    reference_character: ReferenceCharacterAssessment,
    cardinal: CardinalConvergenceAssessment | None,
    diffuse: DiffuseConvergenceAssessment | None,
    cbs: CBSResolvedResult,
    core_valence: CoreValenceAssessment | None = None,
    scalar_relativity: ScalarRelativityAssessment | None = None,
    nuclear_motion: NuclearMotionAssessment | None = None,
    post_cc: PostCCAssessment | None = None,
    correlation_reliability: Review | None = None,
    expanded_reference_diagnostics: Review | None = None,
    mr_capability: MRProductionCapability | None = None,
    physical_correction_overrides: Mapping[str, CorrectionEvidence] | None = None,
    borderline_reference_uncertainty: UncertaintyComponent | None = None,
) -> ProductionEvidenceBundle:
    """Build the production-side evidence bundle from real OpenEA assessments.

    Missing SOC/nuclear-motion/scalar-remainder/beyond-BO nuclear evidence is
    intentionally represented as open/unknown by default. A reviewed ``nuclear_motion``
    assessment is translated directly into D12 correction evidence; callers
    may still use explicit ``physical_correction_overrides`` for externally
    reviewed physical terms.
    """

    overrides = {} if physical_correction_overrides is None else dict(physical_correction_overrides)
    route = plan_production_route(
        reference_character,
        expanded_reference_diagnostics=expanded_reference_diagnostics,
        mr_capability=mr_capability,
    )
    if route.status is not ProductionRouteStatus.READY_SINGLE_REFERENCE:
        raise ValueError(
            'Single-reference production evidence cannot be assembled unless '
            'the Reference Character Gate authorizes READY_SINGLE_REFERENCE; '
            f'got {route.status.value}'
        )

    baseline_ev, baseline_offset, baseline_extra = _cbs_baseline(cbs)
    post = post_cc_evidence(post_cc)
    corr_review = _combine_correlation_reviews(correlation_reliability, post)
    physical = _default_physical_corrections(core_valence, scalar_relativity, nuclear_motion, overrides)

    corrections: list[UncertaintyComponent] = list(baseline_extra)
    for item in physical.values():
        if item.review.status is ReviewStatus.NOT_APPLICABLE:
            continue
        corrections.append(item.component if item.component is not None else _unknown_component(item.name, item.review.evidence_ids))

    if post is not None:
        corrections.append(post.component if post.component is not None else _unknown_component(POST_CC, post.review.evidence_ids))

    borderline_review = None
    if reference_character.status is ReferenceCharacterStatus.BORDERLINE:
        if borderline_reference_uncertainty is None:
            borderline_review = Review(
                ReviewStatus.PENDING,
                reference_character.evidence_ids,
                'BORDERLINE reference-character uncertainty has not been bounded.',
            )
            corrections.append(_unknown_component(REFERENCE_CHARACTER, reference_character.evidence_ids))
        else:
            if borderline_reference_uncertainty.name != REFERENCE_CHARACTER:
                raise ValueError('Borderline reference uncertainty component must be named REFERENCE_CHARACTER')
            if borderline_reference_uncertainty.correction_ev is None:
                raise ValueError('Borderline reference uncertainty must be bounded, not UNKNOWN')
            corrections.append(borderline_reference_uncertainty)
            borderline_review = Review(
                ReviewStatus.CLEARED,
                borderline_reference_uncertainty.evidence_ids,
                'BORDERLINE reference-character uncertainty is explicitly represented in the error budget.',
            )

    estimate = EAEstimate(
        baseline_ev=baseline_ev,
        baseline_offset_ev=baseline_offset,
        baseline_evidence_ids=tuple(cbs.evidence),
        corrections=tuple(corrections),
    )
    budget = ErrorBudget(
        baseline_offset_ev=baseline_offset,
        baseline_evidence_ids=tuple(cbs.evidence),
        components=tuple(corrections),
        baseline_name='VALENCE_CCSD(T)_CBS_MODEL',
    )

    reassessment_triggers: dict[str, Review] = {}
    if post is not None and post.review.status is ReviewStatus.CONFIRMED:
        reassessment_triggers['POST_CC_WARNING'] = post.review

    gates = build_energy_reliability_gates(
        reference_character=reference_character,
        expanded_reference_diagnostics=expanded_reference_diagnostics,
        cardinal=cardinal,
        diffuse=diffuse,
        cbs=cbs,
        correlation_reliability=corr_review,
        physical_corrections={name: item.review for name, item in physical.items()},
        error_budget=budget,
        mr_capability=mr_capability,
        borderline_uncertainty_review=borderline_review,
        reference_reassessment_triggers=reassessment_triggers,
    )

    actions = _closure_actions(
        route=route,
        cardinal=cardinal,
        diffuse=diffuse,
        cbs=cbs,
        post_cc=post,
        correlation_review=corr_review,
        physical=physical,
        reference_character=reference_character,
        borderline_reference_uncertainty=borderline_reference_uncertainty,
    )

    return ProductionEvidenceBundle(
        route=route,
        estimate=estimate,
        error_budget=budget,
        energy_gates=gates,
        physical_corrections=physical,
        correlation_review=corr_review,
        closure_actions=actions,
    )


def plan_production_evidence(
    bundle: ProductionEvidenceBundle,
    *,
    requested_half_width_ev: float,
    precision_candidates: tuple[PrecisionActionCandidate, ...] = (),
) -> ProductionEvidencePlan:
    """Select gate closure before optional precision refinement.

    This planner never converts a missed precision target into UNRESOLVED.  If
    scientific gates are closed and no valid precision-refinement action is
    available, the result remains ready for the scientific decision engine.
    """

    if not bundle.energy_gates.is_closed or bundle.closure_actions:
        next_action = bundle.next_closure_action
        reason = 'Scientific gate closure is required before precision optimization.'
        if next_action is not None:
            reason += f' Next action: {next_action.action_id}.'
        return ProductionEvidencePlan(
            ProductionEvidencePlanningStatus.GATE_CLOSURE_REQUIRED,
            bundle,
            None,
            reason,
        )

    precision = plan_precision_refinement(
        bundle.interval,
        requested_half_width_ev,
        bundle.error_budget,
        precision_candidates,
    )
    if precision.status is PrecisionPlanningStatus.RECOMMEND_CALCULATION:
        return ProductionEvidencePlan(
            ProductionEvidencePlanningStatus.PRECISION_REFINEMENT_RECOMMENDED,
            bundle,
            precision,
            precision.reason,
        )
    return ProductionEvidencePlan(
        ProductionEvidencePlanningStatus.READY_FOR_SCIENTIFIC_DECISION,
        bundle,
        precision,
        'Scientific energy gates are closed; precision status is reported separately from scientific resolution.',
    )
