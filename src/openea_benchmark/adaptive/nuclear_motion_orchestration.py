"""D12 orchestration between nuclear-motion diagnostics and Stage-3 PEC execution.

The radial solver in :mod:`openea_benchmark.adaptive.nuclear_motion` can decide
that more electronic PEC evidence is required, but it must not invent new
high-level points or silently choose a more expensive electronic model.  This
module closes that control loop using the existing Stage-3 request, execution,
identity and continuity infrastructure.

Two operations are intentionally separate:

``REFINE_NUCLEAR_PEC``
    Extend or densify an already identity-cleared high-level PEC only where the
    nuclear solver says additional electronic support is required.

``ASSESS_NUCLEAR_PEC_MODEL_CONVERGENCE``
    Evaluate an explicitly authorized second electronic PEC level on matched
    neutral/anion geometries, then compare the resulting DeltaZPE values.  The
    comparison level is never guessed by this module.

Scientific invariants
---------------------
* Electronic-state identity is never inferred from energy proximity.
* New PEC points are seeded from existing high-level checkpoints and are
  re-reviewed by the canonical Stage-3 identity/continuity resolver.
* Element-specific basis assignments are preserved during refinement.
* A comparison electronic level requires explicit authorization and explicit
  basis provenance; no cardinal/method escalation is automatic.
* Cross-model state identity is not declared from matching labels alone.  A
  separate evidence record is required before a DeltaZPE model bound may close.
* Every executable Stage-3 batch uses the existing bounded retry semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Callable, Mapping, Sequence

from .nuclear_motion import (
    DiatomicMassSpecification,
    NuclearMotionAssessment,
    NuclearMotionModelEvidence,
    NuclearMotionSettings,
    NuclearMotionStatus,
    VibrationalSolveStatus,
    derive_nuclear_motion_model_evidence,
    nuclear_motion_pec_from_high_level,
    run_diatomic_nuclear_motion,
)
from .stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3ExecutionSettings,
    Stage3PointResult,
)
from .stage3_identity import resolve_high_level_identity_and_pec
from .stage3_loop import (
    Stage3ExecutionAttempt,
    Stage3LoopRetrySettings,
    execute_stage3_batch_with_retries,
)
from .stage3_naming import (
    nuclear_model_comparison_job_id,
    nuclear_model_comparison_request_id,
)
from .stage3_pec import (
    HighLevelPEC,
    HighLevelPECPoint,
    HighLevelPECStatus,
    HighLevelPointStatus,
    IdentityReviewStatus,
)
from .stage3_refinement import (
    ProposedGeometry,
    RefinementAction,
    Stage3RefinementPlan,
    Stage3RefinementSettings,
    build_stage3_refinement_requests,
)


class NuclearPECRefinementRunStatus(str, Enum):
    COMPLETED = "COMPLETED"
    NO_REFINEMENT_REQUIRED = "NO_REFINEMENT_REQUIRED"
    EXECUTION_INCOMPLETE = "EXECUTION_INCOMPLETE"
    IDENTITY_UNRESOLVED = "IDENTITY_UNRESOLVED"
    UNRESOLVED = "UNRESOLVED"


class NuclearPECModelRunStatus(str, Enum):
    MODEL_EVIDENCE_READY = "MODEL_EVIDENCE_READY"
    CROSS_MODEL_IDENTITY_REVIEW_REQUIRED = "CROSS_MODEL_IDENTITY_REVIEW_REQUIRED"
    COMPARISON_PEC_REFINEMENT_REQUIRED = "COMPARISON_PEC_REFINEMENT_REQUIRED"
    COMPARISON_NUMERICAL_REFINEMENT_REQUIRED = "COMPARISON_NUMERICAL_REFINEMENT_REQUIRED"
    EXECUTION_INCOMPLETE = "EXECUTION_INCOMPLETE"
    IDENTITY_UNRESOLVED = "IDENTITY_UNRESOLVED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class NuclearPECStage3Context:
    role: str
    state_id: str
    pec: HighLevelPEC
    requests: tuple[Stage3ExecutionRequest, ...]
    results: tuple[Stage3PointResult, ...]

    def __post_init__(self) -> None:
        if self.role not in {"neutral", "anion"}:
            raise ValueError("Nuclear PEC context role must be neutral or anion")
        if not self.state_id.strip():
            raise ValueError("Nuclear PEC context requires state_id")
        expected_charge = 0 if self.role == "neutral" else -1
        if self.pec.charge != expected_charge:
            raise ValueError(
                f"{self.role} nuclear PEC context requires charge {expected_charge}, "
                f"got {self.pec.charge}"
            )
        if not self.requests or not self.results:
            raise ValueError("Nuclear PEC context requires existing Stage-3 requests/results")
        if len({item.request_id for item in self.requests}) != len(self.requests):
            raise ValueError("Nuclear PEC context contains duplicate request IDs")
        if len({item.request_id for item in self.results}) != len(self.results):
            raise ValueError("Nuclear PEC context contains duplicate result IDs")
        if {item.request_id for item in self.requests} != {item.request_id for item in self.results}:
            raise ValueError("Nuclear PEC context request/result IDs do not match")
        if any(item.job_id != self.pec.job_id for item in self.requests):
            raise ValueError("Nuclear PEC requests belong to another Stage-3 job")
        if any(item.job_id != self.pec.job_id for item in self.results):
            raise ValueError("Nuclear PEC results belong to another Stage-3 job")
        if self.pec.status is not HighLevelPECStatus.READY_FOR_DISCRETE_MINIMUM_SCOUT:
            raise ValueError("Nuclear PEC context requires an identity/continuity-cleared Stage-3 PEC")
        if self.pec.initialization_identity_status is not IdentityReviewStatus.CLEARED:
            raise ValueError("Nuclear PEC context initialization identity is not cleared")
        if self.pec.geometry_continuity_status is not IdentityReviewStatus.CLEARED:
            raise ValueError("Nuclear PEC context geometry continuity is not cleared")


@dataclass(frozen=True)
class NuclearPECRefinementBatch:
    role: str
    plan: Stage3RefinementPlan
    requests: tuple[Stage3ExecutionRequest, ...]


@dataclass(frozen=True)
class NuclearPECRefinementRun:
    status: NuclearPECRefinementRunStatus
    neutral_context: NuclearPECStage3Context
    anion_context: NuclearPECStage3Context
    batches: tuple[NuclearPECRefinementBatch, ...]
    attempts: tuple[Stage3ExecutionAttempt, ...]
    evidence_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("Nuclear PEC refinement run requires rationale")


@dataclass(frozen=True)
class NuclearPECModelLevel:
    """Explicitly authorized comparison level for DeltaZPE model sensitivity.

    OpenEA-v1 currently varies the orbital basis policy while retaining the
    same Stage-3 CCSD/CCSD(T) energy model.  A different correlated method must
    enter through a separately validated production runner rather than being
    smuggled into D12.
    """

    model_id: str
    basis: str
    basis_by_element: Mapping[str, str] | None
    methods: tuple[str, ...]
    authorization_evidence_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.basis.strip() or not self.rationale.strip():
            raise ValueError("Nuclear PEC comparison level requires model_id, basis and rationale")
        if not self.authorization_evidence_ids or any(
            not item.strip() for item in self.authorization_evidence_ids
        ):
            raise ValueError("Nuclear PEC comparison level requires authorization evidence")
        if "CCSD" not in self.methods or "CCSD(T)" not in self.methods:
            raise ValueError("OpenEA-v1 nuclear PEC model comparison requires CCSD and CCSD(T)")
        if self.basis_by_element is not None:
            if not self.basis_by_element or any(
                not str(k).strip() or not str(v).strip()
                for k, v in self.basis_by_element.items()
            ):
                raise ValueError("basis_by_element must contain non-empty element/basis entries")


@dataclass(frozen=True)
class NuclearPECModelRun:
    status: NuclearPECModelRunStatus
    comparison_neutral_context: NuclearPECStage3Context | None
    comparison_anion_context: NuclearPECStage3Context | None
    comparison_assessment: NuclearMotionAssessment | None
    model_evidence: NuclearMotionModelEvidence | None
    final_primary_assessment: NuclearMotionAssessment | None
    attempts: tuple[Stage3ExecutionAttempt, ...]
    evidence_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("Nuclear PEC model run requires rationale")
        if self.status is NuclearPECModelRunStatus.MODEL_EVIDENCE_READY:
            if self.model_evidence is None or self.final_primary_assessment is None:
                raise ValueError("MODEL_EVIDENCE_READY requires model evidence and a final primary assessment")


def _dedup(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(x) for x in values if str(x).strip()))


def _point_map(pec: HighLevelPEC) -> dict[str, HighLevelPECPoint]:
    result = {}
    for point in pec.points:
        if point.status is not HighLevelPointStatus.ACCEPTED or point.canonical_request_id is None:
            raise ValueError("Nuclear PEC orchestration requires all Stage-3 points to be accepted")
        result[point.canonical_request_id] = point
    return result


def _role_result(assessment: NuclearMotionAssessment, role: str):
    return assessment.neutral if role == "neutral" else assessment.anion


def _filter_proposals(
    proposals: Sequence[ProposedGeometry],
    *,
    existing_r: Sequence[float],
    minimum_separation: float,
) -> tuple[ProposedGeometry, ...]:
    occupied = [float(x) for x in existing_r]
    kept: list[ProposedGeometry] = []
    for proposal in sorted(proposals, key=lambda item: item.r_angstrom):
        if any(abs(proposal.r_angstrom - value) < minimum_separation for value in occupied):
            continue
        kept.append(proposal)
        occupied.append(float(proposal.r_angstrom))
    return tuple(kept)


def _nuclear_refinement_plan(
    *,
    assessment: NuclearMotionAssessment,
    context: NuclearPECStage3Context,
    nuclear_settings: NuclearMotionSettings,
    refinement_settings: Stage3RefinementSettings,
) -> Stage3RefinementPlan | None:
    result = _role_result(assessment, context.role)
    if result.status is not VibrationalSolveStatus.PEC_REFINEMENT_REQUIRED:
        return None

    points = tuple(sorted(context.pec.points, key=lambda item: item.r_angstrom))
    if not points:
        raise ValueError("Cannot refine an empty Stage-3 PEC")
    existing_r = tuple(float(point.r_angstrom) for point in points)
    proposals: list[ProposedGeometry] = []

    if result.requires_lower_r_extension:
        target = existing_r[0] - refinement_settings.extension_step_angstrom
        if target <= 0.0:
            raise ValueError("Nuclear lower-R extension would produce a non-physical bond distance")
        proposals.append(ProposedGeometry(
            target,
            f"D12 {context.role}: extend lower-R boundary requested by the radial solver",
        ))
    if result.requires_upper_r_extension:
        proposals.append(ProposedGeometry(
            existing_r[-1] + refinement_settings.extension_step_angstrom,
            f"D12 {context.role}: extend upper-R boundary requested by the radial solver",
        ))

    interpolation_bad = (
        result.interpolation_difference_ev is not None
        and float(result.interpolation_difference_ev)
        > float(nuclear_settings.interpolation_sensitivity_tolerance_ev)
    )
    if interpolation_bad:
        for left, right in zip(existing_r, existing_r[1:]):
            proposals.append(ProposedGeometry(
                0.5 * (left + right),
                f"D12 {context.role}: bisect PEC interval to reduce interpolation sensitivity",
                (left, right),
            ))

    proposals_tuple = _filter_proposals(
        proposals,
        existing_r=existing_r,
        minimum_separation=refinement_settings.minimum_new_point_separation_angstrom,
    )
    if not proposals_tuple:
        raise ValueError(
            f"{context.role} nuclear PEC is flagged for refinement but no admissible new geometry was generated"
        )

    lower = any(p.r_angstrom < existing_r[0] for p in proposals_tuple)
    upper = any(p.r_angstrom > existing_r[-1] for p in proposals_tuple)
    interior = any(existing_r[0] < p.r_angstrom < existing_r[-1] for p in proposals_tuple)
    if lower and upper:
        action = RefinementAction.EXTEND_BOTH_SIDES
    elif lower:
        action = RefinementAction.EXTEND_LOWER_R
    elif upper:
        action = RefinementAction.EXTEND_UPPER_R
    elif interior:
        action = RefinementAction.REFINE_NONSTRICT_INTERIOR
    else:
        action = RefinementAction.UNRESOLVED

    return Stage3RefinementPlan(
        job_id=context.pec.job_id,
        action=action,
        proposed_geometries=proposals_tuple,
        rationale=(
            f"D12 requested additional {context.role} electronic PEC support; "
            "only solver-indicated boundary extension and/or interval bisection is authorized."
        ),
        source_pec_status=context.pec.status.value,
        source_minimum_status=context.pec.minimum_scout.status.value,
    )


def plan_nuclear_pec_refinement(
    *,
    assessment: NuclearMotionAssessment,
    neutral_context: NuclearPECStage3Context,
    anion_context: NuclearPECStage3Context,
    nuclear_settings: NuclearMotionSettings,
    refinement_settings: Stage3RefinementSettings,
    refinement_round: int,
) -> tuple[NuclearPECRefinementBatch, ...]:
    """Translate a D12 PEC-refinement request into explicit Stage-3 requests."""

    if assessment.status is not NuclearMotionStatus.PEC_REFINEMENT_REQUIRED:
        return ()
    if assessment.action != "REFINE_NUCLEAR_PEC":
        raise ValueError("Nuclear assessment does not authorize REFINE_NUCLEAR_PEC")
    if refinement_round < 1:
        raise ValueError("refinement_round must be >= 1")

    batches: list[NuclearPECRefinementBatch] = []
    for context in (neutral_context, anion_context):
        plan = _nuclear_refinement_plan(
            assessment=assessment,
            context=context,
            nuclear_settings=nuclear_settings,
            refinement_settings=refinement_settings,
        )
        if plan is None:
            continue
        requests = build_stage3_refinement_requests(
            plan=plan,
            pec=context.pec,
            prior_requests=context.requests,
            prior_results=context.results,
            refinement_round=refinement_round,
        )
        if not requests:
            raise ValueError("D12 refinement plan produced no executable Stage-3 requests")
        batches.append(NuclearPECRefinementBatch(context.role, plan, requests))
    if not batches:
        raise ValueError("REFINE_NUCLEAR_PEC was requested but neither PEC requires electronic refinement")
    return tuple(batches)


def _resolve_context(
    *,
    context: NuclearPECStage3Context,
    new_requests: Sequence[Stage3ExecutionRequest],
    new_results: Sequence[Stage3PointResult],
    identity_thresholds: Any,
    branch_thresholds: Any,
    audit_settings: Any,
    duplicate_energy_tolerance_hartree: float,
    resolver: Callable[..., Any] | None,
) -> NuclearPECStage3Context | None:
    all_requests = tuple(context.requests) + tuple(new_requests)
    all_results = tuple(context.results) + tuple(new_results)
    implementation = resolver or resolve_high_level_identity_and_pec
    resolution = implementation(
        requests=all_requests,
        results=all_results,
        identity_thresholds=identity_thresholds,
        branch_thresholds=branch_thresholds,
        audit_settings=audit_settings,
        duplicate_energy_tolerance_hartree=duplicate_energy_tolerance_hartree,
    )
    pec = resolution.pec
    if pec.status is not HighLevelPECStatus.READY_FOR_DISCRETE_MINIMUM_SCOUT:
        return None
    if pec.initialization_identity_status is not IdentityReviewStatus.CLEARED:
        return None
    if pec.geometry_continuity_status is not IdentityReviewStatus.CLEARED:
        return None
    return NuclearPECStage3Context(
        context.role,
        context.state_id,
        pec,
        all_requests,
        all_results,
    )


def run_nuclear_pec_refinement(
    *,
    assessment: NuclearMotionAssessment,
    neutral_context: NuclearPECStage3Context,
    anion_context: NuclearPECStage3Context,
    nuclear_settings: NuclearMotionSettings,
    refinement_settings: Stage3RefinementSettings,
    refinement_round: int,
    execution_settings: Stage3ExecutionSettings,
    identity_thresholds: Any,
    branch_thresholds: Any,
    retry_settings: Stage3LoopRetrySettings | None = None,
    audit_settings: Any = None,
    duplicate_energy_tolerance_hartree: float = 1.0e-7,
    point_runner: Callable[[Stage3ExecutionRequest, Stage3ExecutionSettings], Stage3PointResult] | None = None,
    resolver: Callable[..., Any] | None = None,
) -> NuclearPECRefinementRun:
    """Execute exactly one D12-requested electronic PEC refinement batch."""

    if resolver is None and execution_settings.artifact_dir is None:
        raise ValueError(
            "Real D12 Stage-3 refinement requires execution_settings.artifact_dir so new high-level checkpoints are retained for identity/continuity review"
        )

    batches = plan_nuclear_pec_refinement(
        assessment=assessment,
        neutral_context=neutral_context,
        anion_context=anion_context,
        nuclear_settings=nuclear_settings,
        refinement_settings=refinement_settings,
        refinement_round=refinement_round,
    )
    if not batches:
        return NuclearPECRefinementRun(
            NuclearPECRefinementRunStatus.NO_REFINEMENT_REQUIRED,
            neutral_context,
            anion_context,
            (), (),
            assessment.evidence_ids,
            "The supplied nuclear-motion assessment does not request electronic PEC refinement.",
        )

    contexts = {"neutral": neutral_context, "anion": anion_context}
    attempts: list[Stage3ExecutionAttempt] = []
    evidence = list(assessment.evidence_ids)
    for batch in batches:
        results, batch_attempts = execute_stage3_batch_with_retries(
            batch.requests,
            settings=execution_settings,
            retry_settings=retry_settings,
            runner=point_runner,
        )
        attempts.extend(batch_attempts)
        if any(item.status is not PointExecutionStatus.COMPLETED for item in results):
            return NuclearPECRefinementRun(
                NuclearPECRefinementRunStatus.EXECUTION_INCOMPLETE,
                contexts["neutral"], contexts["anion"],
                batches, tuple(attempts), _dedup(evidence),
                f"At least one {batch.role} Stage-3 nuclear-PEC refinement point did not complete.",
            )
        updated = _resolve_context(
            context=contexts[batch.role],
            new_requests=batch.requests,
            new_results=results,
            identity_thresholds=identity_thresholds,
            branch_thresholds=branch_thresholds,
            audit_settings=audit_settings,
            duplicate_energy_tolerance_hartree=duplicate_energy_tolerance_hartree,
            resolver=resolver,
        )
        if updated is None:
            return NuclearPECRefinementRun(
                NuclearPECRefinementRunStatus.IDENTITY_UNRESOLVED,
                contexts["neutral"], contexts["anion"],
                batches, tuple(attempts), _dedup(evidence),
                f"The refined {batch.role} PEC did not re-clear Stage-3 identity/continuity review.",
            )
        contexts[batch.role] = updated
        evidence.extend(
            evidence_id
            for point in updated.pec.points
            for evidence_id in point.evidence_ids
        )

    return NuclearPECRefinementRun(
        NuclearPECRefinementRunStatus.COMPLETED,
        contexts["neutral"], contexts["anion"],
        batches, tuple(attempts), _dedup(evidence + ["D12_STAGE3_PEC_REFINEMENT_COMPLETED"]),
        "All D12-requested electronic PEC points completed and the updated neutral/anion PECs re-cleared Stage-3 identity/continuity review.",
    )




def _canonical_basis_signature(
    basis: str,
    mapping: Mapping[str, str] | None,
) -> tuple[str, tuple[tuple[str, str], ...] | None]:
    label = str(basis)
    if mapping is None:
        return label, None
    normalized = tuple(sorted((str(k), str(v)) for k, v in mapping.items()))
    # A per-element mapping that assigns the same named basis to every element
    # is physically the same policy as the global basis label.
    if normalized and all(value == label for _, value in normalized):
        return label, None
    return label, normalized


def _context_basis_policy_signature(context: NuclearPECStage3Context) -> tuple[str, tuple[tuple[str, str], ...] | None]:
    signatures = {
        _canonical_basis_signature(request.basis, request.basis_by_element)
        for request in context.requests
    }
    if len(signatures) != 1:
        raise ValueError(f"{context.role} Stage-3 context mixes orbital-basis policies")
    return next(iter(signatures))


def _model_basis_policy_signature(model: NuclearPECModelLevel) -> tuple[str, tuple[tuple[str, str], ...] | None]:
    return _canonical_basis_signature(model.basis, model.basis_by_element)

def _comparison_seed_requests(
    *,
    context: NuclearPECStage3Context,
    model: NuclearPECModelLevel,
) -> tuple[Stage3ExecutionRequest, ...]:
    request_map = {item.request_id: item for item in context.requests}
    result_map = {item.request_id: item for item in context.results}
    job_id = nuclear_model_comparison_job_id(
        source_job_id=context.pec.job_id,
        model_id=model.model_id,
    )
    output: list[Stage3ExecutionRequest] = []
    points = tuple(sorted(context.pec.points, key=lambda item: item.r_angstrom))
    for geometry_index, point in enumerate(points):
        if point.status is not HighLevelPointStatus.ACCEPTED or point.canonical_request_id is None:
            raise ValueError("Comparison PEC generation requires accepted canonical Stage-3 points")
        seed_id = point.canonical_request_id
        source_request = request_map.get(seed_id)
        source_result = result_map.get(seed_id)
        if source_request is None or source_result is None:
            raise ValueError(f"Primary canonical seed request/result unavailable: {seed_id}")
        checkpoint = source_result.high_level_checkpoint_path
        if not checkpoint or not str(checkpoint).strip():
            raise ValueError(f"Primary canonical high-level checkpoint unavailable: {seed_id}")
        expected_elements = {str(x) for x in source_request.atoms}
        basis_by_element = None
        if model.basis_by_element is not None:
            if set(model.basis_by_element) != expected_elements:
                raise ValueError(
                    f"Comparison basis_by_element must provide exactly {sorted(expected_elements)}"
                )
            basis_by_element = dict(model.basis_by_element)

        request_id = nuclear_model_comparison_request_id(
            source_job_id=context.pec.job_id,
            model_id=model.model_id,
            geometry_index=geometry_index,
            seed_index=0,
            seed_request_id=seed_id,
            target_r_angstrom=point.r_angstrom,
        )
        output.append(Stage3ExecutionRequest(
            request_id=request_id,
            job_id=job_id,
            system=source_request.system,
            atoms=source_request.atoms,
            charge=source_request.charge,
            spin_2s=source_request.spin_2s,
            component_id=source_request.component_id,
            r_angstrom=point.r_angstrom,
            basis=model.basis,
            basis_by_element=basis_by_element,
            methods=model.methods,
            requested_reference=source_request.requested_reference,
            scf_reference=source_request.scf_reference,
            source_link_status="NUCLEAR_MODEL_COMPARISON_PRIMARY_SEED",
            source_root_id=seed_id,
            source_checkpoint_path=str(checkpoint),
            source_origin_guess="D12_PRIMARY_HIGH_LEVEL_CONTINUATION",
            grid_index=geometry_index,
            initialization_index=0,
            dft_center_r_angstrom=source_request.dft_center_r_angstrom,
            dft_center_energy_hartree=source_request.dft_center_energy_hartree,
            requires_independent_state_identity_validation=True,
            authorizes_pruning=False,
        ))
    return tuple(output)


def _resolve_new_model_context(
    *,
    role: str,
    state_id: str,
    requests: Sequence[Stage3ExecutionRequest],
    results: Sequence[Stage3PointResult],
    identity_thresholds: Any,
    branch_thresholds: Any,
    audit_settings: Any,
    duplicate_energy_tolerance_hartree: float,
    resolver: Callable[..., Any] | None,
) -> NuclearPECStage3Context | None:
    implementation = resolver or resolve_high_level_identity_and_pec
    resolution = implementation(
        requests=tuple(requests),
        results=tuple(results),
        identity_thresholds=identity_thresholds,
        branch_thresholds=branch_thresholds,
        audit_settings=audit_settings,
        duplicate_energy_tolerance_hartree=duplicate_energy_tolerance_hartree,
    )
    pec = resolution.pec
    if pec.status is not HighLevelPECStatus.READY_FOR_DISCRETE_MINIMUM_SCOUT:
        return None
    if pec.initialization_identity_status is not IdentityReviewStatus.CLEARED:
        return None
    if pec.geometry_continuity_status is not IdentityReviewStatus.CLEARED:
        return None
    return NuclearPECStage3Context(role, state_id, pec, tuple(requests), tuple(results))


def run_nuclear_pec_model_convergence(
    *,
    primary_assessment: NuclearMotionAssessment,
    primary_neutral_context: NuclearPECStage3Context,
    primary_anion_context: NuclearPECStage3Context,
    model: NuclearPECModelLevel,
    masses: DiatomicMassSpecification,
    nuclear_settings: NuclearMotionSettings,
    execution_settings: Stage3ExecutionSettings,
    identity_thresholds: Any,
    branch_thresholds: Any,
    anion_dissociation_channels: Sequence[Any] | None = None,
    dissociation_safety_margin_hartree: float = 0.0,
    retry_settings: Stage3LoopRetrySettings | None = None,
    audit_settings: Any = None,
    duplicate_energy_tolerance_hartree: float = 1.0e-7,
    cross_model_identity_evidence_ids: tuple[str, ...] = (),
    existing_comparison_neutral_context: NuclearPECStage3Context | None = None,
    existing_comparison_anion_context: NuclearPECStage3Context | None = None,
    point_runner: Callable[[Stage3ExecutionRequest, Stage3ExecutionSettings], Stage3PointResult] | None = None,
    resolver: Callable[..., Any] | None = None,
) -> NuclearPECModelRun:
    """Generate/reuse an authorized second PEC level and bound DeltaZPE sensitivity."""

    if resolver is None and execution_settings.artifact_dir is None and (
        existing_comparison_neutral_context is None or existing_comparison_anion_context is None
    ):
        raise ValueError(
            "Real D12 comparison-PEC execution requires execution_settings.artifact_dir so comparison checkpoints are retained for identity/continuity review"
        )
    if primary_assessment.status is not NuclearMotionStatus.MODEL_CONVERGENCE_REQUIRED:
        raise ValueError("Primary D12 assessment does not request PEC model convergence")
    if primary_assessment.action != "ASSESS_NUCLEAR_PEC_MODEL_CONVERGENCE":
        raise ValueError("Primary D12 assessment does not authorize model-convergence work")
    if execution_settings.scalar_relativistic != "NONE":
        raise ValueError("D12 electronic model comparison must not mix scalar-relativity into the PEC level")
    if execution_settings.frozen_core:
        # Nuclear motion follows the already selected base electronic model; a
        # separate CV correction must remain separate from the Born-Oppenheimer PEC.
        raise ValueError("D12 comparison PEC execution must not silently fold core-valence into the base PEC")

    primary_neutral_policy = _context_basis_policy_signature(primary_neutral_context)
    primary_anion_policy = _context_basis_policy_signature(primary_anion_context)
    if primary_neutral_policy != primary_anion_policy:
        raise ValueError("Primary neutral/anion PECs do not share one orbital-basis policy")
    if _model_basis_policy_signature(model) == primary_neutral_policy:
        raise ValueError("Comparison model must differ explicitly from the primary PEC model")

    attempts: list[Stage3ExecutionAttempt] = []
    comparison_contexts: dict[str, NuclearPECStage3Context] = {}
    supplied = (existing_comparison_neutral_context, existing_comparison_anion_context)
    if (supplied[0] is None) != (supplied[1] is None):
        raise ValueError("Existing comparison PEC contexts must be supplied for both neutral and anion")

    if supplied[0] is not None and supplied[1] is not None:
        comparison_contexts = {"neutral": supplied[0], "anion": supplied[1]}
    else:
        if execution_settings.checkpoint_project is not True:
            raise ValueError(
                "Comparison PEC execution requires checkpoint_project=True so primary high-level orbitals are explicitly projected into the authorized comparison basis"
            )
        for context in (primary_neutral_context, primary_anion_context):
            requests = _comparison_seed_requests(context=context, model=model)
            results, batch_attempts = execute_stage3_batch_with_retries(
                requests,
                settings=execution_settings,
                retry_settings=retry_settings,
                runner=point_runner,
            )
            attempts.extend(batch_attempts)
            if any(item.status is not PointExecutionStatus.COMPLETED for item in results):
                return NuclearPECModelRun(
                    NuclearPECModelRunStatus.EXECUTION_INCOMPLETE,
                    comparison_contexts.get("neutral"), comparison_contexts.get("anion"),
                    None, None, None, tuple(attempts),
                    model.authorization_evidence_ids,
                    f"At least one {context.role} comparison-level Stage-3 point did not complete.",
                )
            resolved = _resolve_new_model_context(
                role=context.role,
                state_id=context.state_id,
                requests=requests,
                results=results,
                identity_thresholds=identity_thresholds,
                branch_thresholds=branch_thresholds,
                audit_settings=audit_settings,
                duplicate_energy_tolerance_hartree=duplicate_energy_tolerance_hartree,
                resolver=resolver,
            )
            if resolved is None:
                return NuclearPECModelRun(
                    NuclearPECModelRunStatus.IDENTITY_UNRESOLVED,
                    comparison_contexts.get("neutral"), comparison_contexts.get("anion"),
                    None, None, None, tuple(attempts),
                    model.authorization_evidence_ids,
                    f"The {context.role} comparison PEC did not clear Stage-3 identity/continuity review.",
                )
            comparison_contexts[context.role] = resolved

    neutral_cmp = comparison_contexts["neutral"]
    anion_cmp = comparison_contexts["anion"]
    if neutral_cmp.pec.basis != model.basis or anion_cmp.pec.basis != model.basis:
        raise ValueError("Comparison PEC contexts do not match the explicitly authorized basis label")
    expected_policy = _model_basis_policy_signature(model)
    if _context_basis_policy_signature(neutral_cmp) != expected_policy:
        raise ValueError("Neutral comparison PEC does not match the explicitly authorized orbital-basis policy")
    if _context_basis_policy_signature(anion_cmp) != expected_policy:
        raise ValueError("Anion comparison PEC does not match the explicitly authorized orbital-basis policy")
    if neutral_cmp.state_id != primary_neutral_context.state_id:
        raise ValueError("Neutral comparison state_id does not match primary state_id")
    if anion_cmp.state_id != primary_anion_context.state_id:
        raise ValueError("Anion comparison state_id does not match primary state_id")

    neutral_nuc = nuclear_motion_pec_from_high_level(
        neutral_cmp.pec,
        role="neutral",
        state_id=neutral_cmp.state_id,
        evidence_ids=model.authorization_evidence_ids,
    )
    anion_nuc = nuclear_motion_pec_from_high_level(
        anion_cmp.pec,
        role="anion",
        state_id=anion_cmp.state_id,
        evidence_ids=model.authorization_evidence_ids,
    )
    comparison_assessment = run_diatomic_nuclear_motion(
        neutral_pec=neutral_nuc,
        anion_pec=anion_nuc,
        masses=masses,
        settings=nuclear_settings,
        anion_dissociation_channels=anion_dissociation_channels,
        dissociation_safety_margin_hartree=dissociation_safety_margin_hartree,
        model_evidence=None,
    )

    common_evidence = _dedup(
        tuple(primary_assessment.evidence_ids)
        + tuple(comparison_assessment.evidence_ids)
        + tuple(model.authorization_evidence_ids)
    )
    if comparison_assessment.status is NuclearMotionStatus.PEC_REFINEMENT_REQUIRED:
        return NuclearPECModelRun(
            NuclearPECModelRunStatus.COMPARISON_PEC_REFINEMENT_REQUIRED,
            neutral_cmp, anion_cmp, comparison_assessment,
            None, None, tuple(attempts), common_evidence,
            "The explicitly authorized comparison electronic level was generated, but its nuclear PEC still requires electronic range/density refinement.",
        )
    if comparison_assessment.status is NuclearMotionStatus.NUMERICAL_REFINEMENT_REQUIRED:
        return NuclearPECModelRun(
            NuclearPECModelRunStatus.COMPARISON_NUMERICAL_REFINEMENT_REQUIRED,
            neutral_cmp, anion_cmp, comparison_assessment,
            None, None, tuple(attempts), common_evidence,
            "The comparison PEC is available, but its radial solve requires a denser numerical grid.",
        )
    if comparison_assessment.status not in {
        NuclearMotionStatus.MODEL_CONVERGENCE_REQUIRED,
        NuclearMotionStatus.CLEARED,
    }:
        return NuclearPECModelRun(
            NuclearPECModelRunStatus.UNRESOLVED,
            neutral_cmp, anion_cmp, comparison_assessment,
            None, None, tuple(attempts), common_evidence,
            "The comparison D12 solve is scientifically unresolved and cannot provide a model-sensitivity bound.",
        )

    if not cross_model_identity_evidence_ids or any(
        not item.strip() for item in cross_model_identity_evidence_ids
    ):
        return NuclearPECModelRun(
            NuclearPECModelRunStatus.CROSS_MODEL_IDENTITY_REVIEW_REQUIRED,
            neutral_cmp, anion_cmp, comparison_assessment,
            None, None, tuple(attempts), common_evidence,
            "Both PEC levels are internally identity/continuity-cleared, but a separate cross-model state-identity evidence record is required before their DeltaZPE difference may be interpreted as model sensitivity.",
        )

    model_evidence = derive_nuclear_motion_model_evidence(
        primary_assessment,
        comparison_assessment,
        evidence_ids=_dedup(tuple(cross_model_identity_evidence_ids) + tuple(model.authorization_evidence_ids)),
        rationale=(
            "Observed DeltaZPE shift between explicitly authorized, state-matched primary and comparison electronic PEC levels."
        ),
    )

    primary_neutral_nuc = nuclear_motion_pec_from_high_level(
        primary_neutral_context.pec,
        role="neutral",
        state_id=primary_neutral_context.state_id,
        evidence_ids=primary_assessment.evidence_ids,
    )
    primary_anion_nuc = nuclear_motion_pec_from_high_level(
        primary_anion_context.pec,
        role="anion",
        state_id=primary_anion_context.state_id,
        evidence_ids=primary_assessment.evidence_ids,
    )
    final_primary = run_diatomic_nuclear_motion(
        neutral_pec=primary_neutral_nuc,
        anion_pec=primary_anion_nuc,
        masses=masses,
        settings=nuclear_settings,
        anion_dissociation_channels=anion_dissociation_channels,
        dissociation_safety_margin_hartree=dissociation_safety_margin_hartree,
        model_evidence=model_evidence,
    )
    if final_primary.status is not NuclearMotionStatus.CLEARED:
        return NuclearPECModelRun(
            NuclearPECModelRunStatus.UNRESOLVED,
            neutral_cmp, anion_cmp, comparison_assessment,
            model_evidence, final_primary, tuple(attempts),
            _dedup(common_evidence + tuple(cross_model_identity_evidence_ids)),
            "A model-sensitivity bound was derived, but the final primary D12 assessment still did not clear.",
        )

    return NuclearPECModelRun(
        NuclearPECModelRunStatus.MODEL_EVIDENCE_READY,
        neutral_cmp, anion_cmp, comparison_assessment,
        model_evidence, final_primary, tuple(attempts),
        _dedup(common_evidence + tuple(cross_model_identity_evidence_ids) + ("D12_PEC_MODEL_SENSITIVITY_BOUNDED",)),
        "An explicitly authorized second electronic PEC level was generated/reused, independently identity/continuity-cleared, cross-model identity evidence was supplied, and the observed DeltaZPE shift now bounds D12 electronic-model sensitivity.",
    )
