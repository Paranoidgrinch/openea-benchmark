"""Adaptive refinement of an identity-cleared Stage-3 high-level local PEC.

This layer decides *where to calculate next*.  It does not fit a continuous
PEC, assign a ground state, compute an EA, or prune electronic states.

Scientific invariants
---------------------
* Refinement starts only from a high-level PEC whose execution, same-geometry
  identity, and geometry continuity gates are already cleared.
* A boundary minimum is extended outward; it is never called an equilibrium.
* A single bracketed discrete minimum is refined symmetrically until an
  explicitly configured bracket-width target is met.
* Multiple bracketed minima are all retained and refined.
* Flat/non-strict interior minima trigger local resolution, not arbitrary
  state selection.
* No production refinement thresholds are supplied here.  Step sizes,
  tolerances, and targets are explicit workflow inputs.
* New interior geometries are seeded independently from both neighboring
  high-level HF checkpoints when available, so the existing Stage-3 identity
  review can detect root flips after refinement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from .stage3_execution import Stage3ExecutionRequest, Stage3PointResult
from .stage3_naming import refinement_request_id
from .stage3_pec import (
    HighLevelMinimumStatus,
    HighLevelPEC,
    HighLevelPECPoint,
    HighLevelPECStatus,
    HighLevelPointStatus,
)


class RefinementAction(str, Enum):
    WAIT_FOR_VALID_PEC = "WAIT_FOR_VALID_PEC"
    EXTEND_LOWER_R = "EXTEND_LOWER_R"
    EXTEND_UPPER_R = "EXTEND_UPPER_R"
    EXTEND_BOTH_SIDES = "EXTEND_BOTH_SIDES"
    REFINE_NONSTRICT_INTERIOR = "REFINE_NONSTRICT_INTERIOR"
    REFINE_SINGLE_BRACKET = "REFINE_SINGLE_BRACKET"
    REFINE_MULTIPLE_BRACKETS = "REFINE_MULTIPLE_BRACKETS"
    BRACKET_TARGET_MET = "BRACKET_TARGET_MET"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class Stage3RefinementSettings:
    extension_step_angstrom: float
    target_bracket_width_angstrom: float
    minimum_new_point_separation_angstrom: float
    energy_tie_tolerance_hartree: float

    def __post_init__(self) -> None:
        for name in (
            "extension_step_angstrom",
            "target_bracket_width_angstrom",
            "minimum_new_point_separation_angstrom",
            "energy_tie_tolerance_hartree",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.extension_step_angstrom <= 0.0:
            raise ValueError("extension_step_angstrom must be positive")
        if self.target_bracket_width_angstrom <= 0.0:
            raise ValueError("target_bracket_width_angstrom must be positive")
        if self.minimum_new_point_separation_angstrom <= 0.0:
            raise ValueError("minimum_new_point_separation_angstrom must be positive")


@dataclass(frozen=True)
class ProposedGeometry:
    r_angstrom: float
    reason: str
    source_interval_angstrom: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isfinite(float(self.r_angstrom)) or self.r_angstrom <= 0.0:
            raise ValueError("Proposed geometry must be positive and finite")
        if not self.reason.strip():
            raise ValueError("Proposed geometry requires a reason")


@dataclass(frozen=True)
class Stage3RefinementPlan:
    job_id: str
    action: RefinementAction
    proposed_geometries: tuple[ProposedGeometry, ...]
    rationale: str
    source_pec_status: str
    source_minimum_status: str
    is_production_ea: bool = False
    ground_state_assigned: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("Refinement plan requires job_id")
        if not self.rationale.strip():
            raise ValueError("Refinement plan requires rationale")
        if self.is_production_ea or self.ground_state_assigned or self.authorizes_pruning:
            raise ValueError("PEC refinement cannot assign EA/ground state or authorize pruning")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        return data


def _accepted_points(pec: HighLevelPEC) -> list[HighLevelPECPoint]:
    return sorted(
        [point for point in pec.points if point.status is HighLevelPointStatus.ACCEPTED],
        key=lambda point: point.r_angstrom,
    )


def _filter_new_r(
    candidates: Sequence[ProposedGeometry],
    *,
    existing_r: Sequence[float],
    minimum_separation: float,
) -> tuple[ProposedGeometry, ...]:
    kept: list[ProposedGeometry] = []
    occupied = [float(x) for x in existing_r]
    for candidate in sorted(candidates, key=lambda item: item.r_angstrom):
        if any(abs(candidate.r_angstrom - value) < minimum_separation for value in occupied):
            continue
        kept.append(candidate)
        occupied.append(candidate.r_angstrom)
    return tuple(kept)


def _midpoint(a: float, b: float, reason: str) -> ProposedGeometry:
    return ProposedGeometry(
        r_angstrom=0.5 * (float(a) + float(b)),
        reason=reason,
        source_interval_angstrom=(min(float(a), float(b)), max(float(a), float(b))),
    )


def plan_stage3_pec_refinement(
    pec: HighLevelPEC,
    *,
    settings: Stage3RefinementSettings,
) -> Stage3RefinementPlan:
    """Choose the next high-level geometries without interpreting them as an EA."""
    minimum_status = pec.minimum_scout.status
    base_kwargs = dict(
        job_id=pec.job_id,
        source_pec_status=pec.status.value,
        source_minimum_status=minimum_status.value,
    )

    if pec.status is not HighLevelPECStatus.READY_FOR_DISCRETE_MINIMUM_SCOUT:
        return Stage3RefinementPlan(
            action=RefinementAction.WAIT_FOR_VALID_PEC,
            proposed_geometries=(),
            rationale="Stage-3 PEC execution/identity/continuity prerequisites are not all cleared",
            **base_kwargs,
        )

    points = _accepted_points(pec)
    if len(points) != len(pec.points):
        return Stage3RefinementPlan(
            action=RefinementAction.WAIT_FOR_VALID_PEC,
            proposed_geometries=(),
            rationale="Not every sampled Stage-3 geometry is an accepted high-level point",
            **base_kwargs,
        )
    if not points:
        return Stage3RefinementPlan(
            action=RefinementAction.UNRESOLVED,
            proposed_geometries=(),
            rationale="No accepted Stage-3 points are available for refinement",
            **base_kwargs,
        )

    existing_r = [point.r_angstrom for point in points]
    min_sep = settings.minimum_new_point_separation_angstrom

    if minimum_status is HighLevelMinimumStatus.BRACKETED_SINGLE_MINIMUM:
        candidate = pec.minimum_scout.candidates[0]
        width = candidate.right_r_angstrom - candidate.left_r_angstrom
        if width <= settings.target_bracket_width_angstrom:
            return Stage3RefinementPlan(
                action=RefinementAction.BRACKET_TARGET_MET,
                proposed_geometries=(),
                rationale=(
                    f"Single high-level minimum bracket width {width:.12g} Angstrom "
                    "meets the explicit refinement target"
                ),
                **base_kwargs,
            )
        proposals = _filter_new_r(
            (
                _midpoint(
                    candidate.left_r_angstrom,
                    candidate.r_angstrom,
                    "Refine left half of the single retained minimum bracket",
                ),
                _midpoint(
                    candidate.r_angstrom,
                    candidate.right_r_angstrom,
                    "Refine right half of the single retained minimum bracket",
                ),
            ),
            existing_r=existing_r,
            minimum_separation=min_sep,
        )
        return Stage3RefinementPlan(
            action=RefinementAction.REFINE_SINGLE_BRACKET,
            proposed_geometries=proposals,
            rationale="Single strict sampled minimum is bracketed but its bracket remains wider than the explicit target",
            **base_kwargs,
        )

    if minimum_status is HighLevelMinimumStatus.MULTIPLE_MINIMUM_CANDIDATES:
        raw: list[ProposedGeometry] = []
        for candidate in pec.minimum_scout.candidates:
            raw.extend((
                _midpoint(
                    candidate.left_r_angstrom,
                    candidate.r_angstrom,
                    f"Resolve left side of retained minimum candidate at R={candidate.r_angstrom:.12g} Angstrom",
                ),
                _midpoint(
                    candidate.r_angstrom,
                    candidate.right_r_angstrom,
                    f"Resolve right side of retained minimum candidate at R={candidate.r_angstrom:.12g} Angstrom",
                ),
            ))
        proposals = _filter_new_r(raw, existing_r=existing_r, minimum_separation=min_sep)
        return Stage3RefinementPlan(
            action=RefinementAction.REFINE_MULTIPLE_BRACKETS,
            proposed_geometries=proposals,
            rationale="Multiple strict sampled minima remain; every candidate bracket is retained and refined",
            **base_kwargs,
        )

    if minimum_status is HighLevelMinimumStatus.INSUFFICIENT_POINTS:
        if len(points) == 1:
            center = points[0].r_angstrom
            raw = []
            if center - settings.extension_step_angstrom > 0.0:
                raw.append(ProposedGeometry(
                    center - settings.extension_step_angstrom,
                    "Create a lower-R neighbor for the single sampled high-level point",
                ))
            raw.append(ProposedGeometry(
                center + settings.extension_step_angstrom,
                "Create an upper-R neighbor for the single sampled high-level point",
            ))
            proposals = _filter_new_r(raw, existing_r=existing_r, minimum_separation=min_sep)
            return Stage3RefinementPlan(
                action=RefinementAction.EXTEND_BOTH_SIDES,
                proposed_geometries=proposals,
                rationale="A single sampled point cannot bracket a minimum",
                **base_kwargs,
            )

        # With two points, extend beyond the lower-energy end.  A tie is
        # deliberately expanded on both sides because neither direction is preferred.
        left, right = points[0], points[-1]
        assert left.energy_hartree is not None and right.energy_hartree is not None
        de = left.energy_hartree - right.energy_hartree
        if abs(de) <= settings.energy_tie_tolerance_hartree:
            raw = []
            if left.r_angstrom - settings.extension_step_angstrom > 0.0:
                raw.append(ProposedGeometry(
                    left.r_angstrom - settings.extension_step_angstrom,
                    "Two-point high-level energy tie: extend lower-R boundary",
                ))
            raw.append(ProposedGeometry(
                right.r_angstrom + settings.extension_step_angstrom,
                "Two-point high-level energy tie: extend upper-R boundary",
            ))
            return Stage3RefinementPlan(
                action=RefinementAction.EXTEND_BOTH_SIDES,
                proposed_geometries=_filter_new_r(raw, existing_r=existing_r, minimum_separation=min_sep),
                rationale="Two sampled points are energetically tied within the explicit tolerance; no direction is preferred",
                **base_kwargs,
            )
        if left.energy_hartree < right.energy_hartree:
            target = left.r_angstrom - settings.extension_step_angstrom
            if target <= 0.0:
                return Stage3RefinementPlan(
                    action=RefinementAction.UNRESOLVED,
                    proposed_geometries=(),
                    rationale="Lower-R extension would produce a non-physical bond distance",
                    **base_kwargs,
                )
            return Stage3RefinementPlan(
                action=RefinementAction.EXTEND_LOWER_R,
                proposed_geometries=(ProposedGeometry(target, "Extend beyond the lower-energy endpoint"),),
                rationale="With two points, the lower-energy endpoint must be tested beyond its boundary",
                **base_kwargs,
            )
        return Stage3RefinementPlan(
            action=RefinementAction.EXTEND_UPPER_R,
            proposed_geometries=(ProposedGeometry(
                right.r_angstrom + settings.extension_step_angstrom,
                "Extend beyond the lower-energy endpoint",
            ),),
            rationale="With two points, the lower-energy endpoint must be tested beyond its boundary",
            **base_kwargs,
        )

    if minimum_status is HighLevelMinimumStatus.NO_BRACKETED_MINIMUM:
        energies = [float(point.energy_hartree) for point in points]
        e_min = min(energies)
        min_indices = [
            index for index, energy in enumerate(energies)
            if abs(energy - e_min) <= settings.energy_tie_tolerance_hartree
        ]
        at_lower = 0 in min_indices
        at_upper = (len(points) - 1) in min_indices

        raw: list[ProposedGeometry] = []
        if at_lower:
            target = points[0].r_angstrom - settings.extension_step_angstrom
            if target > 0.0:
                raw.append(ProposedGeometry(
                    target,
                    "Lowest sampled high-level energy reaches the lower-R boundary",
                ))
        if at_upper:
            raw.append(ProposedGeometry(
                points[-1].r_angstrom + settings.extension_step_angstrom,
                "Lowest sampled high-level energy reaches the upper-R boundary",
            ))

        if raw:
            action = (
                RefinementAction.EXTEND_BOTH_SIDES
                if at_lower and at_upper
                else RefinementAction.EXTEND_LOWER_R
                if at_lower
                else RefinementAction.EXTEND_UPPER_R
            )
            return Stage3RefinementPlan(
                action=action,
                proposed_geometries=_filter_new_r(raw, existing_r=existing_r, minimum_separation=min_sep),
                rationale="No strict interior minimum is bracketed; expand only boundary directions supported by the sampled energy ordering",
                **base_kwargs,
            )

        # The sampled minimum is interior but non-strict (usually a plateau or
        # numerical near-tie). Resolve locally on every boundary of that region.
        raw = []
        min_set = set(min_indices)
        for index in min_indices:
            if index > 0 and (index - 1) not in min_set:
                raw.append(_midpoint(
                    points[index - 1].r_angstrom,
                    points[index].r_angstrom,
                    "Resolve lower boundary of a non-strict interior minimum region",
                ))
            if index < len(points) - 1 and (index + 1) not in min_set:
                raw.append(_midpoint(
                    points[index].r_angstrom,
                    points[index + 1].r_angstrom,
                    "Resolve upper boundary of a non-strict interior minimum region",
                ))
        proposals = _filter_new_r(raw, existing_r=existing_r, minimum_separation=min_sep)
        if not proposals:
            return Stage3RefinementPlan(
                action=RefinementAction.UNRESOLVED,
                proposed_geometries=(),
                rationale="Interior non-strict minimum cannot be resolved without violating the configured point-separation policy",
                **base_kwargs,
            )
        return Stage3RefinementPlan(
            action=RefinementAction.REFINE_NONSTRICT_INTERIOR,
            proposed_geometries=proposals,
            rationale="An interior lowest-energy region exists but is not a strict sampled minimum; refine its boundaries without selecting a state",
            **base_kwargs,
        )

    return Stage3RefinementPlan(
        action=RefinementAction.WAIT_FOR_VALID_PEC,
        proposed_geometries=(),
        rationale=f"Minimum scout status {minimum_status.value} is not actionable for adaptive geometry refinement",
        **base_kwargs,
    )


def _result_by_id(results: Sequence[Stage3PointResult]) -> dict[str, Stage3PointResult]:
    ids = [item.request_id for item in results]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Stage-3 result IDs")
    return {item.request_id: item for item in results}


def _request_by_id(requests: Sequence[Stage3ExecutionRequest]) -> dict[str, Stage3ExecutionRequest]:
    ids = [item.request_id for item in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Stage-3 request IDs")
    return {item.request_id: item for item in requests}


def _seed_points_for_r(points: Sequence[HighLevelPECPoint], target_r: float) -> tuple[HighLevelPECPoint, ...]:
    ordered = sorted(points, key=lambda item: item.r_angstrom)
    lower = [point for point in ordered if point.r_angstrom < target_r]
    upper = [point for point in ordered if point.r_angstrom > target_r]
    if lower and upper:
        # Interior refinement gets independent continuation from both sides.
        return lower[-1], upper[0]
    if lower:
        return (lower[-1],)
    if upper:
        return (upper[0],)
    raise ValueError("Cannot select a high-level seed for refinement geometry")


def build_stage3_refinement_requests(
    *,
    plan: Stage3RefinementPlan,
    pec: HighLevelPEC,
    prior_requests: Sequence[Stage3ExecutionRequest],
    prior_results: Sequence[Stage3PointResult],
    refinement_round: int,
) -> tuple[Stage3ExecutionRequest, ...]:
    """Turn proposed geometries into executable requests seeded by HF checkpoints.

    Interior points are seeded from both neighboring accepted high-level points.
    Boundary extensions use the nearest accepted boundary point.  The returned
    requests remain non-pruning and must be passed through Stage-3 identity and
    continuity review again after execution.
    """
    if refinement_round < 1:
        raise ValueError("refinement_round must be >= 1")
    if plan.job_id != pec.job_id:
        raise ValueError("Refinement plan and PEC belong to different jobs")
    if not plan.proposed_geometries:
        return ()

    req_map = _request_by_id(prior_requests)
    res_map = _result_by_id(prior_results)
    accepted = _accepted_points(pec)
    if len(accepted) != len(pec.points):
        raise ValueError("Refinement requests require an all-accepted source PEC")

    output: list[Stage3ExecutionRequest] = []
    geometry_index = 0
    for proposal in plan.proposed_geometries:
        seeds = _seed_points_for_r(accepted, proposal.r_angstrom)
        for seed_index, point in enumerate(seeds):
            if point.canonical_request_id is None:
                raise ValueError("Accepted high-level seed point lacks canonical request")
            seed_id = point.canonical_request_id
            if seed_id not in req_map or seed_id not in res_map:
                raise ValueError(f"Canonical seed request/result unavailable: {seed_id}")
            source_request = req_map[seed_id]
            source_result = res_map[seed_id]
            checkpoint = source_result.high_level_checkpoint_path
            if not checkpoint:
                raise ValueError(f"Canonical high-level seed has no checkpoint: {seed_id}")
            if not str(checkpoint).strip():
                raise ValueError(f"Canonical high-level checkpoint path is empty: {seed_id}")

            request_id = refinement_request_id(
                job_id=pec.job_id,
                refinement_round=refinement_round,
                geometry_index=geometry_index,
                seed_index=seed_index,
                seed_request_id=seed_id,
                target_r_angstrom=proposal.r_angstrom,
            )
            output.append(Stage3ExecutionRequest(
                request_id=request_id,
                job_id=pec.job_id,
                system=source_request.system,
                atoms=source_request.atoms,
                charge=source_request.charge,
                spin_2s=source_request.spin_2s,
                component_id=source_request.component_id,
                r_angstrom=proposal.r_angstrom,
                basis=source_request.basis,
                methods=source_request.methods,
                requested_reference=source_request.requested_reference,
                scf_reference=source_request.scf_reference,
                source_link_status="HIGH_LEVEL_REFINEMENT_SEED",
                source_root_id=seed_id,
                source_checkpoint_path=str(checkpoint),
                source_origin_guess="STAGE3_HIGH_LEVEL_CONTINUATION",
                grid_index=geometry_index,
                initialization_index=seed_index,
                dft_center_r_angstrom=source_request.dft_center_r_angstrom,
                dft_center_energy_hartree=source_request.dft_center_energy_hartree,
                requires_independent_state_identity_validation=True,
                authorizes_pruning=False,
            ))
        geometry_index += 1

    request_ids = [item.request_id for item in output]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Refinement generated duplicate request IDs")
    return tuple(output)
