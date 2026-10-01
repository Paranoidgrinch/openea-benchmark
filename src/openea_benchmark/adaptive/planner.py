"""Deterministic adaptive action planning from reviewed OpenEA evidence.

This module does not run quantum-chemistry jobs and does not reinterpret scout
energies as uncertainty bounds.  It converts reviewed D04/D09 evidence into a
small set of explicit next actions and provides a safe queue of existing
Stage-3 seed candidates.

Scientific invariants
---------------------
* DFT/scout splittings may ORDER high-level candidate jobs, but may not prune
  candidates or clear D04.
* A locally bracketed minimum is a Stage-3 seed, not a production energy
  interval and not an equilibrium-geometry proof.
* Open PEC/asymptote evidence remains a prerequisite even when high-level
  candidate calculations are also useful.
* Missing seed provenance fails closed; the planner requests an audit rather
  than fabricating a geometry/energy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from math import inf, isfinite
from typing import Iterable

from .model import DiagnosticID, DiagnosticRecord, ReviewStatus
from .pec_selection_bridge import CandidateEnergyEvidence, StateSelectionBridgeResult


class ActionPriority(IntEnum):
    """Coarse scientific priority. Lower runs first."""

    P0_PREREQUISITE = 0
    P1_DOMINANT_UNCERTAINTY = 1
    P2_REFINEMENT = 2


class ActionKind(str, Enum):
    AUDIT_STATE_SELECTION = "AUDIT_STATE_SELECTION"
    COMPLETE_STATE_SELECTION = "COMPLETE_STATE_SELECTION"
    RESOLVE_STATE_CONTINUITY = "RESOLVE_STATE_CONTINUITY"
    EXTEND_OPEN_PEC = "EXTEND_OPEN_PEC"
    CONSTRUCT_RELEVANT_PECS = "CONSTRUCT_RELEVANT_PECS"
    RESOLVE_PEC_ASYMPTOTES = "RESOLVE_PEC_ASYMPTOTES"
    HIGH_ACCURACY_CANDIDATE_COMPARISON = "HIGH_ACCURACY_CANDIDATE_COMPARISON"
    INVESTIGATE_DIAGNOSTIC = "INVESTIGATE_DIAGNOSTIC"


@dataclass(frozen=True)
class Stage3SeedCandidate:
    """A conservative seed for an existing high-level Stage-3 calculation.

    `scout_delta_ev` is an ordering hint only.  `authorizes_pruning` is fixed
    false by design: no DFT/scout splitting may remove a state candidate.
    """

    candidate_ref: str
    group_ref: str
    component_id: str
    r_angstrom: float
    sampled_energy_hartree: float
    scout_delta_ev: float | None
    evidence_ids: tuple[str, ...]
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_ref.strip() or not self.component_id.strip():
            raise ValueError("Stage-3 seed requires stable candidate identity")
        if not isfinite(float(self.r_angstrom)):
            raise ValueError("Stage-3 seed geometry must be finite")
        if not isfinite(float(self.sampled_energy_hartree)):
            raise ValueError("Stage-3 seed energy must be finite")
        if self.scout_delta_ev is not None and not isfinite(float(self.scout_delta_ev)):
            raise ValueError("Scout delta must be finite when supplied")
        if not self.evidence_ids:
            raise ValueError("Stage-3 seed requires provenance")
        if self.authorizes_pruning:
            raise ValueError("Scout-derived Stage-3 seed may never authorize pruning")


@dataclass(frozen=True)
class Stage3SeedQueue:
    ready: tuple[Stage3SeedCandidate, ...]
    blocked_candidate_refs: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.blocked_candidate_refs


@dataclass(frozen=True)
class AdaptiveAction:
    kind: ActionKind
    priority: ActionPriority
    sequence_rank: int
    diagnostic_id: DiagnosticID
    target_refs: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    reason: str
    route: str
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if not self.reason.strip() or not self.route.strip():
            raise ValueError("Planner action requires reason and route")
        if self.authorizes_pruning:
            raise ValueError("Planner actions from D04/D09 scout evidence cannot prune states")


@dataclass(frozen=True)
class AdaptivePlan:
    actions: tuple[AdaptiveAction, ...]
    stage3_seeds: Stage3SeedQueue

    @property
    def next_action(self) -> AdaptiveAction | None:
        return self.actions[0] if self.actions else None


_ROUTE_TO_KIND = {
    "audit_state_selection_report": ActionKind.AUDIT_STATE_SELECTION,
    "complete_state_selection": ActionKind.COMPLETE_STATE_SELECTION,
    "resolve_state_continuity": ActionKind.RESOLVE_STATE_CONTINUITY,
    "extend_open_pec_components": ActionKind.EXTEND_OPEN_PEC,
    "extend_open_candidate_pecs": ActionKind.EXTEND_OPEN_PEC,
    "construct_relevant_pecs": ActionKind.CONSTRUCT_RELEVANT_PECS,
    "resolve_pec_asymptotes": ActionKind.RESOLVE_PEC_ASYMPTOTES,
    "high_accuracy_candidate_comparison": ActionKind.HIGH_ACCURACY_CANDIDATE_COMPARISON,
}


def _candidate_order_key(item: CandidateEnergyEvidence) -> tuple[float, str]:
    delta = item.delta_from_lowest_bracketed_ev
    return (inf if delta is None else float(delta), item.candidate_ref)


def build_stage3_seed_queue(
    bridge: StateSelectionBridgeResult,
) -> Stage3SeedQueue:
    """Build a complete, deterministic queue of usable bracketed scout seeds.

    All usable candidates are retained.  DFT/scout delta only controls ordering.
    Missing sampled geometry/energy is reported as blocked instead of skipped
    silently.
    """

    ready: list[Stage3SeedCandidate] = []
    blocked: list[str] = []

    for item in sorted(bridge.candidate_evidence, key=_candidate_order_key):
        if item.sampled_r_angstrom is None or item.sampled_energy_hartree is None:
            blocked.append(item.candidate_ref)
            continue
        ready.append(
            Stage3SeedCandidate(
                candidate_ref=item.candidate_ref,
                group_ref=item.group_ref,
                component_id=item.component_id,
                r_angstrom=float(item.sampled_r_angstrom),
                sampled_energy_hartree=float(item.sampled_energy_hartree),
                scout_delta_ev=item.delta_from_lowest_bracketed_ev,
                evidence_ids=item.evidence_ids,
            )
        )

    return Stage3SeedQueue(tuple(ready), tuple(blocked))


def _action_from_diagnostic(
    record: DiagnosticRecord,
    *,
    candidate_refs: tuple[str, ...],
) -> AdaptiveAction | None:
    status = record.review.status
    if status in (ReviewStatus.CLEARED, ReviewStatus.NOT_APPLICABLE):
        return None

    route = record.recommended_branch or "investigate_diagnostic"
    kind = _ROUTE_TO_KIND.get(route, ActionKind.INVESTIGATE_DIAGNOSTIC)

    # D09 prerequisite work precedes D04 comparison when both are open.
    if record.identifier is DiagnosticID.D09_PEC_ASYMPTOTES:
        sequence_rank = 0
    elif kind in (
        ActionKind.AUDIT_STATE_SELECTION,
        ActionKind.COMPLETE_STATE_SELECTION,
        ActionKind.RESOLVE_STATE_CONTINUITY,
        ActionKind.EXTEND_OPEN_PEC,
        ActionKind.CONSTRUCT_RELEVANT_PECS,
    ):
        sequence_rank = 10
    elif kind is ActionKind.HIGH_ACCURACY_CANDIDATE_COMPARISON:
        sequence_rank = 20
    else:
        sequence_rank = 30

    if kind is ActionKind.HIGH_ACCURACY_CANDIDATE_COMPARISON:
        targets = candidate_refs
    else:
        targets = record.state_refs

    return AdaptiveAction(
        kind=kind,
        priority=ActionPriority.P0_PREREQUISITE,
        sequence_rank=sequence_rank,
        diagnostic_id=record.identifier,
        target_refs=tuple(targets),
        evidence_ids=record.review.evidence_ids,
        reason=record.review.rationale or f"{record.identifier.value} remains open",
        route=route,
    )


def _deduplicate_actions(actions: Iterable[AdaptiveAction]) -> tuple[AdaptiveAction, ...]:
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    kept: list[AdaptiveAction] = []
    for action in actions:
        key = (action.kind.value, action.diagnostic_id.value, action.target_refs)
        if key in seen:
            continue
        seen.add(key)
        kept.append(action)
    return tuple(
        sorted(
            kept,
            key=lambda item: (
                int(item.priority),
                item.sequence_rank,
                item.diagnostic_id.value,
                item.kind.value,
                item.target_refs,
            ),
        )
    )


def plan_from_state_selection_bridge(
    bridge: StateSelectionBridgeResult,
) -> AdaptivePlan:
    """Create deterministic next actions from D04/D09 bridge evidence.

    This is intentionally a planning layer only.  It neither executes the
    existing Stage-3 planner nor changes state-selection output.
    """

    seeds = build_stage3_seed_queue(bridge)
    candidate_refs = tuple(seed.candidate_ref for seed in seeds.ready)

    actions: list[AdaptiveAction] = []

    if seeds.blocked_candidate_refs:
        actions.append(
            AdaptiveAction(
                kind=ActionKind.AUDIT_STATE_SELECTION,
                priority=ActionPriority.P0_PREREQUISITE,
                sequence_rank=-10,
                diagnostic_id=DiagnosticID.D04_STATE_COMPETITION,
                target_refs=seeds.blocked_candidate_refs,
                evidence_ids=bridge.state_competition.review.evidence_ids,
                reason=(
                    "Bracketed candidate evidence lacks sampled geometry and/or "
                    "energy required for a reproducible Stage-3 seed"
                ),
                route="audit_state_selection_report",
            )
        )

    for record in bridge.diagnostics:
        action = _action_from_diagnostic(record, candidate_refs=candidate_refs)
        if action is not None:
            actions.append(action)

    # If D04 requests a high-accuracy comparison but no usable seed exists,
    # fail closed and request completion/audit instead of emitting an empty job.
    if any(a.kind is ActionKind.HIGH_ACCURACY_CANDIDATE_COMPARISON for a in actions) and not seeds.ready:
        actions = [a for a in actions if a.kind is not ActionKind.HIGH_ACCURACY_CANDIDATE_COMPARISON]
        actions.append(
            AdaptiveAction(
                kind=ActionKind.COMPLETE_STATE_SELECTION,
                priority=ActionPriority.P0_PREREQUISITE,
                sequence_rank=5,
                diagnostic_id=DiagnosticID.D04_STATE_COMPETITION,
                target_refs=(),
                evidence_ids=bridge.state_competition.review.evidence_ids,
                reason="High-accuracy candidate comparison requested but no reproducible bracketed seed is available",
                route="complete_state_selection",
            )
        )

    return AdaptivePlan(_deduplicate_actions(actions), seeds)
