"""Tests for deterministic D04/D09 -> adaptive action planning."""

from openea_benchmark.adaptive.model import (
    DiagnosticID,
    DiagnosticRecord,
    Review,
    ReviewStatus,
)
from openea_benchmark.adaptive.pec_selection_bridge import (
    CandidateEnergyEvidence,
    StateSelectionBridgeResult,
)
from openea_benchmark.adaptive.planner import (
    ActionKind,
    build_stage3_seed_queue,
    plan_from_state_selection_bridge,
)


def review(status, tag="e1", rationale="r"):
    return Review(status, (tag,) if status is not ReviewStatus.NOT_APPLICABLE else (), rationale)


def diag(identifier, status, branch=None, refs=()):
    return DiagnosticRecord(
        identifier,
        review(status, identifier.value, f"{identifier.value}-{status.value}"),
        state_refs=tuple(refs),
        recommended_branch=branch,
    )


def candidate(ref, delta, r=1.2, e=-75.0):
    group, component = ref.split(":", 1)
    return CandidateEnergyEvidence(
        candidate_ref=ref,
        group_ref=group,
        component_id=component,
        minimum_status="bracketed_single_minimum",
        sampled_r_angstrom=r,
        sampled_energy_hartree=e,
        delta_from_lowest_bracketed_ev=delta,
        production_interval_ev=None,
        evidence_ids=("sel:" + ref,),
        rationale="scout only",
    )


def bridge(d04, d09, candidates=()):
    return StateSelectionBridgeResult(
        tuple(candidates), d04, d09, 1, True,
    )


def test_seed_queue_keeps_every_candidate_and_uses_scout_delta_only_for_ordering():
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.UNRESOLVED),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.UNRESOLVED),
        [candidate("g:c2", 0.3), candidate("g:c0", 0.0), candidate("g:c1", 0.1)],
    )
    q = build_stage3_seed_queue(b)
    assert [x.candidate_ref for x in q.ready] == ["g:c0", "g:c1", "g:c2"]
    assert all(x.authorizes_pruning is False for x in q.ready)
    assert q.blocked_candidate_refs == ()


def test_missing_scout_delta_does_not_drop_candidate():
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.UNRESOLVED),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.UNRESOLVED),
        [candidate("g:c1", None), candidate("g:c0", 0.0)],
    )
    q = build_stage3_seed_queue(b)
    assert [x.candidate_ref for x in q.ready] == ["g:c0", "g:c1"]


def test_missing_seed_geometry_is_blocked_not_silently_skipped():
    bad = candidate("g:c0", 0.0, r=None)
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.UNRESOLVED),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.UNRESOLVED),
        [bad],
    )
    q = build_stage3_seed_queue(b)
    assert q.ready == ()
    assert q.blocked_candidate_refs == ("g:c0",)


def test_open_pec_routes_to_extension():
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.CONFIRMED, "extend_open_candidate_pecs"),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.CONFIRMED, "extend_open_pec_components"),
        [candidate("g:c0", 0.0)],
    )
    p = plan_from_state_selection_bridge(b)
    assert p.next_action.kind is ActionKind.EXTEND_OPEN_PEC
    assert any(x.diagnostic_id is DiagnosticID.D09_PEC_ASYMPTOTES for x in p.actions)
    assert all(x.authorizes_pruning is False for x in p.actions)


def test_asymptote_resolution_precedes_high_accuracy_comparison():
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.UNRESOLVED, "high_accuracy_candidate_comparison"),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.UNRESOLVED, "resolve_pec_asymptotes"),
        [candidate("g:c0", 0.0), candidate("g:c1", 0.2)],
    )
    p = plan_from_state_selection_bridge(b)
    assert [a.kind for a in p.actions][:2] == [
        ActionKind.RESOLVE_PEC_ASYMPTOTES,
        ActionKind.HIGH_ACCURACY_CANDIDATE_COMPARISON,
    ]


def test_high_accuracy_comparison_targets_all_usable_candidates():
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.UNRESOLVED, "high_accuracy_candidate_comparison"),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.CLEARED),
        [candidate("g:c2", 0.5), candidate("g:c0", 0.0), candidate("g:c1", 0.2)],
    )
    p = plan_from_state_selection_bridge(b)
    action = p.next_action
    assert action.kind is ActionKind.HIGH_ACCURACY_CANDIDATE_COMPARISON
    assert action.target_refs == ("g:c0", "g:c1", "g:c2")


def test_cleared_d04_and_d09_produce_no_actions():
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.CLEARED),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.CLEARED),
        [candidate("g:c0", 0.0)],
    )
    p = plan_from_state_selection_bridge(b)
    assert p.actions == ()
    assert len(p.stage3_seeds.ready) == 1


def test_no_seed_converts_high_accuracy_request_to_completion_action():
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.UNRESOLVED, "high_accuracy_candidate_comparison"),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.CLEARED),
        [],
    )
    p = plan_from_state_selection_bridge(b)
    assert len(p.actions) == 1
    assert p.actions[0].kind is ActionKind.COMPLETE_STATE_SELECTION


def test_blocked_seed_forces_audit_before_other_work():
    bad = candidate("g:c0", 0.0, e=None)
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.UNRESOLVED, "high_accuracy_candidate_comparison"),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.UNRESOLVED, "resolve_pec_asymptotes"),
        [bad],
    )
    p = plan_from_state_selection_bridge(b)
    assert p.next_action.kind is ActionKind.AUDIT_STATE_SELECTION
    assert p.next_action.target_refs == ("g:c0",)


def test_state_continuity_route_is_preserved():
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.CONFIRMED, "resolve_state_continuity", ("s1",)),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.CLEARED),
        [candidate("g:c0", 0.0)],
    )
    p = plan_from_state_selection_bridge(b)
    assert p.next_action.kind is ActionKind.RESOLVE_STATE_CONTINUITY
    assert p.next_action.target_refs == ("s1",)


def test_empty_state_selection_route_is_preserved():
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.UNRESOLVED, "complete_state_selection"),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.UNRESOLVED, "construct_relevant_pecs"),
        [],
    )
    p = plan_from_state_selection_bridge(b)
    kinds = {x.kind for x in p.actions}
    assert ActionKind.CONSTRUCT_RELEVANT_PECS in kinds
    assert ActionKind.COMPLETE_STATE_SELECTION in kinds


def test_unknown_route_fails_to_generic_investigation_not_to_clearance():
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.CONFIRMED, "future_route"),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.CLEARED),
        [candidate("g:c0", 0.0)],
    )
    p = plan_from_state_selection_bridge(b)
    assert p.next_action.kind is ActionKind.INVESTIGATE_DIAGNOSTIC


def test_scout_evidence_never_generates_production_intervals_or_pruning_permission():
    c = candidate("g:c0", 0.0)
    b = bridge(
        diag(DiagnosticID.D04_STATE_COMPETITION, ReviewStatus.UNRESOLVED, "high_accuracy_candidate_comparison"),
        diag(DiagnosticID.D09_PEC_ASYMPTOTES, ReviewStatus.CLEARED),
        [c],
    )
    p = plan_from_state_selection_bridge(b)
    assert c.production_interval_ev is None
    assert not p.stage3_seeds.ready[0].authorizes_pruning
    assert not p.next_action.authorizes_pruning
