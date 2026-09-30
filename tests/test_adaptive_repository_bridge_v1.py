"""Contract tests for the read-only repository -> adaptive bridge."""
from types import SimpleNamespace

import pytest

from openea_benchmark.adaptive.model import (
    GateSet,
    Review,
    ReviewStatus,
)
from openea_benchmark.adaptive.repository_bridge import (
    SpinWarningPolicy,
    branch_diagnostic_from_graph,
    bridge_discovery_records,
    gates_with_state_completeness,
    spin_diagnostic_from_root,
    state_completeness_gate,
)


def cleared(tag: str) -> Review:
    return Review(ReviewStatus.CLEARED, (tag,))


def unresolved(tag: str = "") -> Review:
    refs = (tag,) if tag else ()
    return Review(ReviewStatus.UNRESOLVED, refs)


def component(*, unambiguous: bool = True):
    return SimpleNamespace(is_unambiguous=unambiguous)


def graph(*, ambiguous=False):
    return SimpleNamespace(
        root_ids_by_geometry=((1.0, ("r1",)), (1.1, ("r2",))),
        is_fully_unambiguous=not ambiguous,
        ambiguous_edges=(("r1", "r2"),) if ambiguous else (),
        topology_ambiguous_root_ids=("r1", "r2") if ambiguous else (),
        components=(component(unambiguous=not ambiguous),),
    )


def root(*, s2=0.75, status="CANONICALIZED", internal=True, external=None):
    # spin_2s=1 -> S=1/2 -> expected <S^2>=0.75
    return SimpleNamespace(
        root_id="root-1",
        status=status,
        internal_stable=internal,
        external_stable=external,
        s2=s2,
        spin_2s=1,
        expected_s2=0.75,
    )


def test_spin_policy_has_no_implicit_default():
    with pytest.raises(TypeError):
        SpinWarningPolicy()


def test_low_spin_contamination_does_not_clear_d01():
    result = spin_diagnostic_from_root(
        root(s2=0.75001),
        policy=SpinWarningPolicy(0.01),
    )
    assert result.review.status is ReviewStatus.UNRESOLVED


def test_large_spin_contamination_confirms_warning_only():
    result = spin_diagnostic_from_root(
        root(s2=0.9),
        policy=SpinWarningPolicy(0.01),
    )
    assert result.review.status is ReviewStatus.CONFIRMED
    assert "does not by itself select" in result.review.rationale


def test_unambiguous_graph_does_not_clear_state_competition_by_itself():
    result = branch_diagnostic_from_graph(graph(), evidence_id="graph-1")
    assert result.review.status is ReviewStatus.UNRESOLVED


def test_ambiguous_graph_confirms_continuity_problem():
    result = branch_diagnostic_from_graph(graph(ambiguous=True), evidence_id="graph-1")
    assert result.review.status is ReviewStatus.CONFIRMED
    assert result.recommended_branch == "resolve_state_continuity"


def test_graph_can_clear_d04_only_with_higher_level_competition_review():
    result = branch_diagnostic_from_graph(
        graph(),
        evidence_id="graph-1",
        candidate_competition_review=cleared("competition-review"),
    )
    assert result.review.status is ReviewStatus.CLEARED
    assert set(result.review.evidence_ids) == {"graph-1", "competition-review"}


def test_g1_requires_search_competition_and_pec_asymptotes():
    g1 = state_completeness_gate(
        state_search_review=cleared("search"),
        state_competition_review=cleared("competition"),
        pec_asymptote_review=unresolved("pec"),
    )
    assert g1.status is ReviewStatus.UNRESOLVED
    assert "PEC_ASYMPTOTES=UNRESOLVED" in g1.rationale


def test_g1_clears_only_when_all_three_are_cleared():
    g1 = state_completeness_gate(
        state_search_review=cleared("search"),
        state_competition_review=cleared("competition"),
        pec_asymptote_review=cleared("pec"),
    )
    assert g1.status is ReviewStatus.CLEARED
    assert set(g1.evidence_ids) == {"search", "competition", "pec"}


def test_bridge_preserves_internal_only_stability_as_unresolved():
    bundle = bridge_discovery_records(
        (root(external=None),),
        (("graph-1", graph()),),
        spin_policy=SpinWarningPolicy(0.01),
        state_search_review=cleared("search"),
        candidate_competition_review=cleared("competition"),
        pec_asymptote_review=cleared("pec"),
    )
    # D01 then D02
    assert bundle.root_diagnostics[0].review.status is ReviewStatus.UNRESOLVED
    assert bundle.root_diagnostics[1].review.status is ReviewStatus.UNRESOLVED
    assert bundle.state_completeness_gate.status is ReviewStatus.CLEARED


def test_bridge_does_not_use_local_graph_to_manufacture_g1():
    bundle = bridge_discovery_records(
        (root(),),
        (("graph-1", graph()),),
        spin_policy=SpinWarningPolicy(0.01),
        state_search_review=unresolved("search"),
        candidate_competition_review=cleared("competition"),
        pec_asymptote_review=cleared("pec"),
    )
    assert bundle.branch_diagnostics[0].review.status is ReviewStatus.CLEARED
    assert bundle.state_completeness_gate.status is ReviewStatus.UNRESOLVED


def test_gate_replacement_preserves_g2_g3():
    base = GateSet(
        state_completeness=unresolved(),
        attachment_resolution=cleared("g2"),
        energy_reliability=cleared("g3"),
    )
    updated = gates_with_state_completeness(base, cleared("g1"))
    assert updated.state_completeness.status is ReviewStatus.CLEARED
    assert updated.attachment_resolution is base.attachment_resolution
    assert updated.energy_reliability is base.energy_reliability
