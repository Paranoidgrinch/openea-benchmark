"""Tests for conservative state-selection/PEC -> adaptive evidence bridging."""
import pytest

from openea_benchmark.adaptive.model import Interval, Review, ReviewStatus
from openea_benchmark.adaptive.pec_selection_bridge import (
    CandidateEnergyEvidence,
    bridge_state_selection_report,
    g1_from_state_selection_bridge,
)


def cleared(tag: str) -> Review:
    return Review(ReviewStatus.CLEARED, (tag,))


def unresolved(tag: str) -> Review:
    return Review(ReviewStatus.UNRESOLVED, (tag,))


def candidate(component_id="c0", energy=-75.0, r=1.1, delta=0.0):
    return {
        "component_id": component_id,
        "minimum_status": "bracketed_single_minimum",
        "delta_from_lowest_bracketed_ev": delta,
        "scout": {
            "minimum_scout": {
                "status": "bracketed_single_minimum",
                "candidates": [
                    {"r_angstrom": r, "energy_hartree": energy},
                ],
            }
        },
    }


def report(*, all_bracketed=True, candidates=None):
    if candidates is None:
        candidates = [candidate()]
    return {
        "groups": [
            {
                "group_id": "q0-s1",
                "all_scouted_components_bracketed": all_bracketed,
                "bracketed_candidates": candidates,
            }
        ]
    }


def test_bracketed_scout_energy_never_becomes_production_interval():
    result = bridge_state_selection_report(report(), evidence_id="selection-1")
    item = result.candidate_evidence[0]
    assert item.sampled_energy_hartree == pytest.approx(-75.0)
    assert item.production_interval_ev is None
    assert result.production_intervals == ()


def test_candidate_preserves_delta_only_as_scout_evidence():
    result = bridge_state_selection_report(
        report(candidates=[candidate("a", delta=0.0), candidate("b", -74.99, 1.2, 0.27)]),
        evidence_id="selection-1",
    )
    assert result.candidate_evidence[1].delta_from_lowest_bracketed_ev == pytest.approx(0.27)
    assert result.candidate_evidence[1].production_interval_ev is None


def test_all_local_minima_bracketed_does_not_clear_d04_without_high_level_review():
    result = bridge_state_selection_report(report(), evidence_id="selection-1")
    assert result.state_competition.review.status is ReviewStatus.UNRESOLVED
    assert result.state_competition.recommended_branch == "high_accuracy_candidate_comparison"


def test_all_local_minima_bracketed_does_not_clear_d09_without_asymptote_review():
    result = bridge_state_selection_report(report(), evidence_id="selection-1")
    assert result.pec_asymptotes.review.status is ReviewStatus.UNRESOLVED
    assert result.pec_asymptotes.recommended_branch == "resolve_pec_asymptotes"


def test_open_component_confirms_d04_and_d09_problem():
    result = bridge_state_selection_report(
        report(all_bracketed=False), evidence_id="selection-1"
    )
    assert result.state_competition.review.status is ReviewStatus.CONFIRMED
    assert result.pec_asymptotes.review.status is ReviewStatus.CONFIRMED
    assert result.state_competition.recommended_branch == "extend_open_candidate_pecs"
    assert result.pec_asymptotes.recommended_branch == "extend_open_pec_components"


def test_external_reviews_can_clear_d04_and_d09_only_after_local_bracketing():
    result = bridge_state_selection_report(
        report(),
        evidence_id="selection-1",
        higher_level_competition_review=cleared("cc-state-comparison"),
        asymptote_review=cleared("large-r-review"),
    )
    assert result.state_competition.review.status is ReviewStatus.CLEARED
    assert result.pec_asymptotes.review.status is ReviewStatus.CLEARED
    assert "cc-state-comparison" in result.state_competition.review.evidence_ids
    assert "large-r-review" in result.pec_asymptotes.review.evidence_ids


def test_external_clearance_cannot_override_open_local_pecs():
    result = bridge_state_selection_report(
        report(all_bracketed=False),
        evidence_id="selection-1",
        higher_level_competition_review=cleared("cc-state-comparison"),
        asymptote_review=cleared("large-r-review"),
    )
    assert result.state_competition.review.status is ReviewStatus.CONFIRMED
    assert result.pec_asymptotes.review.status is ReviewStatus.CONFIRMED


def test_empty_report_is_fail_closed():
    result = bridge_state_selection_report({"groups": []}, evidence_id="selection-1")
    assert result.candidate_evidence == ()
    assert result.state_competition.review.status is ReviewStatus.UNRESOLVED
    assert result.pec_asymptotes.review.status is ReviewStatus.UNRESOLVED


def test_inconsistent_bracketed_label_triggers_audit():
    bad = candidate()
    bad["minimum_status"] = "multiple_minimum_candidates"
    result = bridge_state_selection_report(
        report(candidates=[bad]), evidence_id="selection-1"
    )
    assert result.pec_asymptotes.review.status is ReviewStatus.CONFIRMED
    assert result.pec_asymptotes.recommended_branch == "audit_state_selection_report"


def test_g1_requires_search_d04_and_d09_clearance():
    bridge = bridge_state_selection_report(
        report(),
        evidence_id="selection-1",
        higher_level_competition_review=cleared("cc-state-comparison"),
        asymptote_review=cleared("large-r-review"),
    )
    g1 = g1_from_state_selection_bridge(
        state_search_review=unresolved("search"), bridge=bridge
    )
    assert g1.status is ReviewStatus.UNRESOLVED
    assert "STATE_SEARCH=UNRESOLVED" in g1.rationale


def test_g1_can_clear_when_all_three_independent_requirements_clear():
    bridge = bridge_state_selection_report(
        report(),
        evidence_id="selection-1",
        higher_level_competition_review=cleared("cc-state-comparison"),
        asymptote_review=cleared("large-r-review"),
    )
    g1 = g1_from_state_selection_bridge(
        state_search_review=cleared("search"), bridge=bridge
    )
    assert g1.status is ReviewStatus.CLEARED
    assert set(g1.evidence_ids) == {
        "search", "selection-1", "cc-state-comparison", "large-r-review"
    }


def test_direct_group_sequence_contract_is_supported():
    groups = report()["groups"]
    result = bridge_state_selection_report(groups, evidence_id="selection-1")
    assert result.group_count == 1


def test_candidate_dataclass_rejects_nonfinite_scout_energy():
    with pytest.raises(ValueError):
        CandidateEnergyEvidence(
            candidate_ref="x",
            group_ref="g",
            component_id="c",
            minimum_status="bracketed_single_minimum",
            sampled_r_angstrom=1.0,
            sampled_energy_hartree=float("nan"),
            delta_from_lowest_bracketed_ev=0.0,
            production_interval_ev=None,
            evidence_ids=("e",),
            rationale="test",
        )


def test_separately_supplied_production_interval_is_not_created_by_bridge():
    # Contract-level reminder: Interval is allowed in the evidence type, but
    # this bridge intentionally never creates one from scout data.
    standalone = CandidateEnergyEvidence(
        candidate_ref="x",
        group_ref="g",
        component_id="c",
        minimum_status="validated_high_accuracy",
        sampled_r_angstrom=None,
        sampled_energy_hartree=None,
        delta_from_lowest_bracketed_ev=None,
        production_interval_ev=Interval(-0.01, 0.01),
        evidence_ids=("high-level",),
        rationale="supplied by a later validated uncertainty layer",
    )
    assert standalone.production_interval_ev == Interval(-0.01, 0.01)
