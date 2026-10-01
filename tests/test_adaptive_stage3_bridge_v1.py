"""Contract tests for adaptive evidence -> existing Stage-3 job planning."""
from copy import deepcopy

from openea_benchmark.adaptive.model import Review, ReviewStatus
from openea_benchmark.adaptive.pec_selection_bridge import bridge_state_selection_report
from openea_benchmark.adaptive.planner import plan_from_state_selection_bridge
from openea_benchmark.adaptive.stage3_bridge import (
    Stage3ReleaseStatus,
    bridge_adaptive_plan_to_stage3,
    build_adaptive_stage3_plan,
)
from openea_benchmark.stage3_plan import build_stage3_plan


def cleared(tag: str) -> Review:
    return Review(ReviewStatus.CLEARED, (tag,), "independently cleared")


def _sector(charge, spin, component, r, energy, root_id, checkpoint="x.chk"):
    return {
        "charge": charge,
        "spin_2s": spin,
        "tracking_mode": "root",
        "sector_status": "dft_scout_complete",
        "pec_scouts": [{
            "component_id": component,
            "pec": {"points": [{"root_id": root_id, "r_angstrom": r, "energy_hartree": energy}]},
            "minimum_scout": {
                "status": "bracketed_single_minimum",
                "candidates": [{"r_angstrom": r, "energy_hartree": energy}],
            },
        }],
        "representative_roots": [{
            "root_id": root_id,
            "r_angstrom": r,
            "checkpoint_path": checkpoint,
            "origin_guess": "g",
            "energy_hartree": energy,
            "s2": 0.75,
            "internal_stable": True,
        }],
    }


def _candidate(charge, spin, component, r, energy, delta):
    return {
        "charge": charge,
        "spin_2s": spin,
        "component_id": component,
        "minimum_status": "bracketed_single_minimum",
        "r_candidate_angstrom": r,
        "energy_hartree": energy,
        "delta_from_lowest_bracketed_ev": delta,
        "sector_bridged_edges": [],
        "sector_discontinuous_edges": [],
    }


def fixture(two_neutral=False):
    n0 = _candidate(0, 1, "component_shared", 1.10, -10.0, 0.0)
    a0 = _candidate(-1, 0, "component_shared", 1.20, -10.2, 0.0)
    neutral = [n0]
    sectors = [
        _sector(0, 1, "component_shared", 1.10, -10.0, "n0", "n0.chk"),
        _sector(-1, 0, "component_shared", 1.20, -10.2, "a0", "a0.chk"),
    ]
    if two_neutral:
        n1 = _candidate(0, 3, "component_excited", 1.30, -9.98, 0.4)
        neutral.append(n1)
        sectors.append(_sector(0, 3, "component_excited", 1.30, -9.98, "n1", "n1.chk"))

    selection = {
        "system": "X",
        "charge_groups": {
            "neutral": {
                "bracketed_candidates": neutral,
                "unresolved_components": [],
                "all_scouted_components_bracketed": True,
                "provisional_high_level_seed": n0,
            },
            "anion": {
                "bracketed_candidates": [a0],
                "unresolved_components": [],
                "all_scouted_components_bracketed": True,
                "provisional_high_level_seed": a0,
            },
        },
    }
    summary = {"system": "X", "sectors": sectors}
    manifest = {
        "systems": {"X": {"atoms": ["X", "Y"]}},
        "protocol": {
            "stage_3_high_level_local_pec": {
                "reference": "ROHF",
                "methods": ["CCSD", "CCSD(T)"],
                "basis": "def2-TZVPPD",
                "relative_grid_angstrom": [-0.05, 0.0, 0.05],
            }
        },
    }
    return summary, selection, manifest


def test_current_charge_groups_contract_and_direct_seed_fields_are_consumed():
    _, selection, _ = fixture()
    bridge = bridge_state_selection_report(selection, evidence_id="selection:X")
    assert bridge.group_count == 2
    assert bridge.all_scouted_components_bracketed is True
    assert len(bridge.candidate_evidence) == 2
    refs = {x.candidate_ref for x in bridge.candidate_evidence}
    # Same component ID in neutral and anion remains unambiguous via group_ref.
    assert refs == {"neutral:component_shared", "anion:component_shared"}
    values = {(x.group_ref, x.sampled_r_angstrom, x.sampled_energy_hartree) for x in bridge.candidate_evidence}
    assert ("neutral", 1.10, -10.0) in values
    assert ("anion", 1.20, -10.2) in values


def test_default_local_bracketing_defers_stage3_behind_asymptote_prerequisite():
    summary, selection, manifest = fixture()
    result = build_adaptive_stage3_plan(
        system_summary=summary,
        selection_report=selection,
        manifest=manifest,
        evidence_id="selection:X",
    )
    assert result.release_status is Stage3ReleaseStatus.DEFERRED_PREREQUISITE
    assert result.released_job_ids == ()
    assert any("RESOLVE_PEC_ASYMPTOTES" in x for x in result.blocking_reasons)
    assert result.stage3_plan["automatic_pruning_performed"] is False


def test_cleared_asymptotes_release_complete_high_accuracy_candidate_set():
    summary, selection, manifest = fixture(two_neutral=True)
    result = build_adaptive_stage3_plan(
        system_summary=summary,
        selection_report=selection,
        manifest=manifest,
        evidence_id="selection:X",
        asymptote_review=cleared("asym:X"),
    )
    assert result.release_status is Stage3ReleaseStatus.RELEASED
    assert set(result.released_job_ids) == {job["job_id"] for job in result.stage3_plan["jobs"]}
    assert len(result.released_job_ids) == 3
    assert result.blocked_job_ids == ()


def test_dft_splitting_orders_but_does_not_prune_jobs():
    summary, selection, manifest = fixture(two_neutral=True)
    # Make the anion sort after both neutral candidates; all must still survive.
    selection["charge_groups"]["anion"]["bracketed_candidates"][0]["delta_from_lowest_bracketed_ev"] = 0.8
    result = build_adaptive_stage3_plan(
        system_summary=summary,
        selection_report=selection,
        manifest=manifest,
        evidence_id="selection:X",
        asymptote_review=cleared("asym:X"),
    )
    assert result.release_status is Stage3ReleaseStatus.RELEASED
    assert len(result.ordered_job_ids) == result.stage3_plan["n_jobs"] == 3
    assert result.ordered_job_ids[0].endswith("component_shared")  # neutral delta 0.0 wins tie/order
    assert any(job_id.endswith("component_excited") for job_id in result.ordered_job_ids)
    assert result.stage3_plan["automatic_pruning_performed"] is False


def test_unresolved_checkpoint_provenance_blocks_entire_comparison():
    summary, selection, manifest = fixture()
    summary["sectors"][1]["representative_roots"][0]["checkpoint_path"] = None
    result = build_adaptive_stage3_plan(
        system_summary=summary,
        selection_report=selection,
        manifest=manifest,
        evidence_id="selection:X",
        asymptote_review=cleared("asym:X"),
    )
    assert result.release_status is Stage3ReleaseStatus.BLOCKED_PROVENANCE
    assert result.released_job_ids == ()
    assert len(result.blocked_job_ids) == 1
    assert "q-1" in result.blocked_job_ids[0]


def test_higher_level_competition_already_cleared_does_not_request_stage3_comparison():
    summary, selection, manifest = fixture()
    result = build_adaptive_stage3_plan(
        system_summary=summary,
        selection_report=selection,
        manifest=manifest,
        evidence_id="selection:X",
        asymptote_review=cleared("asym:X"),
        higher_level_competition_review=cleared("competition:X"),
    )
    assert result.release_status is Stage3ReleaseStatus.NOT_REQUESTED
    assert result.released_job_ids == ()


def test_open_pecs_are_not_released_to_stage3_comparison():
    summary, selection, manifest = fixture()
    selection["charge_groups"]["neutral"]["all_scouted_components_bracketed"] = False
    selection["charge_groups"]["neutral"]["unresolved_components"] = [
        {"component_id": "open_N", "charge": 0, "spin_2s": 1}
    ]
    result = build_adaptive_stage3_plan(
        system_summary=summary,
        selection_report=selection,
        manifest=manifest,
        evidence_id="selection:X",
    )
    assert result.release_status is Stage3ReleaseStatus.NOT_REQUESTED
    assert result.released_job_ids == ()
    assert result.stage3_plan["n_open_components"] == 1


def test_missing_legacy_job_fails_closed_as_contract_mismatch():
    summary, selection, manifest = fixture()
    bridge = bridge_state_selection_report(
        selection, evidence_id="selection:X", asymptote_review=cleared("asym:X")
    )
    adaptive = plan_from_state_selection_bridge(bridge)
    legacy = build_stage3_plan(system_summary=summary, selection_report=selection, manifest=manifest)
    broken = deepcopy(legacy)
    broken["jobs"] = broken["jobs"][:-1]
    broken["n_jobs"] -= 1
    result = bridge_adaptive_plan_to_stage3(
        adaptive_plan=adaptive,
        stage3_plan=broken,
        selection_bridge=bridge,
    )
    assert result.release_status is Stage3ReleaseStatus.BLOCKED_CONTRACT
    assert any("ADAPTIVE_SEED_JOB_MATCH_0" in x for x in result.blocking_reasons)


def test_extra_legacy_job_fails_closed_instead_of_bypassing_adaptive_layer():
    summary, selection, manifest = fixture()
    bridge = bridge_state_selection_report(
        selection, evidence_id="selection:X", asymptote_review=cleared("asym:X")
    )
    adaptive = plan_from_state_selection_bridge(bridge)
    legacy = build_stage3_plan(system_summary=summary, selection_report=selection, manifest=manifest)
    broken = deepcopy(legacy)
    extra = deepcopy(broken["jobs"][0])
    extra["job_id"] = "X__extra"
    extra["component_id"] = "not_in_adaptive_queue"
    broken["jobs"].append(extra)
    broken["n_jobs"] += 1
    result = bridge_adaptive_plan_to_stage3(
        adaptive_plan=adaptive,
        stage3_plan=broken,
        selection_bridge=bridge,
    )
    assert result.release_status is Stage3ReleaseStatus.BLOCKED_CONTRACT
    assert any("LEGACY_JOB_WITHOUT_ADAPTIVE_SEED" in x for x in result.blocking_reasons)


def test_validation_metadata_is_ignored_by_adaptive_stage3_bridge():
    summary, selection, manifest = fixture()
    first = build_adaptive_stage3_plan(
        system_summary=summary,
        selection_report=selection,
        manifest=manifest,
        evidence_id="selection:X",
        asymptote_review=cleared("asym:X"),
    )
    changed = deepcopy(summary)
    changed["validation"] = {"electron_affinity_ev": 999.0}
    second = build_adaptive_stage3_plan(
        system_summary=changed,
        selection_report=selection,
        manifest=manifest,
        evidence_id="selection:X",
        asymptote_review=cleared("asym:X"),
    )
    assert first.release_status == second.release_status
    assert first.ordered_job_ids == second.ordered_job_ids
    assert first.stage3_plan == second.stage3_plan
