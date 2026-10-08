"""Acquisition controller contract: autonomous but never scientifically permissive."""
import json
from pathlib import Path

import pytest

from openea_benchmark.benchmark_auto import (
    AcquisitionSettings, build_benchmark_acquisition_plan, run_benchmark_acquisition,
)
from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus, _empty_result,
)
from openea_benchmark.workflow import build_parser


def fixture(tmp_path, open_highspin=True):
    source = str(tmp_path / "source.chk")
    Path(source).write_bytes(b"test checkpoint stand-in")

    def sector(charge, spin, component, emin, root, bracketed=True):
        p = [{"r_angstrom": r, "energy_center_hartree": e, "manifold_id": f"{component}_{i}"}
             for i, (r, e) in enumerate(((1.0, emin + .05), (1.1, emin), (1.2, emin + .05)))]
        return {
            "charge": charge, "spin_2s": spin, "sector_status": "dft_scout_complete",
            "tracking_mode": "manifold", "manifold_branch_graph_unambiguous": True,
            "manifold_branch_graph": {"discontinuous_edges": [], "bridged_edges": []},
            "adaptive_grid": {"stop_reason": "MINIMUM_BRACKETED" if bracketed else "EXTENSION_LIMIT_REACHED"},
            "manifold_pec_scouts": [{"component_id": component, "pec": {"points": p},
                "minimum_scout": {"status": "bracketed_single_minimum" if bracketed else "decreases_toward_higher_r",
                    "candidates": [{"r_angstrom": 1.1, "energy_hartree": emin}] if bracketed else []}}],
            "manifolds": [{"manifold_id": f"{component}_1", "r_angstrom": 1.1,
                            "member_root_ids": [root], "unique_state_root_ids": [root]}],
            "representative_roots": [{"root_id": root, "checkpoint_path": source,
                                      "r_angstrom": 1.1, "energy_hartree": emin, "origin_guess": "seed"}],
        }

    summary = {
        "system": "HX", "validation_metadata_used_in_computation": False,
        "sectors": [sector(0, 1, "N", -10.0, "n"), sector(-1, 0, "A", -10.4, "a")],
    }
    if open_highspin:
        summary["sectors"].append(sector(0, 3, "N3", -9.0, "n3", False))
    manifest = {"systems": {"HX": {"atoms": ["H", "H"]}},
                "protocol": {"stage_3_high_level_local_pec": {
                    "reference": "ROHF", "methods": ["CCSD", "CCSD(T)"],
                    "basis": "sto-3g", "relative_grid_angstrom": [-0.05, 0, 0.05],
                }}}
    return summary, manifest


def test_cli_exposes_one_command_mode():
    args = build_parser().parse_args(["m.yaml", "--mode", "benchmark-auto", "--system", "HX"])
    assert args.mode == "benchmark-auto"
    assert args.max_stage3_points == 100


def test_automatic_selection_and_plan_detect_open_highspin(tmp_path):
    summary, manifest = fixture(tmp_path)
    selection, bridge = build_benchmark_acquisition_plan(summary=summary, manifest=manifest)
    assert len(selection["charge_groups"]["neutral"]["unresolved_components"]) == 1
    assert bridge.release_status.value == "DEFERRED_PREREQUISITE" or bridge.release_status.value == "NOT_REQUESTED"
    assert len(bridge.stage3_plan["jobs"]) == 2


def test_dry_run_queues_all_bracketed_seeds_without_pruning(tmp_path):
    summary, manifest = fixture(tmp_path)
    state = run_benchmark_acquisition(
        summary=summary, manifest=manifest, output_dir=tmp_path / "output",
        settings=AcquisitionSettings(compute_points=False),
    )
    assert state["outcome"] == "UNRESOLVED"
    assert state["candidate_job_count"] == 2
    assert state["point_request_count"] == 6
    assert state["pending_point_count"] == 6
    assert state["charge_groups"]["neutral"]["open_components"] == 1
    assert not state["automatic_pruning_performed"]
    assert not state["ea_computed"]
    assert state["charge_groups"]["neutral"]["open_diagnostics"][0]["observed_gap_to_bracketed_seed_ev_not_lower_bound"] > 0
    assert state["stage3_purpose"] == "PROVISIONAL_EVIDENCE_ONLY_NOT_SCIENTIFIC_RELEASE"


def test_no_provisional_blocks_points_when_prerequisites_open(tmp_path):
    summary, manifest = fixture(tmp_path)
    state = run_benchmark_acquisition(
        summary=summary, manifest=manifest, output_dir=tmp_path / "output",
        settings=AcquisitionSettings(enable_provisional_points=False),
    )
    assert state["point_request_count"] == 0
    assert not state["ea_computed"]


def test_bounded_execution_and_resume_preserve_result_without_rerun(tmp_path):
    summary, manifest = fixture(tmp_path)
    count = []
    def fake_runner(req, settings):
        count.append(req.request_id)
        return _empty_result(req, PointExecutionStatus.ERROR, error_type="MockError", error_message="simulated")
    path = tmp_path / "output"
    one = run_benchmark_acquisition(
        summary=summary, manifest=manifest, output_dir=path,
        settings=AcquisitionSettings(max_points=2), point_runner=fake_runner)
    assert one["failed_point_count"] == 2
    assert one["pending_point_count"] == 4
    two = run_benchmark_acquisition(
        summary=summary, manifest=manifest, output_dir=path,
        settings=AcquisitionSettings(max_points=10), point_runner=fake_runner)
    assert two["failed_point_count"] == 6
    assert two["pending_point_count"] == 0
    assert len(count) == 12  # canonical one-retry policy for ERROR
    assert len(list((path / "points").glob("*.json"))) == 6
    assert all(json.loads(p.read_text())["scientific_release"] is False for p in (path / "points").glob("*.json"))


def test_missing_source_checkpoints_are_blocked_not_executed(tmp_path):
    summary, manifest = fixture(tmp_path)
    source = Path(summary["sectors"][0]["representative_roots"][0]["checkpoint_path"])
    source.unlink()
    state = run_benchmark_acquisition(summary=summary, manifest=manifest,
                                      output_dir=tmp_path / "output")
    assert state["point_request_count"] == 0
    assert len(state["blocked_jobs"]) == 2
    assert state["outcome"] == "UNRESOLVED"


def test_changed_manifest_refuses_cached_run(tmp_path):
    summary, manifest = fixture(tmp_path)
    path = tmp_path / "output"
    run_benchmark_acquisition(summary=summary, manifest=manifest, output_dir=path,
                              settings=AcquisitionSettings(compute_points=False))
    manifest["protocol"]["stage_3_high_level_local_pec"]["basis"] = "cc-pvdz"
    with pytest.raises(ValueError, match="different manifest"):
        run_benchmark_acquisition(summary=summary, manifest=manifest, output_dir=path,
                                  settings=AcquisitionSettings(compute_points=False))


def test_rejects_unblinded_summary(tmp_path):
    summary, manifest = fixture(tmp_path)
    summary["validation_metadata_used_in_computation"] = True
    with pytest.raises(ValueError, match="validation isolation"):
        run_benchmark_acquisition(summary=summary, manifest=manifest, output_dir=tmp_path / "output")


def test_cli_uses_previous_scout_and_writes_plan(monkeypatch, tmp_path):
    import openea_benchmark.workflow as workflow
    summary, manifest = fixture(tmp_path)
    manifest["benchmark"] = {"name": "test_benchmark"}
    root = tmp_path / "run"
    (root / "HX").mkdir(parents=True)
    (root / "HX" / "summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(workflow, "load_manifest", lambda _path: manifest)
    monkeypatch.setattr(workflow, "run_system_dft_scout", lambda **kwargs: pytest.fail("Unexpected DFT rescout"))
    rc = workflow.main(["dummy.yaml", "--mode", "benchmark-auto", "--system", "HX",
                        "--output", str(root), "--auto-dry-run"])
    assert rc == 0
    state = json.loads((root / "HX" / "benchmark_auto" / "auto_state.json").read_text())
    assert state["point_request_count"] == 6
    assert state["outcome"] == "UNRESOLVED"


def test_source_checkpoint_change_invalidates_point_cache(tmp_path):
    summary, manifest = fixture(tmp_path)
    path = tmp_path / "output"
    def fake(req, settings):
        return _empty_result(req, PointExecutionStatus.ERROR, error_type="ValueError", error_message="test")
    run_benchmark_acquisition(summary=summary, manifest=manifest, output_dir=path,
                              settings=AcquisitionSettings(max_points=1), point_runner=fake)
    Path(summary["sectors"][0]["representative_roots"][0]["checkpoint_path"]).write_bytes(b"modified original checkpoint")
    with pytest.raises(ValueError, match="Stale/conflicting point artifact"):
        run_benchmark_acquisition(summary=summary, manifest=manifest, output_dir=path,
                                  settings=AcquisitionSettings(max_points=1), point_runner=fake)
