"""D09 EXTEND_OPEN_PEC executor: continuation, asymptotes, merge and review."""
from pathlib import Path

import pytest

from openea_benchmark.adaptive.model import ReviewStatus
from openea_benchmark.benchmark_auto import AcquisitionSettings, build_benchmark_acquisition_plan, run_benchmark_acquisition
from openea_benchmark.open_pec_extension import (
    DISSOCIATIVE_STATUS, OpenPECExtensionPolicy, apply_open_pec_extension, asymptote_review_from_extension,
    atom_key, classify_continuation, compatible_asymptote, continue_component, run_open_pec_extension,
)
from openea_benchmark.state_selection import build_state_selection_report

from test_benchmark_auto_v1 import fixture

POLICY = OpenPECExtensionPolicy()


def _summary(tmp_path):
    summary, manifest = fixture(tmp_path, open_highspin=True)
    summary["atoms"] = ["H", "H"]
    # the fixture's open sector must really be open: monotone decrease to higher R
    for i, p in enumerate(summary["sectors"][2]["manifold_pec_scouts"][0]["pec"]["points"]):
        p["energy_center_hartree"] = -9.0 - 0.05 * i
    for sector in summary["sectors"]:
        sector["method"] = {"functional": "PBE0", "basis": "sto-3g"}
        root = sector["representative_roots"][0]
        # give every scout point a unique manifold so edges link to a root
        sector["manifolds"] = [
            {"manifold_id": p["manifold_id"], "r_angstrom": p["r_angstrom"],
             "member_root_ids": [f"{root['root_id']}@{p['r_angstrom']}"],
             "unique_state_root_ids": [f"{root['root_id']}@{p['r_angstrom']}"]}
            for p in sector["manifold_pec_scouts"][0]["pec"]["points"]]
        sector["representative_roots"] = [
            {**root, "root_id": f"{root['root_id']}@{p['r_angstrom']}", "r_angstrom": p["r_angstrom"]}
            for p in sector["manifold_pec_scouts"][0]["pec"]["points"]]
    return summary, manifest


def test_compatible_asymptote_respects_spin_coupling():
    atomic = {atom_key("H", 0, 1): {"energy_hartree": -0.5}, atom_key("H", -1, 0): {"energy_hartree": -0.52}}
    assert compatible_asymptote(atomic, "H", "H", 0, 0)["energy_hartree"] == pytest.approx(-1.0)
    assert compatible_asymptote(atomic, "H", "H", 0, 2)["energy_hartree"] == pytest.approx(-1.0)
    assert compatible_asymptote(atomic, "H", "H", 0, 4) is None
    assert compatible_asymptote(atomic, "H", "H", -1, 1)["energy_hartree"] == pytest.approx(-1.02)


def _point(r, e, s2=0.75):
    return {"r_angstrom": r, "energy_hartree": e, "converged": True, "internal_stable": True,
            "s2": s2, "checkpoint_path": f"p{r}.chk"}


def test_continuation_finds_minimum_and_stops():
    energies = {1.25: -1.05, 1.35: -1.08, 1.45: -1.07}
    scan, end = continue_component(edge=_point(1.2, -1.0), direction=+1, policy=POLICY,
                                   compute=lambda r, seed: _point(r, energies[round(r, 2)]))
    assert end == "MINIMUM_BRACKETED"
    result = classify_continuation(scan, direction=1, asymptote=None, policy=POLICY, terminated=end)
    assert result["classification"] == "MINIMUM_BRACKETED"
    assert result["lowest_sampled_r_angstrom"] == pytest.approx(1.35)


def test_plateau_on_asymptote_is_dissociative_but_mismatch_is_not():
    scan = [_point(2.0, -0.90), _point(2.6, -0.9995), _point(3.1, -0.99995)]
    ok = classify_continuation(scan, direction=1, asymptote={"energy_hartree": -1.0, "channel": "x"},
                               policy=POLICY, terminated="PLATEAU")
    assert ok["classification"] == "DISSOCIATIVE_TO_ASYMPTOTE"
    bad = classify_continuation(scan, direction=1, asymptote={"energy_hartree": -1.1, "channel": "x"},
                                policy=POLICY, terminated="PLATEAU")
    assert bad["classification"] == "UNRESOLVED"


def test_spin_jump_blocks_classification():
    scan = [_point(1.0, -1.0, 0.75), _point(1.1, -1.1, 1.9), _point(1.2, -1.05, 1.9)]
    result = classify_continuation(scan, direction=1, asymptote=None, policy=POLICY,
                                   terminated="MINIMUM_BRACKETED")
    assert result["classification"] == "UNRESOLVED"
    assert "S2_JUMP_POSSIBLE_STATE_SWITCH" in result["reasons"]


def _fake_backends(tmp_path, shape):
    def atom_energy(symbol, charge, spin_2s, **kw):
        return {"energy_hartree": {0: -4.6, -1: -4.7}[charge] - 0.001 * spin_2s, "guess": "fake", "s2": 0.0}

    def point(*, r_angstrom, checkpoint, **kw):
        Path(checkpoint).write_bytes(b"chk")
        return {"r_angstrom": r_angstrom, "energy_hartree": shape(r_angstrom), "converged": True,
                "internal_stable": True, "s2": 3.75, "checkpoint_path": checkpoint}
    return atom_energy, point


def test_executor_merges_new_minimum_into_seed_candidates(tmp_path):
    summary, manifest = _summary(tmp_path)
    summary["atoms"] = ["O", "H"]  # fragments able to couple to every synthetic sector spin
    atom_energy, point = _fake_backends(tmp_path, lambda r: -9.2 + 0.5 * (r - 1.5) ** 2)
    ext = run_open_pec_extension(summary=summary, output_dir=tmp_path / "sys", policy=POLICY,
                                 atom_energy=atom_energy, continuation_point=point)
    (component,) = ext["components"].values()
    assert component["classification"] == "MINIMUM_BRACKETED"
    merged = apply_open_pec_extension(summary, ext)
    selection = build_state_selection_report(merged)
    neutral = selection["charge_groups"]["neutral"]
    assert not neutral["unresolved_components"]
    assert {c["spin_2s"] for c in neutral["bracketed_candidates"]} == {1, 3}
    # the original summary is untouched
    assert summary["sectors"][2]["manifold_pec_scouts"][0]["minimum_scout"]["status"] == "decreases_toward_higher_r"
    review = asymptote_review_from_extension(merged, ext, selection)
    assert review.status is ReviewStatus.CLEARED
    assert "AUTOMATED_SCOUT_LEVEL" in review.rationale
    # The merged candidate is a real Stage-3 job with checkpoint provenance.
    _, bridge = build_benchmark_acquisition_plan(summary=summary, manifest=manifest, open_pec_extension=ext)
    assert len(bridge.stage3_plan["jobs"]) == 3
    assert bridge.release_status.value == "RELEASED"


def test_executor_resumes_without_recomputing(tmp_path):
    summary, _ = _summary(tmp_path)
    calls = []
    atom_energy, point = _fake_backends(tmp_path, lambda r: -9.2 + 0.5 * (r - 1.5) ** 2)

    def counting(**kw):
        calls.append(kw["r_angstrom"])
        return point(**kw)
    run_open_pec_extension(summary=summary, output_dir=tmp_path / "sys", policy=POLICY,
                           atom_energy=atom_energy, continuation_point=counting)
    n = len(calls)
    run_open_pec_extension(summary=summary, output_dir=tmp_path / "sys", policy=POLICY,
                           atom_energy=lambda *a, **k: pytest.fail("atoms recomputed"),
                           continuation_point=counting)
    assert len(calls) == n


def test_dissociative_component_resolved_but_never_a_seed(tmp_path):
    summary, _ = _summary(tmp_path)
    # a quartet needs fragments that can couple to 2S=3: O(2S=2) + H(2S=1)
    summary["atoms"] = ["O", "H"]
    _, point = _fake_backends(tmp_path, lambda r: -9.202 + 0.2 * 2.718 ** (-3 * r))
    lowest = {("O", 0, 2): -4.601, ("H", 0, 1): -4.601}
    atom_energy_low = lambda symbol, charge, spin_2s, **kw: {
        "energy_hartree": lowest.get((symbol, charge, spin_2s), -4.0), "guess": "f", "s2": 0.0}
    ext = run_open_pec_extension(summary=summary, output_dir=tmp_path / "sys", policy=POLICY,
                                 atom_energy=atom_energy_low, continuation_point=point)
    (component,) = ext["components"].values()
    assert component["classification"] == "DISSOCIATIVE_TO_ASYMPTOTE", component
    merged = apply_open_pec_extension(summary, ext)
    selection = build_state_selection_report(merged)
    neutral = selection["charge_groups"]["neutral"]
    assert not neutral["unresolved_components"]
    assert [c["minimum_status"] for c in neutral["asymptote_resolved_components"]] == [DISSOCIATIVE_STATUS]
    assert {c["spin_2s"] for c in neutral["bracketed_candidates"]} == {1}


def test_unresolved_extension_keeps_d09_open(tmp_path):
    summary, manifest = _summary(tmp_path)
    atom_energy, point = _fake_backends(tmp_path, lambda r: -9.0 - 0.3 * r)  # never flattens
    ext = run_open_pec_extension(summary=summary, output_dir=tmp_path / "sys", policy=POLICY,
                                 atom_energy=atom_energy, continuation_point=point)
    (component,) = ext["components"].values()
    assert component["classification"] == "UNRESOLVED"
    merged = apply_open_pec_extension(summary, ext)
    selection = build_state_selection_report(merged)
    assert selection["charge_groups"]["neutral"]["unresolved_components"]
    assert asymptote_review_from_extension(merged, ext, selection).status is ReviewStatus.UNRESOLVED
    state = run_benchmark_acquisition(summary=summary, manifest=manifest, output_dir=tmp_path / "out",
                                      settings=AcquisitionSettings(compute_points=False), open_pec_extension=ext)
    assert state["open_pec_extension_applied"] is True
    assert state["stage3_release"] != "RELEASED"
