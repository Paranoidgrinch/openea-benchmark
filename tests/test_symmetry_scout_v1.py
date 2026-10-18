"""Symmetry scout: block planning and seeded R scans (fake SCF backends)."""
from pathlib import Path

from openea_benchmark.state_candidates import MolecularSymmetry
from openea_benchmark.symmetry_scout import (
    SymmetryScoutPolicy, _symmetry_from_label, covalent_start_r, plan_blocks, scan_block,
)
from openea_benchmark.symmetry_states import BlockOccupation, BlockSCFResult

POLICY = SymmetryScoutPolicy()


def test_label_round_trip_and_start_geometry():
    for label in ("3Sigmag-", "2Pi", "1Deltag", "2Sigma+", "4Piu"):
        assert _symmetry_from_label(label).label == label
    assert abs(covalent_start_r("O", "H") - 0.97) < 0.02


def fake_reference(atoms, r, charge, two_s, policy):
    # OH-like valence in C2v: 1s, 2s, 3s (A1), pi x/y (B1/B2), 4s* (A1), pi* x/y, ...
    from openea_benchmark.symmetry_states import OrbitalInfo
    spec = [("A1", 0, -20.0), ("A1", 0, -1.2), ("A1", 0, -0.6), ("B1", 1, -0.4), ("B2", 1, -0.4),
            ("A1", 0, 0.1), ("B1", 1, 0.3), ("B2", 1, 0.3), ("A1", 0, 0.5), ("A1", 0, 0.8)]
    table = [OrbitalInfo(i, e, irrep, m, None) for i, (irrep, m, e) in enumerate(spec)]
    n = 9 if charge == 0 else 10
    return table, ((n + two_s) // 2, (n - two_s) // 2)


def test_plan_finds_oh_ground_blocks_and_reports_unrepresentable_labels():
    plan = plan_blocks("O", "H", 0, 0.97, POLICY, reference=fake_reference)
    labels = {o.symmetry.label for o in plan["blocks"]}
    assert "2Pi" in labels and "4Sigma-" in labels
    assert all(o.symmetry.label in plan["candidate_labels"] for o in plan["blocks"])


def _occ():
    return BlockOccupation(MolecularSymmetry(1, 1), (("A1", (3, 3)), ("E1x", (1, 1)), ("E1y", (1, 0))), 0.0)


def _runner(shape, label="2Pi"):
    def run(r, seed):
        sym = _symmetry_from_label(label) if label else None
        return BlockSCFResult(symmetry=sym, target=_occ().symmetry, irrep_nelec={}, converged=True,
                              energy_hartree=shape(r), stable_within_symmetry=True,
                              symmetry_breaking_instability=False, checkpoint=f"{r}.chk")
    return run


def test_scan_brackets_minimum(tmp_path):
    res = scan_block(("O", "H"), 0, _occ(), 1.0, POLICY, tmp_path, _runner(lambda r: (r - 0.97) ** 2))
    assert res["status"] == "BRACKETED_SINGLE_MINIMUM"
    assert abs(res["minimum"]["r_angstrom"] - 0.95) < 1e-9 or abs(res["minimum"]["r_angstrom"] - 1.0) < 1e-9


def test_scan_refines_coarse_bracket_from_grown_steps(tmp_path):
    # Minimum far from the start: the grown steps bracket it coarsely; the
    # bracket is bisected until both neighbours lie within one fine step.
    res = scan_block(("Cl", "O"), -1, _occ(), 1.6, POLICY, tmp_path, _runner(lambda r: (r - 2.37) ** 2))
    assert res["status"] == "BRACKETED_SINGLE_MINIMUM"
    rs = [p["r_angstrom"] for p in res["points"]]
    i = rs.index(res["minimum"]["r_angstrom"])
    assert rs[i] - rs[i - 1] <= POLICY.step_angstrom + 1e-9 and rs[i + 1] - rs[i] <= POLICY.step_angstrom + 1e-9
    assert abs(res["minimum"]["r_angstrom"] - 2.37) <= POLICY.step_angstrom


def test_scan_detects_dissociative_plateau(tmp_path):
    import math
    res = scan_block(("O", "H"), 0, _occ(), 1.0, POLICY, tmp_path, _runner(lambda r: math.exp(-6 * r)))
    assert res["status"] == "DISSOCIATIVE_PLATEAU"


def test_other_state_is_never_accepted_under_the_block_label(tmp_path):
    # The block converged onto 2Sigma+ everywhere: a curve of 2Sigma+, not of 2Pi.
    res = scan_block(("O", "H"), 0, _occ(), 1.0, POLICY, tmp_path,
                     _runner(lambda r: (r - 0.97) ** 2, label="2Sigma+"))
    assert res["status"] == "BRACKETED_RELABELLED"


def _labelled_runner(shape, label_at):
    def run(r, seed):
        label = label_at(r)
        return BlockSCFResult(symmetry=_symmetry_from_label(label) if label else None, target=_occ().symmetry,
                              irrep_nelec={}, converged=True, energy_hartree=shape(r), stable_within_symmetry=True,
                              symmetry_breaking_instability=False, checkpoint=f"{r}.chk")
    return run


def test_label_loss_at_scan_edge_keeps_bracketed_minimum(tmp_path):
    # verify_v1 IO X2Pi: minimum and its neighbours are 2Pi, only a far edge point is
    # not an Lz^2 eigenfunction (label None) -> the minimum is still bracketed.
    res = scan_block(("I", "O"), 0, _occ(), 1.9, POLICY, tmp_path,
                     _labelled_runner(lambda r: (r - 1.85) ** 2, lambda r: None if r < 1.8 else "2Pi"))
    assert res["status"] == "BRACKETED_SINGLE_MINIMUM"


def test_other_label_next_to_minimum_is_rejected(tmp_path):
    res = scan_block(("O", "H"), 0, _occ(), 1.0, POLICY, tmp_path,
                     _labelled_runner(lambda r: (r - 0.97) ** 2, lambda r: "2Sigma+" if r > 0.99 else "2Pi"))
    assert res["status"] == "LABEL_CHANGED"


def test_cached_record_is_reclassified_with_current_rule():
    from openea_benchmark.symmetry_scout import to_legacy_summary
    block = _block("2Pi", 1, [(1.775, -0.95), (1.85, -1.0), (1.9, -0.99), (2.0, -0.97)], "LABEL_CHANGED")
    block["points"][0]["label"] = None
    summary = to_legacy_summary(_record({"g": block}))
    assert summary["sectors"][0]["manifold_pec_scouts"][0]["minimum_scout"]["status"] == "bracketed_single_minimum"


def _record(blocks):
    return {"system": "OH", "atoms": ["O", "H"], "r0_angstrom": 0.97,
            "policy": {"functional": "PBE0", "basis": "def2-TZVPPD"},
            "charges": {"0": {"plan": {"mr_only_or_outside_window": [{"label": "2Delta", "count": 1, "channels": []}]},
                              "blocks": blocks}}}


def _block(label, two_s, energies, status, higher="RISES"):
    pts = [{"r_angstrom": r, "energy_hartree": e, "converged": True, "stable_within_symmetry": True,
            "checkpoint": f"{r}.chk", "label": label} for r, e in energies]
    i = min(range(len(pts)), key=lambda k: pts[k]["energy_hartree"])
    return {"label": label, "two_s": two_s, "irrep_nelec": [["A1", [3, 3]], ["B1", [1, 1]], ["B2", [1, 0]]],
            "status": status, "side_status": {"lower_r": "RISES", "higher_r": higher},
            "minimum": {"r_angstrom": pts[i]["r_angstrom"], "energy_hartree": pts[i]["energy_hartree"]},
            "points": pts}


def test_export_excludes_blocks_above_lowest_dissociation_limit_and_keeps_ground():
    from openea_benchmark.open_pec_extension import atom_key
    from openea_benchmark.state_selection import build_state_selection_report
    from openea_benchmark.symmetry_scout import to_legacy_summary
    atomic = {atom_key("O", 0, 2): {"energy_hartree": -75.0}, atom_key("H", 0, 1): {"energy_hartree": -0.5}}
    ground = _block("2Pi", 1, [(0.92, -75.60), (0.97, -75.65), (1.02, -75.62)], "BRACKETED_SINGLE_MINIMUM")
    junk = _block("2Sigma+", 1, [(2.1, -74.80), (2.16, -74.85), (2.2, -74.82)], "BRACKETED_SINGLE_MINIMUM")
    summary = to_legacy_summary(_record({"g": ground, "j": junk}), atomic=atomic)
    statuses = {sc["component_id"]: sc["minimum_scout"]["status"] for sec in summary["sectors"]
                for sc in sec["manifold_pec_scouts"]}
    assert statuses == {"g": "bracketed_single_minimum", "j": "above_lowest_dissociation_limit"}
    sel = build_state_selection_report(summary)["charge_groups"]["neutral"]
    assert [c["component_id"] for c in sel["bracketed_candidates"]] == ["g"]
    assert not sel["unresolved_components"]
    assert summary["mr_only_candidates"][0]["label"] == "2Delta"
    sc = next(sc for sec in summary["sectors"] for sc in sec["manifold_pec_scouts"] if sc["component_id"] == "g")
    assert sc["symmetry_block"]["point_group"] == "C2v"


def test_legacy_summary_independent_of_record_key_order():
    """Fresh (insertion-ordered) and reloaded (sorted) scout records give one summary."""
    import json
    from pathlib import Path
    from openea_benchmark.symmetry_scout import to_legacy_summary
    candidates = sorted(Path(__file__).resolve().parents[1].glob("../openea_two_atom*/*/symmetry_scout/symmetry_scout.json"))
    if not candidates:
        import pytest
        pytest.skip("no stored scout record")
    record = json.loads(candidates[0].read_text())
    reversed_order = json.loads(json.dumps(record), object_pairs_hook=lambda kv: dict(reversed(kv)))
    assert json.dumps(to_legacy_summary(record), sort_keys=True) == json.dumps(to_legacy_summary(reversed_order), sort_keys=True)


def test_cached_scan_stopped_at_r_limit_is_continued(tmp_path):
    # verify_v1 Li2 a3Sigmau+: still decreasing at the former 4.0 A limit.  A wider
    # extent continues the stored scan instead of rescanning it.
    from dataclasses import replace
    calls = []
    def shape(r):
        calls.append(r)
        return (r - 4.6) ** 2
    small = replace(POLICY, max_r_angstrom=4.0)
    first = scan_block(("Li", "Li"), 0, _occ(), 2.6, small, tmp_path, _runner(shape))
    assert first["side_status"]["higher_r"] == "R_LIMIT" and first["status"] == "UNRESOLVED"
    n_first = len(calls)
    wide = replace(POLICY, max_r_angstrom=8.0)
    second = scan_block(("Li", "Li"), 0, _occ(), 2.6, wide, tmp_path, _runner(shape), existing=first)
    assert second["status"] == "BRACKETED_SINGLE_MINIMUM"
    assert abs(second["minimum"]["r_angstrom"] - 4.6) <= POLICY.step_angstrom
    assert all(round(r, 4) not in {p["r_angstrom"] for p in first["points"]} for r in calls[n_first:])


def test_consistently_relabelled_bracket_becomes_candidate_of_that_state():
    from openea_benchmark.symmetry_scout import to_legacy_summary
    block = _block("2Delta", 1, [(1.8, -0.95), (1.9, -1.0), (2.0, -0.97)], "LABEL_CHANGED")
    for p in block["points"]:
        p["label"] = "2Sigma+"
    sc = to_legacy_summary(_record({"d": block}))["sectors"][0]["manifold_pec_scouts"][0]
    assert sc["minimum_scout"]["status"] == "bracketed_single_minimum"
    assert sc["symmetry_block"]["label"] == "2Sigma+" and sc["symmetry_block"]["block_target_label"] == "2Delta"


def test_scf_failure_is_retried_at_half_step(tmp_path):
    # verify_v1 NiO- 2Delta: SCF failed one step inside the minimum -> unbracketed.
    def run(r, seed):
        ok = abs(r - 0.95) > 1e-6
        return BlockSCFResult(symmetry=_symmetry_from_label("2Pi"), target=_occ().symmetry, irrep_nelec={},
                              converged=ok, energy_hartree=(r - 0.985) ** 2 if ok else None,
                              stable_within_symmetry=True, symmetry_breaking_instability=False,
                              checkpoint=f"{r}.chk")
    res = scan_block(("O", "H"), 0, _occ(), 1.0, POLICY, tmp_path, run)
    assert res["status"] == "BRACKETED_SINGLE_MINIMUM"
    assert any(abs(p["r_angstrom"] - 0.975) < 1e-9 for p in res["points"])


def test_like_atom_anion_below_localized_asymptote_is_dissociative():
    # verify_v1 Li2- 2Sigmag+: still falling at the R limit, 0.28 eV below Li + Li-.
    from openea_benchmark.open_pec_extension import atom_key
    from openea_benchmark.symmetry_scout import to_legacy_summary
    atomic = {atom_key("Li", 0, 1): {"energy_hartree": -7.46705}, atom_key("Li", -1, 0): {"energy_hartree": -7.47936}}
    block = _block("2Sigmag+", 1, [(5.0, -14.950), (6.0, -14.953), (7.0, -14.955), (7.75, -14.9565)],
                   "UNRESOLVED", higher="R_LIMIT")
    record = {"system": "LiLi", "atoms": ["Li", "Li"], "r0_angstrom": 2.6,
              "policy": {"functional": "PBE0", "basis": "def2-TZVPPD"},
              "charges": {"-1": {"plan": {"mr_only_or_outside_window": []}, "blocks": {"d": block}}}}
    sc = to_legacy_summary(record, atomic=atomic)["sectors"][0]["manifold_pec_scouts"][0]
    assert sc["minimum_scout"]["status"] == "dissociative_to_asymptote"
    assert sc["minimum_scout"]["asymptote_evidence"]["basis"].startswith("SYMMETRIC_CHARGE")


def test_descending_into_scf_failure_routes_to_mr_layer():
    from openea_benchmark.symmetry_scout import to_legacy_summary
    block = _block("2Delta", 1, [(1.725, -1.0), (1.75, -0.99), (1.8, -0.98)], "UNRESOLVED")
    block["side_status"] = {"lower_r": "SCF_FAILED", "higher_r": "RISES"}
    summary = to_legacy_summary(_record({"d": block}))
    assert summary["sectors"][0]["manifold_pec_scouts"][0]["minimum_scout"]["status"] == "scf_failed_edge_mr_route"
    assert any(m["label"] == "2Delta" and m["charge"] == 0 for m in summary["mr_only_candidates"])
