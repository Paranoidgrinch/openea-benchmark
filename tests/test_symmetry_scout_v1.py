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


def test_label_change_is_never_accepted(tmp_path):
    res = scan_block(("O", "H"), 0, _occ(), 1.0, POLICY, tmp_path,
                     _runner(lambda r: (r - 0.97) ** 2, label="2Sigma+"))
    assert res["status"] == "LABEL_CHANGED"


def _record(blocks):
    return {"system": "OH", "atoms": ["O", "H"], "r0_angstrom": 0.97,
            "policy": {"functional": "PBE0", "basis": "def2-TZVPPD"},
            "charges": {"0": {"plan": {"mr_only_or_outside_window": [{"label": "2Delta", "count": 1, "channels": []}]},
                              "blocks": blocks}}}


def _block(label, two_s, energies, status, higher="RISES"):
    pts = [{"r_angstrom": r, "energy_hartree": e, "converged": True, "stable_within_symmetry": True,
            "checkpoint": f"{r}.chk"} for r, e in energies]
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
