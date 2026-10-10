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


def test_scan_detects_dissociative_plateau(tmp_path):
    import math
    res = scan_block(("O", "H"), 0, _occ(), 1.0, POLICY, tmp_path, _runner(lambda r: math.exp(-6 * r)))
    assert res["status"] == "DISSOCIATIVE_PLATEAU"


def test_label_change_is_never_accepted(tmp_path):
    res = scan_block(("O", "H"), 0, _occ(), 1.0, POLICY, tmp_path,
                     _runner(lambda r: (r - 0.97) ** 2, label="2Sigma+"))
    assert res["status"] == "LABEL_CHANGED"
