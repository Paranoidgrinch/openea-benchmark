"""Patch 51: MR-only competitor blocks (fake SA-CASSCF/NEVPT2 energies)."""
from types import SimpleNamespace

from openea_benchmark.mr_competitors import candidate_block, run_mr_competitor_stage, selected_block

H = 1.0 / 27.211386245988


def test_blocks():
    assert candidate_block("q=0 1Deltag") == (0, 1, "A2")
    assert candidate_block("q=-1 2Sigmau-") == (-1, 2, "A2")
    assert candidate_block("q=0 1Sigmau+") == (0, 1, "A1")
    assert candidate_block("q=0 3Phig") == (0, 3, "B1")
    assert selected_block("3Sigmag-") == (3, "A2") and selected_block("2Pig") == (2, "B1")


def run(tmp_path, gap_ev):
    def block_fn(atoms, charge, mult, irrep, r, policy, cache):
        base = {0: -150.0, -1: -150.02}[charge]
        if (charge, mult, irrep) == (0, 3, "A2"):  # selected neutral block: X, then a state 2 eV up
            return {"status": "COMPLETED", "energies_hartree": [base, base + 2.0 * H]}
        if (charge, mult, irrep) == (-1, 2, "B1"):
            return {"status": "COMPLETED", "energies_hartree": [base, base + 1.0 * H]}
        return {"status": "COMPLETED", "energies_hartree": [base + gap_ev * H, base + (gap_ev + 1) * H]}

    ea = {"status": "ELECTRONIC_EA_EVALUATED", "system": "OO", "selected_neutral_job": "N", "selected_anion_job": "A",
          "candidates": {"neutral": [{"job_id": "N", "r_central_angstrom": 1.21}],
                         "anion": [{"job_id": "A", "r_central_angstrom": 1.35}]}}
    reqs = [SimpleNamespace(job_id="N", symmetry_label="3Sigmag-", charge=0, atoms=("O", "O")),
            SimpleNamespace(job_id="A", symmetry_label="2Pig", charge=-1, atoms=("O", "O"))]
    return run_mr_competitor_stage(electronic_ea=ea, requests=reqs, output_dir=tmp_path, block_fn=block_fn,
                                   mr_only_labels=["q=0 1Deltag", "q=0 3Sigmau-", "q=-1 2Deltag"])


def test_far_blocks_are_excluded_by_lower_bound(tmp_path):
    res = run(tmp_path, gap_ev=0.9)
    assert res["status"] == "CLEARED"
    same = next(b for b in res["blocks"] if b["mult"] == 3)  # 3Sigmau- shares the selected block: second root
    assert same["bound_root"] == "second" and abs(same["lower_bound_margin_ev"] - 2.0) < 1e-9


def test_close_block_keeps_assignment_open(tmp_path):
    res = run(tmp_path, gap_ev=0.1)
    assert res["status"] == "UNRESOLVED_MR_COMPETITOR"
    assert {b["status"] for b in res["blocks"]} == {"EXCLUDED", "UNRESOLVED_MR_COMPETITOR"}
