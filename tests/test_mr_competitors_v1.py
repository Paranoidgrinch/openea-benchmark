"""Patch 51: MR-only competitor blocks (fake SA-CASSCF/NEVPT2 energies)."""
from types import SimpleNamespace

import numpy as np

from openea_benchmark.mr_competitors import (candidate_block, candidate_parity, root_lambda, root_parity,
                                             run_mr_competitor_stage, selected_block)

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


def test_lambda_resolution_skips_computed_pi_state_below_phi_candidates(tmp_path):
    """C2: the 3B1 block's lowest root is a3Piu (Stage 3 state), the 3Phi candidates lie far above."""
    def roots(base, spec):
        return {"status": "COMPLETED", "energies_hartree": [base + d * H for d, *_ in spec],
                "roots": [{"casci_index": i, "e_hartree": base + d * H, "lz2": float(lam or 0) ** 2, "lambda": lam,
                           "parity": par} for i, (d, lam, par) in enumerate(spec)]}

    def block_fn(atoms, charge, mult, irrep, r, policy, cache):
        base = {0: -75.79, -1: -75.90}[charge]
        if (charge, mult, irrep) == (0, 1, "A1"):
            return roots(base, [(0.0, 0, 1), (2.0, 2, 1)])
        if (charge, mult, irrep) == (0, 3, "B1"):  # a3Piu, d3Pig, unassigned, Pi
            return roots(base, [(0.17, 1, -1), (2.6, 1, 1), (7.5, None, None), (7.6, 1, 1)])
        return roots(base, [(0.0, 0, 1), (3.0, 0, -1)])

    ea = {"status": "ELECTRONIC_EA_EVALUATED", "system": "CC", "selected_neutral_job": "N", "selected_anion_job": "A",
          "candidates": {"neutral": [{"job_id": "N", "r_central_angstrom": 1.25}],
                         "anion": [{"job_id": "A", "r_central_angstrom": 1.27}]}}
    reqs = [SimpleNamespace(job_id="N", symmetry_label="1Sigmag+", charge=0, atoms=("C", "C")),
            SimpleNamespace(job_id="A", symmetry_label="2Sigmag+", charge=-1, atoms=("C", "C"))]
    res = run_mr_competitor_stage(electronic_ea=ea, requests=reqs, output_dir=tmp_path, block_fn=block_fn,
                                  mr_only_labels=["q=0 3Phig", "q=0 3Phiu"])
    blk = next(b for b in res["blocks"] if b["mult"] == 3)
    assert blk["lambda_resolved"] and abs(blk["lower_bound_margin_ev"] - 7.5) < 1e-6
    assert res["status"] == "CLEARED"
    # A 3Pig candidate is bounded by the lowest gerade Pi root (d3Pig, 2.6 eV), not by a3Piu.
    res = run_mr_competitor_stage(electronic_ea=ea, requests=reqs, output_dir=tmp_path / "pig", block_fn=block_fn,
                                  mr_only_labels=["q=0 3Pig"])
    blk = next(b for b in res["blocks"] if b["mult"] == 3)
    assert abs(blk["lower_bound_margin_ev"] - 2.6) < 1e-6 and res["status"] == "CLEARED"
    # A 3Piu candidate is bounded by a3Piu itself (0.17 eV): kept open.
    res = run_mr_competitor_stage(electronic_ea=ea, requests=reqs, output_dir=tmp_path / "piu", block_fn=block_fn,
                                  mr_only_labels=["q=0 3Piu"])
    assert res["status"] == "UNRESOLVED_MR_COMPETITOR"


def test_root_classification_helpers():
    assert candidate_parity("q=0 3Pig") == 1 and candidate_parity("q=0 1Sigmau-") == -1
    assert candidate_parity("q=0 2Pi") is None
    assert root_lambda(1.01, 0.25, "B1") == 1 and root_lambda(8.9, 0.25, "B1") == 3
    assert root_lambda(4.0, 0.25, "B1") is None  # Pi/Phi mixture, not Delta
    assert root_lambda(4.0, 0.25, "A2") == 2 and root_lambda(1.0, 0.25, "A1") is None
    assert root_lambda(2.7, 0.25, "B1") is None
    # two active orbitals (g, u), one alpha + one beta electron
    c = np.zeros((2, 2)); c[0, 0] = 1.0  # both in the g orbital -> g
    assert root_parity(c, 2, (1, 1), [1, -1]) == 1
    c = np.zeros((2, 2)); c[0, 1] = c[1, 0] = 2 ** -0.5  # one g, one u -> u
    assert root_parity(c, 2, (1, 1), [1, -1]) == -1
    c = np.zeros((2, 2)); c[0, 0] = c[0, 1] = 2 ** -0.5  # mixed -> unassigned
    assert root_parity(c, 2, (1, 1), [1, -1]) is None


def test_inversion_parity_from_ci_vector():
    """g/u of a CI root when the active orbitals themselves mix g and u (Patch 55b)."""
    import numpy as np
    from pyscf import gto
    from openea_benchmark.mr_competitors import inversion_matrix_active, root_parity_general
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    s = mol.intor("int1e_ovlp")
    g = np.array([1.0, 1.0]) / np.sqrt(2 + 2 * s[0, 1])
    u = np.array([1.0, -1.0]) / np.sqrt(2 - 2 * s[0, 1])
    mixed = np.column_stack([g + u, g - u]) / np.sqrt(2)  # localized: neither g nor u
    inv = inversion_matrix_active(mol, mixed)
    # sigma_g^2 in the localized basis: (|aa> + |bb> + |ab> + |ba>)/2 is gerade
    ci_g = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert root_parity_general(ci_g, 2, (1, 1), inv) == 1
    # open-shell 1Sigma_u+ (sigma_g sigma_u singlet) = (|aa> - |bb>)/sqrt2 is ungerade
    ci_u = np.array([[1.0, 0.0], [0.0, -1.0]]) / np.sqrt(2)
    assert root_parity_general(ci_u, 2, (1, 1), inv) == -1
