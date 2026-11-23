"""Patch 51: MR-only competitor blocks (fake SA-CASSCF/NEVPT2 energies)."""
from types import SimpleNamespace

import numpy as np

from openea_benchmark.mr_competitors import (candidate_block, candidate_parity, root_lambda, root_parity,
                                             run_mr_competitor_stage, selected_block, MRCompetitorPolicy)

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


def test_collapsed_candidate_is_bounded_at_its_own_r_too(tmp_path):
    """Li2-: 2Sigmag+ (collapsed CCSD(T)) shares the A1 doublet block with the selected
    2Sigmau+ and lies lowest far outside its window; the bound includes that R."""
    def roots(base, spec):
        return {"status": "COMPLETED", "r": None, "energies_hartree": [base + d * H for d, *_ in spec],
                "roots": [{"casci_index": i, "e_hartree": base + d * H, "lz2": 0.0, "lambda": lam, "parity": par}
                          for i, (d, lam, par) in enumerate(spec)]}

    def block_fn(atoms, charge, mult, irrep, r, policy, cache):
        if charge == 0:
            return roots(-14.9, [(0.0, 0, 1), (2.0, 0, -1)])
        if r > 4.0:  # the 2Sigmag+ well: below 2Sigmau+ there, 0.6 eV above its minimum
            return {**roots(-14.98, [(0.6, 0, 1), (0.9, 0, -1)]), "r": r}
        return {**roots(-14.98, [(0.0, 0, -1), (1.5, 0, 1)]), "r": r}

    ea = {"status": "ELECTRONIC_EA_EVALUATED", "system": "LiLi", "selected_neutral_job": "N", "selected_anion_job": "A",
          "candidates": {"neutral": [{"job_id": "N", "r_central_angstrom": 2.67}],
                         "anion": [{"job_id": "A", "r_central_angstrom": 3.08}]}}
    reqs = [SimpleNamespace(job_id="N", symmetry_label="1Sigmag+", charge=0, atoms=("Li", "Li")),
            SimpleNamespace(job_id="A", symmetry_label="2Sigmau+", charge=-1, atoms=("Li", "Li"))]
    kw = dict(electronic_ea=ea, requests=reqs, block_fn=block_fn, mr_only_labels=["q=-1 2Sigmag+"])
    window_only = run_mr_competitor_stage(output_dir=tmp_path / "w", **kw)
    blk = window_only["blocks"][0]
    assert abs(blk["lower_bound_margin_ev"] - 1.5) < 1e-9 and window_only["status"] == "CLEARED"
    res = run_mr_competitor_stage(output_dir=tmp_path / "x", candidate_r_angstrom={"q=-1 2Sigmag+": 5.0}, **kw)
    blk = res["blocks"][0]
    # lowest at the outer edge: one step out (doubling the distance from the window), flat there
    assert blk["extra_r_angstrom"] == [5.0, 6.67]
    assert abs(blk["lower_bound_margin_ev"] - 0.6) < 1e-9 and res["status"] == "CLEARED"
    # Selected energy stays the window minimum; a well within the margin keeps it open.
    res = run_mr_competitor_stage(output_dir=tmp_path / "y", policy=MRCompetitorPolicy(exclusion_margin_ev=0.7),
                                  candidate_r_angstrom={"q=-1 2Sigmag+": 5.0}, **kw)
    assert res["status"] == "UNRESOLVED_MR_COMPETITOR"


def _falling_tail_run(tmp_path, turn_r, policy=None, asymptote=False):
    """Anion competitor in its own block, falling with R until ``turn_r`` (or towards an
    asymptote 1 eV above the selected state)."""
    def block_fn(atoms, charge, mult, irrep, r, policy, cache):
        if charge == 0 or mult == 2:
            return {"status": "COMPLETED", "energies_hartree": [-10.0 - (0.02 if charge else 0.0)]}
        if asymptote:
            return {"status": "COMPLETED", "energies_hartree": [-10.02 + (1.0 + 0.5 * np.exp(-(r - 2.0))) * H]}
        e = -10.02 + 1.0 * H + 0.2 * H * abs(min(r, turn_r) - turn_r) + (0.5 * H * (r - turn_r) if r > turn_r else 0.0)
        return {"status": "COMPLETED", "energies_hartree": [e]}

    ea = {"status": "ELECTRONIC_EA_EVALUATED", "system": "XY", "selected_neutral_job": "N", "selected_anion_job": "A",
          "candidates": {"neutral": [{"job_id": "N", "r_central_angstrom": 2.0}],
                         "anion": [{"job_id": "A", "r_central_angstrom": 2.0}]}}
    reqs = [SimpleNamespace(job_id="N", symmetry_label="1Sigma+", charge=0, atoms=("X", "Y")),
            SimpleNamespace(job_id="A", symmetry_label="2Sigma+", charge=-1, atoms=("X", "Y"))]
    return run_mr_competitor_stage(electronic_ea=ea, requests=reqs, output_dir=tmp_path, block_fn=block_fn,
                                   mr_only_labels=["q=-1 4Sigma+"], policy=policy or MRCompetitorPolicy())


def test_bound_falling_at_the_window_edge_is_followed_outward(tmp_path):
    res = _falling_tail_run(tmp_path / "a", turn_r=2.9)
    blk = res["blocks"][0]
    assert blk["extra_r_angstrom"] == [2.5, 2.75, 3.25] and res["status"] == "CLEARED"
    assert abs(blk["lower_bound_margin_ev"] - 1.0) < 0.06  # the well bottom, not the window edge (1.13 eV)
    res = _falling_tail_run(tmp_path / "b", turn_r=9.0, policy=MRCompetitorPolicy(max_edge_extensions=3))
    assert res["blocks"][0]["status"] == "UNRESOLVED_BOUND_AT_WINDOW_EDGE" and res["status"] != "CLEARED"
    # Falling towards its dissociation limit: followed to R >= 10 A, the last value bounds it.
    res = _falling_tail_run(tmp_path / "c", turn_r=None, asymptote=True)
    blk = res["blocks"][0]
    assert blk["extra_r_angstrom"] == [2.5, 2.75, 3.25, 4.25, 6.25, 10.25] and blk["falling_to_asymptote"]
    assert blk["status"] == "EXCLUDED" and abs(blk["lower_bound_margin_ev"] - 1.0) < 1e-3


def test_divergent_nevpt2_root_is_bounded_from_its_casci_energy():
    from openea_benchmark.mr_competitors import bound_nevpt2_divergence
    # SH- B1 at 1.59 A: a high CASCI root with an NEVPT2 "correction" of -36 Eh.
    rec = {"status": "COMPLETED", "roots": [
        {"e_casci_hartree": -397.998, "nevpt2_hartree": -0.327, "e_hartree": -398.325},
        {"e_casci_hartree": -393.766, "nevpt2_hartree": -36.41, "e_hartree": -430.176}]}
    bound_nevpt2_divergence(rec, -0.334)
    low, high = rec["roots"]
    assert not low.get("nevpt2_diverged") and abs(low["e_hartree"] + 398.325) < 1e-9
    assert high["nevpt2_diverged"] and abs(high["e_hartree"] - (-393.766 - 0.334 - 0.5)) < 1e-9
    assert rec["energies_hartree"] == [low["e_hartree"], high["e_hartree"]] and rec["nevpt2_diverged_roots"] == 1
    bound_nevpt2_divergence(rec, -0.334)  # idempotent: the raw correction is kept
    assert abs(rec["roots"][1]["e_hartree"] - (-394.6)) < 1e-9 and rec["roots"][1]["nevpt2_raw_hartree"] == -36.41


def test_high_lambda_labels_are_classified():
    # verify_v1 CuO: excited asymptotes give H (Lambda 5) and I (Lambda 6) states.
    assert candidate_block("q=0 2H") == (0, 2, "B1")
    assert candidate_block("q=0 4I") == (0, 4, "A2")
    assert candidate_block("q=0 2Ig") == (0, 2, "A2") and candidate_parity("q=0 2Hu") == -1


def test_high_lambda_asymptote_labels_are_classified():
    # verify_v1 ScO: Sc-(1H) + O(1D) gives Lambda = 7; it was labelled "1L7" and left unclassified.
    from openea_benchmark.mr_competitors import LAMBDA_IRREP, candidate_lambda
    from openea_benchmark.state_candidates import _LAMBDA_LETTER
    assert _LAMBDA_LETTER[7] == "K" and _LAMBDA_LETTER[8] == "L"
    assert candidate_lambda("q=-1 1K") == 7 and LAMBDA_IRREP["K"] == "B1" and LAMBDA_IRREP["L"] == "A2"
    assert candidate_lambda("2Deltag") == 2 and candidate_lambda("1Sigma+") == 0
