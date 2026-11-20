"""CC route for MR-only competitor states (fake CCSD / EOM-CCSD block energies)."""
from types import SimpleNamespace

from openea_benchmark import cc_competitors
from openea_benchmark.cc_competitors import (CCCompetitorPolicy, cc_block_task, merge_competitor_records,
                                             run_cc_competitor_stage, stage_references)

H = 1.0 / 27.211386245988


def _ea_and_requests():
    ea = {"status": "ELECTRONIC_EA_EVALUATED", "system": "CuO", "selected_neutral_job": "N", "selected_anion_job": "A",
          "candidates": {"neutral": [{"job_id": "N", "r_central_angstrom": 1.72, "energy_central_hartree": -1714.0}],
                         "anion": [{"job_id": "A", "r_central_angstrom": 1.67, "energy_central_hartree": -1714.1},
                                   {"job_id": "A3", "r_central_angstrom": 1.70, "energy_central_hartree": -1714.05}]}}
    occ_a = {"A1": (10, 10), "A2": (1, 1), "B1": (4, 4), "B2": (4, 4)}
    reqs = [SimpleNamespace(job_id="N", symmetry_label="2Pi", charge=0, spin_2s=1, atoms=("Cu", "O"),
                            irrep_nelec={"A1": (10, 10), "A2": (1, 1), "B1": (4, 3), "B2": (4, 4)}),
            SimpleNamespace(job_id="A", symmetry_label="1Sigma+", charge=-1, spin_2s=0, atoms=("Cu", "O"),
                            irrep_nelec=occ_a),
            SimpleNamespace(job_id="A3", symmetry_label="3Pi", charge=-1, spin_2s=2, atoms=("Cu", "O"),
                            irrep_nelec={"A1": (10, 9), "A2": (1, 1), "B1": (5, 4), "B2": (4, 4)})]
    return ea, reqs


def test_references_are_the_selected_states_and_the_lowest_candidate_per_spin():
    ea, reqs = _ea_and_requests()
    refs = stage_references(ea, reqs)
    assert set(refs["selected"]) == {"0:2:B1", "-1:1:A1"}
    assert refs["lowest"]["-1:0"]["A1"] == [10, 10] and refs["lowest"]["-1:2"]["B1"] == [5, 4]


def test_eom_singlet_well_above_the_selected_state_clears_the_block(tmp_path):
    ea, reqs = _ea_and_requests()

    def block_fn(atoms, charge, mult, irrep, r, policy, cache):
        base = {0: -1714.0, -1: -1714.1}[charge]
        if (charge, mult, irrep) in ((0, 2, "B1"), (-1, 1, "A1")):   # selected blocks: CCSD, then EOM 2 eV up
            energies = [base, base + 2.0 * H]
        else:                                                        # 1Pi of the anion 0.9 eV up
            energies = [base + 0.9 * H]
        return {"status": "COMPLETED", "energies_hartree": energies,
                "roots": [{"e_hartree": e, "lambda": None, "parity": None} for e in energies]}

    res = run_cc_competitor_stage(electronic_ea=ea, requests=reqs, mr_only_labels=["q=-1 1Pi"],
                                  output_dir=tmp_path, block_fn=block_fn)
    assert res["status"] == "CLEARED" and res["stage"] == "CC_COMPETITORS"
    block = res["blocks"][0]
    assert abs(block["lower_bound_margin_ev"] - 0.9) < 1e-9 and block["status"] == "EXCLUDED"
    assert (tmp_path / "cc_competitors.json").is_file()


def test_block_within_the_margin_stays_open(tmp_path):
    ea, reqs = _ea_and_requests()

    def block_fn(atoms, charge, mult, irrep, r, policy, cache):
        base = {0: -1714.0, -1: -1714.1}[charge]
        energies = [base, base + 2.0 * H] if (charge, mult, irrep) in ((0, 2, "B1"), (-1, 1, "A1")) \
            else [base + 0.2 * H]
        return {"status": "COMPLETED", "energies_hartree": energies,
                "roots": [{"e_hartree": e, "lambda": None, "parity": None} for e in energies]}

    res = run_cc_competitor_stage(electronic_ea=ea, requests=reqs, mr_only_labels=["q=-1 1Pi"],
                                  output_dir=tmp_path, block_fn=block_fn)
    assert res["status"] == "UNRESOLVED_MR_COMPETITOR"


def test_selected_block_uses_its_own_determinant_and_no_spin_flip(tmp_path, monkeypatch):
    ea, reqs = _ea_and_requests()
    refs = stage_references(ea, reqs)
    seen = []

    def fake_roots(atoms, charge, mult, irrep, r, policy, keep, flip):
        seen.append((charge, mult, irrep, keep, flip))
        return [{"e_hartree": -1.0, "method": "CCSD", "singles_weight": 1.0, "lambda": None, "parity": None}]

    monkeypatch.setattr(cc_competitors, "block_roots", fake_roots)
    pol = CCCompetitorPolicy()
    cc_block_task(refs, pol, ("Cu", "O"), -1, 1, "A1", 1.67, None, tmp_path / "a.json")
    cc_block_task(refs, pol, ("Cu", "O"), -1, 1, "B1", 1.67, None, tmp_path / "b.json")
    cc_block_task(refs, pol, ("Cu", "O"), -1, 5, "A2", 1.67, None, tmp_path / "c.json")
    cc_block_task(refs, pol, ("Cu", "O"), 0, 1, "A2", 1.72, None, tmp_path / "d.json")
    sel, pi, quintet, neutral_singlet = seen
    assert sel[3]["irrep_nelec"]["A1"] == [10, 10] and sel[4] is None
    # a Stage-3 singlet of the anion exists: spin-adapted singlets only, no spin flip
    assert pi[3]["irrep_nelec"]["A1"] == [10, 10] and pi[4] is None
    assert quintet[3] == {"two_s": 4, "irrep_nelec": None} and quintet[4] is None   # S = 2: no spin flip
    # no Stage-3 singlet of the neutral: aufbau singlet plus spin flip from the triplet
    assert neutral_singlet[3] == {"two_s": 0, "irrep_nelec": None} and neutral_singlet[4]["two_s"] == 2
    again = cc_block_task(refs, pol, ("Cu", "O"), -1, 1, "B1", 1.67, None, tmp_path / "b.json")
    assert len(seen) == 4 and again["status"] == "COMPLETED"                       # cached


def test_doubly_excited_root_is_lowered_by_the_allowance(monkeypatch):
    pol = CCCompetitorPolicy()
    roots = [{"e_hartree": -1.0, "singles_weight": 0.5}, {"e_hartree": -0.9, "singles_weight": 0.95}]
    for root in roots:  # the post-processing of block_roots
        root["lambda"], root["parity"] = None, None
        if root["singles_weight"] < pol.min_singles_weight:
            root["e_hartree"] -= pol.low_singles_allowance_ev / 27.211386245988
    assert abs(roots[0]["e_hartree"] - (-1.0 - 0.7 * H)) < 1e-12 and roots[1]["e_hartree"] == -0.9


def _block(q, m, irrep, status, margin=1.0):
    return {"charge": q, "mult": m, "irrep_c2v": irrep, "candidates": [f"q={q} x"], "status": status,
            "lower_bound_margin_ev": margin}


def test_merge_excludes_a_block_when_either_layer_excludes_it():
    cc = {"status": "UNRESOLVED_MR_COMPETITOR", "blocks": [_block(-1, 1, "B1", "EXCLUDED"),
                                                          _block(0, 6, "A2", "UNRESOLVED_NOT_COMPUTED", None)]}
    mr = {"status": "UNRESOLVED_MR_COMPETITOR", "selected": {}, "policy": {},
          "blocks": [_block(-1, 1, "B1", "UNRESOLVED_MR_COMPETITOR", -2.6), _block(0, 6, "A2", "EXCLUDED")]}
    merged = merge_competitor_records(cc, mr)
    assert merged["status"] == "CLEARED"
    assert {b["source"] for b in merged["blocks"]} == {"CC", "MR"}
    mr["blocks"][1]["status"] = "UNRESOLVED_MR_COMPETITOR"
    assert merge_competitor_records(cc, mr)["status"] == "UNRESOLVED_MR_COMPETITOR"


def test_merge_with_a_layer_that_did_not_get_to_the_blocks():
    cc = {"status": "CLEARED", "blocks": [_block(-1, 1, "B1", "EXCLUDED")]}
    assert merge_competitor_records(cc, {"status": "NOT_ATTEMPTED"})["status"] == "CLEARED"
    assert merge_competitor_records({"status": "NOT_ATTEMPTED"}, {"status": "NOT_ATTEMPTED"})["status"] == "NOT_ATTEMPTED"
    mr = {"status": "UNRESOLVED_UNCLASSIFIED_LABELS", "blocks": [_block(-1, 1, "B1", "EXCLUDED")],
          "unclassified_labels": ["q=0 9X"]}
    assert merge_competitor_records(cc, mr)["status"] == "UNRESOLVED_UNCLASSIFIED_LABELS"


def test_lower_eom_root_in_the_selected_block_never_replaces_the_selected_energy(tmp_path, monkeypatch):
    ea, reqs = _ea_and_requests()
    refs = stage_references(ea, reqs)

    def fake_roots(atoms, charge, mult, irrep, r, policy, keep, flip):
        return [{"e_hartree": -2.0, "method": "EOM-EE-CCSD spin-keeping", "singles_weight": 0.9,
                 "lambda": None, "parity": None},
                {"e_hartree": -1.0, "method": "CCSD", "singles_weight": 1.0, "lambda": None, "parity": None}]

    monkeypatch.setattr(cc_competitors, "block_roots", fake_roots)
    rec = cc_block_task(refs, CCCompetitorPolicy(), ("Cu", "O"), -1, 1, "A1", 1.67, None, tmp_path / "s.json")
    assert rec["energies_hartree"] == [-1.0, -2.0]      # selected first; the lower state is a competitor
