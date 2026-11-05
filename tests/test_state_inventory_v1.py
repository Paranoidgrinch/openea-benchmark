from openea_benchmark.electronic_ea_stage import HARTREE_TO_EV
from openea_benchmark.state_inventory import build_state_inventory, format_state_table


def _cand(job, charge, e, r=1.2, status="RESOLVED"):
    return {"job_id": job, "charge": charge, "spin_2s": 0, "branch_of": None, "equilibrium_status": status,
            "r_central_angstrom": None if e is None else r, "energy_central_hartree": e,
            "energy_lower_hartree": None if e is None else e - 1e-6,
            "energy_upper_hartree": None if e is None else e + 1e-6}


ELECTRONIC = {
    "basis": "def2-TZVPPD", "selected_neutral_job": "XY__q+0__2S1__q+0__2Sigma+__a",
    "selected_anion_job": "XY__q-1__2S0__q-1__1Sigma+__b",
    "candidates": {
        "neutral": [_cand("XY__q+0__2S1__q+0__2Sigma+__a", 0, -100.0),
                    _cand("XY__q+0__2S1__q+0__2Pi__c", 0, -99.9),
                    _cand("XY__q+0__2S1__q+0__2Pi__d", 0, None, status="SINGLE_REFERENCE_COLLAPSED")],
        "anion": [_cand("XY__q-1__2S0__q-1__1Sigma+__b", -1, -100.1),
                  _cand("XY__q-1__2S2__q-1__3Pi__e", -1, -100.05)],
    },
}


def test_term_energies_and_thresholds():
    inv = build_state_inventory(ELECTRONIC)
    by = {r["job_id"]: r for r in inv["states"]}
    assert by["XY__q+0__2S1__q+0__2Sigma+__a"]["is_ground_state"]
    assert abs(by["XY__q+0__2S1__q+0__2Pi__c"]["t_e_ev"] - 0.1 * HARTREE_TO_EV) < 1e-9
    # detachment threshold anion ground -> neutral excited state
    assert abs(by["XY__q+0__2S1__q+0__2Pi__c"]["detachment_threshold_ev"] - 0.2 * HARTREE_TO_EV) < 1e-9
    # excited anion state still below the neutral ground state: bound at this level
    assert abs(by["XY__q-1__2S2__q-1__3Pi__e"]["binding_vs_neutral_ground_ev"] - 0.05 * HARTREE_TO_EV) < 1e-9
    collapsed = by["XY__q+0__2S1__q+0__2Pi__d"]
    assert collapsed["t_e_ev"] is None and collapsed["status"] == "SINGLE_REFERENCE_COLLAPSED"
    assert inv["counts"] == {"total": 5, "with_minimum": 4, "mr_only": 0}
    assert inv["final_ea_ev_used_for_scale"] is None


def test_final_ea_scale_only_with_issued_ea():
    inv = build_state_inventory(ELECTRONIC, final_verdict={"outcome": "EA", "ea_ev": 2.5})
    by = {r["job_id"]: r for r in inv["states"]}
    assert abs(by["XY__q+0__2S1__q+0__2Pi__c"]["detachment_threshold_final_scale_ev"]
               - (2.5 + 0.1 * HARTREE_TO_EV)) < 1e-9
    assert abs(by["XY__q-1__2S2__q-1__3Pi__e"]["binding_final_scale_ev"] - (2.5 - 0.05 * HARTREE_TO_EV)) < 1e-9
    inv = build_state_inventory(ELECTRONIC, final_verdict={"outcome": "UNRESOLVED"})
    assert all(r.get("detachment_threshold_final_scale_ev") is None for r in inv["states"])


def test_mr_only_states_listed_with_status():
    comp = {"blocks": [{"charge": 0, "status": "EXCLUDED", "lower_bound_margin_ev": 1.5,
                        "candidates": ["q=0 2Delta"]}]}
    gap = {"gaps": [{"charge": -1, "label": "q=-1 1Delta", "status": "COMPLETED", "gap0_ev": 0.4,
                     "gap0_lower_ev": 0.3, "gap0_upper_ev": 0.5}]}
    inv = build_state_inventory(ELECTRONIC, mr_competitors=comp, mr_term_gap=gap,
                                mr_only_labels=["q=0 2Delta", "q=-1 1Delta", "q=-1 3Phi"])
    status = {(m.get("term") or m.get("label")): m["status"] for m in inv["mr_only_states"]}
    assert status == {"q=-1 1Delta": "COMPLETED", "q=0 2Delta": "EXCLUDED", "q=-1 3Phi": "NOT_COMPUTED"}
    text = format_state_table(inv)
    assert "2Pi" in text and "NOT_COMPUTED" in text and "at least 1.50 eV above" in text


def test_no_electronic_stage_no_inventory():
    assert build_state_inventory(None) is None
    assert build_state_inventory({"candidates": {}}) is None
