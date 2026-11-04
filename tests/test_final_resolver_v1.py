"""Patch 52: final verdict from the layer results."""
from openea_benchmark.final_resolver import resolve_final_verdict

EA = {"status": "ELECTRONIC_EA_EVALUATED",
      "high_level_competition": {"neutral": {"status": "CLEARED"}, "anion": {"status": "CLEARED"}}}
CBS = {"status": "CLEARED", "multireference_exceeded": ["neutral:T1"]}
ADI = {"status": "CLEARED", "ea0_ev": 1.83, "ea0_lower_ev": 1.82, "ea0_upper_ev": 1.84,
       "includes_core_valence": True, "includes_scalar_relativity": True, "includes_soc": True,
       "includes_post_ccsd_t": True}


def test_everything_cleared_gives_ea_with_warning():
    res = resolve_final_verdict(electronic_ea=EA, cbs=CBS, adiabatic=ADI, mr_competitors={"status": "CLEARED"},
                                mr_only_labels=["q=0 2Delta"])
    assert res["outcome"] == "EA" and res["ea_ev"] == 1.83 and res["warnings"]


def test_missing_correction_and_open_mr_state_block():
    adi = {**ADI, "includes_post_ccsd_t": False, "post_ccsd_t_status": "QUADRUPLES_NOT_AFFORDABLE"}
    res = resolve_final_verdict(electronic_ea=EA, cbs=CBS, adiabatic=adi,
                                mr_competitors={"status": "UNRESOLVED_MR_COMPETITOR"}, mr_only_labels=["q=0 1Delta"])
    assert res["outcome"] == "UNRESOLVED"
    assert any("post-CCSD(T)" in b for b in res["blocking"]) and any("MR-only" in b for b in res["blocking"])


def test_unbound_only_with_closed_state_question():
    cbs = {"status": "ANION_UNBOUND_AT_CBS", "binding_reason": "CONTINUUM"}
    assert resolve_final_verdict(electronic_ea=EA, cbs=cbs, adiabatic=None, mr_competitors=None)["outcome"] == "UNBOUND"
    ea = {**EA, "high_level_competition": {"neutral": {"status": "UNRESOLVED"}, "anion": {"status": "CLEARED"}}}
    assert resolve_final_verdict(electronic_ea=ea, cbs=cbs, adiabatic=None, mr_competitors=None)["outcome"] == "UNRESOLVED"


def test_pending_mr_competitors_need_the_mr_layer():
    ea = {**EA, "high_level_competition": {"neutral": {"status": "CLEARED"},
                                           "anion": {"status": "PENDING_MR_COMPETITORS"}}}
    res = resolve_final_verdict(electronic_ea=ea, cbs=CBS, adiabatic=ADI, mr_competitors={"status": "CLEARED"},
                                mr_only_labels=["q=-1 2Sigmag+"])
    assert res["outcome"] == "EA"
    res = resolve_final_verdict(electronic_ea=ea, cbs=CBS, adiabatic=ADI,
                                mr_competitors={"status": "UNRESOLVED_MR_COMPETITOR"}, mr_only_labels=["q=-1 2Sigmag+"])
    assert res["outcome"] == "UNRESOLVED" and any("MR-only" in b for b in res["blocking"])
    res = resolve_final_verdict(electronic_ea=ea, cbs=CBS, adiabatic=ADI, mr_competitors=None)
    assert res["outcome"] == "UNRESOLVED" and any("MR layer not run" in b for b in res["blocking"])


def test_quadruples_allowance_is_reported_as_a_warning():
    adi = {**ADI, "post_ccsd_t_status": "CLEARED_WITH_QUADRUPLES_ALLOWANCE",
           "post_ccsd_t_quadruples_allowance_ev": 0.027}
    res = resolve_final_verdict(electronic_ea=EA, cbs=CBS, adiabatic=adi, mr_competitors=None)
    assert res["outcome"] == "EA" and any("quadruples not computed" in w and "0.027" in w for w in res["warnings"])


def test_same_state_twins_keep_the_label_of_their_determinant():
    from openea_benchmark.electronic_ea_stage import _competition
    from openea_benchmark.cbs_stage import _states_to_compute, CBSStagePolicy
    def cand(job, e, match):
        return {"job_id": job, "charge": 0, "spin_2s": 1, "equilibrium_status": "RESOLVED", "branch_of": None,
                "energy_central_hartree": e, "energy_lower_hartree": e - 1e-5, "energy_upper_hartree": e + 1e-5,
                "r_central_angstrom": 1.8324, "label_matches_determinant": match}
    delta = cand("LaO__q+0__2S1__q+0__2Delta__c68", -8565.94005160, False)  # converged onto 2Sigma+
    sigma = cand("LaO__q+0__2S1__q+0__2Sigma+__c68", -8565.94005154, True)
    comp = _competition("neutral", [delta, sigma], 0.3)
    assert comp["status"] == "CLEARED" and comp["selected_job_id"] == sigma["job_id"]
    ea = {"selected_neutral_job": sigma["job_id"], "selected_anion_job": "A",
          "candidates": {"neutral": [delta, sigma], "anion": [{**cand("A", -8565.97, True), "charge": -1}]}}
    assert [s["job_id"] for s in _states_to_compute(ea, CBSStagePolicy()) if s["role"] == "neutral"] == [sigma["job_id"]]
