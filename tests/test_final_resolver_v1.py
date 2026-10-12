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
