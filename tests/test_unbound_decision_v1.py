"""The sign of the EA is decided on EA_0 (all corrections), not on the electronic CBS EA,
unless the CBS level alone is decisive (continuum anion or far below zero)."""
import pytest

from openea_benchmark.cbs_stage import CBSStagePolicy, _binding
from openea_benchmark.final_resolver import CBS_NEGATIVE_WITHIN_CORRECTION_MARGIN, resolve_final_verdict
from openea_benchmark.mr_term_gap import apply_mr_term_gaps
from openea_benchmark.vibrational_stage import assemble_adiabatic_ea

EA = {"status": "ELECTRONIC_EA_EVALUATED",
      "high_level_competition": {"neutral": {"status": "CLEARED"}, "anion": {"status": "CLEARED"}}}
LAYERS = dict(core_valence={"status": "CORE_VALENCE_CLEARED", "delta_cv_ev": 0.0, "delta_cv_bound_ev": 0.001},
              scalar_relativity={"status": "SCALAR_RELATIVITY_CLEARED", "delta_sr_ev": 0.0,
                                 "delta_sr_bound_ev": 0.001},
              spin_orbit={"status": "CLEARED", "delta_ea_so_ev": 0.0, "bound_ev": 0.0},
              post_ccsd_t={"status": "CLEARED", "delta_post_ev": 0.0, "bound_ev": 0.002})


def near_cbs(ea, lower, upper, reason=CBS_NEGATIVE_WITHIN_CORRECTION_MARGIN):
    return {"status": "NEED_MORE_EVIDENCE_BINDING", "binding_at_cbs": "UNDETERMINED", "binding_reason": reason,
            "cbs_model_status": "CLEARED", "ea_cbs_ev": ea, "ea_cbs_lower_ev": lower, "ea_cbs_upper_ev": upper}


def nuclear(dz, hw=0.002, vib="BOUND"):
    return {"status": "CLEARED", "delta_zpe_ev": dz, "delta_zpe_half_width_ev": hw, "anion_vibrational_binding": vib}


def test_negative_cbs_interval_is_decisive_only_beyond_the_margin():
    policy = CBSStagePolicy()
    assert _binding(-0.80, -0.81, -0.79, 0.002, policy) == ("UNBOUND", "CBS_INTERVAL_NEGATIVE")
    assert _binding(-0.02, -0.03, -0.01, 0.002, policy) == ("UNDETERMINED", CBS_NEGATIVE_WITHIN_CORRECTION_MARGIN)
    assert _binding(0.01, -0.01, 0.03, 0.002, policy) == ("UNDETERMINED", "CBS_INTERVAL_CONTAINS_ZERO")
    # continuum-discretizing anion (verify_v1 CO: diffuse correction 0.82 eV): no localized state to correct
    assert _binding(-0.37, -0.39, -0.35, 0.82, policy)[0] == "UNBOUND"


def test_zero_point_term_can_bind_an_anion_that_is_unbound_at_cbs():
    # EA_e = -0.02 eV, Delta ZPE = +0.05 eV: EA_0 = +0.03 eV, interval above zero
    adi = assemble_adiabatic_ea(cbs=near_cbs(-0.02, -0.025, -0.015), nuclear=nuclear(0.05), **LAYERS)
    assert adi["status"] == "CLEARED" and adi["binding"] == "BOUND"
    assert adi["ea0_ev"] == pytest.approx(0.03)
    verdict = resolve_final_verdict(electronic_ea=EA, cbs=near_cbs(-0.02, -0.025, -0.015), adiabatic=adi,
                                    mr_competitors=None)
    assert verdict["outcome"] == "EA" and verdict["ea_ev"] == pytest.approx(0.03)


def test_complete_ea0_interval_below_zero_is_unbound():
    # still unbound after the corrections: UNBOUND from EA_0, not UNRESOLVED
    cbs = near_cbs(-0.20, -0.21, -0.19)
    adi = assemble_adiabatic_ea(cbs=cbs, nuclear=nuclear(0.03), **LAYERS)
    assert adi["status"] == "ANION_UNBOUND" and adi["ea0_upper_ev"] < 0.0
    verdict = resolve_final_verdict(electronic_ea=EA, cbs=cbs, adiabatic=adi, mr_competitors=None)
    assert verdict["outcome"] == "UNBOUND" and verdict["reason"].startswith("EA0_INTERVAL_NEGATIVE")
    # bound at CBS, pulled below zero by the corrections (stiffer anion): also UNBOUND
    cleared = {"status": "CLEARED", "binding_at_cbs": "BOUND", "binding_reason": "CBS_INTERVAL_POSITIVE",
               "ea_cbs_ev": 0.010, "ea_cbs_lower_ev": 0.009, "ea_cbs_upper_ev": 0.011}
    adi = assemble_adiabatic_ea(cbs=cleared, nuclear=nuclear(-0.03), **LAYERS)
    assert adi["status"] == "ANION_UNBOUND"
    assert resolve_final_verdict(electronic_ea=EA, cbs=cleared, adiabatic=adi,
                                 mr_competitors=None)["outcome"] == "UNBOUND"


def test_ea0_unbound_needs_every_correction_and_a_closed_state_question():
    cbs = near_cbs(-0.20, -0.21, -0.19)
    missing = assemble_adiabatic_ea(cbs=cbs, nuclear=nuclear(0.03),
                                    **{**LAYERS, "post_ccsd_t": {"status": "EXECUTION_BLOCKED"}})
    verdict = resolve_final_verdict(electronic_ea=EA, cbs=cbs, adiabatic=missing, mr_competitors=None)
    assert verdict["outcome"] == "UNRESOLVED"
    adi = assemble_adiabatic_ea(cbs=cbs, nuclear=nuclear(0.03), **LAYERS)
    open_anion = resolve_final_verdict(electronic_ea=EA, cbs=cbs, adiabatic=adi,
                                       mr_competitors={"status": "UNRESOLVED_MR_COMPETITOR"},
                                       mr_only_labels=["q=-1 2Deltag"])
    assert open_anion["outcome"] == "UNRESOLVED"
    open_neutral = resolve_final_verdict(electronic_ea=EA, cbs=cbs, adiabatic=adi,
                                         mr_competitors={"status": "UNRESOLVED_MR_COMPETITOR"},
                                         mr_only_labels=["q=0 1Deltag"])
    assert open_neutral["outcome"] == "UNBOUND" and open_neutral["warnings"]


def test_cbs_status_other_than_the_sign_still_blocks():
    diffuse = near_cbs(0.02, -0.01, 0.05, reason="DIFFUSE_CORRECTION_ABOVE_CONTINUUM_THRESHOLD")
    adi = assemble_adiabatic_ea(cbs=diffuse, nuclear=nuclear(0.05), **LAYERS)
    assert adi["status"] == "NEED_MORE_EVIDENCE_BINDING"
    assert resolve_final_verdict(electronic_ea=EA, cbs=diffuse, adiabatic=adi,
                                 mr_competitors=None)["outcome"] == "UNRESOLVED"
    order = {"status": "UNRESOLVED_STATE_ORDER_AT_CBS", "binding_at_cbs": "BOUND", "ea_cbs_ev": 1.0,
             "ea_cbs_lower_ev": 0.99, "ea_cbs_upper_ev": 1.01}
    adi = assemble_adiabatic_ea(cbs=order, nuclear=nuclear(0.0), **LAYERS)
    assert adi["status"] == "UNRESOLVED_STATE_ORDER_AT_CBS"


def test_mr_ground_state_decides_the_sign_again():
    # SR states: EA_0 below zero; an anion MR state 0.1 eV below the SR anion binds it
    adi = assemble_adiabatic_ea(cbs=near_cbs(-0.05, -0.055, -0.045), nuclear=nuclear(0.0), **LAYERS)
    assert adi["status"] == "ANION_UNBOUND"
    gaps = {"gaps": [{"role": "anion", "label": "q=-1 1Sigma+", "status": "COMPLETED", "gap0_ev": -0.10,
                      "gap0_lower_ev": -0.11, "gap0_upper_ev": -0.09}]}
    out = apply_mr_term_gaps(adi, gaps)
    assert out["status"] == "CLEARED" and out["binding"] == "BOUND" and out["ea0_ev"] == pytest.approx(0.05)
