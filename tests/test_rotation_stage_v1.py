"""Rotational energy of the lowest rovibronic levels."""
from types import SimpleNamespace

import pytest

from openea_benchmark.rotation_stage import (
    CM1_TO_EV, lowest_level_cm1, pi_regularity, rotational_constant_cm1, run_rotation_stage,
    state_rotation_offset)


def test_hill_van_vleck_levels_of_known_2pi_molecules():
    assert lowest_level_cm1(1, 1, -139.2, 18.9) == pytest.approx(-37.66, abs=0.02)   # OH X2Pi3/2 J=3/2
    assert lowest_level_cm1(1, 1, 28.0, 14.2) == pytest.approx(0.2, abs=1e-9)        # CH X2Pi1/2 J=1/2
    assert lowest_level_cm1(1, 1, 142.8, 7.5) == pytest.approx(-63.9, abs=1e-9)      # SiH


def test_case_b_and_sigma_limits():
    assert lowest_level_cm1(1, 0, 0.0, 10.0) == pytest.approx(10.0)   # 1Pi: B[J(J+1) - 1], J=1
    assert lowest_level_cm1(2, 0, 0.0, 10.0) == pytest.approx(20.0)   # 1Delta: 2B
    assert lowest_level_cm1(0, 2, 0.0, 10.0) == 0.0                   # 3Sigma-: N=0
    assert lowest_level_cm1(1, 2, 500.0, 1.0) == pytest.approx(-499.0, abs=0.01)   # 3Pi0, case (a)


def test_offset_relative_to_the_applied_first_order_shift():
    oh = state_rotation_offset(lam=1, two_s=1, b_cm1=18.9, delta_so_ev=-69.6 * CM1_TO_EV, regularity="INVERTED")
    assert oh["offset_cm1"] == pytest.approx(-37.66 + 69.6, abs=0.05)
    ch = state_rotation_offset(lam=1, two_s=1, b_cm1=14.2, delta_so_ev=-14.0 * CM1_TO_EV, regularity="REGULAR")
    assert ch["offset_cm1"] == pytest.approx(14.2, abs=1e-6)
    unknown = state_rotation_offset(lam=1, two_s=1, b_cm1=14.2, delta_so_ev=-14.0 * CM1_TO_EV, regularity=None)
    assert unknown["bound_cm1"] > ch["bound_cm1"]
    assert state_rotation_offset(lam=0, two_s=2, b_cm1=14.0, delta_so_ev=0.0, regularity=None)["offset_cm1"] == 0.0


def test_pi_regularity_from_the_irrep_occupation():
    assert pi_regularity({"A1": (3, 3), "B1": (1, 0), "B2": (0, 0), "A2": (0, 0)}, 1) == "REGULAR"    # pi^1
    assert pi_regularity({"A1": (3, 3), "B1": (1, 1), "B2": (1, 0), "A2": (0, 0)}, 1) == "INVERTED"   # pi^3
    assert pi_regularity({"A1": (3, 3), "B1": (1, 1), "B2": (1, 0), "A2": (0, 0)}, 2) is None
    assert pi_regularity(None, 1) is None


def test_stage_gives_the_ea_contribution_neutral_minus_anion():
    # OH-like: neutral 2Pi (inverted), anion 1Sigma+
    reqs = [SimpleNamespace(job_id="N", symmetry_label="2Pi",
                            irrep_nelec={"A1": (3, 3), "B1": (1, 1), "B2": (1, 0), "A2": (0, 0)}),
            SimpleNamespace(job_id="A", symmetry_label="1Sigma+", irrep_nelec=None)]
    ea = {"status": "ELECTRONIC_EA_EVALUATED", "selected_neutral_job": "N", "selected_anion_job": "A",
          "candidates": {"neutral": [{"job_id": "N", "r_central_angstrom": 0.97}],
                         "anion": [{"job_id": "A", "r_central_angstrom": 0.96}]}}
    nuclear = {"masses": {"mass_a_u": 15.994915, "mass_b_u": 1.007825},
               "cross_model_identity_checks": {"neutral": {"re_primary_angstrom": 0.9697},
                                               "anion": {"re_primary_angstrom": 0.964}}}
    so = {"status": "CLEARED", "states": {"neutral": {"status": "CLEARED", "delta_so_ev": -0.00827},
                                          "anion": {"status": "FIRST_ORDER_ZERO", "delta_so_ev": 0.0}}}
    rec = run_rotation_stage(electronic_ea=ea, requests=reqs, nuclear=nuclear, spin_orbit=so)
    assert rec["status"] == "CLEARED" and rec["states"]["neutral"]["case"] == "INVERTED"
    assert rotational_constant_cm1(15.994915, 1.007825, 0.9697) == pytest.approx(18.9, abs=0.1)
    assert rec["delta_rot_ev"] == pytest.approx(0.0039, abs=0.0002)
    assert rec["bound_ev"] < 0.5 * rec["delta_rot_ev"]


def test_assembly_includes_the_rotation_layer():
    from openea_benchmark.vibrational_stage import assemble_adiabatic_ea
    cbs = {"binding_at_cbs": "BOUND", "status": "CLEARED", "ea_cbs_ev": 1.0, "ea_cbs_lower_ev": 0.99,
           "ea_cbs_upper_ev": 1.01}
    nuc = {"status": "CLEARED", "delta_zpe_ev": 0.0, "delta_zpe_half_width_ev": 0.0, "anion_vibrational_binding": "BOUND"}
    out = assemble_adiabatic_ea(cbs=cbs, nuclear=nuc, rotation={"status": "CLEARED", "delta_rot_ev": 0.004,
                                                                "bound_ev": 0.001})
    assert out["includes_rotation"] and out["ea0_ev"] == pytest.approx(1.004)
    assert out["ea0_upper_ev"] == pytest.approx(1.015)
    out = assemble_adiabatic_ea(cbs=cbs, nuclear=nuc, rotation={"status": "FAILED"})
    assert not out["includes_rotation"] and out["ea0_ev"] == pytest.approx(1.0)
