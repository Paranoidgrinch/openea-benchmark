"""Patch 47: first-order spin-orbit layer (no OpenMolcas run in unit tests)."""
from openea_benchmark.spin_orbit_stage import (
    SpinOrbitStagePolicy, full_valence_space, molcas_input, parse_rassi, parse_term_symbol, state_spin_orbit_shift,
)
from openea_benchmark.vibrational_stage import assemble_adiabatic_ea


def test_term_symbols():
    assert parse_term_symbol("2Pi") == (2, 1) and parse_term_symbol("3Sigmag-") == (3, 0)
    assert parse_term_symbol("2Pig") == (2, 1) and parse_term_symbol("1Deltau") == (1, 2)
    assert parse_term_symbol("weird") is None


def test_full_valence_space_rule():
    assert full_valence_space(("O", "H"), 0) == {"inactive": (1, 0, 0, 0), "ras2": (3, 1, 1, 0),
                                                 "active_electrons": 7, "rule": "FULL_VALENCE_NS_NP"}
    cl_o = full_valence_space(("Cl", "O"), 0)
    assert cl_o["inactive"] == (4, 1, 1, 0) and cl_o["ras2"] == (4, 2, 2, 0) and cl_o["active_electrons"] == 13
    assert full_valence_space(("Fe", "H"), 0) is None


def test_sigma_states_have_no_first_order_shift(tmp_path):
    res = state_spin_orbit_shift(atoms=("O", "O"), r_angstrom=1.2, charge=0, symmetry_label="3Sigmag-",
                                 irrep_nelec=None, workdir=tmp_path, policy=SpinOrbitStagePolicy())
    assert res["status"] == "FIRST_ORDER_ZERO" and res["delta_so_ev"] == 0.0
    res = state_spin_orbit_shift(atoms=("O", "O"), r_angstrom=1.2, charge=0, symmetry_label="1Deltag",
                                 irrep_nelec=None, workdir=tmp_path, policy=SpinOrbitStagePolicy())
    assert res["status"] == "FIRST_ORDER_ZERO"  # singlet


def test_input_and_parser():
    text = molcas_input(("O", "H"), 0.97, 0, 2, full_valence_space(("O", "H"), 0), SpinOrbitStagePolicy())
    assert "AMFI" in text and text.count("&RASSCF") == 2 and "Inactive\n1 0 0 0" in text and "Ras2\n3 1 1 0" in text
    out = ("::    RASSI State    1     Total energy:     -75.49201258322125\n"
           "::    RASSI State    2     Total energy:     -75.49201258322120\n"
           "::    SO-RASSI State    1     Total energy:     -75.49232128971342\n"
           "::    SO-RASSI State    2     Total energy:     -75.49232128971342\n")
    sf, so = parse_rassi(out)
    assert len(sf) == 2 and len(so) == 2 and so[0] < sf[0]


def test_assembly_adds_soc_and_widens():
    cbs = {"binding_at_cbs": "BOUND", "status": "CLEARED", "ea_cbs_ev": 1.0, "ea_cbs_lower_ev": 0.99,
           "ea_cbs_upper_ev": 1.01}
    nuc = {"status": "CLEARED", "delta_zpe_ev": 0.0, "delta_zpe_half_width_ev": 0.0, "anion_vibrational_binding": "BOUND"}
    so = {"status": "CLEARED", "delta_ea_so_ev": -0.008, "bound_ev": 0.0024}
    res = assemble_adiabatic_ea(cbs=cbs, nuclear=nuc, spin_orbit=so)
    assert res["includes_soc"] and abs(res["ea0_ev"] - 0.992) < 1e-12
    assert abs(res["ea0_upper_ev"] - (1.01 - 0.008 + 0.0024)) < 1e-12
    assert res["verdict"] == "NOT_ISSUED_NO_CV_SR_POST_CCSD_T_LAYERS"
