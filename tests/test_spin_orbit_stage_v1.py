"""Patch 47: first-order spin-orbit layer (no OpenMolcas run in unit tests)."""
from openea_benchmark.spin_orbit_stage import (
    SpinOrbitStagePolicy, c2_space, component_pair, full_valence_space, identify_input, parse_angmom_z, parse_rassi,
    parse_term_symbol, spin_orbit_rassi_input, state_spin_orbit_shift,
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
    fe_o = full_valence_space(("Fe", "O"), 0)  # 3d metals: 4s 3d, [Ar] core (Patch 55)
    assert fe_o["inactive"] == (6, 2, 2, 0) and fe_o["ras2"] == (5, 2, 2, 1) and fe_o["active_electrons"] == 14
    assert fe_o["rule"] == "FULL_VALENCE_NS_NP_ND"
    assert full_valence_space(("Ag", "H"), 0) is None  # 4d: def2 ECP, fail closed
    assert c2_space(fe_o)["ras2"] == (6, 4) and c2_space(fe_o)["inactive"] == (6, 4)


def test_sigma_states_have_no_first_order_shift(tmp_path):
    res = state_spin_orbit_shift(atoms=("O", "O"), r_angstrom=1.2, charge=0, symmetry_label="3Sigmag-",
                                 irrep_nelec=None, workdir=tmp_path, policy=SpinOrbitStagePolicy())
    assert res["status"] == "FIRST_ORDER_ZERO" and res["delta_so_ev"] == 0.0
    res = state_spin_orbit_shift(atoms=("O", "O"), r_angstrom=1.2, charge=0, symmetry_label="1Deltag",
                                 irrep_nelec=None, workdir=tmp_path, policy=SpinOrbitStagePolicy())
    assert res["status"] == "FIRST_ORDER_ZERO"  # singlet


def test_input_and_parser():
    space = full_valence_space(("Ti", "O"), 0)
    text = identify_input(("Ti", "O"), 1.62, 0, 3, space, 2, 4, SpinOrbitStagePolicy())
    assert "Group = XY" in text and "AngMom" in text and "MEES" in text and "Symmetry\n1" in text
    assert "Ras2\n6 4" in text and "CIRoot\n4 4 1" in text
    assert "Symmetry\n2" in identify_input(("N", "O"), 1.15, 0, 2, full_valence_space(("N", "O"), 0), 1, 4,
                                            SpinOrbitStagePolicy())
    assert "1 2\n1 2" in spin_orbit_rassi_input((1, 2))
    out = ("::    RASSI State    1     Total energy:     -75.49201258322125\n"
           "::    RASSI State    2     Total energy:     -75.49201258322120\n"
           "::    SO-RASSI State    1     Total energy:     -75.49232128971342\n"
           "::    SO-RASSI State    2     Total energy:     -75.49232128971342\n")
    sf, so = parse_rassi(out)
    assert len(sf) == 2 and len(so) == 2 and so[0] < sf[0]


TIO_ANGMOM = """
 PROPERTY: ANGMOM     COMPONENT:   3
 ORIGIN:   0.000000000E+00  0.000000000E+00  0.000000000E+00

  STATE          1                2                3                4

    1        2.866065555E-16  2.000000000E+00  3.252047538E-08 -1.021189758E-07
    2       -2.000000000E+00 -2.569224504E-16  6.568428230E-08 -5.388086025E-07
    3       -3.252047538E-08 -6.568428230E-08  5.246759138E-16  4.455665557E-12
    4        1.021189758E-07  5.388086025E-07 -4.455665557E-12  4.540074275E-16
"""


def test_lz_identification_picks_the_lambda_pair():
    lz = parse_angmom_z(TIO_ANGMOM, 4)
    sf = [-927.8419832001985, -927.8419831920479, -927.7505864703858, -927.6638350832561]
    assert component_pair(sf, lz, 2, 4) == (1, 2)  # TiO X3Delta
    assert component_pair(sf, lz, 1, 4) is None  # no Pi pair among these roots
    assert component_pair([-1.0, -0.9, -1.0, -0.8], lz, 2, 4) is None  # components not degenerate


def test_assembly_adds_soc_and_widens():
    cbs = {"binding_at_cbs": "BOUND", "status": "CLEARED", "ea_cbs_ev": 1.0, "ea_cbs_lower_ev": 0.99,
           "ea_cbs_upper_ev": 1.01}
    nuc = {"status": "CLEARED", "delta_zpe_ev": 0.0, "delta_zpe_half_width_ev": 0.0, "anion_vibrational_binding": "BOUND"}
    so = {"status": "CLEARED", "delta_ea_so_ev": -0.008, "bound_ev": 0.0024}
    res = assemble_adiabatic_ea(cbs=cbs, nuclear=nuc, spin_orbit=so)
    assert res["includes_soc"] and abs(res["ea0_ev"] - 0.992) < 1e-12
    assert abs(res["ea0_upper_ev"] - (1.01 - 0.008 + 0.0024)) < 1e-12
    assert res["verdict"] == "NOT_ISSUED_NO_CV_SR_POST_CCSD_T_LAYERS"
