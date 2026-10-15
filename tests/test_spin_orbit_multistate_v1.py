"""Multistate spin-orbit layer (heavy-element step 2): parsing and bookkeeping."""
from openea_benchmark.spin_orbit_multistate import (applies, multistate_input, parse_so_weights, reference_roots,
                                                    spin_blocks, MultistateSpinOrbitPolicy)

RASSI_TABLE = """ Weights of the five most important spin-orbit-free states for each spin-orbit state.

 SO State  Total energy (au)           Spin-free states, spin, and weights
 -------------------------------------------------------------------------------------------------------
    1        -0.005431       1 0.5  0.4945    2 0.5  0.4945   13 1.5  0.0053    5 0.5  0.0013    6 0.5  0.0013
    2        -0.005431       1 0.5  0.4945    2 0.5  0.4945   13 1.5  0.0053    5 0.5  0.0013    6 0.5  0.0013

"""


def test_weights_table_io():  # IO X2Pi3/2, SO-XMS-CASPT2 run 2026-10-15
    w = parse_so_weights(RASSI_TABLE)
    assert w[1][0] == (1, 0.5, 0.4945) and len(w[1]) == 5
    assert abs(sum(x[2] for x in w[1] if x[0] in (1, 2)) - 0.989) < 1e-9


def test_reference_roots():
    assert reference_roots([-1.0, -1.0, -0.9, -0.8], 1, 2e-5) == [0, 1]       # Pi pair
    assert reference_roots([-1.0, -0.95, -0.95], 0, 2e-5) == [0]               # Sigma single
    assert reference_roots([-1.0, -0.95, -0.95], 2, 2e-5) == [1, 2]            # Delta above a Sigma
    assert reference_roots([-1.0, -0.9], 1, 2e-5) is None


def test_spin_blocks():
    assert spin_blocks(2, 13, 8) == [2, 4]          # IO: doublet + quartet (no 0)
    assert spin_blocks(1, 12, 8) == [1, 3]
    assert spin_blocks(3, 12, 8) == [1, 3, 5]


def test_applies_only_beyond_kr():
    assert applies(("I", "O")) and applies(("Cs", "H"))
    assert not applies(("Cl", "O")) and not applies(("Br", "O"))


def test_input_has_one_jobiph_per_block():
    from openea_benchmark.spin_orbit_stage import full_valence_space
    text = multistate_input(("I", "O"), 1.868, 0, full_valence_space(("I", "O"), 0),
                            [(2, 2, 4), (2, 1, 4), (4, 2, 4), (4, 1, 1)], MultistateSpinOrbitPolicy())
    assert text.count("&CASPT2") == 4 and "XMultistate" in text and "MultiState" in text
    assert "Nr of JobIph\n4 4 4 4 1" in text and "EJob" in text and ">>> COPY $Project.JobMix JOB004" in text
