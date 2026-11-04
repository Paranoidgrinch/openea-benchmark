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


def test_hund_case_c_widens_the_bound_instead_of_failing(monkeypatch, tmp_path):
    import openea_benchmark.spin_orbit_multistate as ms
    # 2Pi reference (roots 1, 2 of its block); lowest SO level only 40 % reference.
    def fake_molcas(workdir, inp, policy, name):
        return "text"
    def blocks_sf(n_per, n_blocks, ref):
        e = [-1.0 + 0.01 * k for k in range(n_per * n_blocks)]
        return e
    monkeypatch.setattr(ms, "_run_molcas", fake_molcas)
    monkeypatch.setattr(ms, "full_valence_space", lambda atoms, charge: {"inactive": (5, 2, 2, 0), "ras2": (2, 1, 1, 0), "active_electrons": 3})
    monkeypatch.setattr(ms, "spin_blocks", lambda mult, ne, no: [mult])
    sf = [-1.0, -1.0] + [-0.9 + 0.01 * k for k in range(10)]
    monkeypatch.setattr(ms, "parse_rassi", lambda text: (sf, [-1.02, -1.02, -0.99]))
    monkeypatch.setattr(ms, "reference_roots", lambda e, lam, tol: [0, 1])
    monkeypatch.setattr(ms, "parse_so_weights", lambda text: {1: [(1, 0.0, 0.2), (2, 0.0, 0.2), (3, 0.0, 0.6)]})
    rec = ms.state_multistate_shift(atoms=("Bi", "Bi"), r_angstrom=2.66, charge=0, symmetry_label="2Pi",
                                    workdir=tmp_path)
    p = ms.MultistateSpinOrbitPolicy()
    assert rec["status"] == "CLEARED_HUND_CASE_C" and rec["delta_so_ev"] is not None
    expected = p.relative_bound * abs(rec["delta_so_ev"]) + 0.6 * p.mixing_gap_ev * 2.0
    assert abs(rec["bound_ev"] - expected) < 1e-9
