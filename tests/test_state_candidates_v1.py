"""Wigner-Witmer candidates validated against Herzberg's correlation tables."""
from collections import Counter

from openea_benchmark.state_candidates import (
    AtomicState, AtomicTerm, atomic_states, candidate_spin_sectors, enumerate_state_candidates,
    ground_configuration, terms_of_configuration, wigner_witmer,
)


def labels(syms):
    return Counter(s.label for s in syms)


def ground(element, charge=0):
    states = [s for s in atomic_states(element, charge, include_promotions=False)]
    # Hund: highest multiplicity, then highest L is the ground term of the ground configuration
    return max(states, key=lambda s: (s.term.two_s, s.term.l))


def test_atomic_terms_by_microstates():
    assert {t.label for t in terms_of_configuration(ground_configuration(8))} == {"3P", "1D", "1S"}
    assert {t.label for t in terms_of_configuration(ground_configuration(7))} == {"4So", "2Do", "2Po"}
    assert {t.label for t in terms_of_configuration(ground_configuration(26))} >= {"5D", "3H", "1I"}
    assert ground("Cl", -1).term.label == "1S"
    assert ground("Mg", -1).term.label == "2Po"


def test_oh_from_ground_atoms():
    syms = wigner_witmer(ground("O"), ground("H"), identical_nuclei=False)
    assert labels(syms) == Counter({"2Sigma-": 1, "2Pi": 1, "4Sigma-": 1, "4Pi": 1})


def test_cn_from_ground_atoms():
    syms = wigner_witmer(ground("C"), ground("N"), identical_nuclei=False)
    assert labels(syms) == Counter({"2Sigma+": 1, "2Pi": 1, "4Sigma+": 1, "4Pi": 1, "6Sigma+": 1, "6Pi": 1})


def test_o2_from_3p_3p():
    syms = wigner_witmer(ground("O"), ground("O"), identical_nuclei=True, electrons_per_atom=8)
    assert labels(syms) == Counter({
        "1Sigmag+": 2, "1Sigmau-": 1, "1Pig": 1, "1Piu": 1, "1Deltag": 1,
        "3Sigmau+": 2, "3Sigmag-": 1, "3Pig": 1, "3Piu": 1, "3Deltau": 1,
        "5Sigmag+": 2, "5Sigmau-": 1, "5Pig": 1, "5Piu": 1, "5Deltag": 1,
    })


def test_n2_from_4s_4s():
    syms = wigner_witmer(ground("N"), ground("N"), identical_nuclei=True, electrons_per_atom=7)
    assert labels(syms) == Counter({"1Sigmag+": 1, "3Sigmau+": 1, "5Sigmag+": 1, "7Sigmau+": 1})


def test_f2_from_2p_2p():
    syms = wigner_witmer(ground("F"), ground("F"), identical_nuclei=True, electrons_per_atom=9)
    got = labels(syms)
    assert got["1Sigmag+"] == 2 and got["1Sigmau-"] == 1 and got["1Deltag"] == 1
    assert got["3Sigmau+"] == 2 and got["3Sigmag-"] == 1 and got["3Deltau"] == 1
    assert got["1Piu"] == 1 and got["3Piu"] == 1


def test_unlike_states_of_like_atoms_give_both_g_and_u():
    o3p, o1d = ground("O"), AtomicState("O", 0, "x", AtomicTerm(0, 2, 1), "GROUND_AUFBAU")
    got = labels(wigner_witmer(o3p, o1d, identical_nuclei=True, electrons_per_atom=8))
    # Herzberg: 3P + 1D gives two 3Delta (m=(0,2),(1,1)), each as g and u
    assert got["3Deltag"] == got["3Deltau"] == 2 and got["3Phig"] == got["3Phiu"] == 1


def test_candidates_cover_known_ground_states_and_spins():
    def has(a, b, q, label):
        return any(c.symmetry.label == label for c in enumerate_state_candidates(a, b, q))
    assert has("O", "H", 0, "2Pi") and has("O", "H", -1, "1Sigma+")
    assert has("Mg", "H", 0, "2Sigma+") and has("Mg", "H", -1, "1Sigma+")
    assert has("O", "O", 0, "3Sigmag-") and has("O", "O", -1, "2Pig")
    assert has("N", "O", 0, "2Pi") and has("N", "O", -1, "3Sigma-")
    assert has("Al", "O", 0, "2Sigma+") and has("Al", "O", -1, "1Sigma+")
    assert has("C", "N", -1, "1Sigma+") and has("Si", "C", 0, "3Pi")
    assert candidate_spin_sectors("O", "H", 0)[:2] == [1, 3]
