"""Symmetry-block labels from occupied orbitals (abelian subgroups) and constrained SCF."""
import pytest

from openea_benchmark.state_candidates import MolecularSymmetry
from openea_benchmark.symmetry_states import (
    BlockOccupation, OrbitalInfo, build_symmetric_molecule, enumerate_block_occupations, label_from_orbitals,
    orbital_table, run_block_scf,
)


def orb(i, irrep, m, gu=None, e=0.0):
    return OrbitalInfo(i, e, irrep, m, gu)


# O2 valence in D2h: sigma_g (Ag), sigma_u (B1u), pi_u (B2u/B3u), pi_g* (B2g/B3g)
CORE = [orb(0, "Ag", 0, "g"), orb(1, "B1u", 0, "u"), orb(2, "Ag", 0, "g"), orb(3, "B1u", 0, "u"),
        orb(4, "Ag", 0, "g"), orb(5, "B2u", 1, "u"), orb(6, "B3u", 1, "u")]
PIG_X, PIG_Y = orb(7, "B2g", 1, "g"), orb(8, "B3g", 1, "g")


def test_o2_labels():
    assert label_from_orbitals(CORE + [PIG_X, PIG_Y], CORE, like_nuclei=True).label == "3Sigmag-"
    assert label_from_orbitals(CORE + [PIG_X], CORE + [PIG_X], like_nuclei=True) is None  # (pi_gx)^2: MR
    assert label_from_orbitals(CORE + [PIG_X, PIG_Y], CORE + [PIG_Y], like_nuclei=True).label == "2Pig"


def test_heteronuclear_labels():
    core = [orb(0, "A1", 0), orb(1, "A1", 0), orb(2, "A1", 0), orb(3, "B1", 1)]
    piy = orb(4, "B2", 1)
    assert label_from_orbitals(core + [piy], core, like_nuclei=False).label == "2Pi"
    two_pix = [orb(0, "A1", 0), orb(3, "B1", 1), orb(9, "B1", 1)]
    assert label_from_orbitals(two_pix, two_pix, like_nuclei=False) is None
    pi_delta = [orb(0, "A1", 0), orb(3, "B1", 1), orb(9, "A1", 2)]
    assert label_from_orbitals(pi_delta, [orb(0, "A1", 0)], like_nuclei=False) is None


@pytest.fixture(scope="module")
def o2():
    pytest.importorskip("pyscf")
    return build_symmetric_molecule(("O", "O"), 1.21, basis="cc-pvdz", charge=0, two_s=2)


def test_measured_m_and_enumeration_on_o2(o2):
    from pyscf import scf
    ref = scf.ROHF(o2); ref.verbose = 0; ref.kernel()
    table = orbital_table(o2, ref.mo_coeff, ref.mo_energy)
    assert sorted({o.m for o in table[:10]}) == [0, 1]
    occs = enumerate_block_occupations(table, n_alpha=9, n_beta=7, like_nuclei=True)
    labels = {o.symmetry.label for o in occs}
    assert {"3Sigmag-", "3Sigmau+", "3Pig", "3Piu"} <= labels
    assert min(occs, key=lambda o: o.orbital_energy_sum).symmetry.label == "3Sigmag-"


def test_block_scf_o2_ground_state(o2):
    occ = BlockOccupation(MolecularSymmetry(2, 0, "g", "-"), tuple(sorted({
        "Ag": (3, 3), "B1u": (2, 2), "B2u": (1, 1), "B3u": (1, 1), "B2g": (1, 0), "B3g": (1, 0)}.items())), 0.0)
    res = run_block_scf(o2, occ)
    assert res.converged and res.symmetry.label == "3Sigmag-" and res.stable_within_symmetry


def test_block_scf_oh_pi_converges_in_c2v():
    pytest.importorskip("pyscf")
    mol = build_symmetric_molecule(("O", "H"), 0.97, basis="cc-pvdz", charge=0, two_s=1)
    occ = BlockOccupation(MolecularSymmetry(1, 1), (("A1", (3, 3)), ("B1", (1, 1)), ("B2", (1, 0))), 0.0)
    res = run_block_scf(mol, occ)
    assert res.converged and res.symmetry.label == "2Pi"


def test_xy_partner_components_are_not_separate_states():
    table = [orb(0, "A1", 0, e=-20), orb(1, "A1", 0, e=-1.2), orb(2, "A1", 0, e=-0.6),
             orb(3, "B1", 1, e=-0.4), orb(4, "B2", 1, e=-0.4), orb(5, "A1", 0, e=0.1)]
    occs = enumerate_block_occupations(table, n_alpha=5, n_beta=4, like_nuclei=False, keep_per_label=2)
    from openea_benchmark.symmetry_states import swap_xy
    pi = [o.irrep_nelec for o in occs if o.symmetry.label == "2Pi"]
    assert (("A1", (3, 3)), ("B1", (1, 1)), ("B2", (1, 0))) in pi
    assert all(swap_xy(a) != b for a in pi for b in pi)  # no x/y partner kept twice
