"""Symmetry-block labels from irrep occupations and constrained SCF."""
import pytest

from openea_benchmark.state_candidates import MolecularSymmetry
from openea_benchmark.symmetry_states import (
    BlockOccupation, build_symmetric_molecule, enumerate_block_occupations, label_from_irrep_nelec,
    parse_irrep, run_block_scf,
)

O2_CORE = {"A1g": (3, 3), "A1u": (2, 2), "E1ux": (1, 1), "E1uy": (1, 1)}


def test_parse_linear_irreps():
    assert parse_irrep("E1gy").sigma_v_odd and parse_irrep("E1gy").m == 1
    assert not parse_irrep("A1").sigma_v_odd and parse_irrep("A2u").sigma_v_odd


def test_o2_labels_from_counts():
    x = label_from_irrep_nelec({**O2_CORE, "E1gx": (1, 0), "E1gy": (1, 0)}, like_nuclei=True)
    assert x.label == "3Sigmag-"
    closed_pix2 = label_from_irrep_nelec({**O2_CORE, "E1gx": (1, 1), "E1gy": (0, 0)}, like_nuclei=True)
    assert closed_pix2 is None  # (pi_x)^2 mixes 1Sigma_g+ and 1Delta_g -> multi-determinant
    anion = label_from_irrep_nelec({**O2_CORE, "E1gx": (1, 0), "E1gy": (1, 1)}, like_nuclei=True)
    assert anion.label == "2Pig"


def test_heteronuclear_labels():
    oh = label_from_irrep_nelec({"A1": (3, 3), "E1x": (1, 1), "E1y": (1, 0)}, like_nuclei=False)
    assert oh.label == "2Pi"
    alo = label_from_irrep_nelec({"A1": (7, 6), "E1x": (2, 2), "E1y": (2, 2)}, like_nuclei=False)
    assert alo.label == "2Sigma+"
    mixed = label_from_irrep_nelec({"A1": (3, 3), "E1x": (1, 0), "E2x": (1, 0)}, like_nuclei=False)
    assert mixed is None  # Lambda^2 = 1 + 4 is no perfect square


@pytest.fixture(scope="module")
def o2():
    pytest.importorskip("pyscf")
    return build_symmetric_molecule(("O", "O"), 1.21, basis="cc-pvdz", charge=0, two_s=2)


def test_enumeration_contains_o2_ground_and_excited_blocks(o2):
    from pyscf import scf
    ref = scf.ROHF(o2); ref.verbose = 0; ref.kernel()
    from pyscf.scf import hf_symm
    names = dict(zip(o2.irrep_id, o2.irrep_name))
    irreps = [names[i] for i in hf_symm.get_orbsym(o2, ref.mo_coeff)]
    occs = enumerate_block_occupations(irreps, ref.mo_energy, n_alpha=9, n_beta=7, like_nuclei=True)
    labels = {o.symmetry.label for o in occs}
    assert {"3Sigmag-", "3Sigmau+", "3Pig", "3Piu"} <= labels
    lowest = min(occs, key=lambda o: o.orbital_energy_sum)
    assert lowest.symmetry.label == "3Sigmag-"


def test_block_scf_o2_ground_state_is_clean_and_stable(o2):
    occ = BlockOccupation(MolecularSymmetry(2, 0, "g", "-"),
                          tuple(sorted({**O2_CORE, "E1gx": (1, 0), "E1gy": (1, 0)}.items())), 0.0)
    res = run_block_scf(o2, occ)
    assert res.converged and res.symmetry.label == "3Sigmag-"
    assert res.stable_within_symmetry is True
    assert not res.warnings or res.warnings == ["SYMMETRY_BREAKING_INSTABILITY_RECORDED_NOT_FOLLOWED"]
