"""Element rules beyond Kr (heavy-element step 1)."""
from __future__ import annotations

import pytest

from openea_benchmark.heavy_elements import (
    ROUTE_ALL_ELECTRON, ROUTE_PSEUDOPOTENTIAL, core_irreps_c2v, element_route, frozen_orbitals,
    full_valence_space, load_basis, molecule_basis, shell_irreps_c2v, valence_core_orbitals)


def test_shell_irreps_follow_lz_components():
    assert shell_irreps_c2v(0) == (1, 0, 0, 0)
    assert shell_irreps_c2v(1) == (1, 1, 1, 0)
    assert shell_irreps_c2v(2) == (2, 1, 1, 1)
    assert shell_irreps_c2v(3) == (2, 2, 2, 1)


def test_core_irreps_match_the_existing_light_element_table():
    from openea_benchmark.spin_orbit_stage import CORE_ORBITALS_BY_COUNT
    for n, irreps in CORE_ORBITALS_BY_COUNT.items():
        assert core_irreps_c2v(n) == irreps
    assert core_irreps_c2v(18) == (9, 4, 4, 1)      # [Kr]
    assert core_irreps_c2v(39) == (17, 9, 9, 4)     # [Xe]4f14 5d10
    assert core_irreps_c2v(7) is None               # not a whole-shell core


@pytest.mark.parametrize("z, n", [(1, 0), (8, 1), (17, 5), (22, 9), (35, 14), (39, 18), (47, 18),
                                  (53, 23), (55, 27), (79, 34), (83, 39), (57, None), (92, None)])
def test_valence_frozen_core(z, n):
    assert valence_core_orbitals(z) == n


def test_full_valence_space_light_elements_as_before():
    # values of the pre-heavy-element rule (Patch 55)
    assert full_valence_space(("Cl", "O"), 0) == {"inactive": (4, 1, 1, 0), "ras2": (4, 2, 2, 0),
                                                  "active_electrons": 13, "rule": "FULL_VALENCE_NS_NP"}
    assert full_valence_space(("Ti", "O"), 0) == {"inactive": (6, 2, 2, 0), "ras2": (5, 2, 2, 1),
                                                  "active_electrons": 10, "rule": "FULL_VALENCE_NS_NP_ND"}
    assert full_valence_space(("Br", "H"), -1)["inactive"] == (7, 3, 3, 1)


def test_full_valence_space_heavy():
    agO = full_valence_space(("Ag", "O"), -1)
    assert agO["inactive"] == (10, 4, 4, 1) and agO["ras2"] == (5, 2, 2, 1)
    assert agO["active_electrons"] == 11 + 6 + 1 and agO["rule"] == "FULL_VALENCE_NS_NP_ND"
    assert full_valence_space(("I", "O"), 0)["ras2"] == (4, 2, 2, 0)
    assert full_valence_space(("La", "O"), 0) is None


def test_routes():
    assert element_route("O")["route"] == ROUTE_ALL_ELECTRON
    for el, nelec, frozen in (("Ag", 28, 4), ("I", 28, 9), ("Au", 60, 4), ("Bi", 60, 9)):
        r = element_route(el)
        assert r["route"] == ROUTE_PSEUDOPOTENTIAL and r["ecp_electrons"] == nelec
        assert frozen_orbitals(el, nelec) == frozen
    assert element_route("Cs")["route"] is None     # no aug-cc-pVXZ-PP series
    assert element_route("Ce")["reason"] == "OPEN_F_SHELL_OR_BEYOND_RN"


def test_ecp_basis_never_loses_its_ecp():
    loaded = load_basis("def2-TZVPPD", "Ag")
    assert loaded.ecp_electrons == 28
    kw = molecule_basis(("Ag", "O"), {"Ag": "aug-cc-pVTZ-PP", "O": "aug-cc-pVTZ"})
    assert set(kw["ecp"]) == {"Ag"}
    from pyscf import gto
    mol = gto.M(atom="Ag 0 0 0; O 0 0 2.0", basis=kw["basis"], ecp=kw["ecp"], charge=-1, spin=0, verbose=0)
    assert mol.nelectron == 47 - 28 + 8 + 1


def test_basis_names_and_fail_closed_shells():
    from openea_benchmark.heavy_elements import basis_name_for, mole_basis_kwargs
    assert basis_name_for("Ag", "aug-cc-pvtz") == "aug-cc-pvtz-PP"
    assert basis_name_for("Ag", "def2-TZVPPD") == "def2-TZVPPD"
    assert basis_name_for("O", "aug-cc-pvtz") == "aug-cc-pvtz"
    assert mole_basis_kwargs(("O", "H"), "def2-TZVPPD") == {"basis": "def2-TZVPPD"}  # light: untouched
    with pytest.raises(ValueError):
        mole_basis_kwargs(("Ag", "O"), {"Ag": [[0, [1.0, 1.0]]], "O": "def2-TZVPPD"})
    d_aug = mole_basis_kwargs(("Au", "Au"), "d-aug-cc-pVQZ-PP")
    assert d_aug["ecp"]["Au"][0] == 60


def test_all_electron_member_keeps_the_ecp_core_frozen():
    from pyscf import gto
    from openea_benchmark.heavy_elements import implicit_core_orbitals, mole_basis_kwargs
    ae = gto.M(atom="Ag 0 0 0; H 0 0 1.6", **mole_basis_kwargs(("Ag", "H"), {"Ag": "dyall-av3z", "H": "aug-cc-pvtz"}), verbose=0)
    pp = gto.M(atom="Ag 0 0 0; H 0 0 1.6", **mole_basis_kwargs(("Ag", "H"), {"Ag": "aug-cc-pVTZ-PP", "H": "aug-cc-pvtz"}), verbose=0)
    assert implicit_core_orbitals(ae) == 14 and implicit_core_orbitals(pp) == 0
    assert ae.nelectron - 2 * 14 == pp.nelectron


def test_pyscf_valence_space_removes_ecp_cores():
    from pyscf import gto
    from openea_benchmark.heavy_elements import mole_basis_kwargs, pyscf_valence_space
    space = pyscf_valence_space(("Au", "Au"), 0, "def2-TZVPPD")
    assert space["inactive"] == (4, 2, 2, 0) and space["active_electrons"] == 22
    mol = gto.M(atom="Au 0 0 0; Au 0 0 2.5", **mole_basis_kwargs(("Au", "Au"), "def2-TZVPPD"), verbose=0)
    assert 2 * sum(space["inactive"]) + space["active_electrons"] == mol.nelectron
    assert pyscf_valence_space(("Cl", "O"), 0, "def2-TZVPPD") == full_valence_space(("Cl", "O"), 0)


def test_stage_basis_rules_beyond_kr():
    from openea_benchmark.cbs_stage import element_basis
    from openea_benchmark.core_valence_basis import core_orbital_count, cv_element_series, sr_element_series
    assert element_basis("Ag", 4)[0] == "aug-cc-pVQZ-PP"
    assert element_basis("Ag", 5, daug=True)[0] == "d-aug-cc-pV5Z-PP"
    assert core_orbital_count("Ag") == 4 and core_orbital_count("I") == 9
    assert cv_element_series("Au", (3, 4))[4][0] == "aug-cc-pwCVQZ-PP"
    assert sr_element_series("Au", (3, 4))[3][0] == ("dyall-av3z", "aug-cc-pVTZ-PP")
    assert element_basis("O", 4)[0] == "aug-cc-pvqz"  # light elements unchanged


def test_closed_d_shell_promotion():
    from openea_benchmark.state_candidates import atomic_states
    pd = {(s.configuration.split()[-1], s.term.label) for s in atomic_states("Pd", 0)}
    assert ("4d10", "1S") in pd


def test_unsupported_element_fails_before_computing(tmp_path):
    import json
    from openea_benchmark.workflow import main
    assert main(["--mode", "symmetry-auto", "--atoms", "Ce", "O", "--output", str(tmp_path)]) == 2
    verdict = json.loads((tmp_path / "CeO" / "final_verdict.json").read_text())
    assert verdict["final_verdict"] == "ELEMENT_NOT_SUPPORTED"
