"""Patch 50: multireference nuclear curve fallback (no CASSCF in unit tests)."""
from types import SimpleNamespace

import numpy as np

from openea_benchmark.vibrational_stage import ZPEStagePolicy
from openea_benchmark.mr_nuclear_curve import MRCurvePolicy, continuity_problems, mr_zpe_fallback, target_c2v_irrep
from openea_benchmark.vibrational_stage import main_isotopologue_masses


def test_target_irreps():
    assert target_c2v_irrep("2Sigma+") == "A1" and target_c2v_irrep("3Sigma-") == "A2"
    assert target_c2v_irrep("3Sigmag-") == "A2" and target_c2v_irrep("2Pig") == "B1"
    assert target_c2v_irrep("1Delta") is None


def morse(r, re=1.62, de=0.2, a=2.0):
    return de * (1 - np.exp(-a * (np.asarray(r) - re))) ** 2


def test_continuity_detects_kink_and_occupation_jump():
    r = [1.5 + 0.025 * i for i in range(12)]
    e = list(morse(r))
    occ = [[1.95, 1.0, 0.05]] * len(r)
    assert continuity_problems(r, e, occ, MRCurvePolicy()) == []
    e_bad = list(e)
    e_bad[6] -= 0.002
    occ_bad = [list(o) for o in occ]
    occ_bad[7] = [1.7, 1.0, 0.3]
    problems = continuity_problems(r, e_bad, occ_bad, MRCurvePolicy())
    assert any("kink" in p for p in problems) and any("occupation" in p for p in problems)


def test_fallback_assembles_delta_zpe_with_model_term(tmp_path):
    r = [1.30 + 0.025 * i for i in range(30)]

    def curve_fn(*, charge, r_center, **_):
        re, a = (1.62, 2.0) if charge == 0 else (1.66, 1.9)
        return {"status": "COMPLETED", "r": r, "e": list(morse(r, re=re, a=a)), "method": "SC-NEVPT2/CASSCF",
                "basis": "def2-TZVPPD"}

    def loop(re, a, charge):  # single-reference curve with slightly different curvature
        rr = [re + 0.01 * i for i in range(-4, 5)]
        return SimpleNamespace(requests=[SimpleNamespace(r_angstrom=re, atoms=("Al", "O"), charge=charge, spin_2s=1,
                                                         irrep_nelec={}, symmetry_label="2Sigma+")],
                               results=[SimpleNamespace(high_level_checkpoint_path="/x")],
                               final_pec=SimpleNamespace(points=[SimpleNamespace(r_angstrom=x, energy_hartree=y)
                                                                 for x, y in zip(rr, morse(rr, re=re, a=a))]))
    ea = {"selected_neutral_job": "N", "selected_anion_job": "A",
          "candidates": {"neutral": [{"job_id": "N", "r_central_angstrom": 1.62}],
                         "anion": [{"job_id": "A", "r_central_angstrom": 1.66}]}}
    loops = {"N": loop(1.62, 2.02, 0), "A": loop(1.66, 1.9, -1)}
    res = mr_zpe_fallback(electronic_ea=ea, loops=loops, masses=main_isotopologue_masses(("Al", "O")), channels=(),
                          nuclear_settings=ZPEStagePolicy().nuclear_settings(), output_dir=tmp_path, curve_fn=curve_fn)
    assert res["status"] == "CLEARED", res
    assert res["delta_zpe_ev"] > 0  # stiffer neutral
    assert res["model_terms_relative_frequency"]["neutral"] > res["model_terms_relative_frequency"]["anion"]
