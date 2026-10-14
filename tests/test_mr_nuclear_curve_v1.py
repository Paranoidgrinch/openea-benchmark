"""Patch 50: multireference nuclear curve fallback (no CASSCF in unit tests)."""
from types import SimpleNamespace

import numpy as np

from openea_benchmark.vibrational_stage import ZPEStagePolicy
from openea_benchmark.mr_nuclear_curve import (MRCurvePolicy, continuity_problems, mr_vibrational_binding,
                                              mr_zpe_fallback, target_c2v_irrep)
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
    assert res["anion_vibrational_binding"] == "UNRESOLVED"  # no electronic margin supplied
    ea["anion_binding"] = {"status": "BOUND", "margin_hartree": 0.232}
    res = mr_zpe_fallback(electronic_ea=ea, loops=loops, masses=main_isotopologue_masses(("Al", "O")), channels=(),
                          nuclear_settings=ZPEStagePolicy().nuclear_settings(), output_dir=tmp_path, curve_fn=curve_fn)
    assert res["anion_vibrational_binding"] == "BOUND", res["vibrational_binding"]


def test_vibrational_binding_from_electronic_margin():
    # AlO-: 0.232 Eh electronic margin, ZPE 0.059 eV -> deep well, bound
    assert mr_vibrational_binding({"status": "BOUND", "margin_hartree": 0.232}, 0.059, 1e-4)["status"] == "BOUND"
    # margin smaller than 2 ZPE -> not decidable from these inputs
    weak = mr_vibrational_binding({"status": "BOUND", "margin_hartree": 0.003}, 0.05, 1e-4)
    assert weak["status"] == "UNRESOLVED" and weak["v0_margin_lower_ev"] < 0
    assert mr_vibrational_binding({"status": "UNBOUND"}, 0.05, 1e-4)["status"] == "UNBOUND"
    assert mr_vibrational_binding(None, 0.05, 1e-4)["status"] == "UNRESOLVED"


def _fake_point(atoms, r, charge, spin_2s, irrep_nelec, wfnsym, space, basis, mo_guess, mol_guess, chk, policy,
                projected=None):
    return "mol", "mo", float(morse(r)), np.array([1.95, 1.0, 0.05])


def test_cached_curve_is_extended_to_larger_clearance(tmp_path, monkeypatch):
    import openea_benchmark.mr_nuclear_curve as mod
    calls = []
    monkeypatch.setattr(mod, "_point", lambda *a, **k: calls.append(a[1]) or _fake_point(*a, **k))
    kw = dict(atoms=("Al", "O"), charge=0, spin_2s=1, irrep_nelec={"A1": (5, 4)}, label="2Sigma+", r_center=1.62,
              chk="/x", policy=MRCurvePolicy(), cache=tmp_path / "neutral.json")
    first = mod.mr_state_curve(**kw)
    assert first["status"] == "COMPLETED" and first["clearance_ev"] == 0.6
    n_first = len(calls)
    assert mod.mr_state_curve(**kw)["r"] == first["r"] and len(calls) == n_first  # cache hit
    wide = mod.mr_state_curve(**kw, clearance_ev=1.2)
    assert wide["status"] == "COMPLETED" and wide["clearance_ev"] == 1.2
    assert wide["r"][0] < first["r"][0] and wide["r"][-1] > first["r"][-1]
    assert set(first["r"]) <= set(wide["r"]) and len(calls) == n_first + len(wide["r"]) - len(first["r"])
    emin = min(wide["e"])
    assert (wide["e"][-1] - emin) * 27.211386 > 1.2


def test_fallback_widens_curves_until_domain_converged(tmp_path):
    seen = []

    def curve_fn(*, charge, r_center, clearance_ev, **_):
        seen.append((charge, clearance_ev))
        half = 0.125 if clearance_ev < 0.8 else 0.40  # narrow box first
        re, a = (1.62, 2.0) if charge == 0 else (1.66, 1.9)
        r = list(np.arange(re - half, re + half + 1e-9, 0.025))
        return {"status": "COMPLETED", "r": r, "e": list(morse(r, re=re, a=a)), "method": "SC-NEVPT2/CASSCF",
                "basis": "def2-TZVPPD"}

    def loop(re, charge):
        rr = [re + 0.01 * i for i in range(-4, 5)]
        return SimpleNamespace(requests=[SimpleNamespace(r_angstrom=re, atoms=("Al", "O"), charge=charge, spin_2s=1,
                                                         irrep_nelec={}, symmetry_label="2Sigma+")],
                               results=[SimpleNamespace(high_level_checkpoint_path="/x")],
                               final_pec=SimpleNamespace(points=[SimpleNamespace(r_angstrom=x, energy_hartree=y)
                                                                 for x, y in zip(rr, morse(rr, re=re))]))
    ea = {"selected_neutral_job": "N", "selected_anion_job": "A",
          "candidates": {"neutral": [{"job_id": "N", "r_central_angstrom": 1.62}],
                         "anion": [{"job_id": "A", "r_central_angstrom": 1.66}]}}
    res = mr_zpe_fallback(electronic_ea=ea, loops={"N": loop(1.62, 0), "A": loop(1.66, -1)},
                          masses=main_isotopologue_masses(("Al", "O")), channels=(),
                          nuclear_settings=ZPEStagePolicy().nuclear_settings(), output_dir=tmp_path, curve_fn=curve_fn)
    assert res["status"] == "CLEARED", res
    assert res["extension_rounds"] and max(c for _, c in seen) == 0.9


def test_extension_is_seeded_and_cold_failures_are_rewalked(tmp_path, monkeypatch):
    import json
    import openea_benchmark.mr_nuclear_curve as mod
    seeds = []

    def point(atoms, r, charge, spin_2s, irrep_nelec, wfnsym, space, basis, mo_guess, mol_guess, chk, policy,
              projected=None):
        seeds.append((round(r, 4), mo_guess is not None))
        return "mol", np.full(3, r), float(morse(r)), np.array([1.95, 1.0, 0.05])
    monkeypatch.setattr(mod, "_point", point)
    monkeypatch.setattr(mod, "_mol", lambda *a, **k: "mol")
    kw = dict(atoms=("Al", "O"), charge=0, spin_2s=1, irrep_nelec={"A1": (5, 4)}, label="2Sigma+", r_center=1.62,
              chk="/x", policy=MRCurvePolicy(), cache=tmp_path / "anion.json")
    first = mod.mr_state_curve(**kw)
    assert (tmp_path / "anion.json.ends.npz").is_file()
    seeds.clear()
    wide = mod.mr_state_curve(**kw, clearance_ev=0.9)
    assert wide["seeded"] and seeds and all(seeded for _, seeded in seeds)
    assert len(seeds) == len(wide["r"]) - len(first["r"])  # only new points
    # A failed curve from a cold (pre-50c) extension is walked again from the centre.
    (tmp_path / "anion.json").write_text(json.dumps({"status": "MR_CURVE_CONTINUITY_FAILED", "clearance_ev": 0.9}))
    seeds.clear()
    again = mod.mr_state_curve(**kw, clearance_ev=0.9)
    assert again["status"] == "COMPLETED" and seeds[0] == (1.62, False) and len(seeds) == len(again["r"])
