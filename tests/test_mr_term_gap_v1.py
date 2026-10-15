"""Patch 54: MR term-gap route (no CASSCF in unit tests)."""
import numpy as np
import pytest

from openea_benchmark.final_resolver import resolve_final_verdict
from openea_benchmark.mr_term_gap import (MRTermGapPolicy, apply_mr_term_gaps, ground_state_offsets,
                                          harmonic_zpe_ev, matched_energy, run_mr_term_gap_stage,
                                          state_minimum, term_gap)

EV = 27.211386245988


def morse(r, re, de=0.2, a=2.0, e0=0.0):
    return e0 + de * (1 - np.exp(-a * (r - re))) ** 2


def fake_blocks(basis_shift=None):
    """X3Sigma- (A2, Lambda 0) at re 1.21; 1Delta (A2 singlet, Lambda 2) 1.0 eV higher at re 1.22;
    a lower 1Sigma- root (Lambda 0) in the singlet block that must not be taken for the Delta state."""
    basis_shift = basis_shift or {}

    def fn(atoms, charge, mult, irrep, r, policy, cache):
        shift = basis_shift.get((policy.basis, mult), 0.0)
        if mult == 3:
            roots = [{"lambda": 0, "parity": 1, "e_hartree": morse(r, 1.21) + shift}]
        else:
            roots = [{"lambda": 0, "parity": -1, "e_hartree": morse(r, 1.25, e0=0.5 / EV)},
                     {"lambda": 2, "parity": 1, "e_hartree": morse(r, 1.22, e0=1.0 / EV) + shift}]
        return {"status": "COMPLETED", "roots": roots}
    return fn


X = {"label": "3Sigmag-", "mult": 3, "irrep": "A2", "lambda": 0, "parity": 1}
D = {"label": "1Deltag", "mult": 1, "irrep": "A2", "lambda": 2, "parity": 1}


def test_matched_energy_requires_lambda_and_parity():
    rec = {"status": "COMPLETED", "roots": [{"lambda": 0, "parity": 1, "e_hartree": -1.0},
                                             {"lambda": None, "parity": 1, "e_hartree": -2.0},
                                             {"lambda": 2, "parity": -1, "e_hartree": -0.5}]}
    assert matched_energy(rec, 0, 1) == -1.0
    assert matched_energy(rec, 2, 1) is None  # unassigned root is never taken as the state
    assert matched_energy({"status": "FAILED"}, 0, None) is None


def test_state_minimum_walks_downhill_and_fits():
    res = state_minimum(lambda r: morse(r, 1.337), 1.10, MRTermGapPolicy())
    assert res["status"] == "COMPLETED"
    assert res["r_e"] == pytest.approx(1.337, abs=2e-3)
    assert res["k_hartree_per_a2"] == pytest.approx(2 * 0.2 * 4.0, rel=0.03)
    assert state_minimum(lambda r: None, 1.1, MRTermGapPolicy())["status"] == "STATE_NOT_FOUND"


def test_harmonic_zpe_matches_o2_scale():
    k = 1177.0 / 1556.893 / 0.529177210903 ** 2  # O2 force constant 1177 N/m in Eh/A^2
    assert harmonic_zpe_ev(k, (15.9949, 15.9949)) == pytest.approx(0.098, rel=0.02)  # omega ~ 1580 cm-1


def test_term_gap_follows_the_lambda_root(tmp_path):
    res = term_gap(atoms=("O", "O"), charge=0, reference=X, state=D, r_start=1.2, policy=MRTermGapPolicy(),
                   cache_dir=tmp_path, block_fn=fake_blocks({("def2-QZVPPD", 1): -0.02 / EV}))
    assert res["status"] == "COMPLETED", res
    assert res["gap_e_tz_ev"] == pytest.approx(1.0, abs=1e-3)  # not the 0.5 eV Sigma- root
    assert res["basis_shift_ev"] == pytest.approx(-0.02, abs=1e-6)
    assert abs(res["delta_zpe_ev"]) < 1e-3  # same Morse shape
    assert res["gap0_half_width_ev"] == pytest.approx(0.02 + 0.15, abs=1e-3)


def test_stage_runs_only_open_candidates(tmp_path):
    comp = {"status": "UNRESOLVED_MR_COMPETITOR", "policy": {"r_offsets_angstrom": [-0.075, 0.0, 0.075]},
            "selected": {"0": {"role": "neutral", "label": "3Sigmag-", "r": 1.21, "atoms": ["O", "O"]}},
            "blocks": [{"charge": 0, "status": "UNRESOLVED_MR_COMPETITOR", "candidates": ["q=0 1Deltag"]},
                       {"charge": 0, "status": "EXCLUDED", "candidates": ["q=0 3Deltau"]}]}
    seen = []

    def task(*args):
        seen.append(args[3]["label"])
        return {"status": "COMPLETED", "gap0_ev": 1.0, "gap0_lower_ev": 0.8, "gap0_upper_ev": 1.2}
    res = run_mr_term_gap_stage(mr_competitors=comp, output_dir=tmp_path, task_fn=task)
    assert res["status"] == "CLEARED" and seen == ["1Deltag"]
    assert (tmp_path / "mr_term_gap.json").is_file()
    cleared = run_mr_term_gap_stage(mr_competitors={**comp, "status": "CLEARED", "blocks": []}, output_dir=tmp_path)
    assert cleared["status"] == "CLEARED" and cleared["gaps"] == []


ADIABATIC = {"status": "CLEARED", "binding": "BOUND", "ea0_ev": 1.0, "ea0_lower_ev": 0.95, "ea0_upper_ev": 1.05,
             "delta_cv_ev": 0.01, "delta_sr_ev": -0.005, "delta_post_ccsd_t_ev": 0.02}


def gap(role, label, g, half):
    return {"status": "COMPLETED", "role": role, "label": label, "gap0_ev": g, "gap0_lower_ev": g - half,
            "gap0_upper_ev": g + half}


def test_mr_state_above_leaves_ea_unchanged():
    out = apply_mr_term_gaps(ADIABATIC, {"gaps": [gap("neutral", "q=0 1Delta", 0.9, 0.15)]})
    assert out["ea0_ev"] == 1.0 and out["ea0_lower_ev"] == 0.95 and out["status"] == "CLEARED"


def test_anion_mr_ground_state_raises_ea():
    out = apply_mr_term_gaps(ADIABATIC, {"gaps": [gap("anion", "q=-1 1Delta", -0.4, 0.15)]})
    assert out["ea0_ev"] == pytest.approx(1.4)
    assert out["ea0_lower_ev"] == pytest.approx(0.95 + 0.25 - 0.035)
    assert out["ea0_upper_ev"] == pytest.approx(1.05 + 0.55 + 0.035)
    assert out["status"] == "CLEARED"
    assert ground_state_offsets({"gaps": [gap("anion", "q=-1 1Delta", -0.4, 0.15)]})["anion"]["mr_ground_state"]


def test_overlapping_gap_widens_and_neutral_mr_lowers_ea():
    out = apply_mr_term_gaps(ADIABATIC, {"gaps": [gap("neutral", "q=0 1Sigma-", 0.05, 0.15)]})
    assert out["ea0_ev"] == 1.0  # central value: SR state stays lower
    assert out["ea0_lower_ev"] == pytest.approx(0.95 - 0.10 - 0.035)
    assert out["ea0_upper_ev"] == pytest.approx(1.05 + 0.035)


def test_open_shell_mr_state_with_orbital_momentum_blocks():
    out = apply_mr_term_gaps(ADIABATIC, {"gaps": [gap("anion", "q=-1 3Phi", -0.4, 0.15)]})
    assert out["status"] == "NEED_MORE_EVIDENCE_MR_STATE_SPIN_ORBIT"


def test_resolver_accepts_term_gap_route():
    ea = {"status": "ELECTRONIC_EA_EVALUATED", "high_level_competition": {}}
    adiabatic = {**ADIABATIC, "includes_core_valence": True, "includes_scalar_relativity": True,
                 "includes_soc": True, "includes_post_ccsd_t": True}
    common = dict(electronic_ea=ea, cbs={"status": "CLEARED"}, adiabatic=adiabatic,
                  mr_competitors={"status": "UNRESOLVED_MR_COMPETITOR"}, mr_only_labels=["q=0 1Delta"])
    assert resolve_final_verdict(**common)["outcome"] == "UNRESOLVED"
    assert resolve_final_verdict(**common, mr_term_gap={"status": "CLEARED"})["outcome"] == "EA"
    blocked = resolve_final_verdict(**common, mr_term_gap={"status": "UNRESOLVED_GAP_NOT_COMPUTED"})
    assert "term gap UNRESOLVED_GAP_NOT_COMPUTED" in blocked["blocking"][0]
