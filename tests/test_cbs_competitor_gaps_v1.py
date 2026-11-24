"""CBS competitor states: the selected state gets every correction, the competitor only its
frozen-core CBS energy (and its first-order spin-orbit shift); a competitor that can swap
with the selected state widens the EA instead of being ignored."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from openea_benchmark import point_pool, spin_orbit_stage
from openea_benchmark.final_resolver import resolve_final_verdict
from openea_benchmark.vibrational_stage import assemble_adiabatic_ea, competitor_state_offsets

EA = {"status": "ELECTRONIC_EA_EVALUATED",
      "high_level_competition": {"neutral": {"status": "CLEARED"}, "anion": {"status": "CLEARED"}}}
NUC = {"status": "CLEARED", "delta_zpe_ev": 0.01, "delta_zpe_half_width_ev": 0.002, "anion_vibrational_binding": "BOUND"}
LAYERS = dict(core_valence={"status": "CORE_VALENCE_CLEARED", "delta_cv_ev": -0.05, "delta_cv_bound_ev": 0.01},
              scalar_relativity={"status": "SCALAR_RELATIVITY_CLEARED", "delta_sr_ev": 0.10, "delta_sr_bound_ev": 0.01},
              post_ccsd_t={"status": "CLEARED", "delta_post_ev": 0.04, "bound_ev": 0.01})
COMP = {"role": "anion", "job_id": "XO__q-1__2Pi", "gap_tz_ev": 0.245, "gap_cbs_ev": 0.135}


def cbs(*competitors):
    return {"status": "CLEARED", "binding_at_cbs": "BOUND", "binding_reason": "CBS_INTERVAL_POSITIVE",
            "ea_cbs_ev": 1.40, "ea_cbs_lower_ev": 1.39, "ea_cbs_upper_ev": 1.41,
            "competitor_ranking_at_cbs": list(competitors)}


def soc(selected_anion=0.0, competitor=-0.02, *, with_competitor=True):
    rec = {"stage": "SPIN_ORBIT_FIRST_ORDER", "status": "CLEARED", "delta_ea_so_ev": -selected_anion,
           "bound_ev": 0.3 * abs(selected_anion),
           "states": {"neutral": {"delta_so_ev": 0.0, "bound_ev": 0.0},
                      "anion": {"delta_so_ev": selected_anion, "bound_ev": 0.3 * abs(selected_anion)}}}
    if with_competitor:
        rec["competitor_states"] = {COMP["job_id"]: {"role": "anion", "status": "CLEARED", "delta_so_ev": competitor,
                                                     "bound_ev": 0.3 * abs(competitor)}}
    return rec


def test_competitor_within_the_uncorrected_corrections_widens_the_ea():
    # transfer = |ZPE| + |CV| + |SR| + |post| = 0.01 + 0.05 + 0.10 + 0.04 = 0.20 eV > gap0 = 0.135 - 0.02
    adi = assemble_adiabatic_ea(cbs=cbs(COMP), nuclear=NUC, spin_orbit=soc(), **LAYERS)
    gap = adi["cbs_competitors"]["gaps"][0]
    assert gap["gap0_ev"] == pytest.approx(0.115) and gap["half_width_ev"] == pytest.approx(0.206)
    selected = 1.40 + 0.01 - 0.05 + 0.10 + 0.04  # no SO for the selected Sigma states
    assert adi["ea0_ev"] == pytest.approx(selected)          # gap0 > 0: same central value
    width = 0.002 + 0.01 + 0.01 + 0.01                        # ZPE, CV, SR, post bounds
    assert adi["ea0_lower_ev"] == pytest.approx(1.39 + selected - 1.40 - width)
    # a lower anion state raises the EA: upper bound + (half - gap0)
    assert adi["ea0_upper_ev"] == pytest.approx(1.41 + selected - 1.40 + width + 0.206 - 0.115)
    assert adi["status"] == "CLEARED"
    assert resolve_final_verdict(electronic_ea=EA, cbs=cbs(COMP), adiabatic=adi,
                                 mr_competitors=None)["outcome"] == "EA"


def test_competitor_far_above_changes_nothing():
    far = {**COMP, "gap_cbs_ev": 0.55}
    adi = assemble_adiabatic_ea(cbs=cbs(far), nuclear=NUC, spin_orbit=soc(), **LAYERS)
    ref = assemble_adiabatic_ea(cbs=cbs(), nuclear=NUC, spin_orbit=soc(), **LAYERS)
    for key in ("ea0_ev", "ea0_lower_ev", "ea0_upper_ev"):
        assert adi[key] == pytest.approx(ref[key])
    assert "ea0_selected_states_ev" not in adi


def test_spin_orbit_can_make_the_competitor_the_ground_state():
    out = competitor_state_offsets(cbs({**COMP, "gap_cbs_ev": 0.02}), soc(competitor=-0.05), 0.0)
    anion = out["offsets"]["anion"]
    assert anion["central_ev"] == pytest.approx(-0.03) and anion["ground_state"] == COMP["job_id"]


def test_missing_competitor_spin_orbit_fails_closed():
    adi = assemble_adiabatic_ea(cbs=cbs(COMP), nuclear=NUC, spin_orbit=soc(with_competitor=False), **LAYERS)
    assert adi["status"] == "NEED_MORE_EVIDENCE_COMPETITOR_SPIN_ORBIT" and "spin-orbit not computed" in adi["reason"]
    assert resolve_final_verdict(electronic_ea=EA, cbs=cbs(COMP), adiabatic=adi,
                                 mr_competitors=None)["outcome"] == "UNRESOLVED"
    # far enough above that even the largest first-order shift cannot make it the ground state
    far = {**COMP, "gap_cbs_ev": 0.55}
    adi = assemble_adiabatic_ea(cbs=cbs(far), nuclear=NUC, spin_orbit=soc(with_competitor=False), **LAYERS)
    assert adi["status"] == "CLEARED"
    # beyond Kr the multistate treatment of the selected state covers its low states
    multi = {**soc(with_competitor=False), "stage": "SPIN_ORBIT_MULTISTATE"}
    adi = assemble_adiabatic_ea(cbs=cbs(COMP), nuclear=NUC, spin_orbit=multi, **LAYERS)
    assert adi["status"] == "CLEARED" and adi["cbs_competitors"]["gaps"][0]["gap0_ev"] == pytest.approx(0.135)


def test_spin_orbit_stage_computes_the_competitor_shift(tmp_path, monkeypatch):
    calls = []

    def fake(**kw):
        calls.append(kw["symmetry_label"])
        return {"status": "CLEARED", "delta_so_ev": -0.01 if kw["symmetry_label"] == "2Pi" else 0.0, "bound_ev": 0.0}

    monkeypatch.setattr(spin_orbit_stage, "state_spin_orbit_shift", fake)
    monkeypatch.setattr(point_pool, "core_slot", nullcontext)
    req = lambda job, charge, label: SimpleNamespace(job_id=job, system="XO", atoms=("O", "O"), charge=charge,
                                                     symmetry_label=label, irrep_nelec=None)
    cand = lambda job: {"job_id": job, "r_central_angstrom": 1.6}
    ea = {"status": "ELECTRONIC_EA_EVALUATED", "selected_neutral_job": "N", "selected_anion_job": "A",
          "candidates": {"neutral": [cand("N")], "anion": [cand("A"), cand(COMP["job_id"])]}}
    rec = spin_orbit_stage.run_spin_orbit_stage(
        electronic_ea=ea, requests=[req("N", 0, "3Sigma-"), req("A", -1, "4Sigma-"),
                                    req(COMP["job_id"], -1, "2Pi")],
        output_dir=tmp_path, competitors=[COMP, {"role": "anion", "job_id": "X", "status": "CBS_POINTS_MISSING"}])
    assert rec["status"] == "CLEARED" and calls == ["3Sigma-", "4Sigma-", "2Pi"]
    assert rec["competitor_states"][COMP["job_id"]]["delta_so_ev"] == -0.01
