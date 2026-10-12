"""ZPE layer: curve shape from the followed reference branch when the seed branch ends."""
from types import SimpleNamespace

import pytest

from openea_benchmark import vibrational_stage
from openea_benchmark.vibrational_stage import harmonic_curvature, run_vibrational_stage


def pec(k, r0=1.64, n=9, step=0.01):
    return SimpleNamespace(points=[SimpleNamespace(r_angstrom=r0 + (i - n // 2) * step,
                                                   energy_hartree=-1.0 + 0.5 * k * ((i - n // 2) * step) ** 2)
                                   for i in range(n)])


def test_harmonic_curvature_of_parabola():
    assert harmonic_curvature(pec(1.14)) == pytest.approx(1.14, rel=1e-6)
    assert harmonic_curvature(SimpleNamespace(points=[])) is None


def run(monkeypatch, tmp_path, loops, first_status="PEC_EXTENSION_EXECUTION_INCOMPLETE"):
    calls = []

    def fake(**kw):
        calls.append(dict(kw["jobs"]))
        if len(calls) == 1:
            return {"status": first_status, "reason": "seed branch ended",
                    "failed_extension": [{"role": "neutral", "points": [{"status": "SCF_NOT_CONVERGED"}]},
                                         {"role": "anion", "points": [{"status": "COMPLETED"}],
                                          "continuity_review": {"status": "CLEARED"}}]}
        return {"status": "CLEARED", "delta_zpe_ev": 0.001, "delta_zpe_half_width_ev": 0.0005,
                "zpe_neutral_ev": 0.06, "zpe_anion_ev": 0.05}

    monkeypatch.setattr(vibrational_stage, "_run_vibrational_stage", fake)
    monkeypatch.setattr(vibrational_stage, "_mr_fallback",
                        lambda record, *a, **k: {**record, "multireference_fallback": {"status": "MR_FAKE"}})
    ea = {"selected_neutral_job": "X__refS", "selected_anion_job": "Y", "system": "XY"}
    res = run_vibrational_stage(electronic_ea=ea, loops=loops, atoms=("X", "Y"), stage3_settings=None,
                                output_dir=tmp_path)
    return res, calls


def test_followed_branch_supplies_the_shape_with_widened_interval(monkeypatch, tmp_path):
    loops = {"X__refS": SimpleNamespace(final_pec=pec(1.14)), "X__refF": SimpleNamespace(final_pec=pec(1.164)),
             "Y": SimpleNamespace(final_pec=pec(0.8))}
    res, calls = run(monkeypatch, tmp_path, loops)
    assert calls[1] == {"neutral": "X__refF", "anion": "Y"}
    sub = res["reference_branch_shape_substitution"]
    rel = (1.164 / 1.14) ** 0.5 - 1.0
    assert sub["roles"]["neutral"]["relative_frequency_difference"] == pytest.approx(rel, rel=1e-4)
    assert res["delta_zpe_half_width_ev"] == pytest.approx(0.0005 + 0.06 * rel, rel=1e-4)
    assert sub["selected_branch_attempt"]["status"] == "PEC_EXTENSION_EXECUTION_INCOMPLETE"


def test_no_sibling_branch_keeps_the_failure(monkeypatch, tmp_path):
    loops = {"X__refS": SimpleNamespace(final_pec=pec(1.14)), "Y": SimpleNamespace(final_pec=pec(0.8))}
    res, calls = run(monkeypatch, tmp_path, loops)
    assert len(calls) == 1 and res["status"] == "PEC_EXTENSION_EXECUTION_INCOMPLETE"
    assert res["multireference_fallback"]["status"] == "MR_FAKE"  # MR curves are tried next


def test_other_failures_do_not_switch_branch(monkeypatch, tmp_path):
    loops = {"X__refS": SimpleNamespace(final_pec=pec(1.14)), "X__refF": SimpleNamespace(final_pec=pec(1.164))}
    res, calls = run(monkeypatch, tmp_path, loops, first_status="MODEL_EXECUTION_INCOMPLETE")
    assert len(calls) == 1
