"""Patch 39: automated frozen-core CBS layer (fake CCSD(T) backend)."""
from dataclasses import replace

from openea_benchmark.adaptive.stage3_execution import PointExecutionStatus, Stage3ExecutionSettings, _empty_result
from openea_benchmark.attachment.component_resolved_cbs import PRIMARY_EXPONENTS
from openea_benchmark.cbs_stage import CBSStagePolicy, _even_tempered_extra_diffuse, element_basis, run_cbs_stage

H = 1.0 / 27.211386245988
CBS = {"N": (-75.0, -0.40, -0.010), "A": (-75.02, -0.45, -0.012), "C": (-74.99, -0.41, -0.011)}


def request(job, charge, two_s):
    from test_reference_branches_v1 import request as base
    return replace(base(0, 1.0, job=job), atoms=("O", "H"), system="OH", charge=charge, spin_2s=two_s,
                   scf_reference="RHF" if two_s == 0 else "ROHF")


def backend(req, settings):
    assert settings.frozen_core and settings.checkpoint_project
    assert req.reference_branch == "SEED_CONTINUOUS"
    assert req.source_checkpoint_path.startswith("/hl/") or req.request_id.endswith("__RESEED")
    if FAIL_DAUG_FROM_TZ and req.job_id == "A" and req.basis.startswith("d-") and "__RESEED" not in req.request_id:
        return _empty_result(req, PointExecutionStatus.SCF_NOT_CONVERGED, reference_warnings=())
    key = req.job_id[0]
    if "MRDIAG" in req.request_id:
        t1 = MR_T1.get(key, 0.01)
        return _empty_result(req, PointExecutionStatus.COMPLETED, ccsd_correlation_hartree=-0.4,
                             triples_correction_hartree=-0.01, t1_diagnostic=t1, t1_diagnostic_definition="fake",
                             max_abs_t2_amplitude=0.05, reference_warnings=())
    scf, corr, trip = CBS[key]
    x = 4 if "pvqz" in req.basis else 5
    shift = -DAUG_SHIFT if req.basis.startswith("d-") else 0.0  # diffuse lowers the anion only
    shift = shift if key == "A" else 0.0
    return _empty_result(req, PointExecutionStatus.COMPLETED,
                         scf_energy_hartree=scf + 0.01 * x ** -PRIMARY_EXPONENTS["scf"] + shift,
                         ccsd_correlation_hartree=corr + 0.1 * x ** -PRIMARY_EXPONENTS["ccsd_correlation"],
                         triples_correction_hartree=trip + 0.01 * x ** -PRIMARY_EXPONENTS["triples"],
                         ccsd_t_total_hartree=(scf + corr + trip + shift + 0.01 * x ** -PRIMARY_EXPONENTS["scf"]
                                               + 0.1 * x ** -PRIMARY_EXPONENTS["ccsd_correlation"]
                                               + 0.01 * x ** -PRIMARY_EXPONENTS["triples"]),
                         reference_warnings=(), high_level_checkpoint_path=f"/cbs/{req.request_id}")


FAIL_DAUG_FROM_TZ = False
MR_T1: dict = {}
DAUG_SHIFT = 0.0005


def ea_record(competitor_gap_ev, sensitivity=0.0):
    cand = lambda job, e, r: {"job_id": job, "equilibrium_status": "RESOLVED", "energy_central_hartree": e,
                              "r_central_angstrom": r, "branch_of": None}
    return {"status": "ELECTRONIC_EA_EVALUATED", "selected_neutral_job": "N", "selected_anion_job": "A",
            "reference_sensitivity_ev": sensitivity,
            "candidates": {"neutral": [cand("N", -75.4, 0.97), cand("C", -75.4 + competitor_gap_ev * H, 1.1)],
                           "anion": [cand("A", -75.48, 0.96)]}}


def run(tmp_path, gap, sensitivity=0.0):
    reqs = [request("N", 0, 1), request("C", 0, 1), request("A", -1, 0)]
    recs = {f"{j}-{i}": {"job_id": j, "status": "COMPLETED", "r_angstrom": r, "high_level_checkpoint_path": f"/hl/{j}{r}"}
            for j in "NCA" for i, r in enumerate((0.9, 1.0))}
    return run_cbs_stage(electronic_ea=ea_record(gap, sensitivity), requests=reqs, point_records=recs,
                         refinement_dir=tmp_path / "none", stage3_settings=Stage3ExecutionSettings(),
                         output_dir=tmp_path / "cbs", point_backend=backend)


def test_cbs_recovers_component_limits_and_diffuse_shift(tmp_path):
    res = run(tmp_path, gap=2.0)  # competitor outside the window: not computed
    assert res["status"] == "CLEARED"
    expected = (sum(CBS["N"]) - sum(CBS["A"])) / H + 0.0005 / H
    assert abs(res["ea_cbs_ev"] - expected) < 1e-6
    assert abs(res["cbs"]["diffuse_correction_ev"] - 0.0005 / H) < 1e-9
    assert res["competitor_ranking_at_cbs"] == []


def test_competitor_within_window_reranked_at_cbs(tmp_path):
    res = run(tmp_path, gap=0.3)
    (rank,) = res["competitor_ranking_at_cbs"]
    assert rank["job_id"] == "C" and abs(rank["gap_cbs_ev"] - (sum(CBS["C"]) - sum(CBS["N"])) / H) < 1e-6
    assert res["status"] == "UNRESOLVED_STATE_ORDER_AT_CBS"  # C is lower at CBS in this fake


def test_not_attempted_without_single_basis_ea(tmp_path):
    res = run_cbs_stage(electronic_ea={"status": "UNRESOLVED"}, requests=[], point_records={},
                        refinement_dir=tmp_path, stage3_settings=Stage3ExecutionSettings(), output_dir=tmp_path)
    assert res["status"] == "NOT_ATTEMPTED"


def test_basis_policy():
    assert element_basis("Cl", 4)[0] == "aug-cc-pv(q+d)z" and element_basis("O", 5)[0] == "aug-cc-pv5z"
    assert element_basis("O", 5, daug=True)[1] == "d-aug-cc-pv5z"
    assert element_basis("Al", 5, daug=True)[1].endswith("(EVEN_TEMPERED_CONSTRUCTED)")
    shells = [[0, [1.0, 1.0]], [0, [0.1, 1.0]], [2, [2.0, 1.0, 0.0], [0.5, 0.0, 1.0]]]
    added = _even_tempered_extra_diffuse(shells)[len(shells):]
    assert [(l, round(e, 12), c) for l, (e, c) in added] == [(0, 0.01, 1.0), (2, 0.125, 1.0)]


def test_failed_point_reseeded_from_previous_basis(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__(__name__), "FAIL_DAUG_FROM_TZ", True)
    res = run(tmp_path, gap=2.0)
    assert res["status"] == "CLEARED"
    import json
    states = json.loads((tmp_path / "cbs" / "cbs.json").read_text())["states"]
    anion = next(s for s in states if s["job_id"] == "A")
    reseed = anion["points"]["d-aug-cc-pv5z"]["reseeded"]
    assert reseed["first_status"] == "SCF_NOT_CONVERGED"
    assert reseed["seed_checkpoint"] == "/cbs/A__CBS__aug-cc-pv5z"


def test_error_results_are_not_cached(tmp_path):
    from openea_benchmark.electronic_ea_stage import cached_point_runner
    calls = []

    def flaky(req, settings):
        calls.append(req.request_id)
        status = PointExecutionStatus.ERROR if len(calls) == 1 else PointExecutionStatus.COMPLETED
        return _empty_result(req, status, reference_warnings=())

    runner = cached_point_runner(tmp_path, "sig", flaky)
    req = request("A", -1, 0)
    assert runner(req, Stage3ExecutionSettings()).status is PointExecutionStatus.ERROR
    assert runner(req, Stage3ExecutionSettings()).status is PointExecutionStatus.COMPLETED
    assert runner(req, Stage3ExecutionSettings()).status is PointExecutionStatus.COMPLETED
    assert len(calls) == 2


def test_mr_diagnostic_recorded_and_clean(tmp_path):
    res = run(tmp_path, gap=2.0)
    assert res["binding_at_cbs"] == "BOUND" and res["multireference_exceeded"] == []
    assert res["multireference_diagnostics"]["anion"]["t1"] == 0.01


def test_open_shell_t1_above_threshold_is_a_warning_only(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__(__name__), "MR_T1", {"N": 0.05})  # neutral is a doublet: limit 0.044
    res = run(tmp_path, gap=2.0)
    assert res["status"] == "CLEARED"
    assert res["multireference_screening"] == "ABOVE_SCREENING_VALUES_WARNING"
    assert res["multireference_exceeded"] == ["neutral:T1"]


def test_closed_shell_t1_threshold_is_stricter(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__(__name__), "MR_T1", {"A": 0.03})
    assert run(tmp_path, gap=2.0)["multireference_exceeded"] == ["anion:T1"]


def test_diffuse_sensitive_negative_ea_is_unbound(tmp_path, monkeypatch):
    # d-aug lowers the anion by 0.5 eV, as for N2-: the continuum, not a bound state.
    monkeypatch.setattr(__import__(__name__), "DAUG_SHIFT", 0.5 * H)
    monkeypatch.setitem(CBS, "A", (-74.9, -0.40, -0.010))
    res = run(tmp_path, gap=2.0)
    assert res["status"] == "ANION_UNBOUND_AT_CBS" and res["binding_at_cbs"] == "UNBOUND"
    assert res["ea_cbs_ev"] is None and res["ea_cbs_raw_ev"] < 0


def test_diffuse_sensitive_positive_ea_is_undetermined(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__(__name__), "DAUG_SHIFT", 0.1 * H)
    res = run(tmp_path, gap=2.0)
    assert res["status"] == "NEED_MORE_EVIDENCE_BINDING"
    assert res["binding_reason"] == "DIFFUSE_CORRECTION_ABOVE_CONTINUUM_THRESHOLD"


def test_reference_sensitivity_widens_cbs_interval(tmp_path):
    base = run(tmp_path / "a", gap=2.0)
    wide = run(tmp_path / "b", gap=2.0, sensitivity=0.06)
    assert abs((base["ea_cbs_lower_ev"] - wide["ea_cbs_lower_ev"]) - 0.06) < 1e-12
    assert abs((wide["ea_cbs_upper_ev"] - base["ea_cbs_upper_ev"]) - 0.06) < 1e-12
    assert wide["reference_sensitivity_source"] == "SINGLE_BASIS_REFERENCE_BRANCH_OFFSET"


def test_point_cache_signature_unchanged_by_interpretation_thresholds(tmp_path, monkeypatch):
    import hashlib
    import json
    seen = []
    import openea_benchmark.cbs_stage as cbs_stage
    monkeypatch.setattr(cbs_stage, "cached_point_runner", lambda d, sig, b: seen.append(sig) or b)
    run(tmp_path, gap=2.0)
    patch39 = {"low_cardinal": 4, "high_cardinal": 5, "diffuse_correction": True, "competitor_window_ev": 0.5,
               "max_memory_mb": 48000, "model_spread_target_ev": 0.005}
    assert seen == [hashlib.sha256(json.dumps(patch39, sort_keys=True).encode()).hexdigest()]
