"""Patch 46: automated core-valence layer (basis rule + fake CCSD(T) backend)."""
from openea_benchmark.adaptive.stage3_execution import PointExecutionStatus, Stage3ExecutionSettings, _empty_result
from openea_benchmark.core_valence_basis import core_orbital_count, cv_element_series
from openea_benchmark.core_valence_stage import CoreValenceStagePolicy, basis_specs_for, run_core_valence_stage
from openea_benchmark.vibrational_stage import assemble_adiabatic_ea

from test_cbs_stage_v1 import ea_record, request

H = 1.0 / 27.211386245988


def test_basis_rule_per_element():
    assert core_orbital_count("H") == 0 and core_orbital_count("O") == 1 and core_orbital_count("Mg") == 5
    assert cv_element_series("H", (3, 4))[3][1].startswith("aug-cc-pVTZ")
    assert cv_element_series("O", (3, 4))[4][1].startswith("aug-cc-pwCVQZ")
    assert cv_element_series("Mg", (3, 4))[3][1].startswith("aug-cc-pCVTZ")  # no pwCV for Mg
    assert cv_element_series("K", (3, 4)) is None


def test_basis_specs_share_one_family_and_stop_at_common_cardinal():
    specs, prov = basis_specs_for(("O", "H"), CoreValenceStagePolicy())
    assert [s.cardinal for s in specs] == [3, 4, 5]
    assert len({s.family_id for s in specs}) == 1 and "aug-cc-pwCVXZ" in specs[0].family_id
    specs, _ = basis_specs_for(("Li", "H"), CoreValenceStagePolicy())  # aug-cc-pCV5Z missing for Li
    assert [s.cardinal for s in specs] == [3, 4]
    assert basis_specs_for(("K", "H"), CoreValenceStagePolicy()) is None


CV_SHIFT = {3: 0.0000, 4: 0.0040, 5: 0.0045}  # eV, AE minus FC EA per cardinal


def backend(req, settings):
    assert req.reference_branch == "SEED_CONTINUOUS" and settings.checkpoint_project
    x = int(req.job_id.split("__X")[1][0])
    base = {"N": -75.40, "A": -75.48}["N" if req.charge == 0 else "A"]
    if not settings.frozen_core:
        base -= 0.05 + (CV_SHIFT[x] * H if req.charge == 0 else 0.0)
    return _empty_result(req, PointExecutionStatus.COMPLETED, ccsd_t_total_hartree=base, reference_warnings=(),
                         high_level_checkpoint_path=f"/cv/{req.request_id}")


def run(tmp_path):
    reqs = [request("N", 0, 1), request("A", -1, 0)]
    recs = {f"{j}-{i}": {"job_id": j, "request_id": f"{j}-{i}", "status": "COMPLETED", "r_angstrom": r,
                         "high_level_checkpoint_path": f"/hl/{j}{r}"} for j in "NA" for i, r in enumerate((0.9, 1.0))}
    return run_core_valence_stage(electronic_ea=ea_record(2.0), requests=reqs, point_records=recs,
                                  refinement_dir=tmp_path / "none", stage3_settings=Stage3ExecutionSettings(),
                                  output_dir=tmp_path / "cv", point_backend=backend)


def test_stage_runs_series_until_converged(tmp_path):
    res = run(tmp_path)
    # T->Q change 4 meV > 2 meV target: X=5 requested; Q->5 0.5 meV, contraction 0.125 -> cleared
    assert res["status"] == "CORE_VALENCE_CLEARED" and res["highest_cardinal"] == 5
    assert abs(res["delta_cv_ev"] + 0.0045) < 1e-9  # lower AE neutral energy: smaller EA and abs(res["delta_cv_bound_ev"] - 0.0005) < 1e-9
    assert res["states"]["neutral"]["r_angstrom"] == 0.97


def test_assembly_adds_cleared_cv_only():
    cbs = {"binding_at_cbs": "BOUND", "status": "CLEARED", "ea_cbs_ev": 1.0, "ea_cbs_lower_ev": 0.99,
           "ea_cbs_upper_ev": 1.01}
    nuc = {"status": "CLEARED", "delta_zpe_ev": 0.01, "delta_zpe_half_width_ev": 0.001,
           "anion_vibrational_binding": "BOUND"}
    cv = {"status": "CORE_VALENCE_CLEARED", "delta_cv_ev": 0.02, "delta_cv_bound_ev": 0.002}
    res = assemble_adiabatic_ea(cbs=cbs, nuclear=nuc, core_valence=cv)
    assert res["includes_core_valence"] and abs(res["ea0_ev"] - 1.03) < 1e-12
    assert abs(res["ea0_lower_ev"] - (0.99 + 0.03 - 0.003)) < 1e-12
    res = assemble_adiabatic_ea(cbs=cbs, nuclear=nuc, core_valence={**cv, "status": "CORE_VALENCE_UNRESOLVED"})
    assert not res["includes_core_valence"] and abs(res["ea0_ev"] - 1.01) < 1e-12
    assert res["core_valence_status"] == "CORE_VALENCE_UNRESOLVED"
