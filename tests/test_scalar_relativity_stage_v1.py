"""Patch 48: automated scalar-relativity layer (fake CCSD(T) backend)."""
from openea_benchmark.adaptive.stage3_execution import PointExecutionStatus, Stage3ExecutionSettings, _empty_result
from openea_benchmark.core_valence_basis import sr_element_series
from openea_benchmark.scalar_relativity_stage import run_scalar_relativity_stage
from openea_benchmark.vibrational_stage import assemble_adiabatic_ea

from test_cbs_stage_v1 import ea_record, request

H = 1.0 / 27.211386245988
SR = {3: -0.0030, 4: -0.0032, 5: -0.0032}


def test_dk_basis_rule():
    assert sr_element_series("O", (3, 4))[3][1].startswith("aug-cc-pVTZ-DK")
    assert sr_element_series("H", (3, 4, 5))[5][1].startswith("aug-cc-pV5Z-DK")


def backend(req, settings):
    assert not settings.frozen_core and req.reference_branch == "SEED_CONTINUOUS"
    x = int(req.job_id.split("__X")[1][0])
    e = -75.40 if req.charge == 0 else -75.48
    if settings.scalar_relativistic == "SFX2C1E":
        e -= 0.05 + (SR[x] * H if req.charge == 0 else 0.0)
    return _empty_result(req, PointExecutionStatus.COMPLETED, ccsd_t_total_hartree=e, reference_warnings=(),
                         high_level_checkpoint_path=f"/sr/{req.request_id}")


def test_stage_and_assembly(tmp_path):
    reqs = [request("N", 0, 1), request("A", -1, 0)]
    recs = {f"{j}-{i}": {"job_id": j, "request_id": f"{j}-{i}", "status": "COMPLETED", "r_angstrom": r,
                         "high_level_checkpoint_path": f"/hl/{j}{r}"} for j in "NA" for i, r in enumerate((0.9, 1.0))}
    res = run_scalar_relativity_stage(electronic_ea=ea_record(2.0), requests=reqs, point_records=recs,
                                      refinement_dir=tmp_path / "none", stage3_settings=Stage3ExecutionSettings(),
                                      output_dir=tmp_path / "sr", point_backend=backend)
    assert res["status"] == "SCALAR_RELATIVITY_CLEARED"
    assert abs(res["delta_sr_ev"] - 0.0032) < 1e-9  # neutral lowered: EA = E_N - E_A drops? no: +|shift|
    cbs = {"binding_at_cbs": "BOUND", "status": "CLEARED", "ea_cbs_ev": 1.0, "ea_cbs_lower_ev": 0.99,
           "ea_cbs_upper_ev": 1.01}
    nuc = {"status": "CLEARED", "delta_zpe_ev": 0.0, "delta_zpe_half_width_ev": 0.0, "anion_vibrational_binding": "BOUND"}
    out = assemble_adiabatic_ea(cbs=cbs, nuclear=nuc, scalar_relativity=res)
    assert out["includes_scalar_relativity"] and abs(out["ea0_ev"] - (1.0 + res["delta_sr_ev"])) < 1e-12
