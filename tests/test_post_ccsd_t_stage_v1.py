"""Patch 49: post-CCSD(T) layer (fake CCpy points)."""
from types import SimpleNamespace

from openea_benchmark import post_ccsd_t_stage
from openea_benchmark.post_ccsd_t_stage import PostCCSDTStagePolicy, quadruples_storage_mb, run_post_ccsd_t_stage
from openea_benchmark.vibrational_stage import assemble_adiabatic_ea

from test_cbs_stage_v1 import ea_record, request

H = 1.0 / 27.211386245988
# neutral-minus-anion energy shifts (eV) per method and basis, relative to CCSD(T)
SHIFT = {("aug-cc-pvdz", "CCSDT"): -0.004, ("aug-cc-pvtz", "CCSDT"): -0.006,
         ("aug-cc-pvdz", "CCSDTQ"): -0.004 + 0.010, ("cc-pvdz", "CCSDT"): -0.003,
         ("cc-pvdz", "CCSDTQ"): -0.003 + 0.008}


def fake_point(spec, basis, method, policy, path):
    e = -75.4 if spec.role == "neutral" else -75.48
    if spec.role == "neutral":
        e += SHIFT.get((basis, method), 0.0) * H
    return {"role": spec.role, "basis": basis, "method": method, "status": "COMPLETED", "total_hartree": e}


def run(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(post_ccsd_t_stage, "quadruples_storage_mb",
                        lambda spec, basis: 50000.0 if basis == "aug-cc-pvdz" else 10000.0)
    reqs = [request("N", 0, 1), request("A", -1, 0)]
    recs = {f"{j}-{i}": {"job_id": j, "request_id": f"{j}-{i}", "status": "COMPLETED", "r_angstrom": r,
                         "high_level_checkpoint_path": f"/hl/{j}{r}"} for j in "NA" for i, r in enumerate((0.9, 1.0))}
    return run_post_ccsd_t_stage(electronic_ea=ea_record(2.0), requests=reqs, point_records=recs,
                                 refinement_dir=tmp_path / "none", output_dir=tmp_path / "pc",
                                 policy=PostCCSDTStagePolicy(q_memory_limit_mb=limit), point_fn=fake_point)


def test_triples_tz_central_and_quadruples_in_first_affordable_basis(tmp_path, monkeypatch):
    res = run(tmp_path, monkeypatch, limit=60000)
    assert res["status"] == "CLEARED" and res["quadruples_basis"] == "aug-cc-pvdz"
    assert abs(res["delta_t_ev"] + 0.006) < 1e-9 and abs(res["delta_t_bound_ev"] - 0.002) < 1e-9
    assert abs(res["delta_q_ev"] - 0.010) < 1e-9 and abs(res["delta_post_ev"] - 0.004) < 1e-9
    res = run(tmp_path, monkeypatch, limit=20000)  # aug-cc-pVDZ too large: cc-pVDZ
    assert res["quadruples_basis"] == "cc-pvdz" and abs(res["delta_q_ev"] - 0.008) < 1e-9


def test_no_affordable_quadruples_applies_nothing(tmp_path, monkeypatch):
    res = run(tmp_path, monkeypatch, limit=5000)
    assert res["status"] == "QUADRUPLES_NOT_AFFORDABLE" and res["delta_post_ev"] is None
    assert abs(res["delta_t_ev"] + 0.006) < 1e-9  # evidence only
    cbs = {"binding_at_cbs": "BOUND", "status": "CLEARED", "ea_cbs_ev": 1.0, "ea_cbs_lower_ev": 0.99,
           "ea_cbs_upper_ev": 1.01}
    nuc = {"status": "CLEARED", "delta_zpe_ev": 0.0, "delta_zpe_half_width_ev": 0.0, "anion_vibrational_binding": "BOUND"}
    out = assemble_adiabatic_ea(cbs=cbs, nuclear=nuc, post_ccsd_t=res)
    assert not out["includes_post_ccsd_t"] and out["ea0_ev"] == 1.0


def test_storage_estimate_grows_with_basis():
    spec = SimpleNamespace(atoms=("O", "H"), r_angstrom=0.97, charge=-1, spin_2s=0)
    assert quadruples_storage_mb(spec, "aug-cc-pvdz") > quadruples_storage_mb(spec, "cc-pvdz") > 0


def test_point_files_are_distinct_per_method():
    from openea_benchmark.post_ccsd_t_stage import METHOD_TOKEN, METHODS
    assert len({METHOD_TOKEN[m] for m in METHODS}) == len(METHODS)
