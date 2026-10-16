"""Patch 49: post-CCSD(T) layer (fake CCpy points)."""
from types import SimpleNamespace

from openea_benchmark import post_ccsd_t_stage
from openea_benchmark.post_ccsd_t_stage import (
    PostCCSDTStagePolicy,
    run_post_ccsd_t_stage,
)
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
    monkeypatch.setattr(post_ccsd_t_stage, "cc_storage_mb",
                        lambda spec, basis, method: (50000.0 if basis == "aug-cc-pvdz" else 10000.0)
                        if method == "CCSDTQ" else 1000.0)
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


def test_point_files_are_distinct_per_method():
    from openea_benchmark.post_ccsd_t_stage import METHOD_TOKEN, METHODS
    assert len({METHOD_TOKEN[m] for m in METHODS}) == len(METHODS)


def test_reservations_follow_the_estimates(tmp_path, monkeypatch):
    seen = {}

    def fake_parallel(fn, jobs, reserve_mb=None):
        seen["reserves"] = list(reserve_mb)
        return [fn(*job) for job in jobs]

    monkeypatch.setattr(post_ccsd_t_stage, "run_parallel", fake_parallel)
    run(tmp_path, monkeypatch, limit=60000)
    assert 50000.0 in seen["reserves"] and 1000.0 in seen["reserves"]


def test_large_tasks_run_one_at_a_time(tmp_path, monkeypatch):
    import threading
    import time
    from openea_benchmark import point_pool
    monkeypatch.setattr(point_pool, "_SLOT_DIR", str(tmp_path))
    monkeypatch.setattr(point_pool, "_N_SLOTS", 4)
    monkeypatch.setattr(point_pool, "_MEM_LIMIT_FRAC", 0.0)
    active, peak = [0], [0]

    def task():
        with point_pool.core_slot(reserve_mb=150000):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            time.sleep(0.2)
            active[0] -= 1

    threads = [threading.Thread(target=task) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


def test_pending_large_reservation_is_counted_by_others(tmp_path, monkeypatch):
    import fcntl
    import os
    from openea_benchmark import point_pool
    monkeypatch.setattr(point_pool, "_SLOT_DIR", str(tmp_path))
    assert point_pool._pending_big_reservation_mb() == 0.0
    fh = open(os.path.join(tmp_path, "bigmem.lock"), "a+")
    fcntl.flock(fh, fcntl.LOCK_EX)
    fh.write("host/ns 1 150000 PENDING\n")
    fh.flush()
    assert point_pool._pending_big_reservation_mb() == 150000.0
    fh.seek(0); fh.truncate(); fh.write("host/ns 1 150000 ADMITTED\n"); fh.flush()
    assert point_pool._pending_big_reservation_mb() == 0.0  # admitted: counted via its slot instead
    fcntl.flock(fh, fcntl.LOCK_UN)
    fh.close()


def test_cached_points_reserve_no_memory(tmp_path, monkeypatch):
    """A completed point only reads its file; it must not queue for the large-task slot."""
    import json
    seen = []
    base_point = fake_point

    def fake_parallel(fn, jobs, reserve_mb=None):
        seen.append(list(reserve_mb))
        return [fn(*job) for job in jobs]

    def writing_point(spec, basis, method, policy, path):
        rec = dict(base_point(spec, basis, method, policy, path),
                   signature=post_ccsd_t_stage._point_signature(spec, basis, method, policy))
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(rec))
        return rec

    monkeypatch.setattr(post_ccsd_t_stage, "run_parallel", fake_parallel)
    import test_post_ccsd_t_stage_v1 as me
    monkeypatch.setattr(me, "fake_point", writing_point)
    run(tmp_path, monkeypatch, limit=60000)
    assert max(seen[0]) == 50000.0
    run(tmp_path, monkeypatch, limit=60000)  # second pass: every point cached
    assert seen[1] and all(x == 0.0 for x in seen[1])
