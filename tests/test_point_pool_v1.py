"""Shared point pool: parallel results in input order, persisted on completion, core-slot budget."""
import os
import time

from openea_benchmark import point_pool
from openea_benchmark.adaptive.stage3_execution import PointExecutionStatus, Stage3ExecutionSettings, _empty_result
from openea_benchmark.adaptive.stage3_loop import Stage3LoopRetrySettings, _execute_batch_with_retries


def slow_square(x, marker_dir):
    path = os.path.join(marker_dir, f"active{os.getpid()}")
    open(path, "w").close()
    active = len(os.listdir(marker_dir))
    time.sleep(0.3)
    os.remove(path)
    return x * x, active


def fake_backend(req, settings):
    return _empty_result(req, PointExecutionStatus.COMPLETED, ccsd_t_total_hartree=-float(req.r_angstrom),
                         reference_warnings=())


def test_parallel_order_callbacks_and_slot_limit(tmp_path):
    markers = tmp_path / "m"
    markers.mkdir()
    point_pool.configure_point_pool(workers=4, threads=1, slot_dir=tmp_path / "slots", n_slots=2)
    try:
        seen = []
        out = point_pool.run_parallel(slow_square, [(i, str(markers)) for i in range(6)],
                                      on_result=lambda i, r: seen.append(i))
    finally:
        point_pool.shutdown_point_pool()
    assert [r[0] for r in out] == [i * i for i in range(6)]
    assert sorted(seen) == list(range(6))
    assert max(r[1] for r in out) <= 2  # never more concurrent tasks than core slots


def test_stage3_batch_on_pool_matches_serial():
    from test_reference_branches_v1 import request
    reqs = [request(i, 1.0 + 0.1 * i) for i in range(4)]
    args = dict(settings=Stage3ExecutionSettings(), retry_settings=Stage3LoopRetrySettings(), runner=fake_backend)
    serial = _execute_batch_with_retries(reqs, **args)
    point_pool.configure_point_pool(workers=3, threads=1)
    try:
        parallel = _execute_batch_with_retries(reqs, **args)
    finally:
        point_pool.shutdown_point_pool()
    assert [r.ccsd_t_total_hartree for r in parallel[0]] == [r.ccsd_t_total_hartree for r in serial[0]]
    assert [a.request_id for a in parallel[1]] == [a.request_id for a in serial[1]]
