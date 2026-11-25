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


def test_memory_gate_waits_counts_reservations_and_clears(tmp_path, monkeypatch):
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=2,
                                    mem_limit_frac=0.8, mem_reserve_mb=100)
    # another task on this host reserved 50 MB above its current RSS: only that unallocated part counts
    import fcntl
    rss = point_pool._rss_mb(os.getpid())
    (slots / "slot002.lock").write_text(f"{point_pool._machine_id()} {os.getpid()} 10000\n")  # stale: not held
    other = open(slots / "slot001.lock", "a+")
    fcntl.flock(other, fcntl.LOCK_EX)
    other.write(f"{point_pool._machine_id()} {os.getpid()} {int(rss) + 50}\n")
    other.flush()
    meminfo = iter([(1000.0, 100.0), (1000.0, 300.0)])  # used 900 -> wait; used 700 + ~50 + 100 > 800 -> wait
    state = {"mem": next(meminfo)}
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        state["mem"] = next(meminfo, (1000.0, 400.0))  # then used 600 + ~50 + 100 <= 800 -> start

    monkeypatch.setattr(point_pool, "meminfo_mb", lambda: state["mem"])
    monkeypatch.setattr(point_pool.time, "sleep", fake_sleep)
    try:
        with point_pool.core_slot():
            record = (slots / "slot000.lock").read_text().split()
    finally:
        other.close()
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)
    assert len(sleeps) == 2
    assert record == [point_pool._machine_id(), str(os.getpid()), "100"]
    assert (slots / "slot002.lock").read_text() == ""  # stale record dropped
    assert (slots / "slot000.lock").read_text() == ""  # reservation cleared on release


def test_nested_slot_reserves_the_task_estimate_and_restores(tmp_path, monkeypatch):
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=1,
                                    mem_limit_frac=0.8, mem_reserve_mb=100)
    monkeypatch.setattr(point_pool, "_rss_mb", lambda pid: 0.0)
    state = {"mem": (1000.0, 500.0)}  # 500 used
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        state["mem"] = (1000.0, 900.0)  # 100 used: the larger reservation fits

    monkeypatch.setattr(point_pool, "meminfo_mb", lambda: state["mem"])
    monkeypatch.setattr(point_pool.time, "sleep", fake_sleep)
    path = slots / "slot000.lock"
    try:
        with point_pool.core_slot():
            outer = path.read_text().split()[2]
            with point_pool.core_slot(reserve_mb=30):  # smaller estimate: no wait, frees the rest
                small = path.read_text().split()[2]
            assert not sleeps
            with point_pool.core_slot(reserve_mb=400):  # 500 + 400 > 800: waits once
                large = path.read_text().split()[2]
            restored = path.read_text().split()[2]
    finally:
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)
    assert (outer, small, large, restored) == ("100", "30", "400", "100")
    assert len(sleeps) == 1


def test_settled_reservations_count_only_a_tail(tmp_path, monkeypatch):
    import fcntl
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=3,
                                    mem_limit_frac=0.8, mem_reserve_mb=100)
    monkeypatch.setattr(point_pool, "_rss_mb", lambda pid: 0.0)
    monkeypatch.setattr(point_pool, "MEM_GATE_TAIL_MB", 10.0)
    held = []
    try:
        for name, reserve, age in (("slot001.lock", 300, 0.0), ("slot002.lock", 300, 2 * point_pool.MEM_GATE_RAMP_S)):
            fh = open(slots / name, "a+")
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(f"{point_pool._machine_id()} {os.getpid()} {reserve}\n")
            fh.flush()
            stamp = time.time() - age
            os.utime(slots / name, (stamp, stamp))
            held.append(fh)
        own = str(slots / "slot000.lock")
        fresh_plus_tail = point_pool._outstanding_reservations_mb(own)
        fresh_only = point_pool._outstanding_reservations_mb(own, tail_mb=0.0)
    finally:
        for fh in held:
            fh.close()
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)
    assert fresh_plus_tail == 310.0  # the fresh reservation in full, the settled one only up to the tail
    assert fresh_only == 300.0


def test_growing_task_ignores_settled_reservations_of_others(tmp_path, monkeypatch):
    import fcntl
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=2,
                                    mem_limit_frac=0.8, mem_reserve_mb=100)
    monkeypatch.setattr(point_pool, "_rss_mb", lambda pid: 0.0)
    monkeypatch.setattr(point_pool, "meminfo_mb", lambda: (1000.0, 600.0))  # 400 used
    monkeypatch.setattr(point_pool, "MEM_GATE_TAIL_MB", 250.0)
    other = open(slots / "slot001.lock", "a+")
    fcntl.flock(other, fcntl.LOCK_EX)
    other.write(f"{point_pool._machine_id()} {os.getpid()} 300\n")  # settled, unused
    other.flush()
    stamp = time.time() - 2 * point_pool.MEM_GATE_RAMP_S
    os.utime(slots / "slot001.lock", (stamp, stamp))
    sleeps = []
    monkeypatch.setattr(point_pool.time, "sleep", lambda s: sleeps.append(s) or (_ for _ in ()).throw(TimeoutError))
    try:
        with point_pool.core_slot():           # new task: 400 + 250 tail + 100 <= 800
            with point_pool.core_slot(reserve_mb=350):  # growth: 400 + 350 <= 800 without the tail
                grown = (slots / "slot000.lock").read_text().split()[2]
    finally:
        other.close()
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)
    assert grown == "350" and not sleeps


def test_cpu_gate_leaves_room_for_other_users():
    from openea_benchmark.point_pool import cpu_gate_admits
    # measured load counts everybody: 70 busy threads leave room for one more 10-thread task
    assert cpu_gate_admits(load=70.0, ncpu=80, held=7, threads=10, frac=1.0)
    assert not cpu_gate_admits(load=75.0, ncpu=80, held=7, threads=10, frac=1.0)
    # slots whose tasks are mostly serial leave cores free: a 12th slot may start
    assert cpu_gate_admits(load=45.0, ncpu=80, held=11, threads=10, frac=1.0)
    # freshly admitted tasks count with full threads until the load average shows them
    assert not cpu_gate_admits(load=45.0, ncpu=80, held=11, threads=10, frac=1.0, ramping=3)
    assert cpu_gate_admits(load=45.0, ncpu=80, held=11, threads=10, frac=1.0, ramping=2)
    # one task always runs; frac 0 disables the gate
    assert cpu_gate_admits(load=500.0, ncpu=80, held=0, threads=10, frac=1.0)
    assert cpu_gate_admits(load=500.0, ncpu=80, held=8, threads=10, frac=0.0)


def test_cpu_gate_queue_is_first_come_first_served():
    from openea_benchmark.point_pool import cpu_queue_update

    alive = {"a", "b", "c"}.__contains__
    queue, first = cpu_queue_update([], "a", 10.0, registered=False, alive=alive)
    assert first and queue == [("a", 10.0)]
    queue, first = cpu_queue_update(queue, "b", 20.0, registered=False, alive=alive)
    assert not first
    # a later poll of b keeps its original place; a stays ahead
    queue, first = cpu_queue_update(queue, "b", 30.0, registered=True, alive=alive)
    assert not first and queue == [("a", 10.0), ("b", 20.0)]
    # a dead waiter is dropped, so b moves to the head
    queue, first = cpu_queue_update(queue, "b", 40.0, registered=True, alive={"b", "c"}.__contains__)
    assert first and queue == [("b", 20.0)]


def test_cpu_gate_queue_equal_times_have_exactly_one_head():
    from openea_benchmark.point_pool import cpu_queue_update

    # waiters registered within the same instant (restart burst): all must agree on one head
    file_queue = [("s5", 5.0), ("s2", 5.0), ("s3", 5.0), ("s9", 4.0)]
    alive = lambda p: p != "s9"
    heads = [own for own in ("s2", "s3", "s5")
             if cpu_queue_update(file_queue, own, 9.0, registered=True, alive=alive)[1]]
    assert heads == ["s2"]


def test_cpu_gate_queue_replaces_stale_entry_of_reused_slot():
    from openea_benchmark.point_pool import cpu_queue_update

    # slot "a" was left in the queue by an earlier holder; the new holder queues behind "b"
    queue, first = cpu_queue_update([("a", 1.0), ("b", 5.0)], "a", 50.0, registered=False,
                                    alive=lambda p: True)
    assert not first and queue == [("b", 5.0), ("a", 50.0)]


def test_finished_task_leaves_the_cpu_gate_ramp(tmp_path, monkeypatch):
    import types
    from openea_benchmark import point_pool as pp

    monkeypatch.setattr(pp, "_SLOT_DIR", str(tmp_path))
    gate = tmp_path / "cpugate.lock"
    gate.write_text(f"{tmp_path}/slot000.lock 1.0\n{tmp_path}/slot001.lock 2.0\n")
    pp._leave_cpu_gate(types.SimpleNamespace(name=f"{tmp_path}/slot000.lock"))
    assert gate.read_text() == f"{tmp_path}/slot001.lock 2.000000\n"


def test_fair_share_rule():
    assert point_pool.fair_share(16, 1) == 16
    assert point_pool.fair_share(16, 3) == 6
    assert point_pool.fair_share(16, 30) == 1
    assert point_pool.fair_share(2, 0) == 2


def test_run_above_its_fair_share_waits_for_a_slot(tmp_path, monkeypatch):
    import fcntl
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=4)
    monkeypatch.setenv(point_pool.RUN_ID_ENV, "111")
    # run 111 already holds two slots, run 222 waits: share ceil(4 / 2) = 2 -> no third slot
    held = []
    for index in (0, 1):
        fh = open(slots / f"slot{index:03d}.lock", "a+")
        fcntl.flock(fh, fcntl.LOCK_EX)
        (slots / f"slot{index:03d}.run").write_text("111\n")
        held.append(fh)
    other = open(slots / "wait_222_9.lock", "a+")
    fcntl.flock(other, fcntl.LOCK_EX)
    (slots / "wait_333_8.lock").write_text("")  # stale waiter: not held, removed
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        other.close()  # run 222 got its slot elsewhere / gave up: 111 alone, share 4

    monkeypatch.setattr(point_pool.time, "sleep", fake_sleep)
    try:
        with point_pool.core_slot():
            assert (slots / "slot002.run").read_text().strip() == "111"
    finally:
        for fh in held:
            fh.close()
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)
    assert len(sleeps) == 1
    assert not (slots / "wait_333_8.lock").exists()
    assert not list(slots.glob("wait_111_*.lock"))  # own wait file removed


def test_busy_cores_ignores_idle_and_iowait():
    # user nice system idle iowait irq softirq steal: 30 busy, 50 idle, 20 iowait of 100 ticks
    before = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    after = [10, 15, 5, 50, 20, 0, 0, 0, 0, 0]
    assert point_pool.busy_cores(before, after, 80) == 24.0
    assert point_pool.busy_cores(before, before, 80) == 0.0


def _record_of_task(slot_file, *_):
    return float(open(slot_file).read().split()[2])


def test_small_stage3_task_enters_the_gate_with_its_own_estimate(tmp_path, monkeypatch):
    from test_reference_branches_v1 import request
    from openea_benchmark.adaptive import stage3_execution
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=1,
                                    mem_limit_frac=0.8, mem_reserve_mb=100)
    monkeypatch.setattr(point_pool, "meminfo_mb", lambda: (1000.0, 900.0))
    budgets = {1.0: 40, 2.0: 400}
    monkeypatch.setattr(stage3_execution, "point_memory_budget_mb", lambda req, s: budgets[req.r_angstrom])
    settings = Stage3ExecutionSettings()
    path = str(slots / "slot000.lock")
    try:
        small = point_pool._slotted_call(_record_of_task, (path, request(0, 1.0), settings))
        chain = point_pool._slotted_call(_record_of_task, (path, [request(0, 1.0), request(1, 2.0)], settings))
        plain = point_pool._slotted_call(_record_of_task, (path,))
        explicit = point_pool._slotted_call(_record_of_task, (path, request(0, 1.0), settings), 300)
        explicit_small = point_pool._slotted_call(_record_of_task, (path,), 30)
    finally:
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)
    assert small == 40 * stage3_execution.GATE_OVERSHOOT  # below the run-wide default: own estimate
    assert chain == 100  # largest point of the chain exceeds the default: default as before
    assert plain == 100  # no requests: run-wide default
    assert explicit == 300  # an explicit reservation is unchanged
    assert explicit_small == 30  # also when it is below the run-wide default


def test_mr_task_reserves_its_estimate_not_the_pyscf_limit():
    from openea_benchmark.mr_competitors import MRCompetitorPolicy, mr_task_reserve_mb
    policy = MRCompetitorPolicy()
    small = mr_task_reserve_mb(("P", "H"), policy)
    assert 4000 <= small < policy.max_memory_mb
    assert mr_task_reserve_mb(("P", "H"), MRCompetitorPolicy(max_memory_mb=3000)) == 3000
    assert mr_task_reserve_mb(("Cr", "Cr"), policy) > mr_task_reserve_mb(("Fe", "O"), policy) > small


def test_large_tasks_share_ram_and_leave_room_for_small_ones(tmp_path, monkeypatch):
    import fcntl
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=3,
                                    mem_limit_frac=0.8, mem_reserve_mb=10)
    monkeypatch.setattr(point_pool, "LARGE_TASK_MIN_MB", 100.0)
    monkeypatch.setenv(point_pool.LARGE_TASK_SHARE_ENV, "0.55")
    monkeypatch.setattr(point_pool, "meminfo_mb", lambda: (1000.0, 950.0))  # 50 used: the total limit 800 never binds here
    monkeypatch.setattr(point_pool, "_rss_mb", lambda pid: 300.0)  # the other large task holds 300
    slots.mkdir(exist_ok=True)
    other = open(slots / "slot002.lock", "a+")
    fcntl.flock(other, fcntl.LOCK_EX)
    other.write(f"{point_pool._machine_id()} {os.getpid()} 300\n")
    other.flush()
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        other.truncate(0)  # the other large task ends
        fcntl.flock(other, fcntl.LOCK_UN)

    monkeypatch.setattr(point_pool.time, "sleep", fake_sleep)
    try:
        with point_pool.core_slot():  # small task: not limited by the large-task share
            assert not sleeps
        with point_pool.core_slot(reserve_mb=200):  # 300 + 200 <= 550: starts
            assert not sleeps
        with point_pool.core_slot(reserve_mb=400):  # 300 + 400 > 550: waits until the other ends
            pass
        assert len(sleeps) == 1
        with point_pool.core_slot(reserve_mb=700):  # alone a large task may exceed the share
            pass
        assert len(sleeps) == 1
    finally:
        other.close()
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)


def test_settled_tails_count_together_with_at_most_the_total_cap(tmp_path, monkeypatch):
    import fcntl
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=6,
                                    mem_limit_frac=0.8, mem_reserve_mb=100)
    monkeypatch.setattr(point_pool, "_rss_mb", lambda pid: 0.0)
    monkeypatch.setattr(point_pool, "MEM_GATE_TAIL_MB", 10.0)
    monkeypatch.setattr(point_pool, "MEM_GATE_TAIL_TOTAL_MB", 25.0)
    held = []
    try:
        records = [("slot001.lock", 300, 0.0)] + [(f"slot00{i}.lock", 300, 2 * point_pool.MEM_GATE_RAMP_S)
                                                  for i in range(2, 6)]
        for name, reserve, age in records:
            fh = open(slots / name, "a+")
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(f"{point_pool._machine_id()} {os.getpid()} {reserve}\n")
            fh.flush()
            stamp = time.time() - age
            os.utime(slots / name, (stamp, stamp))
            held.append(fh)
        outstanding = point_pool._outstanding_reservations_mb(str(slots / "slot000.lock"))
    finally:
        for fh in held:
            fh.close()
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)
    assert outstanding == 325.0  # fresh 300 in full, four settled tails of 10 capped at 25


def test_limit_group_admits_at_most_n_tasks(tmp_path, monkeypatch):
    import fcntl
    from openea_benchmark import point_pool
    monkeypatch.setattr(point_pool, "_SLOT_DIR", str(tmp_path))

    def free(index):
        with open(tmp_path / f"group_g_{index}.lock", "a+") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(fh, fcntl.LOCK_UN)
            return True

    with point_pool.limit_group(("g", 2)):
        assert not free(0) and free(1)
        with point_pool.limit_group(("g", 2)):
            assert not free(0) and not free(1)
    assert free(0) and free(1)
    with point_pool.limit_group(None):   # no group: no lock files
        pass


def test_held_reservations_count_in_full_after_the_ramp(tmp_path, monkeypatch):
    import fcntl
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=4,
                                    mem_limit_frac=0.8, mem_reserve_mb=100)
    monkeypatch.setattr(point_pool, "_rss_mb", lambda pid: 100.0)
    monkeypatch.setattr(point_pool, "MEM_GATE_TAIL_MB", 10.0)
    held = []
    try:
        for name, mark in (("slot001.lock", " HOLD"), ("slot002.lock", "")):
            fh = open(slots / name, "a+")
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(f"{point_pool._machine_id()} {os.getpid()} 400{mark}\n")
            fh.flush()
            stamp = time.time() - 2 * point_pool.MEM_GATE_RAMP_S
            os.utime(slots / name, (stamp, stamp))
            held.append(fh)
        outstanding = point_pool._outstanding_reservations_mb(str(slots / "slot000.lock"))
    finally:
        for fh in held:
            fh.close()
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)
    assert outstanding == 300.0 + 10.0  # held: 400 - 100 resident in full; settled: tail of 10


def test_held_reservations_fall_back_to_the_tail_after_the_hold_time(tmp_path, monkeypatch):
    import fcntl
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=4,
                                    mem_limit_frac=0.8, mem_reserve_mb=100)
    monkeypatch.setattr(point_pool, "_rss_mb", lambda pid: 100.0)
    monkeypatch.setattr(point_pool, "MEM_GATE_TAIL_MB", 10.0)
    held = []
    try:
        for name, age in (("slot001.lock", 0.5 * point_pool.MEM_GATE_HOLD_S),
                          ("slot002.lock", 1.5 * point_pool.MEM_GATE_HOLD_S)):
            fh = open(slots / name, "a+")
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(f"{point_pool._machine_id()} {os.getpid()} 400 HOLD\n")
            fh.flush()
            stamp = time.time() - age
            os.utime(slots / name, (stamp, stamp))
            held.append(fh)
        outstanding = point_pool._outstanding_reservations_mb(str(slots / "slot000.lock"))
    finally:
        for fh in held:
            fh.close()
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)
    assert outstanding == 300.0 + 10.0  # within the hold time in full; after it the tail of 10


def test_core_slot_marks_held_reservations(tmp_path):
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=2,
                                    mem_limit_frac=0.99, mem_reserve_mb=1)
    try:
        with point_pool.core_slot(5.0, hold=True):
            text = point_pool._LOCAL.handle.name
            line = open(text).read().split()
        with point_pool.core_slot(5.0):
            plain = open(point_pool._LOCAL.handle.name).read().split()
    finally:
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)
    assert line[3:] == ["HOLD"] and len(plain) == 3


def test_run_beyond_its_share_borrows_a_free_slot_when_nobody_starves(tmp_path, monkeypatch):
    import fcntl
    slots = tmp_path / "slots"
    point_pool.configure_point_pool(workers=1, threads=1, slot_dir=slots, n_slots=5)
    monkeypatch.setenv(point_pool.RUN_ID_ENV, "111")
    held = []
    for index, run in ((0, "111"), (1, "111"), (2, "111"), (3, "222")):   # share ceil(5/2) = 3
        fh = open(slots / f"slot{index:03d}.lock", "a+")
        fcntl.flock(fh, fcntl.LOCK_EX)
        (slots / f"slot{index:03d}.run").write_text(run + "\n")
        held.append(fh)
    monkeypatch.setattr(point_pool.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("waited")))
    try:
        with point_pool.core_slot():   # 222 holds one slot but does not wait: nobody starves
            assert (slots / "slot004.run").read_text().strip() == "111"
    finally:
        for fh in held:
            fh.close()
        point_pool.shutdown_point_pool()
        point_pool.configure_point_pool(workers=1, threads=1)
