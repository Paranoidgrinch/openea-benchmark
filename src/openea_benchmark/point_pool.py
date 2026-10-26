"""Parallel execution of independent quantum-chemistry tasks (scout blocks, Stage-3 points).

One process pool per workflow run (``configure_point_pool``), shared by every stage.
Each task runs in a spawned worker with its own OpenMP/BLAS thread budget.  A
machine-wide core budget is enforced with *core slots*: lock files under a shared
directory; a task holds one slot (``flock``) while it computes, so several
concurrent workflow runs together never use more than ``n_slots * threads`` cores.
Locks are released by the kernel when a process dies, so a crash never leaks a slot.

Memory gate (``mem_limit_frac`` > 0, needs core slots): a task that has its slot
starts only when the machine-wide memory in use (``MemTotal - MemAvailable``) plus
the not yet allocated part of every running task's reservation plus its own
reservation stays below ``mem_limit_frac * MemTotal``; otherwise it waits.  The
holder writes ``host/pid-namespace pid reserve_mb`` into its slot file, so concurrent runs see
each other's reservations.  ``scripts/openea_mem_watchdog.py`` guards against
tasks that outgrow their reservation.  A task that knows its own need (a CC point
from its basis size, see ``stage3_execution.point_memory_budget_mb``) re-enters
``core_slot(reserve_mb)`` while it holds the slot: its record then carries that
reservation instead of the run-wide default, smaller or larger, until it leaves.

CPU gate (``OPENEA_CPU_LIMIT_FRAC``, default 1.0; 0 disables): a task that got its slot
starts only when the measured machine load (all users, ours included: the larger of the
1-min load average and the current number of running threads) plus its own ``threads``
plus ``threads`` for each of our tasks admitted within the last minute that still runs
(the load average lags behind them) stays within that fraction of the cores.  Short tasks
release their slot quickly and stop counting at once.  Slots
then only cap memory/parallelism; tasks with mostly serial phases leave cores to further
slots.  One task always runs, so the workflow never stalls completely.  Running tasks are
never stopped; under nice 19 they only yield to the other users.

Without a configured pool everything runs serially in the calling process (the
behaviour of all tests and of ``--point-workers 1``); slots still apply when set.

Scratch: every process writes PySCF scratch into its own ``$PYSCF_TMPDIR/<host>_<pid>``
(``scratch.py``); directories of dead processes are swept, and a pool worker deletes its
no longer open scratch files after each task.
"""
from __future__ import annotations

import fcntl
import multiprocessing
import os
import socket
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .scratch import prune_process_scratch, use_process_scratch

_POOL: ProcessPoolExecutor | None = None
_WORKERS = 1
_SLOT_DIR: str | None = None
_N_SLOTS = 0
_THREADS = 1
_MEM_LIMIT_FRAC = 0.0
_MEM_RESERVE_MB = 0
# Tasks reserving more than this run one at a time machine-wide (bigmem.lock in the
# slot directory): the memory gate alone would admit two 150-200 GB points side by
# side, which leaves the shared server no headroom.
BIG_TASK_MB = 100000
_LOCAL = threading.local()
_IN_WORKER = False


def _init_worker(threads: int, slot_dir: str | None, n_slots: int,
                 mem_limit_frac: float = 0.0, mem_reserve_mb: int = 0) -> None:
    global _SLOT_DIR, _N_SLOTS, _THREADS, _MEM_LIMIT_FRAC, _MEM_RESERVE_MB, _IN_WORKER
    _SLOT_DIR, _N_SLOTS, _THREADS, _IN_WORKER = slot_dir, n_slots, threads, True
    _MEM_LIMIT_FRAC, _MEM_RESERVE_MB = mem_limit_frac, mem_reserve_mb
    try:
        os.nice(19 - os.nice(0))  # inherited anyway; enforce the project's nice 19 rule
    except OSError:
        pass
    from .compute_threads import set_compute_threads
    set_compute_threads(threads)
    use_process_scratch()


def configure_point_pool(*, workers: int, threads: int, slot_dir: str | Path | None = None,
                         n_slots: int = 0, mem_limit_frac: float = 0.0, mem_reserve_mb: int = 0) -> None:
    """Create the shared pool (workers > 1), the core-slot budget (n_slots > 0) and the memory gate."""
    global _POOL, _WORKERS, _SLOT_DIR, _N_SLOTS, _THREADS, _MEM_LIMIT_FRAC, _MEM_RESERVE_MB
    shutdown_point_pool()
    _WORKERS, _THREADS = max(1, int(workers)), max(1, int(threads))
    _SLOT_DIR = str(slot_dir) if slot_dir and n_slots > 0 else None
    _N_SLOTS = int(n_slots) if _SLOT_DIR else 0
    _MEM_LIMIT_FRAC = float(mem_limit_frac) if _SLOT_DIR else 0.0
    _MEM_RESERVE_MB = max(0, int(mem_reserve_mb))
    if _SLOT_DIR:
        Path(_SLOT_DIR).mkdir(parents=True, exist_ok=True)
    use_process_scratch()
    if _WORKERS > 1:
        # spawn, not fork: forking a process whose OpenMP runtime is initialised can deadlock.
        _POOL = ProcessPoolExecutor(max_workers=_WORKERS, mp_context=multiprocessing.get_context("spawn"),
                                    initializer=_init_worker,
                                    initargs=(_THREADS, _SLOT_DIR, _N_SLOTS, _MEM_LIMIT_FRAC, _MEM_RESERVE_MB))


def shutdown_point_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=True)
    _POOL = None


def meminfo_mb() -> tuple[float, float]:
    """(MemTotal, MemAvailable) of this machine in MB."""
    values = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, rest = line.split(":", 1)
            values[key] = float(rest.split()[0]) / 1024.0
    return values["MemTotal"], values["MemAvailable"]


def _rss_mb(pid: int) -> float | None:
    """Resident memory of a live process in MB, None if it does not exist."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return 0.0


def _machine_id() -> str:
    """Host plus PID namespace: recorded pids are only meaningful within both."""
    try:
        namespace = os.readlink("/proc/self/ns/pid")
    except OSError:
        namespace = "pid:[?]"
    return f"{socket.gethostname()}/{namespace}"


MEM_GATE_RAMP_S = 900.0     # a reservation counts in full this long after it was recorded
MEM_GATE_TAIL_MB = 4096.0   # afterwards at most this much of it above the task's resident memory


def _outstanding_reservations_mb(own_path: str, tail_mb: float | None = None) -> float:
    """Reserved but not yet resident memory of the tasks running on this host.

    A reservation is a guess made at the start; a task that has not grown into it after
    ``MEM_GATE_RAMP_S`` counts with at most ``tail_mb`` beyond its resident memory (the memory
    watchdog catches the rest).  Counted in full, unused reservations of out-of-core CC tasks
    added up to ~225 GB while 390 GB were free, and every task waited (2026-10-25)."""
    machine, total, now = _machine_id(), 0.0, time.time()
    tail_mb = MEM_GATE_TAIL_MB if tail_mb is None else tail_mb
    for name in os.listdir(_SLOT_DIR):
        path = os.path.join(_SLOT_DIR, name)
        if not name.startswith("slot") or not name.endswith(".lock") or path == own_path:
            continue
        try:
            with open(path, "a+") as fh:
                try:  # a slot nobody holds carries a stale record of a dead task
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fh.truncate(0)
                    continue
                except BlockingIOError:
                    pass
                fh.seek(0)
                fields = fh.read().split()
                age = now - os.fstat(fh.fileno()).st_mtime
        except OSError:
            continue
        if len(fields) != 3 or fields[0] != machine:
            continue
        rss = _rss_mb(int(fields[1]))
        if rss is not None:
            rest = max(0.0, float(fields[2]) - rss)
            total += rest if age < MEM_GATE_RAMP_S else min(rest, tail_mb)
    return total


def _pending_big_reservation_mb() -> float:
    """Reservation of a large task that holds bigmem.lock but has not been admitted yet.

    Other tasks count it, so the memory it waits for drains instead of being taken
    by smaller tasks again and again (2026-10-13: the ClO CCSDTQ point starved)."""
    path = os.path.join(_SLOT_DIR, "bigmem.lock")
    if not os.path.exists(path):
        return 0.0
    try:
        with open(path, "a+") as fh:
            try:  # free lock: nobody is waiting, any content is stale
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh, fcntl.LOCK_UN)
                return 0.0
            except BlockingIOError:
                pass
            fh.seek(0)
            fields = fh.read().split()
    except OSError:
        return 0.0
    if len(fields) != 4 or fields[3] != "PENDING":
        return 0.0
    return float(fields[2])


def _record(handle, reserve_mb: float) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(f"{_machine_id()} {os.getpid()} {reserve_mb:.0f}\n")
    handle.flush()


def _wait_for_memory(handle, reserve_mb: float | None = None, big=None, *, exact: bool = False) -> None:
    """Block until this task's reservation fits under the memory limit, then record it.

    ``reserve_mb`` (per task, e.g. a CCSDTQ storage estimate) replaces the run-wide
    default when it is larger; with ``exact`` (a running task re-reserving its own
    estimate) it replaces the default in either direction.  The task's resident
    memory is already part of the machine's memory in use, so only the rest of its
    reservation is added."""
    if _MEM_LIMIT_FRAC <= 0.0:
        return
    reserve = float(reserve_mb) if exact else max(float(_MEM_RESERVE_MB), float(reserve_mb or 0.0))
    if exact:
        if reserve <= _held_reserve_mb(handle):  # shrinking: already covered, no wait
            _record(handle, reserve)
            return
    if big is not None:  # announce the pending large reservation to everyone else
        big.seek(0)
        big.truncate()
        big.write(f"{_machine_id()} {os.getpid()} {reserve:.0f} PENDING\n")
        big.flush()
    announced = False
    while True:
        with open(os.path.join(_SLOT_DIR, "memgate.lock"), "a+") as gate:
            fcntl.flock(gate, fcntl.LOCK_EX)  # serialise admissions: no two tasks pass on the same free memory
            total, available = meminfo_mb()
            # A running task that grows (``exact``) holds its slot while it waits: it ignores the
            # settled reservations of the others, or tasks that all grow wait for each other.
            others = _outstanding_reservations_mb(handle.name, tail_mb=0.0 if exact else None)
            projected = (total - available) + others + (
                max(0.0, reserve - (_rss_mb(os.getpid()) or 0.0)) if exact else reserve)
            if big is None:
                projected += _pending_big_reservation_mb()
            if projected <= _MEM_LIMIT_FRAC * total:
                if big is not None:
                    big.seek(0)
                    big.truncate()
                    big.write(f"{_machine_id()} {os.getpid()} {reserve:.0f} ADMITTED\n")
                    big.flush()
                _record(handle, reserve)
                return
        if not announced:
            print(f"[MEMORY GATE] pid {os.getpid()} waiting: projected {projected / 1024:.0f} GB > "
                  f"{_MEM_LIMIT_FRAC:.0%} of {total / 1024:.0f} GB", flush=True)
            announced = True
        time.sleep(15.0)


CPU_LIMIT_ENV = "OPENEA_CPU_LIMIT_FRAC"


def _held_slots(own_path: str) -> int:
    """Number of other core slots currently held (by any run sharing the slot directory)."""
    held = 0
    for name in os.listdir(_SLOT_DIR):
        path = os.path.join(_SLOT_DIR, name)
        if not name.startswith("slot") or not name.endswith(".lock") or path == own_path:
            continue
        try:
            with open(path, "a+") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    held += 1
        except OSError:
            continue
    return held


CPU_GATE_RAMP_S = 60.0   # a task admitted this recently may not show in the load average yet
CPU_GATE_HEAD_POLL_S = 0.25


def cpu_gate_admits(load: float, ncpu: int, held: int, threads: int, frac: float, ramping: int = 0) -> bool:
    """Admission rule of the CPU gate (pure, for tests).

    ``load`` is the measured machine load, ``ramping`` the number of our tasks admitted within
    the last ``CPU_GATE_RAMP_S`` that still run (counted with their full threads, since the
    1-min load average lags behind them)."""
    if frac <= 0.0 or held == 0:
        return True
    return load + (ramping + 1) * threads <= frac * ncpu


def _machine_load() -> float:
    """max(1-min load average, running threads now)."""
    running = 0.0
    try:
        with open("/proc/stat") as fh:
            for line in fh:
                if line.startswith("procs_running"):
                    running = float(line.split()[1]) - 1.0   # minus this reader
                    break
    except OSError:
        pass
    return max(os.getloadavg()[0], running)


def _still_held(path: str) -> bool:
    try:
        with open(path, "a+") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return False
            except BlockingIOError:
                return True
    except OSError:
        return False


def _read_entries(fh) -> list[tuple[str, float]]:
    """``"<slot path> <time>"`` lines of a gate file (malformed lines are skipped)."""
    fh.seek(0)
    entries = []
    for line in fh.read().splitlines():
        try:
            path, stamp = line.rsplit(" ", 1)
            entries.append((path, float(stamp)))
        except ValueError:
            continue
    return entries


def _write_entries(fh, entries: list[tuple[str, float]]) -> None:
    fh.seek(0)
    fh.truncate()
    fh.write("".join(f"{p} {t:.6f}\n" for p, t in entries))
    fh.flush()


def cpu_queue_update(queue: list[tuple[str, float]], own: str, now: float, *, registered: bool,
                     alive: Callable[[str], bool]) -> tuple[list[tuple[str, float]], bool]:
    """FIFO bookkeeping of the CPU gate (pure, for tests).

    Drops entries of slots no longer held (dead waiters), (re-)registers ``own`` on its first
    call (a stale entry left on the same slot file by an earlier holder is replaced) and returns
    the queue ordered by waiting time plus whether ``own`` is at its head.  Equal times are
    ordered by slot path, so every waiter sees the same head (2026-10-23: 16 waiters registered
    within 0.1 s at a restart, each sorted itself behind the others and none was ever admitted)."""
    kept = [(p, t) for p, t in queue if p != own and alive(p)]
    mine = [(p, t) for p, t in queue if p == own]
    kept.append(mine[0] if registered and mine else (own, now))
    kept.sort(key=lambda e: (e[1], e[0]))
    return kept, kept[0][0] == own


def _wait_for_cpu(handle) -> None:
    """Wait until the measured machine load leaves room for one more task.

    Waiters are served first come, first served (queue in ``cpuqueue.lock``): without the queue
    whichever waiter polled first after the load dropped won, and single tasks starved for hours."""
    try:
        frac = float(os.environ.get(CPU_LIMIT_ENV, "1.0"))
    except ValueError:
        frac = 1.0
    if frac <= 0.0:
        return
    ncpu = os.cpu_count() or 1
    announced = registered = False
    while True:
        with open(os.path.join(_SLOT_DIR, "cpugate.lock"), "a+") as gate, \
                open(os.path.join(_SLOT_DIR, "cpuqueue.lock"), "a+") as queue_fh:
            fcntl.flock(gate, fcntl.LOCK_EX)
            now = time.time()
            recent = [(p, t) for p, t in _read_entries(gate)   # "<slot path> <admission time>"
                      if now - t < CPU_GATE_RAMP_S and p != handle.name and _still_held(p)]
            queue, first = cpu_queue_update(_read_entries(queue_fh), handle.name, now,
                                            registered=registered, alive=_still_held)
            registered = True
            held = _held_slots(handle.name)
            load = _machine_load()
            if first and cpu_gate_admits(load, ncpu, held, _THREADS, frac, ramping=len(recent)):
                _write_entries(gate, recent + [(handle.name, now)])
                _write_entries(queue_fh, queue[1:])
                return
            _write_entries(queue_fh, queue)
        # The front of the queue polls often: short tasks (cache replay at a restart) pass one
        # per poll, and the next waiter must notice at once that it moved to the head.
        position = next((i for i, (p, _) in enumerate(queue) if p == handle.name), 0)
        if not announced:
            # Behind others in the queue the load is not the reason to wait (2026-10-26: the
            # "load 38 + 1 x 10 > 96" lines of a cache replay looked like a stuck gate).
            reason = (f"load {load:.0f} + {len(recent) + 1} x {_THREADS} threads > {frac:g} x {ncpu} cores"
                      if first else f"position {position + 1} in the queue")
            print(f"[CPU GATE] pid {os.getpid()} waiting: {reason} ({held} slots held, {len(queue)} in queue)",
                  flush=True)
            announced = True
        time.sleep(CPU_GATE_HEAD_POLL_S if position < 3 else 5.0)


def _leave_cpu_gate(handle) -> None:
    """Drop the own ramp entry when the task ends: the next holder of this slot file must not
    count as a ramping task (2026-10-23: finished short tasks blocked the gate for 60 s each)."""
    if not _SLOT_DIR or not os.path.exists(os.path.join(_SLOT_DIR, "cpugate.lock")):
        return
    with open(os.path.join(_SLOT_DIR, "cpugate.lock"), "a+") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX)
        entries = _read_entries(gate)
        kept = [(p, t) for p, t in entries if p != handle.name]
        if len(kept) != len(entries):
            _write_entries(gate, kept)


def _held_reserve_mb(handle) -> float:
    """Reservation currently recorded in this task's own slot file (0 if none)."""
    try:
        handle.seek(0)
        fields = handle.read().split()
        return float(fields[2]) if len(fields) == 3 else 0.0
    except (OSError, ValueError):
        return 0.0


def pool_workers() -> int:
    return _WORKERS if _POOL is not None else 1


@contextmanager
def core_slot(reserve_mb: float | None = None):
    """Hold one machine-wide core slot (no-op without a slot budget; re-entrant per thread).

    Re-entered with ``reserve_mb`` while the slot is held, the slot's reservation becomes
    ``reserve_mb`` (waiting at the memory gate if it grows) and is restored on exit."""
    if not _SLOT_DIR:
        yield
        return
    if getattr(_LOCAL, "held", False):
        handle = getattr(_LOCAL, "handle", None)
        if reserve_mb is None or handle is None or _MEM_LIMIT_FRAC <= 0.0:
            yield
            return
        previous = _held_reserve_mb(handle)
        _wait_for_memory(handle, reserve_mb, exact=True)
        try:
            yield
        finally:
            _record(handle, previous)
        return
    big = None
    if reserve_mb is not None and float(reserve_mb) > BIG_TASK_MB:
        big = open(os.path.join(_SLOT_DIR, "bigmem.lock"), "a+")
        print(f"[MEMORY GATE] pid {os.getpid()} large task ({float(reserve_mb) / 1024:.0f} GB): "
              f"waiting for the one-at-a-time slot", flush=True)
        fcntl.flock(big, fcntl.LOCK_EX)
    handle = None
    while handle is None:
        for index in range(_N_SLOTS):
            fh = open(os.path.join(_SLOT_DIR, f"slot{index:03d}.lock"), "a+")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            handle = fh
            break
        if handle is None:
            time.sleep(2.0)
    _LOCAL.held, _LOCAL.handle = True, handle
    try:
        _wait_for_cpu(handle)
        _wait_for_memory(handle, reserve_mb, big)
        yield
    finally:
        _LOCAL.held, _LOCAL.handle = False, None
        _leave_cpu_gate(handle)
        handle.seek(0)
        handle.truncate()
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
        if big is not None:
            big.seek(0)
            big.truncate()
            fcntl.flock(big, fcntl.LOCK_UN)
            big.close()
        if _IN_WORKER:
            prune_process_scratch()  # the task is over: its scratch files are garbage


def _slotted_call(fn: Callable[..., Any], args: tuple, reserve_mb: float | None = None) -> Any:
    with core_slot(reserve_mb):
        return fn(*args)


def run_parallel(fn: Callable[..., Any], arg_tuples: Iterable[tuple], *,
                 on_result: Callable[[int, Any], None] | None = None,
                 reserve_mb: Sequence[float | None] | None = None) -> list[Any]:
    """``[fn(*args) for args in arg_tuples]`` on the pool; results in input order.

    ``fn`` and its arguments must be picklable when a pool is configured.
    ``on_result(index, result)`` is called in the calling process as each task
    finishes (used to persist results immediately, so a kill loses only running tasks).
    The first task exception is re-raised after the remaining tasks have finished.
    ``reserve_mb[i]`` is task i's memory reservation for the gate (default: run-wide).
    """
    items: Sequence[tuple] = list(arg_tuples)
    reserves = list(reserve_mb) if reserve_mb is not None else [None] * len(items)
    results: list[Any] = [None] * len(items)
    if _POOL is None or len(items) < 2:
        for index, args in enumerate(items):
            results[index] = _slotted_call(fn, args, reserves[index])
            if on_result is not None:
                on_result(index, results[index])
        return results
    futures = {_POOL.submit(_slotted_call, fn, args, reserves[index]): index for index, args in enumerate(items)}
    error: BaseException | None = None
    pending = set(futures)
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            index = futures[future]
            try:
                results[index] = future.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                error = error or exc
                continue
            if on_result is not None:
                on_result(index, results[index])
    if error is not None:
        raise error
    return results
