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
tasks that outgrow their reservation.

Without a configured pool everything runs serially in the calling process (the
behaviour of all tests and of ``--point-workers 1``); slots still apply when set.
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

_POOL: ProcessPoolExecutor | None = None
_WORKERS = 1
_SLOT_DIR: str | None = None
_N_SLOTS = 0
_THREADS = 1
_MEM_LIMIT_FRAC = 0.0
_MEM_RESERVE_MB = 0
_LOCAL = threading.local()


def _init_worker(threads: int, slot_dir: str | None, n_slots: int,
                 mem_limit_frac: float = 0.0, mem_reserve_mb: int = 0) -> None:
    global _SLOT_DIR, _N_SLOTS, _THREADS, _MEM_LIMIT_FRAC, _MEM_RESERVE_MB
    _SLOT_DIR, _N_SLOTS, _THREADS = slot_dir, n_slots, threads
    _MEM_LIMIT_FRAC, _MEM_RESERVE_MB = mem_limit_frac, mem_reserve_mb
    try:
        os.nice(19 - os.nice(0))  # inherited anyway; enforce the project's nice 19 rule
    except OSError:
        pass
    from .compute_threads import set_compute_threads
    set_compute_threads(threads)


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


def _outstanding_reservations_mb(own_path: str) -> float:
    """Reserved but not yet resident memory of the tasks running on this host."""
    machine, total = _machine_id(), 0.0
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
        except OSError:
            continue
        if len(fields) != 3 or fields[0] != machine:
            continue
        rss = _rss_mb(int(fields[1]))
        if rss is not None:
            total += max(0.0, float(fields[2]) - rss)
    return total


def _wait_for_memory(handle) -> None:
    """Block until this task's reservation fits under the memory limit, then record it."""
    if _MEM_LIMIT_FRAC <= 0.0:
        return
    announced = False
    while True:
        with open(os.path.join(_SLOT_DIR, "memgate.lock"), "a+") as gate:
            fcntl.flock(gate, fcntl.LOCK_EX)  # serialise admissions: no two tasks pass on the same free memory
            total, available = meminfo_mb()
            projected = (total - available) + _outstanding_reservations_mb(handle.name) + _MEM_RESERVE_MB
            if projected <= _MEM_LIMIT_FRAC * total:
                handle.seek(0)
                handle.truncate()
                handle.write(f"{_machine_id()} {os.getpid()} {_MEM_RESERVE_MB}\n")
                handle.flush()
                return
        if not announced:
            print(f"[MEMORY GATE] pid {os.getpid()} waiting: projected {projected / 1024:.0f} GB > "
                  f"{_MEM_LIMIT_FRAC:.0%} of {total / 1024:.0f} GB", flush=True)
            announced = True
        time.sleep(15.0)


def pool_workers() -> int:
    return _WORKERS if _POOL is not None else 1


@contextmanager
def core_slot():
    """Hold one machine-wide core slot (no-op without a slot budget; re-entrant per thread)."""
    if not _SLOT_DIR or getattr(_LOCAL, "held", False):
        yield
        return
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
    _LOCAL.held = True
    try:
        _wait_for_memory(handle)
        yield
    finally:
        _LOCAL.held = False
        handle.seek(0)
        handle.truncate()
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _slotted_call(fn: Callable[..., Any], args: tuple) -> Any:
    with core_slot():
        return fn(*args)


def run_parallel(fn: Callable[..., Any], arg_tuples: Iterable[tuple], *,
                 on_result: Callable[[int, Any], None] | None = None) -> list[Any]:
    """``[fn(*args) for args in arg_tuples]`` on the pool; results in input order.

    ``fn`` and its arguments must be picklable when a pool is configured.
    ``on_result(index, result)`` is called in the calling process as each task
    finishes (used to persist results immediately, so a kill loses only running tasks).
    The first task exception is re-raised after the remaining tasks have finished.
    """
    items: Sequence[tuple] = list(arg_tuples)
    results: list[Any] = [None] * len(items)
    if _POOL is None or len(items) < 2:
        for index, args in enumerate(items):
            results[index] = _slotted_call(fn, args)
            if on_result is not None:
                on_result(index, results[index])
        return results
    futures = {_POOL.submit(_slotted_call, fn, args): index for index, args in enumerate(items)}
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
