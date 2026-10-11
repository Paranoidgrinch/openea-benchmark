"""Parallel execution of independent quantum-chemistry tasks (scout blocks, Stage-3 points).

One process pool per workflow run (``configure_point_pool``), shared by every stage.
Each task runs in a spawned worker with its own OpenMP/BLAS thread budget.  A
machine-wide core budget is enforced with *core slots*: lock files under a shared
directory; a task holds one slot (``flock``) while it computes, so several
concurrent workflow runs together never use more than ``n_slots * threads`` cores.
Locks are released by the kernel when a process dies, so a crash never leaks a slot.

Without a configured pool everything runs serially in the calling process (the
behaviour of all tests and of ``--point-workers 1``); slots still apply when set.
"""
from __future__ import annotations

import fcntl
import multiprocessing
import os
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
_LOCAL = threading.local()


def _init_worker(threads: int, slot_dir: str | None, n_slots: int) -> None:
    global _SLOT_DIR, _N_SLOTS, _THREADS
    _SLOT_DIR, _N_SLOTS, _THREADS = slot_dir, n_slots, threads
    try:
        os.nice(19 - os.nice(0))  # inherited anyway; enforce the project's nice 19 rule
    except OSError:
        pass
    from .compute_threads import set_compute_threads
    set_compute_threads(threads)


def configure_point_pool(*, workers: int, threads: int, slot_dir: str | Path | None = None,
                         n_slots: int = 0) -> None:
    """Create the shared pool (workers > 1) and the core-slot budget (n_slots > 0)."""
    global _POOL, _WORKERS, _SLOT_DIR, _N_SLOTS, _THREADS
    shutdown_point_pool()
    _WORKERS, _THREADS = max(1, int(workers)), max(1, int(threads))
    _SLOT_DIR = str(slot_dir) if slot_dir and n_slots > 0 else None
    _N_SLOTS = int(n_slots) if _SLOT_DIR else 0
    if _SLOT_DIR:
        Path(_SLOT_DIR).mkdir(parents=True, exist_ok=True)
    if _WORKERS > 1:
        # spawn, not fork: forking a process whose OpenMP runtime is initialised can deadlock.
        _POOL = ProcessPoolExecutor(max_workers=_WORKERS, mp_context=multiprocessing.get_context("spawn"),
                                    initializer=_init_worker, initargs=(_THREADS, _SLOT_DIR, _N_SLOTS))


def shutdown_point_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=True)
    _POOL = None


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
        yield
    finally:
        _LOCAL.held = False
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
