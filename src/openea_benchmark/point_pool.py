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
# side, which leaves the shared server no headroom.  100 GB until 2026-11-27; raised to
# 150 GB (user decision) so two 100-150 GB points (CrO/MoO aTZ CCSDT, TiO CCSDTQ) run side
# by side on the 564 GB server (2 x 150 GB stays below the 70 % gate).  OPENEA_BIG_TASK_MB overrides.
BIG_TASK_MB = int(os.environ.get("OPENEA_BIG_TASK_MB", 150000))
# Fair share of the core slots: a run (one workflow process and its workers) takes a free
# slot only while it holds fewer than ceil(n_slots / active runs), active = holding a slot or
# waiting for one.  Running tasks are never stopped; a run above its share just gets no new
# slot while others wait (2026-11-07/08: CeO, GdO and LuO held all 16 slots for a day and
# every light molecule waited).  The run of a held slot is in ``slotNNN.run``; waiting
# workers hold ``wait_<run>_<pid>.lock``.
RUN_ID_ENV = "OPENEA_RUN_ID"
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
    os.environ.setdefault(RUN_ID_ENV, str(os.getpid()))  # inherited by the spawned workers
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


def memory_headroom_mb() -> float | None:
    """Memory the gate could still admit now (limit minus in use, outstanding reservations
    and a pending large one); None without a memory gate."""
    if not _SLOT_DIR or _MEM_LIMIT_FRAC <= 0.0:
        return None
    total, available = meminfo_mb()
    used = (total - available) + _outstanding_reservations_mb("") + _pending_big_reservation_mb()
    return max(0.0, _MEM_LIMIT_FRAC * total - used)


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
# ... and all those tails together at most this much: settled tasks rarely grow at the same time,
# and the memory watchdog catches the rest (2026-11-14: 24 tails held 98 GB on paper while LiH/PH
# ZPE and the Na2 MR blocks waited 3.5 h for 4 GB).
MEM_GATE_TAIL_TOTAL_MB = 16384.0
# A task whose reservation is an exact estimate of memory it grows into slowly marks its
# record HOLD: the reservation then counts in full for MEM_GATE_HOLD_S, not only for
# MEM_GATE_RAMP_S.  In-core CC points fill their DIIS vectors over the first ~6 iterations
# (20-60 min); after the ramp the gate gave the rest away and the watchdog killed the
# growing point (ScO aTZ CCSDT, 73 GB of a 76 GB estimate, 2026-11-18 00:29 and 07:55).
# Not for the task's whole life: estimates above the real need then blocked RAM nobody
# used (IO out-of-core 90 GB reserved / 51 GB resident, a VO point 76 / 35 GB; ScO and OO
# waited 10 h at 38 of 80 busy cores, 2026-11-18/19).  Three hours cover the growth
# (ScO aTZ CCSDT: 20-60 min to its full DIIS); afterwards the normal tail rule applies.
HOLD_MARK = "HOLD"
MEM_GATE_HOLD_S = 3 * 3600.0


def _held_records(own_path: str):
    """(reservation MB, resident MB, age s) of every other task holding a slot on this host;
    stale records of slots nobody holds are cleared on the way."""
    machine, now = _machine_id(), time.time()
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
        if len(fields) not in (3, 4) or fields[0] != machine:
            continue
        rss = _rss_mb(int(fields[1]))
        if rss is not None:
            # A held reservation (CC points) counts in full for MEM_GATE_HOLD_S.
            hold = len(fields) == 4 and fields[3] == HOLD_MARK and age < MEM_GATE_HOLD_S
            yield float(fields[2]), rss, (0.0 if hold else age)


def _unallocated_mb(reserve: float, rss: float, age: float, tail_mb: float) -> float:
    rest = max(0.0, reserve - rss)
    return rest if age < MEM_GATE_RAMP_S else min(rest, tail_mb)


def _outstanding_reservations_mb(own_path: str, tail_mb: float | None = None) -> float:
    """Reserved but not yet resident memory of the tasks running on this host.

    A reservation is a guess made at the start; a task that has not grown into it after
    ``MEM_GATE_RAMP_S`` counts with at most ``tail_mb`` beyond its resident memory (the memory
    watchdog catches the rest); all tails together count with at most ``MEM_GATE_TAIL_TOTAL_MB``.
    Counted in full, unused reservations of out-of-core CC tasks
    added up to ~225 GB while 390 GB were free, and every task waited (2026-10-25)."""
    tail_mb = MEM_GATE_TAIL_MB if tail_mb is None else tail_mb
    fresh = tails = 0.0
    for r, rss, age in _held_records(own_path):
        if age < MEM_GATE_RAMP_S:
            fresh += _unallocated_mb(r, rss, age, tail_mb)
        else:
            tails += _unallocated_mb(r, rss, age, tail_mb)
    return fresh + min(tails, MEM_GATE_TAIL_TOTAL_MB)


# Large tasks (reservation above LARGE_TASK_MIN_MB: CCSDT/CCSDTQ, big FNO points) may together
# use at most this share of RAM; the rest up to the memory limit stays free for small tasks
# (2026-11-14: ~10 out-of-core CC tasks waiting on GPFS held 250 GB at 20 busy cores while
# LiH/SeH ZPE and CV points and the Na2 MR blocks waited a day for a few GB).
LARGE_TASK_MIN_MB = 20000.0
LARGE_TASK_SHARE_ENV = "OPENEA_LARGE_TASK_SHARE"
DEFAULT_LARGE_TASK_SHARE = 0.55


def _large_task_share() -> float:
    try:
        return float(os.environ.get(LARGE_TASK_SHARE_ENV, DEFAULT_LARGE_TASK_SHARE))
    except ValueError:
        return DEFAULT_LARGE_TASK_SHARE


def _large_tasks_mb(own_path: str) -> float:
    """Memory held or still claimed by the other large tasks on this host."""
    return sum(rss + _unallocated_mb(r, rss, age, MEM_GATE_TAIL_MB)
               for r, rss, age in _held_records(own_path) if r > LARGE_TASK_MIN_MB)


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
    hold = f" {HOLD_MARK}" if getattr(_LOCAL, "hold", False) else ""
    handle.write(f"{_machine_id()} {os.getpid()} {reserve_mb:.0f}{hold}\n")
    handle.flush()


def _wait_for_memory(handle, reserve_mb: float | None = None, big=None, *, exact: bool = False,
                     default_mb: float | None = None) -> None:
    """Block until this task's reservation fits under the memory limit, then record it.

    ``reserve_mb`` (per task, e.g. a CCSDTQ storage estimate) replaces the run-wide
    default when it is larger; with ``exact`` (a running task re-reserving its own
    estimate) it replaces the default in either direction.  The task's resident
    memory is already part of the machine's memory in use, so only the rest of its
    reservation is added."""
    if _MEM_LIMIT_FRAC <= 0.0:
        return
    default = float(_MEM_RESERVE_MB) if default_mb is None else float(default_mb)
    reserve = float(reserve_mb) if exact else max(default, float(reserve_mb or 0.0))
    if exact:
        if reserve <= _held_reserve_mb(handle):  # shrinking: already covered, no wait
            _record(handle, reserve)
            return
    def announce(state: str) -> None:
        big.seek(0)
        big.truncate()
        big.write(f"{_machine_id()} {os.getpid()} {reserve:.0f} {state}\n")
        big.flush()

    large = reserve > LARGE_TASK_MIN_MB
    share = _large_task_share()
    if big is not None:  # announce the pending large reservation to everyone else
        announce("PENDING")
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
            # A large task also waits for its share; alone it always may start (its own
            # reservation may exceed the share).
            large_held = _large_tasks_mb(handle.name) if large and share > 0.0 else 0.0
            share_ok = not large or share <= 0.0 or large_held == 0.0 or large_held + reserve <= share * total
            if projected <= _MEM_LIMIT_FRAC * total and share_ok:
                if big is not None:
                    announce("ADMITTED")
                _record(handle, reserve)
                return
            if big is not None:  # waiting for the share is no claim on the small tasks' memory
                announce("PENDING" if share_ok else "SHARE_WAIT")
        if not announced:
            reason = (f"projected {projected / 1024:.0f} GB > {_MEM_LIMIT_FRAC:.0%} of {total / 1024:.0f} GB"
                      if share_ok else
                      f"large tasks {(large_held + reserve) / 1024:.0f} GB > {share:.0%} of {total / 1024:.0f} GB")
            print(f"[MEMORY GATE] pid {os.getpid()} waiting: {reason}", flush=True)
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


CPU_GATE_RAMP_S = 60.0   # a task admitted this recently may still be starting up (SCF setup, I/O)
CPU_GATE_HEAD_POLL_S = 0.25


def cpu_gate_admits(load: float, ncpu: int, held: int, threads: int, frac: float, ramping: int = 0) -> bool:
    """Admission rule of the CPU gate (pure, for tests).

    ``load`` is the measured number of busy cores, ``ramping`` the number of our tasks admitted
    within the last ``CPU_GATE_RAMP_S`` that still run (counted with their full threads, since
    a starting task does not use its cores yet)."""
    if frac <= 0.0 or held == 0:
        return True
    return load + (ramping + 1) * threads <= frac * ncpu


CPU_BUSY_SAMPLE_S = 2.0


def busy_cores(before: Sequence[int], after: Sequence[int], ncpu: int) -> float:
    """Cores in use between two /proc/stat ``cpu`` lines (pure, for tests).

    Fields: user nice system idle iowait irq softirq steal ...; idle and iowait are not use
    (a task waiting for GPFS holds no core; 2026-11-10: load ~25-80, 15 cores busy)."""
    delta = [b - a for a, b in zip(before[:8], after[:8])]
    total = sum(delta)
    if total <= 0:
        return 0.0
    return (total - delta[3] - delta[4]) / total * ncpu


def _cpu_times() -> list[int]:
    with open("/proc/stat") as fh:
        return [int(x) for x in fh.readline().split()[1:]]


_CPU_SAMPLE: dict[str, Any] = {}


def _machine_load() -> float:
    """Cores in use machine-wide (all users) since this process's previous reading, at least
    ``CPU_BUSY_SAMPLE_S`` back (the first call waits that long; later calls return the last
    value until the interval has passed, so the 0.25 s polls of the queue head stay cheap).

    Not the load average: it counts runnable and I/O-blocked threads, and spinning or waiting
    OpenMP threads of nice-19 tasks made it read 2-5x the cores really in use, so the gate held
    tasks back on an idle machine."""
    ncpu = os.cpu_count() or 1
    try:
        if not _CPU_SAMPLE:
            _CPU_SAMPLE.update(t=time.monotonic(), times=_cpu_times())
            time.sleep(CPU_BUSY_SAMPLE_S)
        now, times = time.monotonic(), _cpu_times()
        if "busy" not in _CPU_SAMPLE or now - _CPU_SAMPLE["t"] >= CPU_BUSY_SAMPLE_S:
            _CPU_SAMPLE.update(busy=busy_cores(_CPU_SAMPLE["times"], times, ncpu), t=now, times=times)
        return _CPU_SAMPLE["busy"]
    except (OSError, ValueError):
        return os.getloadavg()[0]


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
            reason = (f"busy {load:.0f} + {len(recent) + 1} x {_THREADS} threads > {frac:g} x {ncpu} cores"
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
        return float(fields[2]) if len(fields) in (3, 4) else 0.0
    except (OSError, ValueError):
        return 0.0


def pool_workers() -> int:
    return _WORKERS if _POOL is not None else 1


@contextmanager
def core_slot(reserve_mb: float | None = None, *, default_mb: float | None = None, hold: bool = False):
    """Hold one machine-wide core slot (no-op without a slot budget; re-entrant per thread).

    Re-entered with ``reserve_mb`` while the slot is held, the slot's reservation becomes
    ``reserve_mb`` (waiting at the memory gate if it grows) and is restored on exit.
    ``default_mb`` replaces the run-wide default reservation for this task (a smaller
    estimate of the task's own need).  ``hold``: the reservation counts in full for
    MEM_GATE_HOLD_S after it was recorded (see ``HOLD_MARK``)."""
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
    big, own_big = None, False
    if reserve_mb is not None and float(reserve_mb) > BIG_TASK_MB:
        big = getattr(_LOCAL, "big", None)   # taken already by big_task_slot (before the limit group)
        if big is None:
            big, own_big = _lock_big_slot(reserve_mb), True
    handle = _acquire_fair_slot()
    _LOCAL.held, _LOCAL.handle, _LOCAL.hold = True, handle, bool(hold)
    try:
        _wait_for_cpu(handle)
        _wait_for_memory(handle, reserve_mb, big, default_mb=default_mb)
        yield
    finally:
        _LOCAL.held, _LOCAL.handle, _LOCAL.hold = False, None, False
        _leave_cpu_gate(handle)
        handle.seek(0)
        handle.truncate()
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
        if own_big:
            _release_big_slot(big)
        if _IN_WORKER:
            prune_process_scratch()  # the task is over: its scratch files are garbage


def _lock_big_slot(reserve_mb: float):
    big = open(os.path.join(_SLOT_DIR, "bigmem.lock"), "a+")
    print(f"[MEMORY GATE] pid {os.getpid()} large task ({float(reserve_mb) / 1024:.0f} GB): "
          f"waiting for the one-at-a-time slot", flush=True)
    fcntl.flock(big, fcntl.LOCK_EX)
    return big


def _release_big_slot(big) -> None:
    big.seek(0)
    big.truncate()
    fcntl.flock(big, fcntl.LOCK_UN)
    big.close()


@contextmanager
def big_task_slot(reserve_mb: float | None):
    """The one-at-a-time slot (bigmem.lock) of a task above BIG_TASK_MB, taken before its
    limit group and core slot (``core_slot`` then uses it).  A large task waiting for its
    turn must not hold a group slot meanwhile: 2026-11-24..27 the MoO aug-cc-pVTZ CCSDT point
    held cc_out_of_core slot 1 for three days behind the FeO point in bigmem.lock, and the
    ClCl and CrO out-of-core points behind it did not start."""
    if (not _SLOT_DIR or reserve_mb is None or float(reserve_mb) <= BIG_TASK_MB
            or getattr(_LOCAL, "held", False) or getattr(_LOCAL, "big", None) is not None):
        yield
        return
    big = _lock_big_slot(reserve_mb)
    _LOCAL.big = big
    try:
        yield
    finally:
        _LOCAL.big = None
        _release_big_slot(big)


def _run_id() -> str:
    return os.environ.get(RUN_ID_ENV) or str(os.getpid())


def fair_share(n_slots: int, active_runs: int) -> int:
    """Slots one run may hold while ``active_runs`` runs hold or wait for slots (pure, for tests)."""
    return max(1, -(-n_slots // max(1, active_runs)))


def _slot_runs() -> tuple[dict[str, int], set[str]]:
    """(held slots per run, runs with a waiting worker); stale wait files are removed."""
    held: dict[str, int] = {}
    waiting: set[str] = set()
    for name in os.listdir(_SLOT_DIR):
        path = os.path.join(_SLOT_DIR, name)
        if name.startswith("slot") and name.endswith(".lock") and _still_held(path):
            try:
                run = Path(path[:-len(".lock")] + ".run").read_text().strip() or "?"
            except OSError:
                run = "?"   # holder of an older version: counts as a run of its own
            held[run] = held.get(run, 0) + 1
        elif name.startswith("wait_") and name.endswith(".lock"):
            if _still_held(path):
                waiting.add(name[len("wait_"):].rsplit("_", 1)[0])
            else:
                try:
                    os.unlink(path)
                except OSError:
                    pass
    return held, waiting


def _acquire_fair_slot():
    """Lock a free core slot once this run is below its fair share, or beyond it while no other
    run waits below its share; returns the slot handle."""
    run = _run_id()
    wait_path = os.path.join(_SLOT_DIR, f"wait_{run}_{os.getpid()}.lock")
    waiter = open(wait_path, "a+")
    fcntl.flock(waiter, fcntl.LOCK_EX)
    announced = False
    try:
        while True:
            with open(os.path.join(_SLOT_DIR, "fairgate.lock"), "a+") as gate:
                fcntl.flock(gate, fcntl.LOCK_EX)  # no two workers of one run pass on the same count
                held, waiting = _slot_runs()
                share = fair_share(_N_SLOTS, len(set(held) | waiting | {run}))
                # Work conserving: beyond its share a run may take a free slot while no other
                # run waits below its own share (verify 11-25: load 26 of 80 with tasks queued
                # behind their run's share).  A run that starts waiting below its share gets
                # the next free slot, since runs at or above theirs then stop taking slots.
                starved = any(held.get(other, 0) < share for other in waiting if other != run)
                if held.get(run, 0) < share or not starved:
                    for index in range(_N_SLOTS):
                        fh = open(os.path.join(_SLOT_DIR, f"slot{index:03d}.lock"), "a+")
                        try:
                            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            fh.close()
                            continue
                        Path(os.path.join(_SLOT_DIR, f"slot{index:03d}.run")).write_text(run + "\n")
                        return fh
                elif not announced:
                    print(f"[SLOT SHARE] pid {os.getpid()} waiting: run {run} holds {held.get(run, 0)} "
                          f"of its {share} slots ({len(set(held) | waiting | {run})} active runs)", flush=True)
                    announced = True
            time.sleep(2.0)
    finally:
        fcntl.flock(waiter, fcntl.LOCK_UN)
        waiter.close()
        try:
            os.unlink(wait_path)
        except OSError:
            pass


def stage3_task_estimate_mb(args: tuple) -> float | None:
    """Memory estimate of a task that runs Stage-3 points (CBS, CV, SR, refinement, ZPE):
    the largest point budget among the requests in its arguments, plus PySCF's overshoot.
    None when the task carries no requests or not exactly one settings object."""
    try:
        from .adaptive.stage3_execution import (GATE_OVERSHOOT, Stage3ExecutionRequest,
                                                Stage3ExecutionSettings, point_memory_budget_mb)
    except Exception:  # noqa: BLE001 - no estimate, the run-wide default applies
        return None
    flat = [x for a in args for x in (a if isinstance(a, (list, tuple)) else (a,))]
    requests = [x for x in flat if isinstance(x, Stage3ExecutionRequest)]
    settings = [x for x in flat if isinstance(x, Stage3ExecutionSettings)]
    if not requests or len(settings) != 1:
        return None
    try:
        return max(point_memory_budget_mb(r, settings[0]) for r in requests) * GATE_OVERSHOOT
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def limit_group(group: tuple[str, int] | None):
    """At most ``n`` tasks of group ``(name, n)`` run at a time machine-wide (no-op without
    a slot directory or group).  Taken before the core slot, so a waiting task holds
    neither a slot nor memory; a large task takes its one-at-a-time slot first
    (``big_task_slot``), so it never holds a group slot while waiting for its turn."""
    if group is None or not _SLOT_DIR:
        yield
        return
    name, n = group
    announced = False
    while True:
        for index in range(max(1, int(n))):
            fh = open(os.path.join(_SLOT_DIR, f"group_{name}_{index}.lock"), "a+")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            return
        if not announced:
            print(f"[LIMIT GROUP] pid {os.getpid()} waiting: {n} tasks of group {name} running", flush=True)
            announced = True
        time.sleep(10.0)


def _slotted_call(fn: Callable[..., Any], args: tuple, reserve_mb: float | None = None,
                  group: tuple[str, int] | None = None, hold: bool = False) -> Any:
    # A small task enters the memory gate with its own estimate instead of the run-wide
    # default (2026-11-13: LiH/SeH ZPE and CV points of ~2 GB waited a day at 397 GB
    # projected vs 395 GB allowed, each reserving 8 GB).  Larger tasks keep the default
    # and grow into their estimate inside the slot, as before.
    # An explicit per-task reservation is the task's own estimate: it replaces the
    # run-wide default also when it is smaller.
    default_mb = reserve_mb
    if reserve_mb is None and _SLOT_DIR and _MEM_LIMIT_FRAC > 0.0:
        estimate = stage3_task_estimate_mb(args)
        if estimate is not None and estimate < _MEM_RESERVE_MB:
            default_mb = estimate
    with big_task_slot(reserve_mb), limit_group(group), core_slot(reserve_mb, default_mb=default_mb, hold=hold):
        return fn(*args)


def run_parallel(fn: Callable[..., Any], arg_tuples: Iterable[tuple], *,
                 on_result: Callable[[int, Any], None] | None = None,
                 reserve_mb: Sequence[float | None] | None = None,
                 limit_groups: Sequence[tuple[str, int] | None] | None = None,
                 hold_reservations: bool = False) -> list[Any]:
    """``[fn(*args) for args in arg_tuples]`` on the pool; results in input order.

    ``fn`` and its arguments must be picklable when a pool is configured.
    ``on_result(index, result)`` is called in the calling process as each task
    finishes (used to persist results immediately, so a kill loses only running tasks).
    The first task exception is re-raised after the remaining tasks have finished.
    ``reserve_mb[i]`` is task i's memory reservation for the gate (default: run-wide).
    ``limit_groups[i]`` = (name, n): at most n tasks of that group run at a time machine-wide.
    ``hold_reservations``: each reservation counts in full for MEM_GATE_HOLD_S (see HOLD_MARK).
    """
    items: Sequence[tuple] = list(arg_tuples)
    reserves = list(reserve_mb) if reserve_mb is not None else [None] * len(items)
    groups = list(limit_groups) if limit_groups is not None else [None] * len(items)
    results: list[Any] = [None] * len(items)
    # One task runs in this process, except from a stage thread (benchmark_auto.run_stages):
    # there process-wide state that point code sets (stdout capture, tempfile.tempdir, the
    # CCpy DIIS class) would reach the stages running beside it, so it goes to a worker.
    if _POOL is None or (len(items) < 2 and threading.current_thread() is threading.main_thread()):
        for index, args in enumerate(items):
            results[index] = _slotted_call(fn, args, reserves[index], groups[index], hold_reservations)
            if on_result is not None:
                on_result(index, results[index])
        return results
    futures = {_POOL.submit(_slotted_call, fn, args, reserves[index], groups[index], hold_reservations): index for index, args in enumerate(items)}
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
