"""Machine memory watchdog for OpenEA runs (the server crashed out of memory on 2026-10-12).

Polls /proc/meminfo and acts only on this user's OpenEA processes (workflow
processes and their spawned pool workers, recognised by cwd in this repository):

* used >= --pause-frac (default 0.70): pause (SIGSTOP) the youngest running
  OpenEA process, one more every --pause-every seconds while it stays high.
  A paused calculation keeps its memory but stops growing.
* used <  --resume-frac (default 0.65): resume (SIGCONT) paused processes,
  the longest paused first, one per --resume-every seconds.
* used >= --kill-frac (default 0.80): kill (SIGKILL) the OpenEA process with the
  largest resident memory.  Its run fails and is resumed from the point cache
  (ERROR results are not cached) by the batch launcher.
* used >= --pause-frac continuously for --stall-kill-after seconds (default 300):
  pausing alone does not free memory (2026-10-13: 90 processes paused at 89 % for
  three hours, the server crawled); the largest OpenEA process is killed, one per
  --kill-every seconds, until memory falls below the pause threshold.

Thresholds were lowered by 10 points on 2026-10-13: a machine above ~80 % used
is already slow for every user.

used = 1 - MemAvailable / MemTotal.  The memory gate in point_pool.py keeps new
calculations from starting above the limit; this watchdog handles calculations
that outgrow their reservation and memory taken by other users.

The watchdog must run in the same process namespace as the runs it guards
(start it from the same launcher, e.g. runs/<run>/run_batch.sh).
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def meminfo_mb() -> tuple[float, float]:
    values = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, rest = line.split(":", 1)
            values[key] = float(rest.split()[0]) / 1024.0
    return values["MemTotal"], values["MemAvailable"]


def openea_processes() -> list[dict]:
    """This user's OpenEA workflow processes and pool workers (pid, start tick, rss, worker flag)."""
    uid, me, found = os.getuid(), os.getpid(), []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        pid = int(entry)
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            cmdline = b" ".join(argv).decode(errors="replace")
            cwd = os.readlink(f"/proc/{pid}/cwd")
            stat = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            rss = 0.0
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    rss = float(line.split()[1]) / 1024.0
        except (OSError, IndexError, ValueError):
            continue
        if not cwd.startswith(str(ROOT)) or b"python" not in os.path.basename(argv[0]):
            continue
        worker = "multiprocessing" in cmdline
        if not worker and "openea_benchmark.workflow" not in cmdline:
            continue
        found.append({"pid": pid, "state": stat[0], "start": int(stat[19]), "rss_mb": rss, "worker": worker})
    return found


def log(fh, message: str) -> None:
    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    fh.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pause-frac", type=float, default=0.70)
    parser.add_argument("--resume-frac", type=float, default=0.65)
    parser.add_argument("--kill-frac", type=float, default=0.80)
    parser.add_argument("--stall-kill-after", type=float, default=300.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--pause-every", type=float, default=20.0)
    parser.add_argument("--resume-every", type=float, default=30.0)
    parser.add_argument("--kill-every", type=float, default=30.0)
    parser.add_argument("--status-every", type=float, default=600.0)
    parser.add_argument("--log", type=Path, default=ROOT / "runs" / "mem_watchdog.log")
    args = parser.parse_args()
    if not args.resume_frac < args.pause_frac < args.kill_frac:
        parser.error("need resume-frac < pause-frac < kill-frac")
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass

    paused: list[int] = []  # in pause order
    last = {"pause": 0.0, "resume": 0.0, "kill": 0.0, "status": 0.0}
    high_since: float | None = None  # start of the current stretch above the pause threshold
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a") as fh:
        log(fh, f"watchdog pid {os.getpid()} on {os.uname().nodename}: pause>={args.pause_frac:.0%} "
                f"resume<{args.resume_frac:.0%} kill>={args.kill_frac:.0%}")

        def release_all(signum, frame):  # never leave calculations stopped behind
            for pid in paused:
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            log(fh, f"watchdog stopped (signal {signum}), resumed {len(paused)} paused")
            sys.exit(0)

        signal.signal(signal.SIGTERM, release_all)
        signal.signal(signal.SIGINT, release_all)

        while True:
            now = time.time()
            total, available = meminfo_mb()
            used = 1.0 - available / total
            procs = openea_processes()
            alive = {p["pid"] for p in procs}
            paused = [pid for pid in paused if pid in alive]
            running = [p for p in procs if p["pid"] not in paused]
            if now - last["status"] >= args.status_every:
                ours = sum(p["rss_mb"] for p in procs) / 1024
                log(fh, f"status used {used:.1%} of {total / 1024:.0f} GB, openea {len(procs)} procs "
                        f"{ours:.0f} GB, paused {len(paused)}")
                last["status"] = now

            high_since = (high_since or now) if used >= args.pause_frac else None
            stalled = high_since is not None and now - high_since >= args.stall_kill_after
            if (used >= args.kill_frac or stalled) and procs and now - last["kill"] >= args.kill_every:
                victim = max(procs, key=lambda p: (p["worker"], p["rss_mb"]))
                os.kill(victim["pid"], signal.SIGKILL)
                if victim["pid"] in paused:
                    paused.remove(victim["pid"])
                reason = (f"used {used:.1%} >= {args.kill_frac:.0%}" if used >= args.kill_frac else
                          f"used >= {args.pause_frac:.0%} for {now - high_since:.0f} s despite pausing")
                log(fh, f"KILL pid {victim['pid']} rss {victim['rss_mb'] / 1024:.1f} GB ({reason})")
                last["kill"] = now
            elif used >= args.pause_frac and running and now - last["pause"] >= args.pause_every:
                victim = max(running, key=lambda p: (p["worker"], p["start"]))
                os.kill(victim["pid"], signal.SIGSTOP)
                paused.append(victim["pid"])
                log(fh, f"PAUSE pid {victim['pid']} rss {victim['rss_mb'] / 1024:.1f} GB "
                        f"(used {used:.1%} >= {args.pause_frac:.0%}, paused {len(paused)})")
                last["pause"] = now
            elif used < args.resume_frac and paused and now - last["resume"] >= args.resume_every:
                pid = paused.pop(0)
                try:
                    os.kill(pid, signal.SIGCONT)
                    log(fh, f"RESUME pid {pid} (used {used:.1%} < {args.resume_frac:.0%}, paused {len(paused)})")
                except ProcessLookupError:
                    pass
                last["resume"] = now
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
