"""Checks before a batch (re)start; start_detached.sh runs it and stops on problems.

Usage: .venv/bin/python scripts/preflight.py RUN_NAME [--fix]

* scratch: free space on the PySCF scratch file system ($OPENEA_PYSCF_SCRATCH, default
  /scratch/$USER/openea_pyscf); leftover batch scratch of batches that no longer run
  (also the old location runs/scratch in the home quota) -- removed with --fix.
* results ($OPENEA_OUTPUT/RUN, default ../openea_results/RUN): JSON files that do not
  parse, HDF5 checkpoints that cannot be read (a full disk or a kill cut them off) and
  stale *.tmp files -- moved to RUN/_quarantine/<stamp>/ with --fix, so the run
  recomputes them instead of crashing on them.  Files already checked unchanged are
  remembered in RUN/.preflight_ok.json.

Exit status 0: ready to start; 1: problems left (rerun with --fix or look at them).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openea_benchmark.scratch import hdf5_readable  # noqa: E402

MIN_FREE_GB = 500.0
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _batch_alive(scratch_dir: Path) -> bool:
    """batch_<YYYYmmdd_HHMMSS>_XXXX belongs to runs/*/batch_<stamp>.pid; alive if that pid runs."""
    parts = scratch_dir.name.split("_")
    if len(parts) < 3:
        return False
    stamp = f"{parts[1]}_{parts[2]}"
    for pid_file in (ROOT / "runs").glob(f"*/batch_{stamp}.pid"):
        try:
            if _alive(int(pid_file.read_text().strip())):
                return True
        except ValueError:
            pass
    return False


def _size_gb(path: Path) -> float:
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                pass
    return total / 1024 ** 3


def check_scratch(fix: bool) -> list[str]:
    problems = []
    root = Path(os.environ.get("OPENEA_PYSCF_SCRATCH", f"/scratch/{os.environ.get('USER', 'openea')}/openea_pyscf"))
    probe = root if root.exists() else root.parent
    if probe.exists():
        free = shutil.disk_usage(probe).free / 1024 ** 3
        print(f"scratch {root}: {free:.0f} GB free")
        if free < MIN_FREE_GB:
            problems.append(f"only {free:.0f} GB free on the scratch file system (< {MIN_FREE_GB:.0f})")
    else:
        problems.append(f"scratch parent {probe} does not exist")
    for base in (root, ROOT / "runs" / "scratch"):
        if not base.is_dir():
            continue
        for d in sorted(base.glob("batch_*")):
            if not d.is_dir() or _batch_alive(d):
                continue
            size = _size_gb(d)
            if fix:
                shutil.rmtree(d, ignore_errors=True)
                print(f"removed leftover scratch {d} ({size:.1f} GB)")
            else:
                problems.append(f"leftover scratch of a finished batch: {d} ({size:.1f} GB)")
    return problems


def _broken(path: Path) -> str | None:
    if path.suffix == ".tmp":
        return "stale temp file"
    if path.suffix == ".json":
        try:
            json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            return f"JSON: {type(exc).__name__}"
        return None
    try:
        with path.open("rb") as fh:
            magic = fh.read(8)
    except OSError as exc:
        return f"unreadable: {exc}"
    if magic == HDF5_MAGIC and not hdf5_readable(path):
        return "HDF5 cannot be read"
    return None


def check_results(run: str, fix: bool) -> list[str]:
    results = Path(os.environ.get("OPENEA_OUTPUT", ROOT.parent / "openea_results")) / run
    if not results.is_dir():
        print(f"results {results}: none yet")
        return []
    memo_path = results / ".preflight_ok.json"
    try:
        memo = json.loads(memo_path.read_text())
    except (OSError, ValueError):
        memo = {}
    quarantine = results / "_quarantine" / time.strftime("%Y%m%d_%H%M%S")
    problems, ok, checked = [], {}, 0
    for path in results.rglob("*"):
        if "_quarantine" in path.parts or path == memo_path or not path.is_file():
            continue
        key, stamp = str(path.relative_to(results)), f"{path.stat().st_mtime_ns}:{path.stat().st_size}"
        if memo.get(key) == stamp:
            ok[key] = stamp
            continue
        checked += 1
        reason = _broken(path)
        if reason is None:
            ok[key] = stamp
            continue
        if fix:
            target = quarantine / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), target)
            print(f"quarantined {key} ({reason})")
        else:
            problems.append(f"{key}: {reason}")
    tmp = memo_path.with_name(memo_path.name + ".tmp")
    tmp.write_text(json.dumps(ok, sort_keys=True))
    tmp.replace(memo_path)
    print(f"results {results}: {checked} files checked, {len(ok)} ok")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    problems = check_scratch(args.fix) + check_results(args.run, args.fix)
    for problem in problems:
        print(f"PROBLEM {problem}")
    print("preflight OK" if not problems else f"preflight: {len(problems)} problem(s); rerun with --fix")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
