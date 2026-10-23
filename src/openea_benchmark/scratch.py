"""Scratch hygiene: per-process PySCF scratch directories and broken-artifact checks.

PySCF puts its temporary files (integrals, DIIS, default chkfiles) into
``lib.param.TMPDIR`` and deletes them only when the owning Python object is
garbage-collected.  A process that is killed (memory watchdog) or crashes
(segfault) leaves them behind until the whole batch ends; on 2026-10-22/23 this
grew to 111 GB, filled the home quota and crashed further workers.

Here every process gets its own directory ``<root>/<host>_<pid>`` under the
batch root (``$PYSCF_TMPDIR``):

* ``use_process_scratch()`` creates it and points PySCF there; it also removes the
  directories of processes on this host that no longer exist (``sweep_dead``).
* ``prune_process_scratch()`` (pool workers, after each task) deletes files in the
  own directory that this process no longer has open: the task is over, so they
  are garbage.  It also sweeps dead processes' directories again, so a killed
  worker's files go away after the next finished task, not only at the next start.

Without ``$PYSCF_TMPDIR`` (tests, interactive use) nothing changes.
"""
from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

_OWN: Path | None = None


def _pid_of(entry: Path) -> int | None:
    parts = entry.name.split("_")
    if len(parts) < 2 or parts[0] != socket.gethostname():
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def sweep_dead(root: Path) -> None:
    """Remove ``<host>_<pid>[_...]`` directories of processes on this host that no longer exist."""
    if not root.is_dir():
        return
    for entry in root.iterdir():
        pid = _pid_of(entry)
        if pid is None or not entry.is_dir():
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(entry, ignore_errors=True)
        except PermissionError:
            pass


def use_process_scratch() -> Path | None:
    """Point PySCF at ``$PYSCF_TMPDIR/<host>_<pid>`` (created); sweep dead processes' directories."""
    global _OWN
    root = os.environ.get("PYSCF_TMPDIR")
    if not root:
        return None
    own = Path(root) / f"{socket.gethostname()}_{os.getpid()}"
    if _OWN != own:
        sweep_dead(Path(root))
        own.mkdir(parents=True, exist_ok=True)
        from pyscf import lib
        lib.param.TMPDIR = str(own)
        _OWN = own
    return _OWN


def _open_paths() -> set[str]:
    paths = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            paths.add(os.path.realpath(f"/proc/self/fd/{fd}"))
        except OSError:
            pass
    return paths


def prune_process_scratch() -> int:
    """Delete files in the own scratch directory that this process has no longer open; returns their count."""
    if _OWN is None or not _OWN.is_dir():
        return 0
    sweep_dead(_OWN.parent)
    held, removed = _open_paths(), 0
    for entry in _OWN.iterdir():
        if os.path.realpath(entry) in held:
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
        removed += 1
    return removed


def hdf5_readable(path: Path) -> bool:
    """True if ``path`` opens as HDF5 and every dataset can be read (catches files cut off by a full disk)."""
    import h5py
    try:
        with h5py.File(path, "r") as fh:
            fh.visititems(lambda name, obj: obj[()] if isinstance(obj, h5py.Dataset) else None)
        return True
    except Exception:
        return False


def discard_broken_checkpoint(path: str | Path | None) -> None:
    """Delete an output checkpoint that a killed/crashed run left unreadable.

    PySCF appends to an existing chkfile; a truncated one makes the rerun fail
    (2026-10-23: 12 such files after the disk-quota crash)."""
    if not path:
        return
    path = Path(path)
    if path.is_file() and not hdf5_readable(path):
        print(f"[SCRATCH] removing unreadable checkpoint {path}", flush=True)
        path.unlink(missing_ok=True)
