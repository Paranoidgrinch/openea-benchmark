from __future__ import annotations

import os
import socket
import subprocess
import sys

import h5py
import numpy as np

from openea_benchmark import scratch


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_sweep_dead_removes_only_directories_of_dead_processes(tmp_path):
    host = socket.gethostname()
    dead = tmp_path / f"{host}_{_dead_pid()}"
    dead_cc = tmp_path / f"{host}_{_dead_pid()}_abcd1234"
    alive = tmp_path / f"{host}_{os.getpid()}"
    other_host = tmp_path / f"otherhost_{_dead_pid()}"
    unrelated = tmp_path / "batch_20261023_000000_XXXX"
    for d in (dead, dead_cc, alive, other_host, unrelated):
        d.mkdir()
        (d / "eri.h5").write_bytes(b"x")
    scratch.sweep_dead(tmp_path)
    assert not dead.exists() and not dead_cc.exists()
    assert alive.exists() and other_host.exists() and unrelated.exists()


def test_process_scratch_points_pyscf_there_and_prunes_closed_files(tmp_path, monkeypatch):
    from pyscf import lib
    monkeypatch.setenv("PYSCF_TMPDIR", str(tmp_path))
    monkeypatch.setattr(scratch, "_OWN", None)
    monkeypatch.setattr(lib.param, "TMPDIR", lib.param.TMPDIR)
    leftover = tmp_path / f"{socket.gethostname()}_{_dead_pid()}"
    leftover.mkdir()
    own = scratch.use_process_scratch()
    assert own == tmp_path / f"{socket.gethostname()}_{os.getpid()}" and own.is_dir()
    assert lib.param.TMPDIR == str(own)
    assert not leftover.exists()
    held = lib.H5TmpFile()  # PySCF scratch file still in use
    held["a"] = np.zeros(3)
    (own / "closed.h5").write_bytes(b"garbage")
    (own / "subdir").mkdir()
    assert scratch.prune_process_scratch() == 2
    assert [p.name for p in own.iterdir()] == [os.path.basename(held.filename)]
    held.close()


def test_process_scratch_is_a_no_op_without_pyscf_tmpdir(monkeypatch):
    monkeypatch.delenv("PYSCF_TMPDIR", raising=False)
    monkeypatch.setattr(scratch, "_OWN", None)
    assert scratch.use_process_scratch() is None
    assert scratch.prune_process_scratch() == 0


def test_discard_broken_checkpoint_keeps_readable_and_removes_truncated(tmp_path):
    good, bad = tmp_path / "good.chk", tmp_path / "bad.chk"
    for path in (good, bad):
        with h5py.File(path, "w") as fh:
            fh["scf/mo_coeff"] = np.arange(200000.0)
    with bad.open("r+b") as fh:  # cut off like a write on a full disk
        fh.truncate(bad.stat().st_size // 2)
    scratch.discard_broken_checkpoint(good)
    scratch.discard_broken_checkpoint(bad)
    scratch.discard_broken_checkpoint(tmp_path / "missing.chk")
    scratch.discard_broken_checkpoint(None)
    assert good.exists() and not bad.exists()


def test_workflow_json_write_is_atomic(tmp_path, monkeypatch):
    from openea_benchmark import workflow
    target = tmp_path / "summary.json"
    workflow._write_json(target, {"a": 1})

    def fail(*args, **kwargs):
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(os, "fsync", fail)
    try:
        workflow._write_json(target, {"a": 2})
    except OSError:
        pass
    assert target.read_text() == '{\n  "a": 1\n}\n'
