"""Layers after CBS run side by side (benchmark_auto.run_stages)."""
import threading
import time

import pytest

from openea_benchmark.benchmark_auto import run_stages


def test_stages_run_concurrently_and_return_by_name():
    started = []

    def make(name, delay):
        def fn():
            started.append((name, threading.current_thread().name))
            time.sleep(delay)
            return name.upper()
        return fn
    t0 = time.perf_counter()
    out = run_stages({"a": make("a", 0.5), "b": make("b", 0.5), "c": make("c", 0.5)}, concurrent=True)
    assert out == {"a": "A", "b": "B", "c": "C"}
    assert time.perf_counter() - t0 < 1.2            # not 1.5 s one after the other
    assert all(name.startswith("stage") for _, name in started)
    assert run_stages({"a": make("a", 0.0)}, concurrent=False) == {"a": "A"}


def test_a_failing_stage_lets_the_others_finish_then_raises():
    done = []

    def ok():
        time.sleep(0.2)
        done.append("ok")
        return 1

    def bad():
        raise RuntimeError("layer failed")
    with pytest.raises(RuntimeError, match="layer failed"):
        run_stages({"bad": bad, "ok": ok}, concurrent=True)
    assert done == ["ok"]
