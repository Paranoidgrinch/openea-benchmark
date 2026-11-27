"""Axial CC: a method that does not converge is retried once with shifted denominators."""
import types

import numpy as np
import pytest


class FakeDriver:
    calls = []

    def __init__(self, system, hamiltonian):
        self.options = {}
        self.correlation_energy = -0.5
        self.T = types.SimpleNamespace(a=np.zeros(1), b=np.zeros(1), aa=np.zeros(1), ab=np.zeros(1), bb=np.zeros(1))

    def run_cc(self, method):
        FakeDriver.calls.append((method, self.options["energy_shift"], self.options["maximum_iterations"]))
        if self.options["energy_shift"] > 0 or not FakeDriver.only_shifted:
            print("CC calculation successfully converged")


def test_retry_with_shift(monkeypatch):
    import ccpy.drivers.driver as drv
    from openea_benchmark.axial import cc
    monkeypatch.setattr(drv, "Driver", FakeDriver)
    FakeDriver.calls, FakeDriver.only_shifted = [], True
    diag = {}
    out = cc._run(None, None, ("ccsd",), 1e-8, 100, diagnostics=diag)
    assert out["ccsd"] == -0.5
    assert FakeDriver.calls == [("ccsd", 0.0, 100), ("ccsd", cc.RETRY_ENERGY_SHIFT, 200)]
    assert diag["retried_with_shift"] == ["ccsd"]
    FakeDriver.calls, FakeDriver.only_shifted = [], False
    cc._run(None, None, ("ccsd",), 1e-8, 100)
    assert len(FakeDriver.calls) == 1


def test_no_convergence_even_shifted_raises(monkeypatch):
    import ccpy.drivers.driver as drv
    from openea_benchmark.axial import cc

    class Never(FakeDriver):
        def run_cc(self, method):
            pass
    monkeypatch.setattr(drv, "Driver", Never)
    with pytest.raises(cc.CCNotConverged):
        cc._run(None, None, ("ccsd",), 1e-8, 10)
