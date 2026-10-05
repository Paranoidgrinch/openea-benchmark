import numpy as np
import pytest

from openea_benchmark.attachment.fragment_execution import (
    _temporary_numpy_linalg_legacy_alias,
)


def test_compat_alias_is_noop_when_legacy_alias_exists(monkeypatch):
    marker = object()
    monkeypatch.setattr(np.linalg, "linalg", marker, raising=False)
    with _temporary_numpy_linalg_legacy_alias() as applied:
        assert applied is False
        assert np.linalg.linalg is marker


def test_compat_alias_restores_numpy24_removed_path(monkeypatch):
    if not hasattr(np.linalg, "_linalg"):
        pytest.skip("NumPy implementation module unavailable")
    monkeypatch.delattr(np.linalg, "linalg", raising=False)
    target = np.linalg._linalg
    with _temporary_numpy_linalg_legacy_alias() as applied:
        assert applied is True
        assert np.linalg.linalg is target
    assert not hasattr(np.linalg, "linalg")


def test_compat_target_exposes_linear_algebra_functions(monkeypatch):
    if not hasattr(np.linalg, "_linalg"):
        pytest.skip("NumPy implementation module unavailable")
    monkeypatch.delattr(np.linalg, "linalg", raising=False)
    with _temporary_numpy_linalg_legacy_alias():
        assert callable(np.linalg.linalg.eigh)
        assert callable(np.linalg.linalg.svd)
