import pytest


@pytest.fixture(autouse=True)
def _no_cpu_gate(monkeypatch):
    """Tests must not wait for the machine's real load average (CPU gate in point_pool)."""
    monkeypatch.setenv("OPENEA_CPU_LIMIT_FRAC", "0")
