from pathlib import Path

from openea_benchmark.adaptive.stage_checkpoint import StageCheckpointStore


def store(tmp_path):
    return StageCheckpointStore(
        tmp_path / "stages",
        signature="test-signature",
    )


def test_remove_existing_checkpoint(tmp_path):
    s = store(tmp_path)
    s.save("x", {"status": "FAILED"})
    assert s.has("x")
    assert s.remove("x") is True
    assert not s.has("x")


def test_remove_missing_checkpoint_is_safe(tmp_path):
    s = store(tmp_path)
    assert s.remove("missing") is False


def test_load_if_valid_reuses_valid_checkpoint(tmp_path):
    s = store(tmp_path)
    s.save("x", {"status": "COMPLETED", "value": 3})
    value = s.load_if_valid(
        "x",
        validator=lambda item: item["status"] == "COMPLETED",
    )
    assert value["value"] == 3
    assert s.has("x")


def test_load_if_valid_invalidates_failed_checkpoint(tmp_path):
    s = store(tmp_path)
    s.save("x", {"status": "SCF_NOT_CONVERGED"})
    value = s.load_if_valid(
        "x",
        validator=lambda item: item["status"] == "COMPLETED",
    )
    assert value is None
    assert not s.has("x")
