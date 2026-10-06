from openea_benchmark.adaptive.stage3_execution import Stage3ExecutionSettings

def test_default_stage3_correlation_space_remains_legacy_all_electron():
    settings = Stage3ExecutionSettings()
    assert settings.frozen_core is False

def test_stage3_can_explicitly_request_frozen_core():
    settings = Stage3ExecutionSettings(frozen_core=True)
    assert settings.frozen_core is True
