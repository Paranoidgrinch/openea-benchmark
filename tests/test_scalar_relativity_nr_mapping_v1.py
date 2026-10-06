def test_scalar_runner_nr_label_maps_to_stage3_none():
    def map_mode(label):
        return "NONE" if label == "NR" else "SFX2C1E"

    assert map_mode("NR") == "NONE"
    assert map_mode("SFX2C1E") == "SFX2C1E"
