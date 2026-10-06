def test_oh_scalar_relativity_uses_first_row_pcv_dk_family():
    policy = {
        3: {"O": "aug-cc-pcvtz-dk", "H": "aug-cc-pvtz-dk"},
        4: {"O": "aug-cc-pcvqz-dk", "H": "aug-cc-pvqz-dk"},
        5: {"O": "aug-cc-pcv5z-dk", "H": "aug-cc-pv5z-dk"},
    }

    assert all("pcv" in policy[x]["O"] for x in policy)
    assert all("pwcv" not in policy[x]["O"] for x in policy)
    assert all(policy[x]["O"].endswith("-dk") for x in policy)
    assert all(policy[x]["H"].endswith("-dk") for x in policy)
