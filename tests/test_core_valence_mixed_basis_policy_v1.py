def test_oh_mixed_core_valence_policy():
    policy = {
        3: {"O": "aug-cc-pwcvtz", "H": "aug-cc-pvtz"},
        4: {"O": "aug-cc-pwcvqz", "H": "aug-cc-pvqz"},
        5: {"O": "aug-cc-pwcv5z", "H": "aug-cc-pv5z"},
    }
    assert all("pwcv" in policy[x]["O"] for x in policy)
    assert all("pwcv" not in policy[x]["H"] for x in policy)
    assert all(policy[x]["H"].startswith("aug-cc-pv") for x in policy)
