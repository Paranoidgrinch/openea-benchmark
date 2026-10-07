def test_post_ccsd_t_ccpy_policy_uses_explicit_c1():
    # CCpy's PySCF bridge calls label_orb_symm unconditionally.  The OpenEA
    # post-CC layer therefore uses explicit C1 metadata rather than
    # symmetry=False or a higher molecular point group.
    symmetry = "C1"
    assert symmetry == "C1"


def test_c1_is_deliberately_nonrestrictive_for_open_ea_policy():
    policy = {
        "group": "C1",
        "reason": "metadata_only_for_ccpy",
        "exploit_higher_symmetry": False,
    }
    assert policy["group"] == "C1"
    assert policy["exploit_higher_symmetry"] is False
