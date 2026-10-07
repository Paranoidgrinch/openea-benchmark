from openea_benchmark.adaptive.core_valence_runner import CoreValenceBasisSpec


def test_explicit_mixed_core_valence_policy_is_supported_without_oh_hardcoding():
    specs = tuple(
        CoreValenceBasisSpec(
            cardinal=x,
            family_id="VALIDATION_MIXED_CVXZ",
            label=f"O:cv{x}|H:v{x}",
            basis_by_element={"O": f"aug-cc-pwcv{x}z", "H": f"aug-cc-pv{x}z"},
        )
        for x in (3, 4, 5)
    )
    assert all("pwcv" in spec.basis_by_element["O"] for spec in specs)
    assert all("pwcv" not in spec.basis_by_element["H"] for spec in specs)
    assert all(spec.family_id == "VALIDATION_MIXED_CVXZ" for spec in specs)
