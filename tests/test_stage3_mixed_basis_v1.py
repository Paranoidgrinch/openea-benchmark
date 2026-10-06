from openea_benchmark.adaptive.stage3_execution import Stage3ExecutionRequest


def build(**overrides):
    data = dict(
        request_id="r", job_id="j", system="OH", atoms=("O", "H"),
        charge=0, spin_2s=1, component_id="c", r_angstrom=0.97,
        basis="O:aug-cc-pwCVTZ|H:aug-cc-pVTZ",
        methods=("CCSD", "CCSD(T)"), requested_reference="ROHF",
        scf_reference="ROHF", source_link_status="SINGLE_DFT_INITIALIZATION",
        source_root_id="root", source_checkpoint_path="/tmp/source.chk",
        source_origin_guess=None, grid_index=0, initialization_index=0,
        dft_center_r_angstrom=0.97, dft_center_energy_hartree=0.0,
        requires_independent_state_identity_validation=False,
    )
    data.update(overrides)
    return Stage3ExecutionRequest(**data)


def test_mixed_basis_mapping_is_accepted():
    req = build(
        basis_by_element={"O": "aug-cc-pwcvtz", "H": "aug-cc-pvtz"}
    )
    assert req.basis_by_element["O"] == "aug-cc-pwcvtz"


def test_mixed_basis_mapping_requires_exact_atoms():
    try:
        build(basis_by_element={"O": "aug-cc-pwcvtz"})
    except ValueError as exc:
        assert "exactly the request atoms" in str(exc)
    else:
        raise AssertionError("incomplete mapping accepted")


def test_legacy_single_basis_request_is_unchanged():
    req = build(basis="aug-cc-pvqz", basis_by_element=None)
    assert req.basis_by_element is None
