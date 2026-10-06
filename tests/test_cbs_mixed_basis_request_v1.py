from openea_benchmark.attachment.cbs_component_evidence import (
    make_cbs_single_point_request,
)


def test_helper_preserves_mixed_basis_mapping(tmp_path):
    mapping = {"O": "aug-cc-pwcvtz", "H": "aug-cc-pvtz"}
    req = make_cbs_single_point_request(
        role="neutral",
        basis="O:aug-cc-pwCVTZ|H:aug-cc-pVTZ",
        cardinal_number=3,
        r_angstrom=0.97,
        source_checkpoint_path=tmp_path / "source.chk",
        basis_by_element=mapping,
    )
    assert req.basis_by_element == mapping


def test_helper_remains_backward_compatible(tmp_path):
    req = make_cbs_single_point_request(
        role="anion", basis="aug-cc-pv5z", cardinal_number=5,
        r_angstrom=0.96, source_checkpoint_path=tmp_path / "source.chk",
    )
    assert req.basis_by_element is None
