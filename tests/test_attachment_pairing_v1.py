from openea_benchmark.attachment.model import ElectronicState, PECBranch
from openea_benchmark.attachment.pairing import generate_attachment_candidates


def branch(state_id, charge, identity="CLEARED"):
    return PECBranch(
        branch_id=state_id,
        state=ElectronicState(
            state_id=state_id,
            charge=charge,
            multiplicity=1,
            spin_2s=0,
            identity_status=identity,
        ),
    )


def test_valid_attachment_pair_is_retained():
    result = generate_attachment_candidates([branch("n", 0)], [branch("a", -1)])
    assert len(result) == 1
    assert result[0].pairing_status == "VALID"


def test_wrong_charge_is_rejected():
    result = generate_attachment_candidates([branch("n", 0)], [branch("a", 0)])
    assert result[0].pairing_status == "REJECTED"


def test_unresolved_state_identity_is_not_promoted_to_valid():
    result = generate_attachment_candidates(
        [branch("n", 0, "AMBIGUOUS")], [branch("a", -1)]
    )
    assert result[0].pairing_status == "AMBIGUOUS"


def test_multiple_anions_remain_multiple_candidates():
    result = generate_attachment_candidates(
        [branch("n", 0)], [branch("a1", -1), branch("a2", -1)]
    )
    assert len(result) == 2
    assert {x.anion_branch.branch_id for x in result} == {"a1", "a2"}
