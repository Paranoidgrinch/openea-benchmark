from types import SimpleNamespace as NS

from openea_benchmark.attachment.stage3_bridge import (
    Stage3AttachmentBridgeStatus,
    bridge_stage3_loop_to_attachment,
    pair_stage3_attachment_records,
)


def ev(x):
    return NS(value=x)


def point(r, e, status="ACCEPTED"):
    return NS(r_angstrom=r, energy_hartree=e, status=ev(status))


def loop(
    *,
    status="CONVERGED",
    action="BRACKET_TARGET_MET",
    pec_status="READY_FOR_DISCRETE_MINIMUM_SCOUT",
    init="CLEARED",
    continuity="CLEARED",
    min_status="BRACKETED_SINGLE_MINIMUM",
    charge=0,
    spin_2s=1,
    job_id="job",
):
    candidate = NS(
        r_angstrom=1.0,
        energy_hartree=-75.0,
        left_r_angstrom=0.99,
        right_r_angstrom=1.01,
        canonical_request_id="req-center",
    )
    pec = NS(
        status=ev(pec_status),
        initialization_identity_status=ev(init),
        geometry_continuity_status=ev(continuity),
        minimum_scout=NS(status=ev(min_status), candidates=(candidate,)),
        points=(point(0.99, -74.9), point(1.0, -75.0), point(1.01, -74.95)),
        system="OH",
        charge=charge,
        spin_2s=spin_2s,
        component_id="component-1",
        basis="aug-cc-pVTZ",
    )
    return NS(
        status=ev(status),
        job_id=job_id,
        final_refinement_plan=NS(action=ev(action)),
        final_pec=pec,
        is_production_ea=False,
        ground_state_assigned=False,
        authorizes_pruning=False,
    )


def test_converged_stage3_loop_becomes_attachment_branch_without_energy_interval():
    result = bridge_stage3_loop_to_attachment(loop())
    assert result.status is Stage3AttachmentBridgeStatus.READY
    assert result.branch is not None
    assert result.branch.state.multiplicity == 2
    assert result.branch.state.identity_status == "CLEARED"
    assert result.minimum_bracket_angstrom == (0.99, 1.01)
    assert result.discrete_minimum_energy_hartree == -75.0
    assert result.energy_interval_hartree is None
    assert "EQUILIBRIUM_ENERGY_INTERVAL_REQUIRED" in result.evidence


def test_nonconverged_loop_is_unresolved():
    result = bridge_stage3_loop_to_attachment(loop(status="ROUND_LIMIT_REACHED"))
    assert result.status is Stage3AttachmentBridgeStatus.UNRESOLVED
    assert result.branch is None


def test_bracket_target_is_required_even_when_loop_claims_converged():
    result = bridge_stage3_loop_to_attachment(loop(action="REFINE_SINGLE_BRACKET"))
    assert result.status is Stage3AttachmentBridgeStatus.UNRESOLVED


def test_identity_gate_is_preserved():
    result = bridge_stage3_loop_to_attachment(loop(init="UNRESOLVED"))
    assert result.status is Stage3AttachmentBridgeStatus.UNRESOLVED


def test_geometry_continuity_gate_is_preserved():
    result = bridge_stage3_loop_to_attachment(loop(continuity="UNRESOLVED"))
    assert result.status is Stage3AttachmentBridgeStatus.UNRESOLVED


def test_single_minimum_is_required():
    result = bridge_stage3_loop_to_attachment(loop(min_status="MULTIPLE_MINIMUM_CANDIDATES"))
    assert result.status is Stage3AttachmentBridgeStatus.UNRESOLVED


def test_neutral_and_anion_records_can_enter_existing_pairing_engine():
    neutral = bridge_stage3_loop_to_attachment(loop(charge=0, spin_2s=1, job_id="neutral"))
    anion = bridge_stage3_loop_to_attachment(loop(charge=-1, spin_2s=0, job_id="anion"))
    candidates = pair_stage3_attachment_records(neutral, anion)
    assert len(candidates) == 1
    assert candidates[0].pairing_status == "VALID"


def test_unresolved_record_does_not_enter_pairing_engine():
    neutral = bridge_stage3_loop_to_attachment(loop(status="ROUND_LIMIT_REACHED", job_id="neutral"))
    anion = bridge_stage3_loop_to_attachment(loop(charge=-1, spin_2s=0, job_id="anion"))
    assert pair_stage3_attachment_records(neutral, anion) == ()
