import math
import pytest

from openea_benchmark.attachment.asymptote import BindingAssessment, BindingStatus
from openea_benchmark.attachment.decision import (
    EADecisionStatus,
    EnergyInterval,
    PrecisionStatus,
    decide_ea,
)
from openea_benchmark.attachment.model import AttachmentCandidate, ElectronicState, PECBranch


def candidate(status="VALID"):
    n = PECBranch("neutral", ElectronicState("n", 0, 2, 1, identity_status="CLEARED"))
    a = PECBranch("anion", ElectronicState("a", -1, 1, 0, identity_status="CLEARED"))
    return AttachmentCandidate("n__a", n, a, status, ("TEST",))


BOUND = BindingAssessment(BindingStatus.BOUND, ("TEST_BOUND",), 0.1)


def test_positive_interval_yields_bound():
    r = decide_ea(
        candidate=candidate(), anion_binding=BOUND,
        neutral_energy=EnergyInterval(-75.00, -74.99),
        anion_energy=EnergyInterval(-75.10, -75.09),
    )
    assert r.status is EADecisionStatus.BOUND
    assert r.ea_lower_ev > 0.0


def test_nonpositive_interval_yields_unbound_without_precise_negative_ea():
    r = decide_ea(
        candidate=candidate(), anion_binding=BOUND,
        neutral_energy=EnergyInterval(-75.10, -75.09),
        anion_energy=EnergyInterval(-75.00, -74.99),
    )
    assert r.status is EADecisionStatus.UNBOUND
    assert r.ea_central_ev is None and r.ea_lower_ev is None and r.ea_upper_ev is None


def test_zero_crossing_is_unresolved():
    r = decide_ea(
        candidate=candidate(), anion_binding=BOUND,
        neutral_energy=EnergyInterval(-75.01, -74.99),
        anion_energy=EnergyInterval(-75.00, -75.00),
    )
    assert r.status is EADecisionStatus.UNRESOLVED
    assert r.ea_lower_ev < 0.0 < r.ea_upper_ev


def test_ambiguous_pairing_blocks_decision():
    r = decide_ea(
        candidate=candidate("AMBIGUOUS"), anion_binding=BOUND,
        neutral_energy=EnergyInterval(-75.00, -74.99),
        anion_energy=EnergyInterval(-75.10, -75.09),
    )
    assert r.status is EADecisionStatus.UNRESOLVED


def test_dissociatively_unbound_anion_is_unbound():
    r = decide_ea(
        candidate=candidate(),
        anion_binding=BindingAssessment(BindingStatus.UNBOUND, ("TEST",), -0.1),
        neutral_energy=EnergyInterval(-75.00, -74.99),
        anion_energy=EnergyInterval(-75.10, -75.09),
    )
    assert r.status is EADecisionStatus.UNBOUND


def test_unresolved_anion_binding_stays_unresolved():
    r = decide_ea(
        candidate=candidate(),
        anion_binding=BindingAssessment(BindingStatus.UNRESOLVED, ("TEST",)),
        neutral_energy=EnergyInterval(-75.00, -74.99),
        anion_energy=EnergyInterval(-75.10, -75.09),
    )
    assert r.status is EADecisionStatus.UNRESOLVED


def test_missing_energy_interval_stays_unresolved():
    r = decide_ea(
        candidate=candidate(), anion_binding=BOUND,
        neutral_energy=None,
        anion_energy=EnergyInterval(-75.10, -75.09),
    )
    assert r.status is EADecisionStatus.UNRESOLVED


def test_precision_target_is_independent_of_bound_status():
    r = decide_ea(
        candidate=candidate(), anion_binding=BOUND,
        neutral_energy=EnergyInterval(-75.0001, -74.9999),
        anion_energy=EnergyInterval(-75.1001, -75.0999),
        target_half_width_ev=0.001,
    )
    assert r.status is EADecisionStatus.BOUND
    assert r.precision_status is PrecisionStatus.TARGET_NOT_MET


def test_precision_target_can_be_met():
    r = decide_ea(
        candidate=candidate(), anion_binding=BOUND,
        neutral_energy=EnergyInterval(-75.00001, -74.99999),
        anion_energy=EnergyInterval(-75.10001, -75.09999),
        target_half_width_ev=0.001,
    )
    assert r.status is EADecisionStatus.BOUND
    assert r.precision_status is PrecisionStatus.TARGET_MET


def test_nominal_ea_is_recorded_only_when_both_nominals_are_given():
    r = decide_ea(
        candidate=candidate(), anion_binding=BOUND,
        neutral_energy=EnergyInterval(-75.01, -74.99),
        anion_energy=EnergyInterval(-75.11, -75.09),
        neutral_nominal_hartree=-75.00,
        anion_nominal_hartree=-75.10,
    )
    assert math.isclose(r.ea_central_ev, 0.1 * 27.211386245988)


def test_single_nominal_is_rejected():
    with pytest.raises(ValueError):
        decide_ea(
            candidate=candidate(), anion_binding=BOUND,
            neutral_energy=EnergyInterval(-75.01, -74.99),
            anion_energy=EnergyInterval(-75.11, -75.09),
            neutral_nominal_hartree=-75.00,
        )


def test_invalid_interval_is_rejected():
    with pytest.raises(ValueError):
        EnergyInterval(-74.0, -75.0)
