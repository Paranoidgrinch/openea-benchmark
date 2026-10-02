import pytest

from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceSettings,
    BasisConvergenceStatus,
)
from openea_benchmark.attachment.basis_series import (
    BasisSeriesDescriptor,
    assess_cardinal_ea_series,
    basis_point_from_ea_decision,
)
from openea_benchmark.attachment.decision import (
    EADecision,
    EADecisionStatus,
    PrecisionStatus,
)


def decision(
    central,
    hw=0.001,
    status=EADecisionStatus.BOUND,
):
    return EADecision(
        status=status,
        precision_status=PrecisionStatus.TARGET_MET,
        rationale=("TEST",),
        ea_lower_ev=(central - hw if status is EADecisionStatus.BOUND else None),
        ea_central_ev=(central if status is EADecisionStatus.BOUND else None),
        ea_upper_ev=(central + hw if status is EADecisionStatus.BOUND else None),
        half_width_ev=(hw if status is EADecisionStatus.BOUND else None),
    )


def descriptor(name, cardinal):
    return BasisSeriesDescriptor(name, cardinal, 1)


def test_bound_decision_becomes_nonproduction_basis_point():
    point = basis_point_from_ea_decision(
        descriptor=descriptor("aug-cc-pvdz", 2),
        method_signature="CCSD(T)",
        decision=decision(1.5),
    )
    assert point.cardinal_number == 2
    assert point.augmentation_level == 1
    assert point.is_production_ea is False


def test_unresolved_decision_cannot_enter_basis_series():
    with pytest.raises(ValueError):
        basis_point_from_ea_decision(
            descriptor=descriptor("aug-cc-pvdz", 2),
            method_signature="CCSD(T)",
            decision=decision(
                1.5,
                status=EADecisionStatus.UNRESOLVED,
            ),
        )


def test_incomplete_numeric_interval_is_rejected():
    broken = EADecision(
        status=EADecisionStatus.BOUND,
        precision_status=PrecisionStatus.NOT_ASSESSED,
        rationale=("TEST",),
        ea_lower_ev=1.4,
        ea_central_ev=None,
        ea_upper_ev=1.6,
    )
    with pytest.raises(ValueError):
        basis_point_from_ea_decision(
            descriptor=descriptor("aug-cc-pvdz", 2),
            method_signature="CCSD(T)",
            decision=broken,
        )


def test_three_real_decisions_delegate_to_cardinal_policy():
    pts = tuple(
        basis_point_from_ea_decision(
            descriptor=descriptor(name, cardinal),
            method_signature="CCSD(T)",
            decision=decision(value),
        )
        for name, cardinal, value in (
            ("aug-cc-pvdz", 2, 1.50),
            ("aug-cc-pvtz", 3, 1.60),
            ("aug-cc-pvqz", 4, 1.61),
        )
    )
    settings = BasisConvergenceSettings(
        cardinal_increment_target_ev=0.02,
        cardinal_contraction_ratio_max=0.75,
        diffuse_increment_target_ev=0.01,
        diffuse_contraction_ratio_max=0.60,
        force_double_augmentation=False,
    )
    result = assess_cardinal_ea_series(
        pts,
        augmentation_level=1,
        settings=settings,
    )
    assert result.status is BasisConvergenceStatus.CLEARED


def test_descriptor_rejects_invalid_cardinal():
    with pytest.raises(ValueError):
        BasisSeriesDescriptor("bad", 1, 1)
