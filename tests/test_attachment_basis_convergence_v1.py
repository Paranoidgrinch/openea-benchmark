import pytest

from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceAction,
    BasisConvergenceSettings,
    BasisConvergenceStatus,
    EAIntervalEV,
    ElectronicEABasisPoint,
    assess_cardinal_convergence,
    assess_diffuse_convergence,
)


SETTINGS = BasisConvergenceSettings(
    cardinal_increment_target_ev=0.02,
    cardinal_contraction_ratio_max=0.60,
    diffuse_increment_target_ev=0.01,
    diffuse_contraction_ratio_max=0.60,
    force_double_augmentation=False,
)


def point(name, x, aug, central, hw=0.001, method="CCSD(T)"):
    return ElectronicEABasisPoint(
        basis_name=name,
        cardinal_number=x,
        augmentation_level=aug,
        method_signature=method,
        ea=EAIntervalEV(central - hw, central, central + hw),
    )


def test_cardinal_three_point_contracting_series_clears():
    pts = (
        point("aDZ", 2, 1, 1.50),
        point("aTZ", 3, 1, 1.60),
        point("aQZ", 4, 1, 1.61),
    )
    result = assess_cardinal_convergence(
        pts, augmentation_level=1, settings=SETTINGS
    )
    assert result.status is BasisConvergenceStatus.CLEARED
    assert result.action is BasisConvergenceAction.NONE
    assert result.residual_estimate_ev is not None
    assert result.expanded_highest_interval.lower_ev < pts[-1].ea.lower_ev


def test_cardinal_large_latest_increment_requests_next_cardinal():
    pts = (
        point("aDZ", 2, 1, 1.40),
        point("aTZ", 3, 1, 1.50),
        point("aQZ", 4, 1, 1.54),
    )
    result = assess_cardinal_convergence(
        pts, augmentation_level=1, settings=SETTINGS
    )
    assert result.status is BasisConvergenceStatus.NEED_MORE_EVIDENCE
    assert result.action is BasisConvergenceAction.COMPUTE_NEXT_CARDINAL


def test_cardinal_oscillation_fails_closed():
    pts = (
        point("aDZ", 2, 1, 1.50),
        point("aTZ", 3, 1, 1.60),
        point("aQZ", 4, 1, 1.595),
    )
    result = assess_cardinal_convergence(
        pts, augmentation_level=1, settings=SETTINGS
    )
    assert result.status is BasisConvergenceStatus.NEED_MORE_EVIDENCE
    assert "CARDINAL_SERIES_OSCILLATORY" in result.evidence


def test_cardinal_two_points_are_not_enough():
    pts = (
        point("aDZ", 2, 1, 1.50),
        point("aTZ", 3, 1, 1.60),
    )
    result = assess_cardinal_convergence(
        pts, augmentation_level=1, settings=SETTINGS
    )
    assert result.action is BasisConvergenceAction.COMPUTE_NEXT_CARDINAL


def test_cardinal_nonconsecutive_highest_three_fail_closed():
    pts = (
        point("aDZ", 2, 1, 1.50),
        point("aQZ", 4, 1, 1.60),
        point("a5Z", 5, 1, 1.61),
    )
    result = assess_cardinal_convergence(
        pts, augmentation_level=1, settings=SETTINGS
    )
    assert result.status is BasisConvergenceStatus.NEED_MORE_EVIDENCE
    assert "HIGHEST_THREE_CARDINALS_NOT_CONSECUTIVE" in result.evidence


def test_mixed_method_signatures_are_rejected():
    pts = (
        point("aDZ", 2, 1, 1.50, method="CCSD(T)"),
        point("aTZ", 3, 1, 1.60, method="CCSD(T)"),
        point("aQZ", 4, 1, 1.61, method="CCSD"),
    )
    with pytest.raises(ValueError):
        assess_cardinal_convergence(
            pts, augmentation_level=1, settings=SETTINGS
        )


def test_small_nonaug_to_aug_shift_can_clear_without_daug():
    pts = (
        point("QZ", 4, 0, 1.600),
        point("aQZ", 4, 1, 1.604),
    )
    result = assess_diffuse_convergence(
        pts, cardinal_number=4, settings=SETTINGS
    )
    assert result.status is BasisConvergenceStatus.CLEARED
    assert result.highest_augmentation_level == 1


def test_large_nonaug_to_aug_shift_requests_double_augmented():
    pts = (
        point("QZ", 4, 0, 1.55),
        point("aQZ", 4, 1, 1.60),
    )
    result = assess_diffuse_convergence(
        pts, cardinal_number=4, settings=SETTINGS
    )
    assert result.action is BasisConvergenceAction.COMPUTE_DOUBLE_AUGMENTED


def test_force_double_augmentation_policy_is_respected():
    strict = BasisConvergenceSettings(
        cardinal_increment_target_ev=0.02,
        cardinal_contraction_ratio_max=0.60,
        diffuse_increment_target_ev=0.01,
        diffuse_contraction_ratio_max=0.60,
        force_double_augmentation=True,
    )
    pts = (
        point("QZ", 4, 0, 1.600),
        point("aQZ", 4, 1, 1.601),
    )
    result = assess_diffuse_convergence(
        pts, cardinal_number=4, settings=strict
    )
    assert result.action is BasisConvergenceAction.COMPUTE_DOUBLE_AUGMENTED


def test_contracting_daug_series_clears():
    pts = (
        point("QZ", 4, 0, 1.50),
        point("aQZ", 4, 1, 1.60),
        point("daQZ", 4, 2, 1.605),
    )
    result = assess_diffuse_convergence(
        pts, cardinal_number=4, settings=SETTINGS
    )
    assert result.status is BasisConvergenceStatus.CLEARED
    assert result.highest_augmentation_level == 2
    assert result.residual_estimate_ev is not None


def test_noncontracting_daug_series_requests_more_diffuse_evidence():
    pts = (
        point("QZ", 4, 0, 1.50),
        point("aQZ", 4, 1, 1.52),
        point("daQZ", 4, 2, 1.55),
    )
    result = assess_diffuse_convergence(
        pts, cardinal_number=4, settings=SETTINGS
    )
    assert result.status is BasisConvergenceStatus.NEED_MORE_EVIDENCE
    assert result.action is BasisConvergenceAction.COMPUTE_MORE_DIFFUSE


def test_interval_uncertainty_participates_in_increment_bound():
    pts = (
        point("QZ", 4, 0, 1.600, hw=0.010),
        point("aQZ", 4, 1, 1.600, hw=0.010),
    )
    result = assess_diffuse_convergence(
        pts, cardinal_number=4, settings=SETTINGS
    )
    assert result.latest_increment_bound_ev == pytest.approx(0.020)
    assert result.action is BasisConvergenceAction.COMPUTE_DOUBLE_AUGMENTED
