from openea_benchmark.adaptive.adaptive_diffuse_runner import (
    AdaptiveDiffuseStatus,
    correlation_consistent_diffuse_basis_name,
    run_adaptive_diffuse_series,
)
from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceSettings,
    EAIntervalEV,
    ElectronicEABasisPoint,
)


def settings(force=False):
    return BasisConvergenceSettings(
        cardinal_increment_target_ev=0.02,
        cardinal_contraction_ratio_max=0.75,
        diffuse_increment_target_ev=0.01,
        diffuse_contraction_ratio_max=0.60,
        force_double_augmentation=force,
    )


def point(level, value, hw=0.001, cardinal=5):
    return ElectronicEABasisPoint(
        basis_name=correlation_consistent_diffuse_basis_name(
            cardinal, augmentation_level=level
        ),
        cardinal_number=cardinal,
        augmentation_level=level,
        method_signature="CCSD(T)|TEST",
        ea=EAIntervalEV(value - hw, value, value + hw),
    )


def test_small_nonaug_aug_shift_clears_without_daug():
    values = {0: 1.812, 1: 1.815}
    calls = []
    result = run_adaptive_diffuse_series(
        cardinal_number=5,
        initial_augmentation_levels=(0, 1),
        maximum_augmentation_level=2,
        convergence_settings=settings(),
        evaluate_point=lambda level: (calls.append(level) or point(level, values[level])),
    )
    assert result.status is AdaptiveDiffuseStatus.DIFFUSE_CLEARED
    assert calls == [0, 1]


def test_large_shift_requests_and_runs_double_augmented():
    values = {0: 1.75, 1: 1.815, 2: 1.820}
    calls = []
    result = run_adaptive_diffuse_series(
        cardinal_number=5,
        initial_augmentation_levels=(0, 1),
        maximum_augmentation_level=2,
        convergence_settings=settings(),
        evaluate_point=lambda level: (calls.append(level) or point(level, values[level])),
    )
    assert result.status is AdaptiveDiffuseStatus.DIFFUSE_CLEARED
    assert calls == [0, 1, 2]
    assert result.iterations[0].requested_next_augmentation_level == 2


def test_force_double_augmentation_is_respected():
    values = {0: 1.812, 1: 1.815, 2: 1.8155}
    calls = []
    result = run_adaptive_diffuse_series(
        cardinal_number=5,
        initial_augmentation_levels=(0, 1),
        maximum_augmentation_level=2,
        convergence_settings=settings(force=True),
        evaluate_point=lambda level: (calls.append(level) or point(level, values[level])),
    )
    assert calls == [0, 1, 2]
    assert result.status is AdaptiveDiffuseStatus.DIFFUSE_CLEARED


def test_noncontracting_daug_hits_explicit_limit():
    values = {0: 1.75, 1: 1.77, 2: 1.80}
    result = run_adaptive_diffuse_series(
        cardinal_number=5,
        initial_augmentation_levels=(0, 1),
        maximum_augmentation_level=2,
        convergence_settings=settings(),
        evaluate_point=lambda level: point(level, values[level]),
    )
    assert result.status is AdaptiveDiffuseStatus.DIFFUSE_LIMIT_REACHED
    assert result.iterations[-1].requested_next_augmentation_level == 3


def test_execution_failure_is_preserved():
    def evaluate(level):
        if level == 2:
            raise RuntimeError("basis backend missing d-aug")
        values = {0: 1.75, 1: 1.815}
        return point(level, values[level])

    result = run_adaptive_diffuse_series(
        cardinal_number=5,
        initial_augmentation_levels=(0, 1),
        maximum_augmentation_level=2,
        convergence_settings=settings(),
        evaluate_point=evaluate,
    )
    assert result.status is AdaptiveDiffuseStatus.EXECUTION_BLOCKED
    assert result.execution_error_type == "RuntimeError"


def test_basis_names_are_explicit():
    assert correlation_consistent_diffuse_basis_name(5, augmentation_level=0) == "cc-pv5z"
    assert correlation_consistent_diffuse_basis_name(5, augmentation_level=1) == "aug-cc-pv5z"
    assert correlation_consistent_diffuse_basis_name(5, augmentation_level=2) == "d-aug-cc-pv5z"


def test_runner_never_promotes_production_ea():
    values = {0: 1.812, 1: 1.815}
    result = run_adaptive_diffuse_series(
        cardinal_number=5,
        initial_augmentation_levels=(0, 1),
        maximum_augmentation_level=2,
        convergence_settings=settings(),
        evaluate_point=lambda level: point(level, values[level]),
    )
    assert result.is_production_ea is False
    assert result.authorizes_pruning is False
