from openea_benchmark.adaptive.stage3_execution import Stage3ExecutionSettings
from openea_benchmark.attachment.scalar_relativity import (
    ScalarRelativityPoint,
    assess_scalar_relativity,
)

def p(x, d):
    return ScalarRelativityPoint(
        cardinal=x,
        basis=f"X{x}",
        ea_nr_ev=1.8,
        ea_sfx2c1e_ev=1.8+d,
        delta_sr_ev=d,
    )

def test_stage3_default_is_nonrelativistic():
    assert Stage3ExecutionSettings().scalar_relativistic == "NONE"

def test_stage3_accepts_sfx2c1e():
    assert Stage3ExecutionSettings(
        scalar_relativistic="SFX2C1E"
    ).scalar_relativistic == "SFX2C1E"

def test_stage3_rejects_unknown_relativistic_mode():
    try:
        Stage3ExecutionSettings(scalar_relativistic="DKH99")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown scalar-relativistic mode accepted")

def test_one_point_requires_next_cardinal():
    a = assess_scalar_relativity([p(3, -0.0010)])
    assert a.status == "NEED_MORE_EVIDENCE"
    assert a.action == "COMPUTE_X4"

def test_converged_two_point_series_clears():
    a = assess_scalar_relativity(
        [p(3, -0.0010), p(4, -0.0012)],
        target_change_ev=0.0005,
    )
    assert a.status == "CLEARED"
    assert abs(a.convergence_bound_ev - 0.0002) < 1e-12

def test_large_change_requests_5z():
    a = assess_scalar_relativity(
        [p(3, -0.0010), p(4, -0.0020)],
        target_change_ev=0.0005,
        max_cardinal=5,
    )
    assert a.status == "NEED_MORE_EVIDENCE"
    assert a.action == "COMPUTE_X5"

def test_max_cardinal_fails_closed():
    a = assess_scalar_relativity(
        [p(4, -0.0010), p(5, -0.0020)],
        target_change_ev=0.0005,
        max_cardinal=5,
    )
    assert a.status == "UNRESOLVED"
