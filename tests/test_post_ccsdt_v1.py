from openea_benchmark.attachment.post_ccsd_t import (
    PostCCPoint,
    assess_post_ccsd_t,
)


def point(x, t3, t4=None):
    return PostCCPoint(
        cardinal=x,
        basis=f"aug-cc-pV{'D' if x == 2 else 'T'}Z",
        ea_ccsd_t_ev=1.80,
        ea_ccsdt_ev=1.80 + t3,
        delta_t3_ev=t3,
        ea_ccsdtq_ev=None if t4 is None else 1.80 + t3 + t4,
        delta_t4_ev=t4,
    )


def test_one_triples_cardinal_requests_next():
    a = assess_post_ccsd_t([point(2, 0.001)])
    assert a.status == "NEED_MORE_EVIDENCE"
    assert a.action == "COMPUTE_T3_X3"


def test_small_dz_quadruples_can_be_conservatively_bounded():
    a = assess_post_ccsd_t([
        point(2, 0.0010, 0.0004),
        point(3, 0.0012),
    ], triples_target_ev=0.001, quadruples_target_ev=0.001)
    assert a.status == "CLEARED"
    assert abs(a.triples_convergence_bound_ev - 0.0002) < 1e-12
    assert abs(a.quadruples_convergence_bound_ev - 0.0004) < 1e-12
    assert abs(a.central_correction_ev - 0.0016) < 1e-12


def test_large_dz_quadruples_requests_tz_ccsdtq():
    a = assess_post_ccsd_t([
        point(2, 0.0010, 0.0020),
        point(3, 0.0012),
    ], quadruples_target_ev=0.001)
    assert a.status == "NEED_MORE_EVIDENCE"
    assert a.action == "COMPUTE_T4_X3"


def test_two_point_quadruples_series_can_clear():
    a = assess_post_ccsd_t([
        point(2, 0.0010, 0.0020),
        point(3, 0.0012, 0.0023),
    ], quadruples_target_ev=0.001)
    assert a.status == "CLEARED"
    assert abs(a.quadruples_convergence_bound_ev - 0.0003) < 1e-12


def test_unconverged_triples_fail_closed():
    a = assess_post_ccsd_t([
        point(2, 0.0000, 0.0002),
        point(3, 0.0030),
    ], triples_target_ev=0.001)
    assert a.status == "UNRESOLVED"
    assert a.action == "REVIEW_T3_BASIS_CONVERGENCE"


def test_result_is_never_production_ea():
    a = assess_post_ccsd_t([
        point(2, 0.0010, 0.0002),
        point(3, 0.0011),
    ])
    assert a.is_production_ea is False
