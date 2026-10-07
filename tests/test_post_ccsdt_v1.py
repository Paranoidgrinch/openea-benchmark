from openea_benchmark.adaptive.model import MethodRole
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


def test_one_triples_cardinal_requests_next_triples_point_only():
    a = assess_post_ccsd_t([point(2, 0.0004)])
    assert a.status == 'NEED_MORE_EVIDENCE'
    assert a.action == 'COMPUTE_T3_X3'
    assert a.central_correction_ev is None
    assert a.quadruples_correction_ev is None


def test_small_and_stable_triples_can_authorize_optional_delta_t3():
    a = assess_post_ccsd_t([
        point(2, 0.0004),
        point(3, 0.0005),
    ], triples_target_ev=0.001)
    assert a.status == 'CLEARED'
    assert a.action == 'NONE'
    assert abs(a.central_correction_ev - 0.0005) < 1e-12
    assert abs(a.triples_convergence_bound_ev - 0.0001) < 1e-12
    assert a.combined_bound_ev == a.triples_convergence_bound_ev


def test_large_but_stable_triples_triggers_method_warning_not_ccsdtq():
    a = assess_post_ccsd_t([
        point(2, 0.0100),
        point(3, 0.0102),
    ], triples_target_ev=0.001)
    assert a.status == 'POST_CC_WARNING'
    assert a.action == 'REASSESS_REFERENCE_CHARACTER'
    assert a.central_correction_ev is None
    assert not a.action.startswith('COMPUTE_T4')


def test_small_but_unstable_triples_triggers_method_warning():
    a = assess_post_ccsd_t([
        point(2, -0.0008),
        point(3, 0.0008),
    ], triples_target_ev=0.001)
    assert a.status == 'POST_CC_WARNING'
    assert a.action == 'REASSESS_REFERENCE_CHARACTER'


def test_legacy_t4_data_is_preserved_but_not_used():
    a = assess_post_ccsd_t([
        point(2, 0.0004, 0.0200),
        point(3, 0.0005),
    ], triples_target_ev=0.001, quadruples_target_ev=0.001, max_quadruples_cardinal=3)
    assert a.status == 'CLEARED'
    assert a.quadruples_correction_ev is None
    assert a.highest_quadruples_cardinal == 2
    assert abs(a.central_correction_ev - 0.0005) < 1e-12
    assert 'LEGACY_CCSDTQ_DATA_PRESENT_BUT_NOT_USED_IN_PRODUCTION_DIAGNOSTIC' in a.evidence


def test_result_is_explicitly_diagnostic_never_production_ea():
    a = assess_post_ccsd_t([
        point(2, 0.0004),
        point(3, 0.0005),
    ])
    assert a.method_role is MethodRole.DIAGNOSTIC
    assert a.is_production_ea is False
