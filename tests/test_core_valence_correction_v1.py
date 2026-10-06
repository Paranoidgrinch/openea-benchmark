from openea_benchmark.attachment.core_valence_correction import (
    CoreValenceStatus, CVCardinalPoint, assess_core_valence, cv_point_from_results,
)

def cv(x,d):
    return CVCardinalPoint(x,f"basis{x}",1.0+d,1.0,d)

def test_tz_qz_clear_when_latest_change_small():
    r=assess_core_valence((cv(3,0.0040),cv(4,0.0048)),target_change_ev=0.002)
    assert r.status is CoreValenceStatus.CLEARED
    assert r.central_correction_ev==0.0048
    assert abs(r.convergence_bound_ev-0.0008)<1e-12

def test_tz_qz_request_5z_when_not_converged():
    r=assess_core_valence((cv(3,0.002),cv(4,0.005)),target_change_ev=0.002)
    assert r.status is CoreValenceStatus.NEED_MORE_EVIDENCE
    assert r.action=="COMPUTE_X5"

def test_three_points_require_contraction():
    r=assess_core_valence((cv(3,0.0),cv(4,0.003),cv(5,0.0045)),
                          target_change_ev=0.002,max_contraction_ratio=0.8)
    assert r.status is CoreValenceStatus.CLEARED
    assert abs(r.contraction_ratio-0.5)<1e-12

def test_fail_closed_at_max_cardinal():
    r=assess_core_valence((cv(3,0.0),cv(4,0.003),cv(5,0.006)),
                          target_change_ev=0.002,max_cardinal=5)
    assert r.status is CoreValenceStatus.NEED_MORE_EVIDENCE
    assert r.action=="MAX_CARDINAL_REACHED_UNRESOLVED"

def test_cv_definition_all_electron_minus_frozen_core():
    r=cv_point_from_results(cardinal=3,basis="x",
        ae_neutral_h=-10.0,ae_anion_h=-10.1,
        fc_neutral_h=-9.9,fc_anion_h=-10.0,
        hartree_to_ev=27.0)
    assert abs(r.delta_cv_ev)<1e-12
