"""Exponential-basis stabilization is diagnostic, never boundness proof."""
from dataclasses import replace

import numpy as np
import pytest

from openea_benchmark.adaptive.attachment_eom_runner import (
    DiffuseShellSelector, G2EOMBasis, G2EOMNeutralState, G2EOMRawResult,
    G2EOMRequest, G2EOMRoot, G2EOMSeries, G2EOMStatus, G2EOMSubpoint,
)
from openea_benchmark.adaptive.attachment_root_continuity import (
    RootContinuityStatus, propose_g2_root_continuity,
)
from openea_benchmark.adaptive.attachment_stabilization_profile import (
    StabilizationProfileSettings, StabilizationProfileStatus,
    analyze_g2_stabilization_profile,
)
from openea_benchmark.adaptive.attachment_continuum import pyscf_eom_eigenvalue_to_attachment_ea_ev
from openea_benchmark.adaptive.model import ReviewStatus


def make_series(*, factors=(0.6,0.8,1.2,1.4), drift=0.001,
                gap=0.30, swap=False):
    neutral = G2EOMNeutralState('LiH', ('Li','H'), 0, 0, 'N', 1.5, 'NROOT',
                                '/tmp/source.chk', True, True, 'RHF')
    basis = G2EOMBasis(2, 'aug-2', 'same-family', {'Li':'a','H':'b'})
    subs=[]
    for j,f in enumerate((None,)+tuple(factors)):
        key = 'aug:2' if f is None else f'scale:2:{f}'
        rid = (1 if swap and j>0 else 0)
        om = -0.10 + (0 if f is None else drift * (f-1))
        leading = G2EOMRoot(rid, om, pyscf_eom_eigenvalue_to_attachment_ea_ev(om),
                            (1.,0.), None, .8)
        other = G2EOMRoot(1-rid, om+gap/27.211386245988,
                          pyscf_eom_eigenvalue_to_attachment_ea_ev(om+gap/27.211386245988),
                          (0.,1.), None, .8)
        req=G2EOMRequest(key,neutral,basis,f,
                         () if f is None else (DiffuseShellSelector('Li',0),), 'a'*64)
        raw=G2EOMRawResult(key,(leading,other),'RHF','FAKE',True,True,True)
        subs.append(G2EOMSubpoint(req,raw,False,f'G2_EOM:{j}'))
    return G2EOMSeries(G2EOMStatus.COMPLETE_ROOT_REVIEW_REQUIRED,tuple(subs),('raw',))


def proposal(series):
    return propose_g2_root_continuity(series,starting_root_index=0,
                                      overlap_provider=lambda a,b:np.eye(2))


def test_complete_profile_is_diagnostic_not_bound():
    series=make_series()
    report=analyze_g2_stabilization_profile(series,proposal(series))
    assert report.status is StabilizationProfileStatus.OBSERVED_REVIEW_REQUIRED
    assert len(report.points)==4
    assert report.energy_span_ev>0
    assert report.review.status is ReviewStatus.UNRESOLVED
    assert report.boundness_decision=='UNRESOLVED'
    assert report.method_role=='DIAGNOSTIC'


def test_root_order_swap_preserves_proposed_root():
    s=make_series(swap=True)
    r=analyze_g2_stabilization_profile(s,proposal(s))
    assert all(p.root_index==1 for p in r.points)
    assert r.status is StabilizationProfileStatus.OBSERVED_REVIEW_REQUIRED


def test_flat_pseudostate_remains_review_only():
    s=make_series(drift=0)
    r=analyze_g2_stabilization_profile(s,proposal(s))
    assert r.energy_span_ev==0
    assert r.review.status is ReviewStatus.UNRESOLVED
    assert r.boundness_decision=='UNRESOLVED'


def test_one_sided_scaling_is_inadequate():
    s=make_series(factors=(.6,.8,.9))
    r=analyze_g2_stabilization_profile(s,proposal(s))
    assert r.status is StabilizationProfileStatus.INSUFFICIENT_SCALE_COVERAGE
    assert 'INSUFFICIENT_BIDIRECTIONAL_SCALE_COVERAGE' in r.flags


def test_root_gap_warning_never_classes_as_unbound():
    s=make_series(gap=0.001)
    r=analyze_g2_stabilization_profile(s,proposal(s))
    assert r.status is StabilizationProfileStatus.POSSIBLE_ROOT_MIXING
    assert 'NEAR_DEGENERATE_EOM_ROOTS' in r.flags
    assert r.boundness_decision=='UNRESOLVED'


def test_large_drift_flag_is_not_unbound():
    s=make_series(drift=0.05)
    r=analyze_g2_stabilization_profile(s,proposal(s))
    assert 'LARGE_SCALE_DEPENDENCE' in r.flags
    assert r.boundness_decision=='UNRESOLVED'


def test_ambiguous_root_path_fails_closed():
    s=make_series()
    p=replace(proposal(s),status=RootContinuityStatus.AMBIGUOUS,
              proposed_roots=(('aug:2',0),))
    r=analyze_g2_stabilization_profile(s,p)
    assert r.status is StabilizationProfileStatus.ROOT_PATH_UNRESOLVED
    assert r.energy_span_ev is None


def test_missing_stabilization_points_raises():
    s=make_series(factors=())
    with pytest.raises(ValueError,match='No exponent scaling'):
        analyze_g2_stabilization_profile(s, proposal(make_series()))


def test_wrong_source_fingerprint_raises():
    s=make_series()
    sub=s.subpoints[-1]
    modified=replace(sub,request=replace(sub.request,source_checkpoint_sha256='b'*64))
    s=replace(s,subpoints=s.subpoints[:-1]+(modified,))
    # Build the proposal from pristine data: analysis must still reject mutated raw.
    with pytest.raises(ValueError,match='Incompatible'):
        analyze_g2_stabilization_profile(s,proposal(make_series()))


def test_multiple_augmentation_baselines_rejected():
    s=make_series()
    sub=s.subpoints[-1]
    modified=replace(sub,request=replace(sub.request,
                       basis=replace(sub.request.basis,augmentation_level=3)))
    s=replace(s,subpoints=s.subpoints[:-1]+(modified,))
    with pytest.raises(ValueError,match='one fixed augmentation'):
        analyze_g2_stabilization_profile(s,proposal(make_series()))


def test_duplicate_scale_factors_rejected():
    s=make_series()
    sub=s.subpoints[-1]
    altered=replace(sub,request=replace(sub.request,scale_factor=.8))
    s=replace(s,subpoints=s.subpoints[:-1]+(altered,))
    with pytest.raises(ValueError,match='Duplicate scale'):
        analyze_g2_stabilization_profile(s,proposal(make_series()))


def test_missing_selectors_rejected():
    s=make_series()
    sub=s.subpoints[-1]
    altered=replace(sub,request=replace(sub.request,selectors=()))
    s=replace(s,subpoints=s.subpoints[:-1]+(altered,))
    with pytest.raises(ValueError,match='Incompatible'):
        analyze_g2_stabilization_profile(s,proposal(make_series()))


def test_short_scan_coverage_configurable():
    s=make_series(factors=(.95,1.05))
    r=analyze_g2_stabilization_profile(s,proposal(s))
    assert r.status is StabilizationProfileStatus.INSUFFICIENT_SCALE_COVERAGE


def test_policy_validation():
    with pytest.raises(ValueError):
        StabilizationProfileSettings(min_scales_below_one=0)
    with pytest.raises(ValueError):
        StabilizationProfileSettings(min_log_scale_span=0)


def test_signature_changes_with_energy_evidence():
    a=make_series(drift=0.001);b=make_series(drift=0.002)
    aa=analyze_g2_stabilization_profile(a,proposal(a))
    bb=analyze_g2_stabilization_profile(b,proposal(b))
    assert aa.evidence_id!=bb.evidence_id



def test_continuity_proposal_provenance_must_match_raw_series():
    s=make_series()
    p=replace(proposal(s),source_evidence_ids=("G2_EOM:fake",))
    with pytest.raises(ValueError,match="does not match raw series"):
        analyze_g2_stabilization_profile(s,p)
