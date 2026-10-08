"""Pure mathematics/policy tests, no PySCF/PySCF-stub computations."""
from dataclasses import replace
import numpy as np
import pytest

from openea_benchmark.adaptive.attachment_eom_runner import (
    G2EOMBasis, G2EOMNeutralState, G2EOMRoot, G2EOMRawResult, G2EOMRequest,
    G2EOMSubpoint, G2EOMSeries, G2EOMStatus, evidence_points_from_review,
)
from openea_benchmark.adaptive.attachment_root_continuity import (
    RootContinuityStatus, RootContinuitySettings, normalized_ao_one_particle_overlap,
    propose_g2_root_continuity,
)
from openea_benchmark.adaptive.attachment_continuum import pyscf_eom_eigenvalue_to_attachment_ea_ev
from openea_benchmark.adaptive.model import Review, ReviewStatus


def root(i, alpha, *, beta=None, fraction=.8, omega=-0.1):
    return G2EOMRoot(i, omega, pyscf_eom_eigenvalue_to_attachment_ea_ev(omega),
                     tuple(alpha) if alpha is not None else None,
                     tuple(beta) if beta is not None else None, fraction)


def make_series(*, swapped=True, poor=False, scale=False):
    state = G2EOMNeutralState('XY', ('X', 'Y'), 0, 0, 'N', 1.3, 'ROOT',
                             '/tmp/source.chk', True, True, 'RHF')
    subs = []
    for k in range(3):
        basis = G2EOMBasis(k, f'aug{k}', 'family', {'X':'X', 'Y':'Y'})
        first = root(0, (1, 0) if k == 0 or not swapped else (0, 1))
        second = root(1, (0, 1) if k == 0 or not swapped else (-1, 0))
        if poor and k == 1:
            first = replace(first, one_particle_amplitude_fraction=.001)
            second = replace(second, one_particle_amplitude_fraction=.001)
        req = G2EOMRequest(f'aug:{k}', state, basis, None, (), 'a'*64)
        subs.append(G2EOMSubpoint(req, G2EOMRawResult(req.key, (first, second),
                         'RHF', 'FAKE', True, True, True), False, f'raw:{k}'))
    if scale:
        for j, f in enumerate((.8, 1.2)):
            parent = subs[-1].request
            req = G2EOMRequest(f'scale:2:{f}', state, parent.basis, f, (), 'a'*64)
            a = root(0, (-1, 0) if swapped else (1, 0))
            b = root(1, (0, 1))
            subs.append(G2EOMSubpoint(req, G2EOMRawResult(req.key,
                (a, b), 'RHF', 'FAKE', True, True, True), False, f'raw-scale:{j}'))
    return G2EOMSeries(G2EOMStatus.COMPLETE_ROOT_REVIEW_REQUIRED, tuple(subs), ('synthetic',))


def overlaps(a, b):
    return np.eye(2)


def test_root_tracking_follows_overlap_not_root_index_or_energy():
    report = propose_g2_root_continuity(make_series(swapped=True),
        starting_root_index=0, overlap_provider=overlaps)
    assert report.status is RootContinuityStatus.CANDIDATE_REVIEW_REQUIRED
    assert report.proposed_roots == (('aug:0', 0), ('aug:1', 1), ('aug:2', 1))
    assert [x.normalized_overlap for x in report.links] == pytest.approx([1., 1.])
    assert report.review.status is ReviewStatus.UNRESOLVED
    assert report.boundness_decision == 'UNRESOLVED'
    assert report.evidence_id.startswith('G2_ROOT_PROPOSAL:')


def test_spinor_overlap_is_phase_invariant():
    a = root(0, (1,0), beta=(0,1))
    b = root(0, (-1,0), beta=(0,-1))
    assert normalized_ao_one_particle_overlap(a,b,saa=np.eye(2),sab=np.eye(2),sbb=np.eye(2)) == pytest.approx(1)


def test_spin_mismatch_rejected():
    a = root(0,(1,0));b=root(0,(1,0),beta=(0,1))
    with pytest.raises(ValueError,match='spin representations'):
        normalized_ao_one_particle_overlap(a,b,saa=np.eye(2),sab=np.eye(2),sbb=np.eye(2))


def test_nonorthogonal_cross_basis_overlap_normalized():
    a=root(0,(1,0));b=root(1,(2,0))
    assert normalized_ao_one_particle_overlap(a,b,saa=np.eye(2),sab=np.eye(2),sbb=np.eye(2)) == pytest.approx(1)


def test_ao_dim_mismatch_rejected():
    a=root(0,(1,0));b=root(0,(1,0))
    with pytest.raises(ValueError,match='AO dimensions'):
        normalized_ao_one_particle_overlap(a,b,saa=np.eye(3),sab=np.eye(2),sbb=np.eye(2))


def test_physically_impossible_overlap_rejected():
    a=root(0,(1,0));b=root(0,(1,0))
    with pytest.raises(ValueError,match='Nonphysical normalized'):
        normalized_ao_one_particle_overlap(a,b,saa=np.eye(2),sab=2*np.eye(2),sbb=np.eye(2))


def test_no_directions_no_candidate():
    series=make_series()
    first=series.subpoints[0]
    raw=replace(first.result, roots=tuple(replace(r,one_particle_ao_alpha=None) for r in first.result.roots))
    series=replace(series,subpoints=(replace(first,result=raw),)+series.subpoints[1:])
    res=propose_g2_root_continuity(series,starting_root_index=0,overlap_provider=overlaps)
    assert res.status is RootContinuityStatus.MISSING_ONE_PARTICLE_DATA
    assert res.review.status is ReviewStatus.UNRESOLVED


def test_low_1p_content_cannot_track():
    res=propose_g2_root_continuity(make_series(poor=True),starting_root_index=0,overlap_provider=overlaps)
    assert res.status is RootContinuityStatus.MISSING_ONE_PARTICLE_DATA


def test_no_match_if_cross_overlap_is_weak():
    def near_orthogonal(a,b):
        return np.eye(2) if a.request.key==b.request.key else np.array([[.1,0],[0,.1]])
    res=propose_g2_root_continuity(make_series(),starting_root_index=0,overlap_provider=near_orthogonal)
    assert res.status is RootContinuityStatus.AMBIGUOUS
    assert res.links[0].normalized_overlap == pytest.approx(.1)


def test_nearly_degenerate_targets_block():
    series=make_series()
    mid=series.subpoints[1]
    # nearly identical target root fingerprints should fail unique match
    raw=replace(mid.result, roots=(replace(mid.result.roots[0], one_particle_ao_alpha=(1.,0.)),
                                   replace(mid.result.roots[1], one_particle_ao_alpha=(1.,0.))))
    series=replace(series, subpoints=(series.subpoints[0],replace(mid,result=raw),series.subpoints[2]))
    res=propose_g2_root_continuity(series,starting_root_index=0,overlap_provider=overlaps)
    assert res.status is RootContinuityStatus.AMBIGUOUS


def test_two_sources_same_target_block_mutual_best():
    series=make_series()
    start=series.subpoints[0]
    raw=replace(start.result, roots=(start.result.roots[0], replace(start.result.roots[1],
                                   one_particle_ao_alpha=(1.,0.))))
    series=replace(series,subpoints=(replace(start,result=raw),)+series.subpoints[1:])
    res=propose_g2_root_continuity(series,starting_root_index=0,overlap_provider=overlaps)
    assert res.status is RootContinuityStatus.AMBIGUOUS
    assert 'competing' in res.links[0].reason


def test_stabilization_compared_to_its_own_baseline():
    series=make_series(swapped=True,scale=True)
    res=propose_g2_root_continuity(series,starting_root_index=0,overlap_provider=overlaps)
    assert res.status is RootContinuityStatus.CANDIDATE_REVIEW_REQUIRED
    # scale root 0 corresponds to aug:2 root 1
    assert dict(res.proposed_roots)['scale:2:0.8'] == 0
    assert res.links[-1].reference_key=='aug:2'


def test_same_provenance_required():
    series=make_series()
    p=series.subpoints[2]
    changed=replace(p,request=replace(p.request,source_checkpoint_sha256='b'*64))
    series=replace(series,subpoints=series.subpoints[:2]+(changed,))
    with pytest.raises(ValueError,match='same neutral/source'):
        propose_g2_root_continuity(series,starting_root_index=0,overlap_provider=overlaps)


def test_incomplete_runner_rejected():
    series=replace(make_series(),status=G2EOMStatus.EXECUTION_BLOCKED)
    with pytest.raises(ValueError,match='Incomplete'):
        propose_g2_root_continuity(series,starting_root_index=0,overlap_provider=overlaps)


def test_explicit_start_index_required_to_exist():
    with pytest.raises(ValueError,match='Starting root absent'):
        propose_g2_root_continuity(make_series(),starting_root_index=88,overlap_provider=overlaps)


def test_proposal_id_cannot_substitute_external_approval():
    series=make_series()
    res=propose_g2_root_continuity(series,starting_root_index=0,overlap_provider=overlaps)
    selections={p.request.key:(0,Review(ReviewStatus.CLEARED,(res.evidence_id,),"Overlap says so"))
                for p in series.subpoints}
    with pytest.raises(ValueError,match='separate reviewed evidence'):
        evidence_points_from_review(series,selections=selections)


def test_root_validation_of_nonfinite_coefficients_and_fraction():
    with pytest.raises(ValueError,match='Invalid AO'):
        root(0,(1,float('nan')))
    with pytest.raises(ValueError,match='amplitude fraction'):
        root(0,(1,0),fraction=2.)


def test_policy_validation():
    with pytest.raises(ValueError,match='thresholds'):
        RootContinuitySettings(min_overlap=1.5)


def test_no_automatic_not_applicable_root_review():
    res=propose_g2_root_continuity(make_series(),starting_root_index=0,overlap_provider=overlaps)
    assert res.review.status is ReviewStatus.UNRESOLVED


def test_single_point_cannot_claim_continuity():
    series=make_series()
    with pytest.raises(ValueError,match='at least two'):
        propose_g2_root_continuity(replace(series,subpoints=series.subpoints[:1]),
                                  starting_root_index=0,overlap_provider=overlaps)


def test_rhf_eom_one_particle_projection_uses_active_virtuals_only():
    from openea_benchmark.adaptive.attachment_eom_runner import _project_eom_one_particle_to_ao

    class C:
        def get_frozen_mask(self):
            return np.array([False, True, True, True])

    class M:
        mo_coeff = np.eye(4)
        mo_occ = np.array([2,2,0,0])

    class E:
        def vector_to_amplitudes(self, v):
            return np.array([3.,4.]), np.array([0.,0.])

    alpha, beta, fraction = _project_eom_one_particle_to_ao(E(),C(),M(),
                              np.array([3.,4.,0.,0.]),'RHF')
    assert alpha == pytest.approx((0.,0.,3.,4.))
    assert beta is None
    assert fraction == pytest.approx(1.)


def test_uhf_eom_projection_retains_spin_channels_separately():
    from openea_benchmark.adaptive.attachment_eom_runner import _project_eom_one_particle_to_ao

    class C:
        def get_frozen_mask(self):
            return (np.array([False,True,True]), np.array([True,True,True]))

    class M:
        mo_coeff = (np.eye(3),np.eye(3))
        mo_occ = (np.array([1,1,0]),np.array([1,0,0]))

    class E:
        def vector_to_amplitudes(self, v):
            return (np.array([2.]),np.array([3.,4.])), ()

    alpha, beta, fraction = _project_eom_one_particle_to_ao(E(),C(),M(),
                  np.array([2.,3.,4.,0.,0.]),'UHF')
    assert alpha == pytest.approx((0.,0.,2.))
    assert beta == pytest.approx((0.,3.,4.))
    assert fraction == pytest.approx(1.)


def test_rhf_projection_rejects_truncated_r1():
    from openea_benchmark.adaptive.attachment_eom_runner import _project_eom_one_particle_to_ao

    class C:
        def get_frozen_mask(self):
            return np.ones(3, dtype=bool)

    class M:
        mo_coeff = np.eye(3)
        mo_occ = np.array([2,0,0])

    class E:
        def vector_to_amplitudes(self,v):
            return np.array([1.]), np.array([0.])

    with pytest.raises(ValueError,match='inconsistent'):
        _project_eom_one_particle_to_ao(E(),C(),M(),np.array([1.,0.]),'RHF')


def test_right_eom_1p_fraction_is_not_physical_dyson_weight():
    # The algebraic fraction is diagnostic, not returned as EOM pole strength.
    a=root(0,(1,0),fraction=.3)
    assert a.one_particle_amplitude_fraction == pytest.approx(.3)
    assert not hasattr(a,'dyson_pole_strength')


def test_new_ao_directions_survive_g2_raw_checkpoint_resume(tmp_path):
    from openea_benchmark.adaptive.attachment_eom_runner import (
        G2EOMAuthorization, G2EOMSettings, run_g2_eom_diagnostics,
    )
    cp = tmp_path / 'neutral.chk'
    cp.write_bytes(b'fixed neutral state')
    state = replace(make_series().subpoints[0].request.state,
                    source_checkpoint_path=str(cp))
    basis = (G2EOMBasis(0, 'X', 'fam', {'X': 'a0', 'Y': 'b0'}),)
    def loader(element, name):
        return [[0, [float(len(name)), 1.0]]]
    calls = []
    def backend(req,settings,material):
        calls.append(req.key)
        return G2EOMRawResult(req.key,
            (root(0,(1.0,0.0),fraction=.25),
             root(1,(0.0,1.0),fraction=.5)),
            'RHF','MOCK',True,True,True)
    args=dict(authorization=G2EOMAuthorization(True,'reviewed',('AUTH',),('aug:0',)),
              neutral=state,basis_specs=basis,
              settings=G2EOMSettings(nroots=2),
              backend=backend,basis_loader=loader,
              backend_id='MOCK_AO_V2',checkpoint_dir=tmp_path/'points')
    first=run_g2_eom_diagnostics(**args)
    restored=run_g2_eom_diagnostics(**args)
    assert calls==['aug:0']
    assert restored.subpoints[0].checkpoint_reused
    assert first.subpoints[0].result.roots == restored.subpoints[0].result.roots
    assert restored.subpoints[0].result.roots[0].one_particle_amplitude_fraction == .25
