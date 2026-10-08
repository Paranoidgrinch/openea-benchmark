"""Nested, independently authorized SOC spin-manifold expansion tests."""
from dataclasses import replace
import pytest

from openea_benchmark.adaptive.mr_casscf_nevpt2_runner import MRPointRequest
from openea_benchmark.adaptive.soc_fci_siso_runner import (
    SOCPointAuthorization, SOCPointRequest, SOCPointResult, SOCPointSettings,
    SOCPointStatus, SOCSpinFreeRoot, SOCSpinManifold, run_soc_fci_siso_point,
)
from openea_benchmark.adaptive.soc_manifold_sensitivity import (
    SOCManifoldSensitivity, SOCManifoldSensitivitySettings,
    SOCManifoldSensitivityStatus as S, compare_soc_manifold_expansion,
    run_soc_manifold_expansion,
)


def pair(tmp_path, role='neutral'):
    c = tmp_path / 'hf.chk'
    c.write_bytes(b'same-checkpoint')
    if role == 'neutral':
        charge, spin, ne, cas = 0, 0, (1, 1), (1, 2)
        small, large = ((0, 1),), ((0, 1), (2, 1))
    else:
        charge, spin, ne, cas = -1, 1, (2, 1), (1, 2, 3)
        small, large = ((1, 1),), ((1, 1), (3, 1))
    p = MRPointRequest(request_id=role.upper(), system='LiH',role=role,
                       atoms=('Li','H'),charge=charge,spin_2s=spin,
                       state_manifold_id='STATE',r_angstrom=1.6,
                       basis_label='STO3G',basis_by_element={'Li':'sto-3g','H':'sto-3g'},
                       source_root_id='HF',source_checkpoint_path=str(c),
                       active_orbital_indices=cas,active_electrons_alpha=ne[0],
                       active_electrons_beta=ne[1],nroots=1,
                       active_space_review_ids=('REVIEW',),
                       state_manifold_review_ids=('REVIEW',))
    def make(ms):
        return SOCPointRequest(p,tuple(SOCSpinManifold(*m) for m in ms),spin,('GROUND_REVIEW',))
    return make(small),make(large)


def auth(authorized=True, limit=100):
    return SOCPointAuthorization(authorized, 'explicit authorized experiment',
                                 ('AUTH',) if authorized else (),
                                 max_fci_determinants=limit, max_spin_orbit_states=30)


def backend(req, settings, *, delta=-0.001, drift=0.0, sha='a'*64):
    roots=[]
    for m in req.spin_manifolds:
        for i in range(m.nroots):
            roots.append(SOCSpinFreeRoot(m.spin_2s, len(roots),
                                         -8.0+ (0.0 if m.spin_2s==req.ground_spin_2s else .2) + i*.07+drift))
    ng = sum(m.nroots*(m.spin_2s+1) for m in req.spin_manifolds)
    e = tuple(sorted([-8.0+delta+drift] + [-7.7+.001*i for i in range(ng-1)]))
    return SOCPointResult(req.point.request_id,SOCPointStatus.COMPLETE_REVIEW_REQUIRED,
                          'diagnostic',spin_free_roots=tuple(roots),soc_energies_hartree=e,
                          spin_free_ground_hartree=-8.0+drift,
                          soc_ground_hartree=-8.0+delta+drift,
                          delta_soc_hartree=delta,scalar_hamiltonian=settings.mr.scalar_relativistic,
                          fci_siso_source_sha256=sha)


def computed(tmp_path, role='neutral'):
    a,b=pair(tmp_path,role)
    sa=SOCPointSettings()
    x=run_soc_fci_siso_point(a,auth(),settings=sa,backend=lambda req,s:backend(req,s,delta=-.001))
    y=run_soc_fci_siso_point(b,auth(),settings=sa,backend=lambda req,s:backend(req,s,delta=-.0012))
    return a,x,b,y


def test_nested_comparison_returns_only_observed_shift(tmp_path):
    a,x,b,y=computed(tmp_path)
    out=compare_soc_manifold_expansion(a,x,b,y)
    assert out.status is S.CANDIDATE_REVIEW_REQUIRED
    assert out.observed_shift_change_hartree==pytest.approx(-.0002)
    assert out.observed_shift_change_ev==pytest.approx(-.0002*27.211386245988)
    assert out.retained_spin_free_max_drift_hartree==pytest.approx(0.)
    assert not out.production_soc_validated and not out.soc_uncertainty_bounded
    assert not out.spin_free_state_identity_cleared and not out.spin_manifold_complete


def test_open_shell_anion_candidate(tmp_path):
    a,x,b,y=computed(tmp_path,'anion')
    assert compare_soc_manifold_expansion(a,x,b,y).status is S.CANDIDATE_REVIEW_REQUIRED


def test_zero_observed_difference_is_not_error_bound(tmp_path):
    a,x,b,y=computed(tmp_path)
    y=replace(y,delta_soc_hartree=x.delta_soc_hartree,
              soc_ground_hartree=x.soc_ground_hartree,
              soc_energies_hartree=(x.soc_ground_hartree,)+y.soc_energies_hartree[1:])
    out=compare_soc_manifold_expansion(a,x,b,y)
    assert out.status is S.CANDIDATE_REVIEW_REQUIRED
    assert out.observed_shift_change_ev==pytest.approx(0.)
    assert out.soc_uncertainty_bounded is False


@pytest.mark.parametrize('change',[
    dict(r_angstrom=1.7), dict(role='anion'), dict(basis_label='different'),
    dict(source_checkpoint_path='other.chk'),dict(active_orbital_indices=(0,1)),
    dict(active_space_review_ids=('DIFFERENT',)),dict(charge=1),
])
def test_changed_physical_point_fails(tmp_path,change):
    a,x,b,y=computed(tmp_path)
    object.__setattr__(b,'point',replace(b.point,**change))
    out=compare_soc_manifold_expansion(a,x,b,y)
    assert out.status is S.UNRESOLVED and out.observed_shift_change_ev is None


def test_same_manifold_fails(tmp_path):
    a,x,b,y=computed(tmp_path)
    assert compare_soc_manifold_expansion(a,x,a,x).status is S.UNRESOLVED


def test_shrinking_manifold_fails(tmp_path):
    a,x,b,y=computed(tmp_path)
    assert compare_soc_manifold_expansion(b,y,a,x).status is S.UNRESOLVED


def test_modified_ground_spin_nroots_fails(tmp_path):
    a,x,b,y=computed(tmp_path)
    b=replace(b, spin_manifolds=(SOCSpinManifold(0,2),SOCSpinManifold(2,1)))
    assert compare_soc_manifold_expansion(a,x,b,y).status is S.UNRESOLVED


def test_changed_review_fails(tmp_path):
    a,x,b,y=computed(tmp_path)
    b=replace(b,ground_state_review_ids=('different',))
    assert compare_soc_manifold_expansion(a,x,b,y).status is S.UNRESOLVED


@pytest.mark.parametrize('field,value',[
    ('source_checkpoint_sha256','b'*64),
    ('source_checkpoint_sha256',None),
    ('fci_siso_source_sha256','b'*64),
    ('fci_siso_source_sha256',None),
    ('result_signature',None),
    ('result_signature','same-as-base'),
    ('scalar_hamiltonian','SFX2C1E'),
    ('request_id','DIFFERENT'),
])
def test_provenance_mismatch_fails(tmp_path,field,value):
    a,x,b,y=computed(tmp_path)
    if field=='result_signature' and value=='same-as-base':value=x.result_signature
    assert compare_soc_manifold_expansion(a,x,b,replace(y,**{field:value})).status is S.UNRESOLVED


def test_lost_root_fails(tmp_path):
    a,x,b,y=computed(tmp_path)
    y=replace(y,spin_free_roots=y.spin_free_roots[:1])
    assert compare_soc_manifold_expansion(a,x,b,y).status is S.UNRESOLVED


def test_incomplete_soc_projection_fails(tmp_path):
    a,x,b,y=computed(tmp_path)
    y=replace(y,soc_energies_hartree=y.soc_energies_hartree[:1])
    assert compare_soc_manifold_expansion(a,x,b,y).status is S.UNRESOLVED


def test_retained_energy_drift_above_threshold_fails(tmp_path):
    a,x,b,y=computed(tmp_path)
    rr=tuple(replace(r,energy_hartree=r.energy_hartree + .0002) for r in y.spin_free_roots)
    # completed result self-consistency requires updating ground reference.
    y=replace(y,spin_free_roots=rr,spin_free_ground_hartree=y.spin_free_ground_hartree+.0002,
              soc_ground_hartree=y.soc_ground_hartree+.0002,
              soc_energies_hartree=tuple(v+.0002 for v in y.soc_energies_hartree))
    out=compare_soc_manifold_expansion(a,x,b,y)
    assert out.status is S.UNRESOLVED
    assert out.retained_spin_free_max_drift_hartree==pytest.approx(.0002)


def test_tight_drift_threshold_can_reject(tmp_path):
    a,x,b,y=computed(tmp_path)
    rr=tuple(replace(r,energy_hartree=r.energy_hartree+1.e-7) for r in y.spin_free_roots)
    y=replace(y,spin_free_roots=rr,spin_free_ground_hartree=y.spin_free_ground_hartree+1.e-7,
              soc_ground_hartree=y.soc_ground_hartree+1.e-7,
              soc_energies_hartree=tuple(v+1.e-7 for v in y.soc_energies_hartree))
    assert compare_soc_manifold_expansion(a,x,b,y,
        settings=SOCManifoldSensitivitySettings(max_retained_spin_free_drift_hartree=1.e-8)).status is S.UNRESOLVED


def test_invalid_threshold_rejected():
    for v in (0,-.1,float('nan'),float('inf'),1):
        with pytest.raises(ValueError):SOCManifoldSensitivitySettings(v)


def test_cannot_promote_accuracy_flags():
    for flag in ('production_soc_validated','soc_uncertainty_bounded',
                 'spin_free_state_identity_cleared','spin_manifold_complete'):
        with pytest.raises(ValueError):
            SOCManifoldSensitivity(S.UNRESOLVED,'no','x','y',**{flag:True})
    with pytest.raises(ValueError):
        SOCManifoldSensitivity(S.UNRESOLVED,'no','x','y',observed_shift_change_ev=0.)


def test_two_independent_authorizations_gate_all_work(tmp_path):
    a,b=pair(tmp_path)
    calls=[]
    res=run_soc_manifold_expansion(a,b,auth(),auth(False),
                                   point_settings=SOCPointSettings(),
                                   backend=lambda req,s:calls.append(req))
    assert res.status is S.UNRESOLVED and not calls


def test_cost_block_prevents_expanded_job(tmp_path):
    a,b=pair(tmp_path)
    calls=[]
    def fake(req,settings):
        calls.append(req)
        return backend(req,settings)
    out=run_soc_manifold_expansion(a,b,auth(),auth(limit=1),
                                   point_settings=SOCPointSettings(),backend=fake)
    assert out.status is S.UNRESOLVED and len(calls)==1


def test_nonnested_job_does_not_execute(tmp_path):
    a,b=pair(tmp_path)
    b=replace(b,spin_manifolds=(SOCSpinManifold(0,1),))
    calls=[]
    out=run_soc_manifold_expansion(a,b,auth(),auth(),
                                   point_settings=SOCPointSettings(),
                                   backend=lambda req,s:calls.append(req))
    assert out.status is S.UNRESOLVED and not calls


def test_real_execution_gate_runs_twice(tmp_path):
    a,b=pair(tmp_path)
    calls=[]
    def fake(req,settings):
        calls.append(req)
        return backend(req,settings,delta=-0.001-.0002*(len(calls)-1))
    out=run_soc_manifold_expansion(a,b,auth(),auth(),
                                   point_settings=SOCPointSettings(),backend=fake)
    assert out.status is S.CANDIDATE_REVIEW_REQUIRED
    assert len(calls)==2
    assert out.observed_shift_change_ev==pytest.approx(-.0002*27.211386245988)
