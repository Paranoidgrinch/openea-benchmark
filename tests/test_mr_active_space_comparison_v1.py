"""Safety and numerical contracts for comparing unlike CAS partitions."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from openea_benchmark.adaptive.mr_casscf_nevpt2_runner import (
    MRPointRequest, MRPointResult, MRRootEnergy, MRPointSettings,
    MRPointStatus, MRPointAuthorization, run_mr_casscf_nevpt2_point,
)
from openea_benchmark.adaptive.mr_active_space_comparison import (
    MRActiveSpaceStatus, MRActiveSpaceThresholds,
    MRActiveSpaceComparison, assess_mr_active_space_pair,
)


def requests(tmp_path):
    chk = tmp_path / 'hf.chk'
    if not chk.exists():
        chk.write_bytes(b'origin-checkpoint')
    common = dict(
        system='Diatomic', role='neutral', atoms=('Li','H'), charge=0,
        spin_2s=0, state_manifold_id='STATE_SET', r_angstrom=1.60,
        basis_label='STO3G', basis_by_element={'Li':'sto-3g','H':'sto-3g'},
        source_root_id='RHF0', source_checkpoint_path=str(chk), nroots=2,
        state_manifold_review_ids=('STATE_MANIFOLD_UNVALIDATED',),
    )
    a = MRPointRequest(
        request_id='SMALL', active_orbital_indices=(1,2),
        active_electrons_alpha=1, active_electrons_beta=1,
        active_space_review_ids=('SMALL_CAS_UNVALIDATED',), **common,
    )
    b = MRPointRequest(
        request_id='LARGE', active_orbital_indices=(0,1,2),
        active_electrons_alpha=2, active_electrons_beta=2,
        active_space_review_ids=('LARGE_CAS_UNVALIDATED',), **common,
    )
    return a,b


def result(request, *, swapped=False, missing_core=False, settings=None, energy_shift=0):
    if request.ncas == 2:
        ca = np.eye(4)[:, [1,2]]
        cc = np.eye(4)[:, [0]]
        gamma = [np.diag([2.,0.]), np.diag([0.,2.])]
    else:
        ca = np.eye(4)[:, [0,1,2]]
        cc = np.zeros((4,0))
        gamma = [np.diag([2.,2.,0.]), np.diag([2.,0.,2.])]
    if swapped:
        gamma = list(reversed(gamma))
    def fake(r,s):
        roots = tuple(MRRootEnergy(
            k, -8 + 0.2*k + energy_shift, -0.02,
            -8.02 + 0.2*k + energy_shift, 0.,
            active_rdm1=tuple(tuple(float(z) for z in row) for row in g)
        ) for k,g in enumerate(gamma))
        return MRPointResult(
            r.request_id, MRPointStatus.COMPLETE_REVIEW_REQUIRED,
            'not validated', cas_orbital_optimization_converged=True,
            roots=roots, active_space_review_ids=r.active_space_review_ids,
            state_manifold_review_ids=r.state_manifold_review_ids,
            active_mo_coeff_ao=tuple(tuple(float(z) for z in row) for row in ca),
            inactive_mo_coeff_ao=None if missing_core else tuple(
                tuple(float(z) for z in row) for row in cc),
        )
    return run_mr_casscf_nevpt2_point(
        request, MRPointAuthorization(True,'test',('AUTH',)),
        settings=settings, backend=fake,
    )


def policy(**changes):
    return MRActiveSpaceThresholds(**{
        **dict(min_density_similarity=0.90, min_root_assignment_margin=0.05,
               max_orbital_metric_error=1e-8, min_ao_metric_eigenvalue=1e-8),
        **changes,
    })


def compare(tmp_path, *, a=None, b=None, ar=None, br=None, provider=None,
            left_settings=None, right_settings=None, thresholds=None):
    aa, bb = requests(tmp_path)
    a = a or aa; b = b or bb
    ar = ar or result(a, settings=left_settings)
    br = br or result(b, settings=right_settings, energy_shift=-0.001)
    return assess_mr_active_space_pair(
        a, ar, b, br, thresholds=thresholds or policy(),
        left_settings=left_settings, right_settings=right_settings,
        overlap_provider=provider or (lambda _: np.eye(4)),
    )


def test_different_cas_ncore_recovered_through_full_ao_density(tmp_path):
    c = compare(tmp_path)
    assert c.status is MRActiveSpaceStatus.CANDIDATE_REVIEW_REQUIRED, c.reason
    assert [(s.left_root,s.right_root) for s in c.candidate_root_shifts] == [(0,0),(1,1)]
    assert all(abs(s.sc_nevpt2_total_shift_hartree + 0.001) < 1e-10
               for s in c.candidate_root_shifts)
    assert not any((c.root_identity_validated,c.active_space_converged,
                    c.method_uncertainty_bounded,c.mr_production_validated))
    assert c.density_similarities[0][0] == pytest.approx(1.0)


def test_swapped_root_order_detected_from_ao_densities(tmp_path):
    _,b = requests(tmp_path)
    c = compare(tmp_path, br=result(b,swapped=True))
    assert c.status is MRActiveSpaceStatus.CANDIDATE_REVIEW_REQUIRED
    assert [(s.left_root,s.right_root) for s in c.candidate_root_shifts] == [(0,1),(1,0)]


def test_near_tied_densities_return_unresolved_even_with_zero_margin(tmp_path):
    _,b=requests(tmp_path)
    original = result(b)
    tied = replace(original, roots=tuple(replace(x,active_rdm1=original.roots[0].active_rdm1)
                                           for x in original.roots))
    c = compare(tmp_path, br=tied, thresholds=policy(min_root_assignment_margin=0.0))
    assert c.status is MRActiveSpaceStatus.UNRESOLVED
    assert not c.candidate_root_shifts


def test_missing_inactive_mos_fail_closed(tmp_path):
    _,b=requests(tmp_path)
    assert compare(tmp_path,br=result(b,missing_core=True)).status is MRActiveSpaceStatus.UNRESOLVED


def test_missing_active_rdm_fail_closed(tmp_path):
    _,b=requests(tmp_path)
    r=result(b)
    r=replace(r,roots=tuple(replace(x,active_rdm1=None) for x in r.roots))
    assert compare(tmp_path,br=r).status is MRActiveSpaceStatus.UNRESOLVED


def test_unchanged_cas_rejected(tmp_path):
    a,_=requests(tmp_path)
    assert compare(tmp_path, b=replace(a,request_id='REPEAT')).status is MRActiveSpaceStatus.UNRESOLVED


@pytest.mark.parametrize('change', [
    {'r_angstrom':1.62}, {'atoms':('B','H'),'basis_by_element':{'B':'sto-3g','H':'sto-3g'}},
    {'basis_label':'OTHER'}, {'basis_by_element':{'Li':'def2-svp','H':'sto-3g'}},
    {'charge':-1}, {'role':'anion'}, {'source_root_id':'other'},
    {'state_manifold_id':'OTHER'}, {'state_manifold_review_ids':('OTHER',)},
    {'system':'OTHER'},
])
def test_mismatched_hamiltonian_or_state_fails_closed(tmp_path,change):
    _,b=requests(tmp_path)
    b=replace(b,**change)
    assert compare(tmp_path,b=b).status is MRActiveSpaceStatus.UNRESOLVED


def test_mismatched_source_checkpoint_fails_closed(tmp_path):
    a,b=requests(tmp_path)
    r=result(b)
    r=replace(r,source_checkpoint_sha256='f'*64)
    assert compare(tmp_path,a=a,b=b,br=r).status is MRActiveSpaceStatus.UNRESOLVED
    ar=result(a); br=result(b)
    Path(b.source_checkpoint_path).write_bytes(b'changed')
    assert compare(tmp_path,a=a,b=b,ar=ar,br=br).status is MRActiveSpaceStatus.UNRESOLVED


def test_signature_includes_settings_and_explicit_review_provenance(tmp_path):
    a,b=requests(tmp_path)
    r=result(b)
    assert compare(tmp_path,br=replace(r,result_signature='f'*64)).status is MRActiveSpaceStatus.UNRESOLVED
    assert compare(tmp_path,br=replace(r,active_space_review_ids=('OTHER',))).status is MRActiveSpaceStatus.UNRESOLVED
    other_settings = MRPointSettings(casscf_conv_tol=1e-9)
    assert compare(tmp_path,right_settings=other_settings).status is MRActiveSpaceStatus.UNRESOLVED


def test_mr_incomplete_or_missing_roots_fails_closed(tmp_path):
    _,b=requests(tmp_path)
    fail=MRPointResult(b.request_id,MRPointStatus.SCF_NOT_CONVERGED,'no')
    assert compare(tmp_path,br=fail).status is MRActiveSpaceStatus.UNRESOLVED
    br=result(b)
    assert compare(tmp_path,br=replace(br,roots=br.roots[:1])).status is MRActiveSpaceStatus.UNRESOLVED


def test_metric_bad_or_near_linearly_dependent_fails_closed(tmp_path):
    for matrix in (np.eye(3),np.zeros((4,4)),np.eye(4)*2,np.zeros((0,0))):
        c=compare(tmp_path,provider=lambda _, m=matrix:m)
        assert c.status is MRActiveSpaceStatus.UNRESOLVED
    c=compare(tmp_path,provider=lambda _:np.diag([1.,1.,1.,1e-11]))
    assert c.status is MRActiveSpaceStatus.UNRESOLVED


def test_unphysical_orbital_or_density_fails_closed(tmp_path):
    _,b=requests(tmp_path)
    r=result(b)
    bad=replace(r,active_mo_coeff_ao=tuple(tuple(2*x for x in row) for row in r.active_mo_coeff_ao))
    assert compare(tmp_path,br=bad).status is MRActiveSpaceStatus.UNRESOLVED
    badgamma=((3.,0.,0.),(0.,1.,0.),(0.,0.,0.))
    bad=replace(r,roots=tuple(replace(x,active_rdm1=badgamma) for x in r.roots))
    assert compare(tmp_path,br=bad).status is MRActiveSpaceStatus.UNRESOLVED


def test_new_inactive_core_result_payload_validation():
    with pytest.raises(ValueError,match='inactive'):
        MRPointResult('x',MRPointStatus.ERROR,'no',inactive_mo_coeff_ao=((1.,),))
    with pytest.raises(ValueError,match='AO'):
        MRPointResult('x',MRPointStatus.COMPLETE_REVIEW_REQUIRED,'yes',
                      roots=(MRRootEnergy(0,-1,-0.1,-1.1,0),),
                      active_mo_coeff_ao=((1.,),(0.,)),
                      inactive_mo_coeff_ao=((1.,),))


def test_cannot_force_scientific_claim_from_diagnostic():
    with pytest.raises(ValueError):
        MRActiveSpaceComparison(MRActiveSpaceStatus.CANDIDATE_REVIEW_REQUIRED,'not validated',
                                'a','b',active_space_converged=True)
    with pytest.raises(ValueError):
        MRActiveSpaceComparison(MRActiveSpaceStatus.UNRESOLVED,'no','a','b',root_identity_validated=True)


def test_single_root_never_automatically_proves_same_state(tmp_path):
    a,b=requests(tmp_path)
    a=replace(a,nroots=1)
    b=replace(b,nroots=1)
    aa,bb=requests(tmp_path)
    assert compare(tmp_path,a=a,b=b,ar=result(aa),br=result(bb)).status is MRActiveSpaceStatus.UNRESOLVED


def test_no_backend_or_energy_call_during_assessment(tmp_path):
    a,b=requests(tmp_path)
    ar=result(a);br=result(b)
    c=compare(tmp_path,a=a,b=b,ar=ar,br=br)
    assert c.status is MRActiveSpaceStatus.CANDIDATE_REVIEW_REQUIRED
    assert ar.roots[0].sc_nevpt2_total_hartree == -8.02


@pytest.mark.parametrize('field,bad',[
    ('min_density_similarity',0),('min_root_assignment_margin',-1),
    ('max_orbital_metric_error',float('nan')),
    ('min_ao_metric_eigenvalue',1.5),
])
def test_policy_rejects_invalid_limits(field,bad):
    with pytest.raises(ValueError):
        policy(**{field:bad})


def test_different_backend_version_and_spin_provenance_fail_closed(tmp_path):
    _,b=requests(tmp_path)
    r=result(b)
    assert compare(tmp_path,br=replace(r,pyscf_version='different')).status is MRActiveSpaceStatus.UNRESOLVED
    bad=replace(r,roots=tuple(replace(root,spin_square=2.0) for root in r.roots))
    assert compare(tmp_path,br=bad).status is MRActiveSpaceStatus.UNRESOLVED
