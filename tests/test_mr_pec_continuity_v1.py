"""Contract tests for fail-closed MR PEC active-space and root fingerprints."""
from dataclasses import replace

import numpy as np
import pytest

from openea_benchmark.adaptive.mr_casscf_nevpt2_runner import (
    MRPointRequest, MRPointResult, MRPointStatus, MRRootEnergy,
)
from openea_benchmark.adaptive.mr_pec_continuity import (
    MRPECContinuityStatus, MRPECContinuityThresholds,
    assess_mr_pec_continuity,
)


def thresholds(**overrides):
    return MRPECContinuityThresholds(**({
        'max_delta_r_angstrom': 0.2,
        'min_active_subspace_singular_value': 0.95,
        'min_root_density_similarity': 0.9,
        'min_root_assignment_margin': 0.15,
        'max_active_orthonormality_error': 1e-6,
    } | overrides))


def request(name, r=1.6):
    return MRPointRequest(
        request_id=name, system='AB', role='neutral', atoms=('Li', 'H'),
        charge=0, spin_2s=0, state_manifold_id='sigma', r_angstrom=r,
        basis_label='TEST', basis_by_element={'Li':'sto-3g','H':'sto-3g'},
        source_root_id='HF-' + name, source_checkpoint_path='/dev/null',
        active_orbital_indices=(1,2), active_electrons_alpha=1,
        active_electrons_beta=1, nroots=2,
        active_space_review_ids=('CAS-reviewed',),
        state_manifold_review_ids=('manifold-reviewed',),
    )


def point(req, *, swapped=False, coefficients=None, densities=None):
    if densities is None:
        densities = (np.diag([2.0,0.0]),np.diag([0.0,2.0]))
    if swapped:
        densities = tuple(reversed(densities))
    if coefficients is None:
        coefficients = np.eye(2)
    roots=[]
    for i,g in enumerate(densities):
        roots.append(MRRootEnergy(
            i,-8.0+i*0.1,-0.01,-8.01+i*0.1,0.0,
            active_rdm1=tuple(tuple(float(x) for x in row) for row in g),
        ))
    return MRPointResult(
        req.request_id, MRPointStatus.COMPLETE_REVIEW_REQUIRED,
        'ROOT_REVIEW_REQUIRED', roots=tuple(roots),
        active_space_review_ids=req.active_space_review_ids,
        state_manifold_review_ids=req.state_manifold_review_ids,
        source_checkpoint_sha256='a'*64, result_signature=('b' if not swapped else 'c')*64,
        active_mo_coeff_ao=tuple(tuple(float(x) for x in row) for row in coefficients),
    )


def identity_metric(*_):
    return np.eye(2), np.eye(2), np.eye(2)


def compare(a=None,b=None,**kwargs):
    a = a or request('p0')
    b = b or request('p1', 1.65)
    ar = kwargs.pop('ar', point(a))
    br = kwargs.pop('br', point(b))
    return assess_mr_pec_continuity(
        a,ar,b,br,thresholds=kwargs.pop('thresholds',thresholds()),
        overlap_provider=kwargs.pop('overlap_provider',identity_metric),**kwargs,
    )


def test_unmodified_root_order_only_candidate_not_certified():
    result=compare()
    assert result.status is MRPECContinuityStatus.CANDIDATE_REVIEW_REQUIRED
    assert result.root_candidates == ((0,0),(1,1))
    assert result.active_subspace_singular_values == (1.0,1.0)
    assert not result.scientific_state_identity_cleared and not result.mr_production_validated


def test_root_swap_identified_from_densities_not_index_or_energy():
    b=request('p1',1.65)
    result=compare(br=point(b,swapped=True))
    assert result.root_candidates == ((0,1),(1,0))
    assert result.status is MRPECContinuityStatus.CANDIDATE_REVIEW_REQUIRED


def test_orbital_gauge_rotation_is_handled_by_cross_density_metric():
    b=request('p1',1.65)
    p=point(b,coefficients=np.array([[0.,1.],[-1.,0.]]),
            densities=(np.diag([0.,2.]),np.diag([2.,0.])))
    result=compare(br=p)
    assert result.status is MRPECContinuityStatus.CANDIDATE_REVIEW_REQUIRED
    assert result.root_candidates == ((0,0),(1,1))


def test_small_cross_geometry_active_subspace_overlap_fails_closed():
    result=compare(overlap_provider=lambda *_: (np.eye(2),0.5*np.eye(2),np.eye(2)))
    assert result.status is MRPECContinuityStatus.UNRESOLVED
    assert not result.root_candidates


def test_ambiguous_same_root_densities_fail_closed():
    b=request('p1',1.65)
    result=compare(br=point(b,densities=(np.eye(2),np.eye(2))))
    assert result.status is MRPECContinuityStatus.UNRESOLVED
    assert not result.root_candidates


def test_missing_rdm_fails_closed():
    b=request('p1',1.65)
    old=point(b)
    old=replace(old,roots=tuple(replace(root,active_rdm1=None) for root in old.roots))
    assert compare(br=old).status is MRPECContinuityStatus.UNRESOLVED


def test_missing_optimized_active_orbitals_fails_closed():
    b=request('p1',1.65)
    assert compare(br=replace(point(b),active_mo_coeff_ao=None)).status is MRPECContinuityStatus.UNRESOLVED


@pytest.mark.parametrize('change', [
    {'charge':-1}, {'spin_2s':1,'active_electrons_alpha':2}, {'role':'anion'},
    {'state_manifold_id':'other'}, {'r_angstrom':1.9},
    {'atoms':('B','H'),'basis_by_element':{'B':'sto-3g','H':'sto-3g'}},
    {'active_electrons_alpha':2,'active_electrons_beta':2},
    {'active_orbital_indices':(1,2,3)}, {'nroots':1},
])
def test_incompatible_or_far_manifolds_fail_closed(change):
    a=request('p0')
    b=replace(request('p1',1.65),**change)
    assert compare(a,b, ar=point(a), br=point(b)).status is MRPECContinuityStatus.UNRESOLVED


def test_bad_identity_provenance_fails_closed():
    b=request('p1',1.65)
    assert compare(br=replace(point(b), result_signature=None)).status is MRPECContinuityStatus.UNRESOLVED
    assert compare(br=replace(point(b), source_checkpoint_sha256=None)).status is MRPECContinuityStatus.UNRESOLVED
    assert compare(br=replace(point(b),state_manifold_review_ids=('other',))).status is MRPECContinuityStatus.UNRESOLVED


def test_invalid_ao_metrics_fail_closed():
    assert compare(overlap_provider=lambda *_: (np.eye(3),np.eye(2),np.eye(2))).status is MRPECContinuityStatus.UNRESOLVED
    assert compare(overlap_provider=lambda *_: (np.eye(2)*0.5,np.eye(2),np.eye(2))).status is MRPECContinuityStatus.UNRESOLVED
    assert compare(overlap_provider=lambda *_: (np.eye(2),np.eye(2)*2,np.eye(2))).status is MRPECContinuityStatus.UNRESOLVED


def test_invalid_root_rdm_occupations_fail_closed():
    b=request('p1',1.65)
    bad=point(b,densities=(np.diag([3.,-1.]),np.diag([0.,2.])))
    assert compare(br=bad).status is MRPECContinuityStatus.UNRESOLVED


def test_comparison_does_not_compute_or_modify_energies():
    a=request('p0'); b=request('p1',1.65)
    ar=point(a); br=point(b)
    result=compare(a,b,ar=ar,br=br)
    assert result.root_candidates and ar.roots[0].sc_nevpt2_total_hartree == -8.01
    assert not result.mr_production_validated


@pytest.mark.parametrize('change', [
    {'max_delta_r_angstrom':0}, {'min_root_density_similarity':1.1},
    {'min_active_subspace_singular_value':0}, {'min_root_assignment_margin':-0.1},
    {'max_active_orthonormality_error':0.2},
    {'min_root_density_similarity':float('nan')},
    {'min_root_assignment_margin':float('nan')},
])
def test_explicit_thresholds_must_be_physical(change):
    with pytest.raises(ValueError): thresholds(**change)


def test_mr_root_result_validates_new_payload_structure():
    with pytest.raises(ValueError,match='1RDM'):
        MRRootEnergy(0,-1,-0.1,-1.1,0,active_rdm1=((float('nan'),),))
    with pytest.raises(ValueError,match='orbital'):
        MRPointResult('x',MRPointStatus.ERROR,'no',active_mo_coeff_ao=((1.,),))


def test_new_result_fields_roundtrip_as_dict():
    from dataclasses import asdict
    d=asdict(point(request('p0')))
    assert len(d['active_mo_coeff_ao'])==2
    assert len(d['roots'][0]['active_rdm1'])==2


def test_system_provenance_must_match():
    b=replace(request('p1',1.65), system='different-system-id')
    assert compare(b=b).status is MRPECContinuityStatus.UNRESOLVED


def test_even_zero_user_margin_cannot_approve_exact_tied_root():
    b=request('p1',1.65)
    # Both left roots see the same right-root fingerprint; a raw argmax
    # cannot infer a unique continuous electronic state.
    result=compare(br=point(b,densities=(np.eye(2),np.eye(2))),
                   thresholds=thresholds(min_root_assignment_margin=0.0,
                                         min_root_density_similarity=0.1))
    assert result.status is MRPECContinuityStatus.UNRESOLVED
