"""Fail-closed SOC scientific contract and real-call boundary tests."""
from dataclasses import replace
from pathlib import Path

import pytest

from openea_benchmark.adaptive.mr_casscf_nevpt2_runner import MRPointRequest
from openea_benchmark.adaptive.soc_fci_siso_runner import (
    FCI_SISO_PIN, HARTREE_TO_EV, SOCPointAuthorization, SOCPointRequest,
    SOCPointResult, SOCPointSettings, SOCPointStatus, SOCSpinFreeRoot,
    SOCSpinManifold, SOCEACandidate, assess_soc_ea_pair, run_soc_fci_siso_point,
)


def soc_req(tmp_path, role='neutral'):
    checkpoint = tmp_path / f'{role}.chk'
    checkpoint.write_bytes(b'HF-checkpoint-' + role.encode())
    if role == 'neutral':
        spin, charge, ne, indices = 0, 0, (1, 1), (1, 2)
        manifolds = (SOCSpinManifold(0, 1), SOCSpinManifold(2, 1))
    else:
        spin, charge, ne, indices = 1, -1, (2, 1), (1, 2, 3)
        manifolds = (SOCSpinManifold(1, 1), SOCSpinManifold(3, 1))
    point = MRPointRequest(
        request_id=role.upper(), system='LiH', role=role,
        atoms=('Li','H'), charge=charge, spin_2s=spin,
        state_manifold_id='MANIFOLD', r_angstrom=1.6,
        basis_label='STO3G', basis_by_element={'Li':'sto-3g','H':'sto-3g'},
        source_root_id='HF', source_checkpoint_path=str(checkpoint),
        active_orbital_indices=indices,
        active_electrons_alpha=ne[0], active_electrons_beta=ne[1],
        nroots=1, active_space_review_ids=('CAS_DIAGNOSTIC',),
        state_manifold_review_ids=('MANIFOLD_DIAGNOSTIC',),
    )
    return SOCPointRequest(point, manifolds, spin, ('ROOT_REVIEW_OPEN',))


def auth(**changes):
    return SOCPointAuthorization(**{**dict(
        authorized=True,rationale='explicit development test',evidence_ids=('AUTH',),
        max_fci_determinants=100,max_spin_orbit_states=20,
    ),**changes})


def complete(req, shift=-0.001, digest='1'*64):
    roots = tuple(SOCSpinFreeRoot(m.spin_2s, i, -7.0 + i*.1)
                  for i,m in enumerate(req.spin_manifolds))
    es = [-7.0+shift] + [-6.95 + 0.01*i for i in range(
        sum(m.nroots*(m.spin_2s+1) for m in req.spin_manifolds)-1)]
    return SOCPointResult(req.point.request_id, SOCPointStatus.COMPLETE_REVIEW_REQUIRED,
                          'review', spin_free_roots=roots, soc_energies_hartree=tuple(es),
                          spin_free_ground_hartree=-7., soc_ground_hartree=-7.+shift,
                          delta_soc_hartree=shift, scalar_hamiltonian="NONE",
                          fci_siso_source_sha256=digest)


def backend(req, settings):
    return complete(req)


def test_authorized_soc_is_diagnostic_and_auditable(tmp_path):
    req=soc_req(tmp_path)
    got=run_soc_fci_siso_point(req,auth(),backend=backend)
    assert got.status is SOCPointStatus.COMPLETE_REVIEW_REQUIRED
    assert got.result_signature and got.source_checkpoint_sha256
    assert not got.production_soc_validated and not got.soc_uncertainty_bounded
    assert len(got.soc_energies_hartree) == 4


def test_unapproved_soc_cannot_run(tmp_path):
    req=soc_req(tmp_path)
    called=[]
    got=run_soc_fci_siso_point(req,auth(authorized=False,evidence_ids=()),
                               backend=lambda r,s:called.append(1))
    assert got.status is SOCPointStatus.BLOCKED_NOT_AUTHORIZED
    assert not called


def test_soc_cost_limit_blocks_backend(tmp_path):
    req=soc_req(tmp_path)
    got=run_soc_fci_siso_point(req,auth(max_fci_determinants=1),backend=backend)
    assert got.status is SOCPointStatus.BLOCKED_INVALID_INPUT
    got=run_soc_fci_siso_point(req,auth(max_spin_orbit_states=2),backend=backend)
    assert got.status is SOCPointStatus.BLOCKED_INVALID_INPUT


def test_checkpoint_must_exist(tmp_path):
    req=soc_req(tmp_path)
    Path(req.point.source_checkpoint_path).unlink()
    assert run_soc_fci_siso_point(req,auth(),backend=backend).status is SOCPointStatus.BLOCKED_INVALID_INPUT


def test_missing_backend_path_fails_closed(tmp_path):
    req=soc_req(tmp_path)
    result=run_soc_fci_siso_point(req,auth())
    assert result.status is SOCPointStatus.BACKEND_UNAVAILABLE


def test_real_backend_missing_checkout_reported(tmp_path):
    req=soc_req(tmp_path)
    s=SOCPointSettings(fci_siso_checkout=str(tmp_path/'nonexistent'))
    assert run_soc_fci_siso_point(req,auth(),settings=s).status is SOCPointStatus.BACKEND_UNAVAILABLE


def test_backend_exception_is_not_success(tmp_path):
    req=soc_req(tmp_path)
    got=run_soc_fci_siso_point(req,auth(),backend=lambda *args:1/0)
    assert got.status is SOCPointStatus.ERROR and got.delta_soc_hartree is None


def test_backend_must_report_all_roots_and_spin_sectors(tmp_path):
    req=soc_req(tmp_path)
    r=complete(req)
    with pytest.raises(ValueError,match='roots'):
        run_soc_fci_siso_point(req,auth(),backend=lambda *a:replace(r,spin_free_roots=r.spin_free_roots[:1]))
    with pytest.raises(ValueError,match='spin sectors'):
        changed=(replace(r.spin_free_roots[0],spin_2s=2),r.spin_free_roots[1])
        run_soc_fci_siso_point(req,auth(),backend=lambda *a:replace(r,spin_free_roots=changed))


def test_backend_requires_complete_soc_spin_multiplets(tmp_path):
    req=soc_req(tmp_path)
    r=complete(req)
    with pytest.raises(ValueError,match='spin projections'):
        run_soc_fci_siso_point(req,auth(),backend=lambda *a:replace(r,soc_energies_hartree=r.soc_energies_hartree[:2]))


def test_backend_mismatched_id_fails(tmp_path):
    req=soc_req(tmp_path)
    with pytest.raises(ValueError,match='mismatched'):
        run_soc_fci_siso_point(req,auth(),backend=lambda *a:replace(complete(req),request_id='OTHER'))


def test_no_unreviewed_binary_revision():
    with pytest.raises(ValueError,match='version'):
        SOCPointSettings(fci_siso_revision='master')
    assert len(FCI_SISO_PIN)==40


@pytest.mark.parametrize('bad', [
    (SOCSpinManifold(0,1),SOCSpinManifold(0,2)),
    (SOCSpinManifold(0,1),SOCSpinManifold(1,1)),
    (SOCSpinManifold(2,1),),
    (SOCSpinManifold(0,10),SOCSpinManifold(2,10),SOCSpinManifold(4,1)),
])
def test_bad_spin_manifolds_rejected(tmp_path,bad):
    p=soc_req(tmp_path).point
    with pytest.raises(ValueError):
        SOCPointRequest(p,bad,0,('REVIEW',))


def test_missing_review_and_wrong_reference_spin_rejected(tmp_path):
    p=soc_req(tmp_path).point
    with pytest.raises(ValueError):
        SOCPointRequest(p,(SOCSpinManifold(0,1),),0,())
    with pytest.raises(ValueError):
        SOCPointRequest(p,(SOCSpinManifold(2,1),),2,('REVIEW',))


def test_never_can_promote_siso_correction():
    with pytest.raises(ValueError,match='cannot certify'):
        SOCEACandidate('CANDIDATE_REVIEW_REQUIRED','no',0.1,production_validated=True)
    with pytest.raises(ValueError,match='cannot certify'):
        SOCEACandidate('UNRESOLVED','no',uncertainty_bounded=True)
    with pytest.raises(ValueError):
        SOCPointResult('x',SOCPointStatus.ERROR,'no',production_soc_validated=True)
    with pytest.raises(ValueError):
        SOCPointResult('x',SOCPointStatus.ERROR,'no',delta_soc_hartree=0.)


def test_soc_pair_is_only_review_candidate(tmp_path):
    n=soc_req(tmp_path,'neutral')
    a=soc_req(tmp_path,'anion')
    nr=run_soc_fci_siso_point(n,auth(),backend=lambda *x:complete(n,shift=-.003))
    ar=run_soc_fci_siso_point(a,auth(),backend=lambda *x:complete(a,shift=-.005))
    got=assess_soc_ea_pair(n,nr,a,ar)
    assert got.status=='CANDIDATE_REVIEW_REQUIRED'
    assert got.delta_ea_soc_ev==pytest.approx(.002*HARTREE_TO_EV)
    assert not got.production_validated and not got.uncertainty_bounded


@pytest.mark.parametrize('field,value',[
    ('role','neutral'),('charge',0),('atoms',('B','H')),
    ('basis_label','OTHER'),
])
def test_soc_pair_inconsistent_neutral_anion_fails(tmp_path,field,value):
    n=soc_req(tmp_path,'neutral'); a=soc_req(tmp_path,'anion')
    ar=run_soc_fci_siso_point(a,auth(),backend=backend)
    nr=run_soc_fci_siso_point(n,auth(),backend=backend)
    overrides={field:value}
    if field=='atoms':
        overrides['basis_by_element']={'B':'sto-3g','H':'sto-3g'}
    a=replace(a,point=replace(a.point,**overrides))
    assert assess_soc_ea_pair(n,nr,a,ar).status=='UNRESOLVED'


def test_soc_pair_provenance_missing_or_mismatched_fails(tmp_path):
    n=soc_req(tmp_path,'neutral'); a=soc_req(tmp_path,'anion')
    nr=run_soc_fci_siso_point(n,auth(),backend=backend)
    ar=run_soc_fci_siso_point(a,auth(),backend=backend)
    for bad in (replace(ar,fci_siso_source_sha256='2'*64),
                replace(ar,result_signature=None)):
        assert assess_soc_ea_pair(n,nr,a,bad).status=='UNRESOLVED'
    assert assess_soc_ea_pair(n,nr,a,replace(ar,status=SOCPointStatus.ERROR,
            spin_free_roots=(),soc_energies_hartree=(),spin_free_ground_hartree=None,
            soc_ground_hartree=None,delta_soc_hartree=None)).status=='UNRESOLVED'


def test_soc_result_energy_conservation_and_sorted_spectrum(tmp_path):
    req=soc_req(tmp_path)
    r=complete(req)
    with pytest.raises(ValueError):
        replace(r,delta_soc_hartree=-.5)
    with pytest.raises(ValueError):
        replace(r,soc_energies_hartree=tuple(reversed(r.soc_energies_hartree)))


def test_settings_are_in_checkpoint_signature(tmp_path):
    req=soc_req(tmp_path)
    x=run_soc_fci_siso_point(req,auth(),backend=backend)
    y=run_soc_fci_siso_point(req,auth(),settings=SOCPointSettings(
        mr=replace(SOCPointSettings().mr, scf_conv_tol=1e-9)), backend=backend)
    assert x.result_signature!=y.result_signature
    Path(req.point.source_checkpoint_path).write_bytes(b'NEW SOURCE')
    z=run_soc_fci_siso_point(req,auth(),backend=backend)
    assert x.result_signature!=z.result_signature


def test_soc_pair_different_scalar_hamiltonians_rejected(tmp_path):
    n=soc_req(tmp_path,'neutral'); a=soc_req(tmp_path,'anion')
    nr=run_soc_fci_siso_point(n,auth(),backend=backend)
    ar=run_soc_fci_siso_point(a,auth(),backend=backend)
    assert assess_soc_ea_pair(n,nr,a,replace(ar,scalar_hamiltonian="SFX2C1E")).status=='UNRESOLVED'


def test_backend_must_report_actual_scalar_hamiltonian(tmp_path):
    n=soc_req(tmp_path)
    with pytest.raises(ValueError,match='Hamiltonian'):
        run_soc_fci_siso_point(n,auth(),settings=SOCPointSettings(
            mr=replace(SOCPointSettings().mr,scalar_relativistic="SFX2C1E")),backend=backend)
