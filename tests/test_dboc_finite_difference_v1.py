from dataclasses import replace
from types import SimpleNamespace
from math import exp
import numpy as np
import pytest

from openea_benchmark.adaptive.dboc_finite_difference import (
    AMU_TO_ELECTRON_MASS, BOHR_TO_ANGSTROM,
    DBOCStatus, HFDBOCRequest, HFDBOCSettings, HFDBOCResult,
    determinant_fidelity, quantum_metric_from_fidelity,
    nuclear_cartesian_displacement, run_hf_dboc_point, assess_hf_dboc_pair,
)
from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV


def req(role='neutral', **kw):
    payload = dict(request_id=f'{role}_HF', role=role, atoms=('Li','H'),
                   charge=0 if role=='neutral' else -1,
                   spin_2s=0 if role=='neutral' else 1,
                   r_angstrom=1.60, basis_by_element={'Li':'sto-3g','H':'sto-3g'},
                   state_id='ground', state_identity_reviewed=True,
                   nuclear_masses_amu=(7.0143579,1.007276466621),
                   nuclear_mass_source='explicit nuclear isotopic masses',
                   scf_reference='RHF' if role=='neutral' else 'ROHF')
    payload.update(kw)
    return HFDBOCRequest(**payload)


def fake_backend(request, xyz, settings):
    # One normalized occupied orbital for each spin (RHF) or only alpha (ROHF).
    val = np.ones((1,1))
    occ = np.array([2.]) if request.scf_reference == 'RHF' else np.array([1.])
    mf = SimpleNamespace(mo_coeff=val, mo_occ=occ, converged=True)
    return np.asarray(xyz), mf


def fake_metric(xyz1, xyz2):
    delta = np.linalg.norm(np.asarray(xyz1)-np.asarray(xyz2))/BOHR_TO_ANGSTROM
    return np.asarray([[exp(-0.2*delta*delta)]])


def evaluate(role='neutral'):
    return run_hf_dboc_point(req(role), HFDBOCSettings(), backend=fake_backend, ao_overlap=fake_metric)


def test_determinant_fidelity_closed_shell_spin_product():
    one = np.ones((1,1))
    assert determinant_fidelity(one, one, one, one, np.asarray([[.9]])) == pytest.approx(.9**4)


def test_determinant_fidelity_open_shell_single_alpha():
    one = np.ones((1,1)); empty = np.zeros((1,0))
    assert determinant_fidelity(one, empty, one, empty, np.asarray([[.9]])) == pytest.approx(.9**2)


def test_determinant_fidelity_phase_and_orbital_rotation_invariance():
    o=np.eye(2); s=np.eye(2)
    rot=np.asarray([[0.,-1.],[1.,0.]])
    assert determinant_fidelity(o, o, o@rot, o@rot, s) == pytest.approx(1)


def test_dimension_mismatch_rejected():
    with pytest.raises(ValueError, match='dimension'):
        determinant_fidelity(np.ones((2,1)),np.ones((2,1)),np.ones((2,1)),np.ones((2,1)),np.ones((3,3)))


def test_electron_count_change_rejected():
    with pytest.raises(ValueError, match='Electron count'):
        determinant_fidelity(np.ones((2,1)),np.zeros((2,0)),np.ones((2,2)),np.zeros((2,0)),np.eye(2))


def test_norm_greater_than_one_rejected():
    with pytest.raises(ValueError, match='Unphysical'):
        determinant_fidelity(np.ones((1,1)),np.ones((1,1)),np.ones((1,1)),np.ones((1,1)),np.asarray([[1.1]]))


@pytest.mark.parametrize('change', [
 {'charge':-1}, {'state_identity_reviewed':False}, {'nuclear_masses_amu':(7.,0.)},
 {'nuclear_masses_amu':(7.,float('nan'))}, {'nuclear_mass_source':''},
 {'r_angstrom':-1.}, {'spin_2s':1}, {'scf_reference':'CCSD'},
 {'basis_by_element':{'Li':'sto-3g'}}, {'request_id':''},
])
def test_invalid_request_rejected(change):
    with pytest.raises(ValueError):
        req(**change)


@pytest.mark.parametrize('change', [
 {'steps_bohr':(.005,.005)}, {'steps_bohr':(.001,)},
 {'steps_bohr':(.003,.004)}, {'steps_bohr':(.2,.1)},
 {'max_scf_calculations':23}, {'scf_conv_tol':1e-4},
 {'min_determinant_fidelity':1.2}, {'max_memory_mb':-3},
])
def test_invalid_settings_rejected(change):
    with pytest.raises(ValueError):
        HFDBOCSettings(**change)


def test_displacement_geometry_units_and_axes():
    xyz=nuclear_cartesian_displacement(1.6,1,0,0.01)
    assert xyz[0] == (0.,0.,0.)
    assert xyz[1][0] == pytest.approx(.01*BOHR_TO_ANGSTROM)
    assert xyz[1][2] == pytest.approx(1.6)


def test_quantum_metric_central_step_and_safety():
    h=.005
    assert quantum_metric_from_fidelity(1-.0004,h) == pytest.approx(4.)
    with pytest.raises(ValueError): quantum_metric_from_fidelity(1.1,h)
    with pytest.raises(ValueError): quantum_metric_from_fidelity(.9,0)


def test_hf_dboc_uses_three_axes_two_nuclei_two_steps_and_nuclear_masses():
    r=evaluate()
    assert r.status == DBOCStatus.COMPLETE_REVIEW_REQUIRED
    assert r.scf_count == 24
    assert len(r.axes) == 12
    # Fidelity model S_AB=exp(-k*(2h)^2). RHF fidelity=S_AB^4,
    # hence g ~= 4*k per axis; sum over 3 axes and both nuclei.
    expect=6*.2*(1/7.0143579+1/1.007276466621)/AMU_TO_ELECTRON_MASS
    assert r.dboc_by_step_hartree[-1] == pytest.approx(expect,rel=5e-5)
    assert r.max_step_difference_ev >= 0
    assert r.scientific_uncertainty_bounded is False
    assert r.is_production_correction is False


def test_open_shell_alpha_only_fidelity_and_no_automatic_physical_ea():
    a=evaluate('anion'); n=evaluate()
    assert a.status == DBOCStatus.COMPLETE_REVIEW_REQUIRED
    assert 0 < a.dboc_by_step_hartree[-1] < n.dboc_by_step_hartree[-1]
    pair=assess_hf_dboc_pair(req(),req('anion'),n,a)
    assert pair.status == DBOCStatus.COMPLETE_REVIEW_REQUIRED
    assert pair.delta_ea_hartree_candidate == pytest.approx(n.dboc_by_step_hartree[-1]-a.dboc_by_step_hartree[-1])
    assert not pair.correlated_remainder_bounded
    assert not pair.nonadiabatic_remainder_bounded
    assert not pair.is_production_correction


def test_mass_and_basis_mismatch_and_provenance_rejected():
    a=evaluate('anion'); n=evaluate()
    with pytest.raises(ValueError,match='isotope'):
        assess_hf_dboc_pair(req(),req('anion',nuclear_masses_amu=(6.0,1.)),n,a)
    with pytest.raises(ValueError,match='basis'):
        assess_hf_dboc_pair(req(),req('anion',basis_by_element={'Li':'6-31g','H':'6-31g'}),n,a)
    with pytest.raises(ValueError,match='provenance'):
        assess_hf_dboc_pair(req(),req('anion'),replace(n,request_id='other'),a)


def test_missing_or_nonconverged_scf_stays_unresolved():
    def broken_backend(*args):
        xyz,mf=fake_backend(*args)
        mf.converged=False
        return xyz,mf
    r=run_hf_dboc_point(req(),backend=broken_backend,ao_overlap=fake_metric)
    assert r.status is DBOCStatus.UNRESOLVED
    assert not r.dboc_by_step_hartree


def test_low_overlap_blocks_root_switch():
    r=run_hf_dboc_point(req(),backend=fake_backend,ao_overlap=lambda *_: np.asarray([[.1]]))
    assert r.status is DBOCStatus.UNRESOLVED
    assert 'overlap' in r.reason


def test_one_failed_species_blocks_ea_correction():
    n=evaluate(); a=evaluate('anion')
    pair=assess_hf_dboc_pair(req(),req('anion'),n,replace(a,status=DBOCStatus.UNRESOLVED))
    assert pair.status is DBOCStatus.UNRESOLVED
    assert pair.delta_ea_hartree_candidate is None


def test_sign_convention_is_neutral_minus_anion():
    n=evaluate(); a=evaluate('anion')
    pair=assess_hf_dboc_pair(req(),req('anion'),n,a)
    assert pair.delta_ea_hartree_candidate > 0
    assert pair.to_dict()['neutral']['status'] == 'COMPLETE_REVIEW_REQUIRED'
