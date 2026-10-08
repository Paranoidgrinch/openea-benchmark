from dataclasses import replace
from math import exp

import numpy as np
import pytest

from openea_benchmark.adaptive.dboc_finite_difference import (
    AMU_TO_ELECTRON_MASS, DBOCStatus, HFDBOCRequest,
    HFDBOCSettings, quantum_metric_from_fidelity,
    determinant_fidelity, zero_step_richardson, run_hf_dboc_point,
    assess_hf_dboc_pair,
)


def test_normalized_primitive_gaussian_exact_metric_and_four_factor():
    alpha = .7
    for h in (.008, .004, .002):
        # Closed-form normalized Gaussian 1s overlap at R-h, R+h:
        # <psi(-h)|psi(+h)> = exp(-2 alpha h^2).
        cross_s = np.asarray([[exp(-2*alpha*h*h)]])
        one = np.ones((1,1)); empty = np.zeros((1,0))
        f = determinant_fidelity(one, empty, one, empty, cross_s)
        assert f == pytest.approx(exp(-4*alpha*h*h), abs=1e-13)
        g = quantum_metric_from_fidelity(f, h)
        assert g == pytest.approx(-np.expm1(-4*alpha*h*h)/(4*h*h), abs=1e-9)


def test_two_electron_gaussian_spin_product_has_double_metric():
    alpha=.7
    h=.005
    s=np.asarray([[exp(-2*alpha*h*h)]])
    one=np.ones((1,1))
    f=determinant_fidelity(one,one,one,one,s)
    g=quantum_metric_from_fidelity(f,h)
    assert g == pytest.approx(-np.expm1(-8*alpha*h*h)/(4*h*h), abs=1e-9)
    assert g == pytest.approx(2*alpha, rel=1e-4)


def test_gaussian_richardson_exact_zero_step_metric():
    alpha=.7
    hs=(.008,.004)
    vals=tuple(-np.expm1(-4*alpha*h*h)/(4*h*h) for h in hs)
    assert zero_step_richardson(hs, vals) == pytest.approx(alpha, abs=1e-8)


@pytest.mark.parametrize('h,v', [((.1,.1),(1.,1.)), ((.01,.02),(1.,1.)),
                                  ((.01,.005),(1.,)), ((.01,.005),(1.,float('nan'))),
                                  ((.01,.005),(float('inf'),1.))])
def test_bad_richardson_inputs_fail_closed(h,v):
    with pytest.raises(ValueError): zero_step_richardson(h,v)


def request(role='neutral', **kw):
    d=dict(request_id=role, role=role, atoms=('Li','H'),
           charge=0 if role=='neutral' else -1,
           spin_2s=0 if role=='neutral' else 1,
           r_angstrom=1.6, basis_by_element={'Li':'sto-3g','H':'sto-3g'},
           state_id='ground', state_identity_reviewed=True,
           nuclear_masses_amu=(7.0143579,1.007276466621),
           nuclear_mass_source='nuclear masses smoke',
           scf_reference='RHF' if role=='neutral' else 'ROHF')
    d.update(kw)
    return HFDBOCRequest(**d)


def evaluate(role='neutral'):
    def fake_backend(req, xyz, settings):
        from types import SimpleNamespace
        occ=np.array([2.]) if req.scf_reference=='RHF' else np.array([1.])
        return np.asarray(xyz), SimpleNamespace(mo_coeff=np.ones((1,1)), mo_occ=occ, converged=True)
    def ao_overlap(a,b):
        diff=np.linalg.norm(np.asarray(a)-np.asarray(b))/0.529177210903
        return np.asarray([[exp(-.2*diff*diff)]])
    return run_hf_dboc_point(request(role), backend=fake_backend, ao_overlap=ao_overlap)


def test_neutral_anion_zero_step_pilot_and_distinct_uncertainty_flag():
    n,a=evaluate(),evaluate('anion')
    pair=assess_hf_dboc_pair(request(),request('anion'),n,a)
    assert pair.status is DBOCStatus.COMPLETE_REVIEW_REQUIRED
    assert pair.zero_step_ea_candidate_hartree is not None
    assert pair.zero_step_observed_shift_ev >= 0
    assert pair.correlated_remainder_bounded is False
    assert pair.nonadiabatic_remainder_bounded is False
    assert pair.is_production_correction is False
    assert abs(pair.zero_step_ea_candidate_hartree-pair.delta_ea_hartree_candidate) < 1e-7


def test_request_digest_prevents_reusing_result_for_other_geometry():
    n,a=evaluate(),evaluate('anion')
    with pytest.raises(ValueError, match='source request signature'):
        assess_hf_dboc_pair(request(r_angstrom=1.7),request('anion'),n,a)
    with pytest.raises(ValueError, match='source request signature'):
        assess_hf_dboc_pair(request(),request('anion',state_id='excited'),n,a)
    with pytest.raises(ValueError, match='source request signature'):
        assess_hf_dboc_pair(request(),request('anion'),replace(n,request_signature='bad'),a)
    with pytest.raises(ValueError, match='SCF settings signature'):
        assess_hf_dboc_pair(request(),request('anion'),replace(n,settings_signature='changed'),a)


def test_mass_scaling_of_fixed_metric_is_exact():
    metric=1.25
    m1,m2=1.,2.
    e1=metric/(2*m1*AMU_TO_ELECTRON_MASS)
    e2=metric/(2*m2*AMU_TO_ELECTRON_MASS)
    assert e1/e2 == pytest.approx(2.)


def test_result_does_not_certify_translation_gauge_or_full_dboc():
    n=evaluate()
    assert not n.correlated_dboc_included
    assert not n.nonadiabatic_off_diagonal_included
    assert not n.is_production_correction
    assert not n.scientific_uncertainty_bounded
