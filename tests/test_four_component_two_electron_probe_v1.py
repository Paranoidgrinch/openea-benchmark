from __future__ import annotations

from dataclasses import replace
from math import nan
import sys
from types import ModuleType, SimpleNamespace

import pytest

from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV
from openea_benchmark.adaptive.four_component_two_electron_probe import (
    FourComponentHamiltonian as H,
    FourComponentPointRequest, FourComponentPointResult,
    FourComponentSettings, FourComponentStatus as S,
    FourComponentPairResult,
    _pyscf_dhf_energy, run_four_component_point, assess_four_component_pair,
)


def request(role='neutral', **updates):
    kw = dict(request_id=f'FOURC_{role.upper()}', role=role, atoms=('Li', 'H'),
              charge=0 if role == 'neutral' else -1,
              spin_2s=0 if role == 'neutral' else 1,
              r_angstrom=1.6 if role == 'neutral' else 1.7,
              basis_by_element={'Li': 'sto-3g', 'H': 'sto-3g'},
              state_id=f'{role}_state', source_state_review_id=f'{role}_review',
              state_identity_reviewed=True)
    kw.update(updates)
    return FourComponentPointRequest(**kw)


def settings(**updates):
    return replace(FourComponentSettings(authorized=True), **updates)


def backend(req, opts, ham):
    shift = {H.DIRAC_COULOMB: 0., H.DIRAC_COULOMB_GAUNT: -0.002,
             H.DIRAC_COULOMB_BREIT: -0.003}[ham]
    return (-7. if req.role == 'neutral' else -6.) + shift * (1 if req.role == 'neutral' else 2), True, 'PySCF-2.14'


def good(role='neutral'):
    return run_four_component_point(request(role), settings(), backend=backend)


def test_real_api_directional_order_and_differences():
    seen = []
    def ordered(req, opts, ham):
        seen.append(ham)
        return backend(req, opts, ham)
    value = run_four_component_point(request(), settings(), backend=ordered)
    assert value.status is S.COMPLETE_REVIEW_REQUIRED
    assert seen == list(H)
    assert value.calculations_completed == 3
    assert value.gaunt_minus_coulomb_hartree == pytest.approx(-0.002)
    assert value.breit_minus_coulomb_hartree == pytest.approx(-0.003)
    assert value.breit_minus_gaunt_hartree == pytest.approx(-0.001)
    assert value.source_signature and value.pyscf_version == 'PySCF-2.14'
    assert tuple(value.to_dict()['energies_hartree']) == tuple(x.value for x in H)
    assert value.to_dict()['status'] == S.COMPLETE_REVIEW_REQUIRED.value


def test_breit_includes_gaunt_do_not_sum_terms():
    v = good()
    assert v.breit_minus_coulomb_hartree == pytest.approx(
        v.gaunt_minus_coulomb_hartree + v.breit_minus_gaunt_hartree)
    assert v.breit_minus_coulomb_hartree != pytest.approx(
        v.gaunt_minus_coulomb_hartree + v.breit_minus_coulomb_hartree)


def test_explicit_authorization_blocks_with_no_backend_execution():
    def forbidden(*args):
        pytest.fail('backend must not run')
    v = run_four_component_point(request(), FourComponentSettings(), backend=forbidden)
    assert v.status is S.POLICY_BLOCKED and not v.energies_hartree


def test_budget_blocks_without_partial_calculations():
    v = run_four_component_point(request(), settings(max_calculations=2), backend=lambda *_: pytest.fail('not allowed'))
    assert v.status is S.POLICY_BLOCKED and v.calculations_completed == 0


@pytest.mark.parametrize('update', [
    {'charge': -1}, {'role': 'invalid'}, {'r_angstrom': 0}, {'r_angstrom': nan},
    {'spin_2s': -1}, {'atoms': ('Li',)}, {'atoms': ('', 'H')},
    {'basis_by_element': {'Li': 'sto-3g'}}, {'basis_by_element': {'Li': '', 'H': 'sto-3g'}},
    {'request_id': ''}, {'source_state_review_id': ''}, {'state_identity_reviewed': False},
])
def test_request_rejects_invalid_state_and_provenance(update):
    with pytest.raises(ValueError):
        request(**update)


@pytest.mark.parametrize('update', [
    {'scf_conv_tol': 1e-4}, {'scf_conv_tol': nan}, {'scf_conv_tol': 0},
    {'max_cycle': 0}, {'max_memory_mb': 0}, {'max_calculations': -1},
    {'nuclear_model': 'FINITE'}, {'with_ssss': False},
])
def test_settings_reject_unsupported_convention(update):
    with pytest.raises(ValueError):
        settings(**update)


@pytest.mark.parametrize('break_case', [
    'unconverged', 'nonfinite', 'exception', 'missingversion', 'versiondrift',
])
def test_fail_closed_backend_result_and_no_partial_energy(break_case):
    def broken(req, opts, ham):
        if ham is H.DIRAC_COULOMB_GAUNT:
            if break_case == 'unconverged':
                return -7.5, False, 'test'
            if break_case == 'nonfinite':
                return nan, True, 'test'
            if break_case == 'exception':
                raise RuntimeError('backend error')
            if break_case == 'missingversion':
                return -7.5, True, ''
            if break_case == 'versiondrift':
                return -7.5, True, 'different'
        return -7.6, True, 'test'
    v = run_four_component_point(request(), settings(), backend=broken)
    assert v.status is S.ERROR and not v.energies_hartree
    assert v.gaunt_minus_coulomb_hartree is None


def test_pair_matches_correct_energy_signs():
    n, a = good(), good('anion')
    pair = assess_four_component_pair(request(), request('anion'), n, a)
    assert pair.status is S.COMPLETE_REVIEW_REQUIRED
    assert pair.gaunt_ea_sensitivity_ev == pytest.approx(0.002 * HARTREE_TO_EV)
    assert pair.breit_ea_sensitivity_ev == pytest.approx(0.003 * HARTREE_TO_EV)
    assert pair.gauge_increment_ea_sensitivity_ev == pytest.approx(0.001 * HARTREE_TO_EV)
    assert pair.to_dict()['status'] == S.COMPLETE_REVIEW_REQUIRED.value
    assert not pair.production_correction_validated
    assert not pair.scalar_two_electron_x2c_remainder_computed
    assert not pair.independent_soc_correction_computed


@pytest.mark.parametrize('bad', [
    {'atoms': ('H', 'Li')}, {'basis_by_element': {'H': '6-31g', 'Li': 'sto-3g'}},
    {'source_state_review_id': 'neutral_review'}, {'request_id': 'FOURC_NEUTRAL'},
])
def test_pair_rejects_mismatched_neutral_anion_provenance(bad):
    with pytest.raises(ValueError):
        assess_four_component_pair(request(), request('anion', **bad), good(), good('anion'))


def test_pair_rejects_swapped_results():
    with pytest.raises(ValueError):
        assess_four_component_pair(request(), request('anion'), good('anion'), good())


def test_pair_fails_closed_when_one_point_unconverged():
    n, a = good(), run_four_component_point(request('anion'), FourComponentSettings())
    pair = assess_four_component_pair(request(), request('anion'), n, a)
    assert pair.status is S.ERROR
    assert pair.breit_ea_sensitivity_ev is None


def test_pair_fails_closed_on_backend_version_drift():
    n, a = good(), good('anion')
    pair = assess_four_component_pair(request(), request('anion'), n,
                                     replace(a, pyscf_version='different version'))
    assert pair.status is S.ERROR


@pytest.mark.parametrize('flag', [
    'two_electron_x2c_picture_change_computed',
    'spin_free_scalar_remainder_computed',
    'independent_soc_correction_computed',
    'correlated_relativistic_correction_computed',
    'production_correction_validated',
    'uncertainty_bounded',
])
def test_point_can_never_claim_missing_relativistic_physics(flag):
    with pytest.raises(ValueError):
        replace(good(), **{flag: True})


@pytest.mark.parametrize('flag', [
    'production_correction_validated', 'uncertainty_bounded',
    'independent_soc_correction_computed',
    'scalar_two_electron_x2c_remainder_computed',
])
def test_pair_never_upgrades_sensitivity_into_ea_correction(flag):
    p = assess_four_component_pair(request(), request('anion'), good(), good('anion'))
    with pytest.raises(ValueError):
        replace(p, **{flag: True})


def test_completed_result_cannot_drop_energy_or_corrupt_difference():
    v = good()
    with pytest.raises(ValueError):
        replace(v, energies_hartree=v.energies_hartree[:-1])
    with pytest.raises(ValueError):
        replace(v, gaunt_minus_coulomb_hartree=100)
    with pytest.raises(ValueError):
        replace(v, calculations_completed=2)


def test_failure_result_never_carries_partially_computed_candidate():
    with pytest.raises(ValueError):
        replace(good(), status=S.ERROR)


def test_pyscf_backend_flags_are_independent_and_breit_contains_gaunt(monkeypatch):
    records = []
    class FakeDHF:
        def __init__(self, mol):
            self.converged = True
            self.mol = mol
        def kernel(self):
            records.append((self.with_gaunt, self.with_breit, self.with_ssss,
                            self.mol.charge, self.mol.spin, self.mol.symmetry,
                            self.mol.max_memory, self.max_memory))
            return -7.0 - 0.001 * self.with_gaunt - 0.002 * self.with_breit
    fake = ModuleType('pyscf')
    fake.__version__ = '2.14.0-test'
    fake.gto = SimpleNamespace(M=lambda **kwargs: SimpleNamespace(**kwargs))
    fake.scf = SimpleNamespace(DHF=FakeDHF)
    monkeypatch.setitem(sys.modules, 'pyscf', fake)
    opts = settings(max_memory_mb=3333)
    energies = [_pyscf_dhf_energy(request('anion'), opts, ham)[0] for ham in H]
    assert energies == pytest.approx([-7.0, -7.001, -7.002])
    assert records == [
        (False, False, True, -1, 1, False, 3333, 3333),
        (True, False, True, -1, 1, False, 3333, 3333),
        (False, True, True, -1, 1, False, 3333, 3333),
    ]
