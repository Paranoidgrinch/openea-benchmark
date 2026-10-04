import pytest
from openea_benchmark.attachment.fragment_execution import AtomicFragmentRequest, AtomicFragmentResult, AtomicFragmentSCFRetrySettings, FragmentExecutionStatus, run_atomic_fragment

def req(): return AtomicFragmentRequest('O_minus_2P','O',-1,1,'d-aug-cc-pv5z',state_label='2P')
def completed(r):
    return AtomicFragmentResult(fragment_id=r.fragment_id,status=FragmentExecutionStatus.COMPLETED,element=r.element,charge=r.charge,spin_2s=r.spin_2s,basis=r.basis,state_label=r.state_label,scf_reference='ROHF',cc_reference='SEMICANONICAL_UHF_FROM_ROHF',scf_converged=True,scf_energy_hartree=-75.0,internal_stable=True,external_stable=None,external_stability_available=False,ccsd_converged=True,ccsd_total_hartree=-75.1,triples_correction_hartree=-.01,ccsd_t_total_hartree=-75.11,s2=.75,multiplicity=2.0,pyscf_version='test',error_type=None,error_message=None,scf_solver_path=('PRIMARY_DIIS','CIAH_NEWTON_FROM_PRIMARY_ORBITALS'),scf_primary_converged=False,scf_newton_attempted=True,scf_newton_converged=True)

def test_retry_defaults():
    x=AtomicFragmentSCFRetrySettings(); assert x.enable_newton_retry and x.newton_cycle_multiplier==2 and x.newton_min_cycles==200

def test_retry_validation():
    with pytest.raises(ValueError): AtomicFragmentSCFRetrySettings(newton_cycle_multiplier=0)

def test_metadata_auditable():
    x=completed(req()).to_dict(); assert x['scf_primary_converged'] is False; assert x['scf_newton_attempted'] is True; assert x['scf_newton_converged'] is True

def test_injected_runner_contract_preserved():
    r=req(); out=run_atomic_fragment(r,runner=lambda item,settings: completed(item)); assert out.status is FragmentExecutionStatus.COMPLETED

def test_no_production_promotion():
    out=completed(req()); assert not out.is_production_threshold and not out.authorizes_pruning
