from openea_benchmark.adaptive.method_basis_advisor import *
from openea_benchmark.adaptive.adaptive_cardinal_runner import *
from openea_benchmark.attachment.basis_convergence import *

SETTINGS=BasisConvergenceSettings(0.02,0.75,0.01,0.60,False)
PROFILE=ChemicalIntelligenceProfile(('O','H'),False,False,8)
EVIDENCE=MethodBasisEvidence(ReferenceCharacter.SINGLE_REFERENCE,AttachmentCharacter.VALENCE_BOUND,FunctionalSensitivity.STABLE,CorrectionNeed.NOT_REQUIRED,CorrectionNeed.NOT_REQUIRED,None,None)

def point(x,v,hw=0.0002):
    return ElectronicEABasisPoint(correlation_consistent_basis_name(x,augmentation_level=1),x,1,'CCSD(T)|TEST',EAIntervalEV(v-hw,v,v+hw))

def test_oh_like_series_runs_5z_and_clears():
    vals={2:1.6295,3:1.7419,4:1.7964,5:1.8120}; calls=[]
    def f(x): calls.append(x); return point(x,vals[x])
    r=run_adaptive_cardinal_series(initial_cardinals=(2,3,4),maximum_cardinal=6,augmentation_level=1,convergence_settings=SETTINGS,chemical_profile=PROFILE,advisor_evidence_template=EVIDENCE,evaluate_point=f)
    assert r.status is AdaptiveCardinalStatus.CARDINAL_CLEARED
    assert calls==[2,3,4,5]
    assert r.iterations[0].requested_next_cardinal==5

def test_can_progress_to_6z():
    vals={2:1.60,3:1.70,4:1.75,5:1.78,6:1.79}; calls=[]
    def f(x): calls.append(x); return point(x,vals[x])
    r=run_adaptive_cardinal_series(initial_cardinals=(2,3,4),maximum_cardinal=6,augmentation_level=1,convergence_settings=SETTINGS,chemical_profile=PROFILE,advisor_evidence_template=EVIDENCE,evaluate_point=f)
    assert calls==[2,3,4,5,6]
    assert r.status is AdaptiveCardinalStatus.CARDINAL_CLEARED

def test_limit_is_explicit():
    vals={2:1.60,3:1.70,4:1.75}
    r=run_adaptive_cardinal_series(initial_cardinals=(2,3,4),maximum_cardinal=4,augmentation_level=1,convergence_settings=SETTINGS,chemical_profile=PROFILE,advisor_evidence_template=EVIDENCE,evaluate_point=lambda x:point(x,vals[x]))
    assert r.status is AdaptiveCardinalStatus.CARDINAL_LIMIT_REACHED
    assert r.final_advisor_plan.requested_next_cardinal==5

def test_execution_failure_is_preserved():
    def f(x):
        if x==5: raise RuntimeError('backend failure')
        return point(x,{2:1.60,3:1.70,4:1.75}[x])
    r=run_adaptive_cardinal_series(initial_cardinals=(2,3,4),maximum_cardinal=6,augmentation_level=1,convergence_settings=SETTINGS,chemical_profile=PROFILE,advisor_evidence_template=EVIDENCE,evaluate_point=f)
    assert r.status is AdaptiveCardinalStatus.EXECUTION_BLOCKED
    assert r.execution_error_type=='RuntimeError'

def test_wrong_evaluator_cardinal_blocks():
    r=run_adaptive_cardinal_series(initial_cardinals=(2,3,4),maximum_cardinal=6,augmentation_level=1,convergence_settings=SETTINGS,chemical_profile=PROFILE,advisor_evidence_template=EVIDENCE,evaluate_point=lambda x:point(x+1,1.0))
    assert r.status is AdaptiveCardinalStatus.EXECUTION_BLOCKED

def test_basis_names():
    assert correlation_consistent_basis_name(2,augmentation_level=1)=='aug-cc-pvdz'
    assert correlation_consistent_basis_name(5,augmentation_level=1)=='aug-cc-pv5z'

def test_never_promotes_production():
    vals={2:1.5,3:1.6,4:1.61}
    r=run_adaptive_cardinal_series(initial_cardinals=(2,3,4),maximum_cardinal=6,augmentation_level=1,convergence_settings=SETTINGS,chemical_profile=PROFILE,advisor_evidence_template=EVIDENCE,evaluate_point=lambda x:point(x,vals[x]))
    assert not r.is_production_ea and not r.authorizes_pruning
