#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from openea_benchmark.adaptive.adaptive_cardinal_runner import correlation_consistent_basis_name,run_adaptive_cardinal_series
from openea_benchmark.adaptive.method_basis_advisor import *
from openea_benchmark.attachment.basis_convergence import *

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--max-rounds',type=int,default=8); ap.add_argument('--max-cardinal',type=int,default=5); ap.add_argument('--target-half-width-ev',type=float,default=0.05); ap.add_argument('--cardinal-increment-target-ev',type=float,default=0.02); ap.add_argument('--cardinal-contraction-ratio-max',type=float,default=0.75); args=ap.parse_args()
    smoke=Path(__file__).with_name('attachment_oh_ea_smoke.py')
    def evaluate(x):
        basis=correlation_consistent_basis_name(x,augmentation_level=1)
        p=subprocess.run([sys.executable,str(smoke),'--basis',basis,'--max-rounds',str(args.max_rounds),'--target-half-width-ev',str(args.target_half_width_ev)],text=True,stdout=subprocess.PIPE,stderr=None)
        if p.returncode: raise RuntimeError(f'{basis}: smoke exited {p.returncode}')
        data=json.loads(p.stdout); ea=data['electronic_ea']
        if ea['decision_status']!='BOUND': raise RuntimeError(f"{basis}: EA {ea['decision_status']}")
        return ElectronicEABasisPoint(basis,x,1,'CCSD(T)|adaptive-equilibrium-PEC|same-basis-molecule-and-fragments|electronic-only',EAIntervalEV(float(ea['lower_ev']),float(ea['central_ev']),float(ea['upper_ev'])),decision_status='BOUND',evidence_quality='CONVERGENCE_ESTIMATED',is_production_ea=False)
    settings=BasisConvergenceSettings(args.cardinal_increment_target_ev,args.cardinal_contraction_ratio_max,0.01,0.60,False)
    profile=ChemicalIntelligenceProfile(('O','H'),False,False,8)
    ev=MethodBasisEvidence(ReferenceCharacter.SINGLE_REFERENCE,AttachmentCharacter.VALENCE_BOUND,FunctionalSensitivity.STABLE,CorrectionNeed.NOT_REQUIRED,CorrectionNeed.NOT_REQUIRED,None,None)
    r=run_adaptive_cardinal_series(initial_cardinals=(2,3,4),maximum_cardinal=args.max_cardinal,augmentation_level=1,convergence_settings=settings,chemical_profile=profile,advisor_evidence_template=ev,evaluate_point=evaluate)
    out={'species':'OH/OH-','validation_only':True,'status':r.status.value,'evaluated_points':[{'basis':p.basis_name,'cardinal_number':p.cardinal_number,'ea_lower_ev':p.ea.lower_ev,'ea_central_ev':p.ea.central_ev,'ea_upper_ev':p.ea.upper_ev} for p in r.points],'iterations':[{'iteration_index':i.iteration_index,'evaluated_cardinals':list(i.evaluated_cardinals),'cardinal_status':i.assessment.status.value,'cardinal_action':i.assessment.action.value,'latest_increment_bound_ev':i.assessment.latest_increment_bound_ev,'contraction_ratio':i.assessment.contraction_ratio,'requested_next_cardinal':i.requested_next_cardinal,'advisor_actions':[a.value for a in i.advisor_plan.actions]} for i in r.iterations],'final_assessment':None if r.final_assessment is None else {'status':r.final_assessment.status.value,'action':r.final_assessment.action.value,'highest_cardinal':r.final_assessment.highest_cardinal,'latest_increment_bound_ev':r.final_assessment.latest_increment_bound_ev,'contraction_ratio':r.final_assessment.contraction_ratio,'residual_estimate_ev':r.final_assessment.residual_estimate_ev,'evidence':list(r.final_assessment.evidence)},'final_advisor':None if r.final_advisor_plan is None else {'requested_next_cardinal':r.final_advisor_plan.requested_next_cardinal,'actions':[a.value for a in r.final_advisor_plan.actions],'high_accuracy_branch':r.final_advisor_plan.high_accuracy_branch.value,'basis_family_policy':r.final_advisor_plan.basis_family_policy.value},'execution_error_type':r.execution_error_type,'execution_error_message':r.execution_error_message,'is_production_ea':r.is_production_ea,'authorizes_pruning':r.authorizes_pruning}
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
