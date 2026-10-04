#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

from openea_benchmark.adaptive.adaptive_diffuse_runner import correlation_consistent_diffuse_basis_name, run_adaptive_diffuse_series
from openea_benchmark.adaptive.basis_point_checkpoint import load_basis_checkpoint, save_basis_checkpoint
from openea_benchmark.adaptive.run_feedback import ProgressReporter
from openea_benchmark.attachment.basis_convergence import BasisConvergenceSettings, EAIntervalEV, ElectronicEABasisPoint

METHOD_SIGNATURE='CCSD(T)|adaptive-equilibrium-PEC|same-basis-molecule-and-fragments|electronic-only'

def _default_run_id():
    return os.environ.get('OPENEA_RUN_ID') or f"oh_diffuse_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

def _preflight(cardinal:int,max_aug:int):
    if max_aug<2: return True,None
    if importlib.util.find_spec('basis_set_exchange') is None:
        return False,'basis-set-exchange is not installed; install it before a run that may request d-aug'
    try:
        import basis_set_exchange as bse
        basis=correlation_consistent_diffuse_basis_name(cardinal,augmentation_level=2)
        bse.api.get_basis(basis,elements=[1,8])
    except Exception as exc:
        return False,f'Basis Set Exchange could not resolve {basis} for H/O: {type(exc).__name__}: {exc}'
    return True,None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cardinal',type=int,default=5); p.add_argument('--max-augmentation',type=int,default=2); p.add_argument('--max-rounds',type=int,default=8)
    p.add_argument('--target-half-width-ev',type=float,default=0.05); p.add_argument('--neutral-center',type=float,default=0.97); p.add_argument('--anion-center',type=float,default=0.97); p.add_argument('--half-width',type=float,default=0.02)
    p.add_argument('--diffuse-increment-target-ev',type=float,default=0.01); p.add_argument('--diffuse-contraction-ratio-max',type=float,default=0.60); p.add_argument('--force-double-augmentation',action='store_true')
    p.add_argument('--run-id',default=_default_run_id()); p.add_argument('--run-root',default=os.environ.get('OPENEA_RUN_ROOT','runs')); p.add_argument('--checkpoint-source',default='LIVE_OPENEA_CALCULATION')
    args=p.parse_args()

    run_dir=Path(args.run_root)/args.run_id; run_dir.mkdir(parents=True,exist_ok=True)
    reporter=ProgressReporter(run_id=args.run_id,workflow='OH_ADAPTIVE_DIFFUSE_VALIDATION',status_file=run_dir/'status.json')
    checkpoint_path=run_dir/'basis_points.json'; result_path=run_dir/'result.json'
    signature=json.dumps({'workflow':'OH_ADAPTIVE_DIFFUSE_VALIDATION_V2','cardinal':args.cardinal,'max_rounds':args.max_rounds,'target_half_width_ev':args.target_half_width_ev,'neutral_center':args.neutral_center,'anion_center':args.anion_center,'half_width':args.half_width,'method_signature':METHOD_SIGNATURE},sort_keys=True,separators=(',',':'))

    reporter.update(current_step='PREFLIGHT',completed_steps=[],next_steps=['LOAD_CHECKPOINT','EVALUATE_NONAUG','EVALUATE_AUG','ASSESS_DIFFUSE','IF_REQUIRED_EVALUATE_DAUG'],details={'cardinal':args.cardinal,'max_augmentation':args.max_augmentation,'run_dir':str(run_dir)})
    ok,msg=_preflight(args.cardinal,args.max_augmentation)
    if not ok:
        reporter.blocked(current_step='PREFLIGHT_BASIS_BACKEND',completed_steps=[],next_steps=['INSTALL_OR_FIX_BASIS_SET_EXCHANGE','RERUN_WITH_SAME_RUN_ID'],details={'reason':msg})
        out={'species':'OH/OH-','validation_only':True,'status':'DEPENDENCY_BLOCKED','execution_error_type':'BasisBackendUnavailable','execution_error_message':msg,'run_id':args.run_id,'run_dir':str(run_dir)}
        result_path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True)); raise SystemExit(2)

    cp=load_basis_checkpoint(checkpoint_path,expected_signature=signature); cached={} if cp is None else {x.augmentation_level:x for x in cp.points}; live=dict(cached)
    reporter.update(current_step='CHECKPOINT_READY',completed_steps=['PREFLIGHT','LOAD_CHECKPOINT'],next_steps=['EVALUATE_OR_REUSE_BASIS_POINTS','ASSESS_DIFFUSE'],details={'cached_levels':sorted(cached),'checkpoint':str(checkpoint_path)})
    smoke=Path(__file__).with_name('attachment_oh_ea_smoke.py')

    def evaluate(level:int):
        basis=correlation_consistent_diffuse_basis_name(args.cardinal,augmentation_level=level)
        if level in live:
            point=live[level]
            if point.basis_name!=basis: raise RuntimeError(f'checkpoint basis mismatch at level {level}: {point.basis_name} != {basis}')
            reporter.update(current_step=f'REUSE_{basis}',completed_steps=['PREFLIGHT','LOAD_CHECKPOINT',*[f'BASIS_LEVEL_{x}_AVAILABLE' for x in sorted(live)]],next_steps=['ASSESS_DIFFUSE'],details={'basis':basis,'source':'CHECKPOINT'})
            return point
        reporter.update(current_step=f'CALCULATE_{basis}',completed_steps=['PREFLIGHT','LOAD_CHECKPOINT',*[f'BASIS_LEVEL_{x}_AVAILABLE' for x in sorted(live)]],next_steps=[f'COMPLETE_{basis}','SAVE_CHECKPOINT','ASSESS_DIFFUSE'],details={'basis':basis,'augmentation_level':level})
        cmd=[sys.executable,str(smoke),'--basis',basis,'--max-rounds',str(args.max_rounds),'--target-half-width-ev',str(args.target_half_width_ev),'--neutral-center',str(args.neutral_center),'--anion-center',str(args.anion_center),'--half-width',str(args.half_width)]
        proc=subprocess.run(cmd,check=False,text=True,stdout=subprocess.PIPE,stderr=None)
        if proc.returncode!=0: raise RuntimeError(f'{basis}: electronic-EA smoke exited with code {proc.returncode}')
        payload=json.loads(proc.stdout); ea=payload['electronic_ea']
        if ea['decision_status']!='BOUND': raise RuntimeError(f"{basis}: electronic EA is {ea['decision_status']}")
        point=ElectronicEABasisPoint(basis_name=basis,cardinal_number=args.cardinal,augmentation_level=level,method_signature=METHOD_SIGNATURE,ea=EAIntervalEV(float(ea['lower_ev']),float(ea['central_ev']),float(ea['upper_ev'])),decision_status='BOUND',evidence_quality='CONVERGENCE_ESTIMATED',is_production_ea=False)
        live[level]=point
        save_basis_checkpoint(checkpoint_path,signature=signature,source=args.checkpoint_source,points=tuple(live[x] for x in sorted(live)))
        reporter.update(current_step=f'COMPLETED_{basis}',completed_steps=['PREFLIGHT','LOAD_CHECKPOINT',*[f'BASIS_LEVEL_{x}_AVAILABLE' for x in sorted(live)]],next_steps=['ASSESS_DIFFUSE'],details={'basis':basis,'ea_central_ev':point.ea.central_ev,'checkpoint_saved':str(checkpoint_path)})
        return point

    reporter.update(current_step='RUN_ADAPTIVE_DIFFUSE_POLICY',completed_steps=['PREFLIGHT','LOAD_CHECKPOINT'],next_steps=['cc-pVXZ','aug-cc-pVXZ','ASSESS','d-aug-if-required'])
    result=run_adaptive_diffuse_series(cardinal_number=args.cardinal,initial_augmentation_levels=(0,1),maximum_augmentation_level=args.max_augmentation,convergence_settings=BasisConvergenceSettings(cardinal_increment_target_ev=0.02,cardinal_contraction_ratio_max=0.75,diffuse_increment_target_ev=args.diffuse_increment_target_ev,diffuse_contraction_ratio_max=args.diffuse_contraction_ratio_max,force_double_augmentation=args.force_double_augmentation),evaluate_point=evaluate)
    out={'species':'OH/OH-','validation_only':True,'run_id':args.run_id,'run_dir':str(run_dir),'cardinal_number':args.cardinal,'status':result.status.value,'evaluated_points':[{'basis':x.basis_name,'augmentation_level':x.augmentation_level,'ea_lower_ev':x.ea.lower_ev,'ea_central_ev':x.ea.central_ev,'ea_upper_ev':x.ea.upper_ev} for x in result.points],'iterations':[{'iteration_index':it.iteration_index,'evaluated_augmentation_levels':list(it.evaluated_augmentation_levels),'diffuse_status':it.assessment.status.value,'diffuse_action':it.assessment.action.value,'latest_increment_bound_ev':it.assessment.latest_increment_bound_ev,'contraction_ratio':it.assessment.contraction_ratio,'requested_next_augmentation_level':it.requested_next_augmentation_level} for it in result.iterations],'final_assessment':None if result.final_assessment is None else {'status':result.final_assessment.status.value,'action':result.final_assessment.action.value,'highest_augmentation_level':result.final_assessment.highest_augmentation_level,'latest_increment_bound_ev':result.final_assessment.latest_increment_bound_ev,'contraction_ratio':result.final_assessment.contraction_ratio,'residual_estimate_ev':result.final_assessment.residual_estimate_ev,'evidence':list(result.final_assessment.evidence)},'execution_error_type':result.execution_error_type,'execution_error_message':result.execution_error_message,'is_production_ea':result.is_production_ea,'authorizes_pruning':result.authorizes_pruning}
    result_path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    if result.status.value=='DIFFUSE_CLEARED':
        reporter.completed(current_step='DIFFUSE_AXIS_CLEARED',completed_steps=['PREFLIGHT','BASIS_POINTS_PERSISTED','DIFFUSE_CONVERGENCE_CLEARED'],next_steps=['RETURN_TO_METHOD_BASIS_ADVISOR','IF_NO_OTHER_BASIS_BLOCKERS_PROCEED_COMPONENT_RESOLVED_CBS'],details={'result_file':str(result_path),'checkpoint_file':str(checkpoint_path)})
    else:
        reporter.blocked(current_step=f'DIFFUSE_{result.status.value}',completed_steps=['PREFLIGHT','BASIS_POINTS_PERSISTED'],next_steps=['INSPECT_EXECUTION_OR_POLICY_BLOCKER','RERUN_WITH_SAME_RUN_ID_TO_REUSE_COMPLETED_POINTS'],details={'error_type':result.execution_error_type,'error_message':result.execution_error_message,'result_file':str(result_path)})
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=='__main__': main()
