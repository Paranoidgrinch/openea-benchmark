#!/usr/bin/env python3
"""Adaptive OH core-valence correction using aug-cc-pwCVXZ.

For each cardinal basis and each molecular role, the same converged SCF
checkpoint is reused for two CCSD(T) calculations:
  * frozen-core
  * all-electron

The additive core-valence correction is the difference between the two EAs.
TZ and QZ are run first.  5Z is requested only if the correction is not yet
converged.

No PECs or atomic fragments are recomputed.
"""
from __future__ import annotations
import argparse, json, os, pickle
from datetime import datetime, timezone
from pathlib import Path

from pyscf import gto, scf

from openea_benchmark.adaptive.run_feedback import ProgressReporter
from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus, Stage3ExecutionSettings, run_stage3_point,
)
from openea_benchmark.attachment.cbs_component_evidence import (
    load_reference_geometries, make_cbs_single_point_request,
)
from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV
from openea_benchmark.attachment.core_valence_correction import (
    assess_core_valence, cv_point_from_results,
)

BASIS_BY_X={
    3:"aug-cc-pwcvtz",
    4:"aug-cc-pwcvqz",
    5:"aug-cc-pwcv5z",
}

def default_run_id():
    return os.environ.get("OPENEA_RUN_ID") or (
        "oh_core_valence_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

def make_source(path, role, basis, r, max_memory):
    if path.is_file():
        return
    charge=0 if role=="neutral" else -1
    spin=1 if role=="neutral" else 0
    mol=gto.M(atom=f"O 0 0 0; H 0 0 {r}",basis=basis,charge=charge,spin=spin,
              unit="Angstrom",symmetry=False,verbose=0,max_memory=max_memory)
    mf=scf.ROHF(mol) if spin else scf.RHF(mol)
    mf.chkfile=str(path); mf.conv_tol=1e-9; mf.max_cycle=100
    mf.kernel()
    if not mf.converged:
        m2=mf.newton(); m2.chkfile=str(path); m2.conv_tol=1e-9; m2.max_cycle=100
        m2.kernel(mo_coeff=mf.mo_coeff,mo_occ=mf.mo_occ)
        if not m2.converged:
            raise RuntimeError(f"SCF failed {role} {basis}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference-run-id",default="oh_diffuse_x5_resume_20261003")
    p.add_argument("--reference-basis",default="d-aug-cc-pv5z")
    p.add_argument("--run-root",default="runs")
    p.add_argument("--run-id",default=default_run_id())
    p.add_argument("--max-memory-mb",type=int,default=12000)
    p.add_argument("--target-change-ev",type=float,default=0.002)
    p.add_argument("--max-cardinal",type=int,default=5)
    args=p.parse_args()

    root=Path(args.run_root); run=root/args.run_id; run.mkdir(parents=True,exist_ok=True)
    points_dir=run/"points"; points_dir.mkdir(exist_ok=True)
    src_dir=run/"source_checkpoints"; src_dir.mkdir(exist_ok=True)
    art_dir=run/"hf_artifacts"; art_dir.mkdir(exist_ok=True)
    reporter=ProgressReporter(run_id=args.run_id,workflow="OH_CORE_VALENCE_CORRECTION",
                              status_file=run/"status.json")

    ng,ag=load_reference_geometries(
        run_dir=root/args.reference_run_id,reference_basis=args.reference_basis
    )
    geoms={"neutral":ng,"anion":ag}
    cv_points=[]
    completed=["REFERENCE_GEOMETRIES_LOADED"]

    for x in range(3,args.max_cardinal+1):
        basis=BASIS_BY_X[x]
        # preflight basis availability before any CC work for this cardinal
        for atom in ("O","H"):
            gto.basis.load(basis,atom)

        totals={}
        for role in ("neutral","anion"):
            src=src_dir/f"{role}__{basis}.chk"
            make_source(src,role,basis,geoms[role].r_angstrom,args.max_memory_mb)

            for frozen in (True,False):
                space="FC" if frozen else "AE"
                label=f"X{x}:{role}:{space}"
                pkl=points_dir/f"{role}__{basis}__{space}.pkl"
                js=points_dir/f"{role}__{basis}__{space}.json"

                result=None
                if pkl.is_file():
                    with pkl.open("rb") as fh: result=pickle.load(fh)
                    if not (
                        getattr(result,"status",None) is PointExecutionStatus.COMPLETED
                        and result.basis==basis
                        and abs(float(result.r_angstrom)-geoms[role].r_angstrom)<1e-10
                    ):
                        pkl.unlink(missing_ok=True); js.unlink(missing_ok=True); result=None

                if result is None:
                    reporter.update(
                        current_step=f"CALCULATE_{label}",
                        completed_steps=completed,
                        next_steps=["SAVE_POINT","CONTINUE_CORE_VALENCE_SERIES"],
                        details={"basis":basis,"role":role,
                                 "correlation_space":"FROZEN_CORE" if frozen else "ALL_ELECTRON"}
                    )
                    request=make_cbs_single_point_request(
                        role=role,basis=basis,cardinal_number=x,
                        r_angstrom=geoms[role].r_angstrom,
                        source_checkpoint_path=src,
                    )
                    settings=Stage3ExecutionSettings(
                        scf_conv_tol=1e-9,scf_conv_tol_grad=1e-6,cc_conv_tol=1e-8,
                        max_memory_mb=args.max_memory_mb,verbose=0,
                        frozen_core=frozen,artifact_dir=str(art_dir),
                    )
                    result=run_stage3_point(request,settings=settings)
                    js.write_text(json.dumps(result.to_dict(),indent=2,sort_keys=True)+"\n")
                    if result.status is not PointExecutionStatus.COMPLETED:
                        reporter.blocked(
                            current_step=f"BLOCKED_{label}",completed_steps=completed,
                            next_steps=["INSPECT_FAILURE","RERUN_SAME_RUN_ID"],
                            details={"status":result.status.value,
                                     "error_type":result.error_type,
                                     "error_message":result.error_message}
                        )
                        raise SystemExit(2)
                    with pkl.open("wb") as fh:
                        pickle.dump(result,fh,pickle.HIGHEST_PROTOCOL)
                else:
                    reporter.update(
                        current_step=f"REUSE_{label}",completed_steps=completed+[label],
                        next_steps=["CONTINUE_CORE_VALENCE_SERIES"],
                        details={"source":"CHECKPOINT"}
                    )
                completed.append(label)
                totals[(role,space)]=float(result.ccsd_t_total_hartree)

        cvp=cv_point_from_results(
            cardinal=x,basis=basis,
            ae_neutral_h=totals[("neutral","AE")],
            ae_anion_h=totals[("anion","AE")],
            fc_neutral_h=totals[("neutral","FC")],
            fc_anion_h=totals[("anion","FC")],
            hartree_to_ev=HARTREE_TO_EV,
        )
        cv_points.append(cvp)
        assessment=assess_core_valence(
            tuple(cv_points),target_change_ev=args.target_change_ev,
            max_cardinal=args.max_cardinal,
        )
        reporter.update(
            current_step=f"ASSESS_CORE_VALENCE_X{x}",
            completed_steps=completed,
            next_steps=[assessment.action],
            details=assessment.to_dict(),
        )
        if assessment.status.value=="CLEARED":
            break
        if assessment.action=="MAX_CARDINAL_REACHED_UNRESOLVED":
            break

    assessment=assess_core_valence(
        tuple(cv_points),target_change_ev=args.target_change_ev,
        max_cardinal=args.max_cardinal,
    )
    payload=assessment.to_dict()
    payload.update({
        "run_id":args.run_id,
        "reference_run_id":args.reference_run_id,
        "basis_family":"aug-cc-pwCVXZ",
        "definition":"Delta_CV = EA_all_electron - EA_frozen_core in same core-valence basis",
        "target_change_ev":args.target_change_ev,
        "is_production_ea":False,
        "next_after_clear":[
            "ADD_DELTA_CV_TO_FROZEN_CORE_CBS_BASELINE",
            "ASSESS_SCALAR_RELATIVITY",
            "ASSESS_POST_CCSD_T",
        ] if assessment.status.value=="CLEARED" else [],
    })
    result_path=run/"result.json"
    result_path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")

    if assessment.status.value=="CLEARED":
        reporter.completed(
            current_step="CORE_VALENCE_CORRECTION_CLEARED",
            completed_steps=completed+["CORE_VALENCE_CONVERGENCE_CLEARED"],
            next_steps=payload["next_after_clear"],
            details={"central_correction_ev":assessment.central_correction_ev,
                     "convergence_bound_ev":assessment.convergence_bound_ev,
                     "result_file":str(result_path)}
        )
    else:
        reporter.blocked(
            current_step="CORE_VALENCE_UNRESOLVED",
            completed_steps=completed,
            next_steps=[assessment.action],
            details={"result_file":str(result_path)}
        )
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
