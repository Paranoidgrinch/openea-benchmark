#!/usr/bin/env python3
"""Manual end-to-end smoke test for the autonomous Stage-3 refinement loop.

The numerical thresholds in this script are validation-only smoke-test policy.
They are deliberately not OpenEA production defaults.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from pyscf import gto, scf

from openea_benchmark.branch_continuity import BranchThresholds
from openea_benchmark.state_identity import IdentityThresholds
from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3ExecutionSettings,
    run_stage3_point,
)
from openea_benchmark.adaptive.stage3_loop import run_stage3_refinement_loop
from openea_benchmark.adaptive.stage3_refinement import Stage3RefinementSettings


IDENTITY_THRESHOLDS = IdentityThresholds(
    same_energy_mev=0.5,
    same_delta_s2=1.0e-5,
    same_total_spectrum_max=1.0e-5,
    same_spin_spectrum_max=1.0e-5,
    same_total_density_rel_fro=1.0e-5,
    same_spin_density_rel_fro=1.0e-5,
    distinct_energy_mev=10.0,
    distinct_delta_s2=0.10,
    distinct_total_spectrum_max=1.0e-2,
    distinct_spin_spectrum_max=1.0e-2,
    distinct_total_density_rel_fro=1.0e-2,
    distinct_spin_density_rel_fro=1.0e-2,
)

BRANCH_THRESHOLDS = BranchThresholds(
    continuous_occ_min=0.95,
    discontinuous_occ_min=0.50,
    continuous_delta_s2=0.02,
    discontinuous_delta_s2=0.20,
    max_step_angstrom=0.05,
)

REFINEMENT_SETTINGS = Stage3RefinementSettings(
    extension_step_angstrom=0.02,
    target_bracket_width_angstrom=0.021,
    minimum_new_point_separation_angstrom=1.0e-7,
    energy_tie_tolerance_hartree=1.0e-8,
)


def case_spec(case: str):
    if case == "h2":
        return ("H", "H"), 0.74, 0, 0, 0.04
    if case == "oh":
        return ("O", "H"), 0.97, 0, 1, 0.02
    raise ValueError(case)


def source_checkpoint(case: str, path: Path):
    atoms, center, charge, spin, half_width = case_spec(case)
    mol = gto.M(
        atom=f"{atoms[0]} 0 0 0; {atoms[1]} 0 0 {center}",
        basis="sto-3g",
        charge=charge,
        spin=spin,
        verbose=0,
    )
    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    mf.chkfile = str(path)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("source SCF did not converge")
    return atoms, center, charge, spin, half_width


def make_request(*, case, atoms, center, charge, spin, source, r, init, grid_index, root):
    return Stage3ExecutionRequest(
        request_id=f"loop_smoke__{case}__init{init:02d}_{root}__g{grid_index:03d}",
        job_id=f"loop_smoke__{case}__job",
        system=case.upper(),
        atoms=atoms,
        charge=charge,
        spin_2s=spin,
        component_id="loop_smoke_component",
        r_angstrom=r,
        basis="sto-3g",
        methods=("CCSD", "CCSD(T)"),
        requested_reference="ROHF",
        scf_reference="RHF" if spin == 0 else "ROHF",
        source_link_status="MULTIPLE_DFT_INITIALIZATIONS",
        source_root_id=root,
        source_checkpoint_path=str(source),
        source_origin_guess="smoke",
        grid_index=grid_index,
        initialization_index=init,
        dft_center_r_angstrom=center,
        dft_center_energy_hartree=0.0,
        requires_independent_state_identity_validation=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("h2", "oh"), default="h2")
    parser.add_argument("--max-rounds", type=int, default=6)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="openea_stage3_loop_") as tmp:
        tmp = Path(tmp)
        source = tmp / f"{args.case}_source.chk"
        artifacts = tmp / "high_level"
        atoms, center, charge, spin, half_width = source_checkpoint(args.case, source)

        grid = (center - half_width, center, center + half_width)
        requests = []
        for grid_index, r in enumerate(grid):
            requests.append(make_request(
                case=args.case,
                atoms=atoms,
                center=center,
                charge=charge,
                spin=spin,
                source=source,
                r=r,
                init=0,
                grid_index=grid_index,
                root="root_A",
            ))
            if grid_index == 1:
                requests.append(make_request(
                    case=args.case,
                    atoms=atoms,
                    center=center,
                    charge=charge,
                    spin=spin,
                    source=source,
                    r=r,
                    init=1,
                    grid_index=grid_index,
                    root="root_B",
                ))

        execution = Stage3ExecutionSettings(
            scf_conv_tol=1.0e-9,
            scf_conv_tol_grad=1.0e-6,
            cc_conv_tol=1.0e-8,
            max_memory_mb=2000,
            verbose=0,
            artifact_dir=str(artifacts),
        )
        initial_results = [
            run_stage3_point(item, settings=execution)
            for item in requests
        ]
        if any(item.status is not PointExecutionStatus.COMPLETED for item in initial_results):
            print(json.dumps([item.to_dict() for item in initial_results], indent=2, sort_keys=True))
            raise SystemExit(2)

        outcome = run_stage3_refinement_loop(
            initial_requests=requests,
            initial_results=initial_results,
            refinement_settings=REFINEMENT_SETTINGS,
            identity_thresholds=IDENTITY_THRESHOLDS,
            branch_thresholds=BRANCH_THRESHOLDS,
            max_refinement_rounds=args.max_rounds,
            execution_settings=execution,
        )

        payload = {
            "case": args.case,
            "status": outcome.status.value,
            "n_evaluations": len(outcome.rounds),
            "n_requests": len(outcome.requests),
            "n_results": len(outcome.results),
            "final_pec_status": outcome.final_pec.status.value,
            "final_minimum_status": outcome.final_pec.minimum_scout.status.value,
            "final_action": outcome.final_refinement_plan.action.value,
            "sampled_r_angstrom": [point.r_angstrom for point in outcome.final_pec.points],
            "minimum_candidates_r_angstrom": [
                item.r_angstrom for item in outcome.final_pec.minimum_scout.candidates
            ],
            "rounds": [
                {
                    "evaluation_index": item.evaluation_index,
                    "refinement_action": item.refinement_action,
                    "proposed_r_angstrom": list(item.proposed_r_angstrom),
                    "new_result_statuses": list(item.new_result_statuses),
                    "initialization_review": item.initialization_review_status,
                    "geometry_continuity_review": item.geometry_continuity_review_status,
                }
                for item in outcome.rounds
            ],
            "is_production_ea": outcome.is_production_ea,
            "ground_state_assigned": outcome.ground_state_assigned,
            "authorizes_pruning": outcome.authorizes_pruning,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))

        if outcome.status.value in ("EXECUTION_BLOCKED", "SCIENTIFICALLY_UNRESOLVED"):
            raise SystemExit(3)
        if outcome.is_production_ea or outcome.ground_state_assigned or outcome.authorizes_pruning:
            raise SystemExit(4)


if __name__ == "__main__":
    main()
