#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from openea_benchmark.adaptive.adaptive_diffuse_runner import (
    correlation_consistent_diffuse_basis_name,
    run_adaptive_diffuse_series,
)
from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceSettings,
    EAIntervalEV,
    ElectronicEABasisPoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cardinal", type=int, default=5)
    parser.add_argument("--max-augmentation", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--target-half-width-ev", type=float, default=0.05)
    parser.add_argument("--neutral-center", type=float, default=0.97)
    parser.add_argument("--anion-center", type=float, default=0.97)
    parser.add_argument("--half-width", type=float, default=0.02)
    parser.add_argument("--diffuse-increment-target-ev", type=float, default=0.01)
    parser.add_argument("--diffuse-contraction-ratio-max", type=float, default=0.60)
    parser.add_argument("--force-double-augmentation", action="store_true")
    args = parser.parse_args()

    smoke_script = Path(__file__).with_name("attachment_oh_ea_smoke.py")

    def evaluate(level):
        basis = correlation_consistent_diffuse_basis_name(
            args.cardinal, augmentation_level=level
        )
        cmd = [
            sys.executable,
            str(smoke_script),
            "--basis", basis,
            "--max-rounds", str(args.max_rounds),
            "--target-half-width-ev", str(args.target_half_width_ev),
            "--neutral-center", str(args.neutral_center),
            "--anion-center", str(args.anion_center),
            "--half-width", str(args.half_width),
        ]
        proc = subprocess.run(
            cmd, check=False, text=True,
            stdout=subprocess.PIPE, stderr=None
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{basis}: electronic-EA smoke exited with code {proc.returncode}"
            )
        payload = json.loads(proc.stdout)
        ea = payload["electronic_ea"]
        if ea["decision_status"] != "BOUND":
            raise RuntimeError(
                f"{basis}: electronic EA is {ea['decision_status']}"
            )
        return ElectronicEABasisPoint(
            basis_name=basis,
            cardinal_number=args.cardinal,
            augmentation_level=level,
            method_signature=(
                "CCSD(T)|adaptive-equilibrium-PEC|"
                "same-basis-molecule-and-fragments|electronic-only"
            ),
            ea=EAIntervalEV(
                float(ea["lower_ev"]),
                float(ea["central_ev"]),
                float(ea["upper_ev"]),
            ),
            decision_status="BOUND",
            evidence_quality="CONVERGENCE_ESTIMATED",
            is_production_ea=False,
        )

    settings = BasisConvergenceSettings(
        cardinal_increment_target_ev=0.02,
        cardinal_contraction_ratio_max=0.75,
        diffuse_increment_target_ev=args.diffuse_increment_target_ev,
        diffuse_contraction_ratio_max=args.diffuse_contraction_ratio_max,
        force_double_augmentation=args.force_double_augmentation,
    )

    result = run_adaptive_diffuse_series(
        cardinal_number=args.cardinal,
        initial_augmentation_levels=(0, 1),
        maximum_augmentation_level=args.max_augmentation,
        convergence_settings=settings,
        evaluate_point=evaluate,
    )

    print(json.dumps({
        "species": "OH/OH-",
        "validation_only": True,
        "cardinal_number": args.cardinal,
        "status": result.status.value,
        "evaluated_points": [
            {
                "basis": p.basis_name,
                "augmentation_level": p.augmentation_level,
                "ea_lower_ev": p.ea.lower_ev,
                "ea_central_ev": p.ea.central_ev,
                "ea_upper_ev": p.ea.upper_ev,
            } for p in result.points
        ],
        "iterations": [
            {
                "iteration_index": it.iteration_index,
                "evaluated_augmentation_levels": list(it.evaluated_augmentation_levels),
                "diffuse_status": it.assessment.status.value,
                "diffuse_action": it.assessment.action.value,
                "latest_increment_bound_ev": it.assessment.latest_increment_bound_ev,
                "contraction_ratio": it.assessment.contraction_ratio,
                "requested_next_augmentation_level": it.requested_next_augmentation_level,
            } for it in result.iterations
        ],
        "final_assessment": None if result.final_assessment is None else {
            "status": result.final_assessment.status.value,
            "action": result.final_assessment.action.value,
            "highest_augmentation_level": result.final_assessment.highest_augmentation_level,
            "latest_increment_bound_ev": result.final_assessment.latest_increment_bound_ev,
            "contraction_ratio": result.final_assessment.contraction_ratio,
            "residual_estimate_ev": result.final_assessment.residual_estimate_ev,
            "evidence": list(result.final_assessment.evidence),
        },
        "execution_error_type": result.execution_error_type,
        "execution_error_message": result.execution_error_message,
        "is_production_ea": result.is_production_ea,
        "authorizes_pruning": result.authorizes_pruning,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
