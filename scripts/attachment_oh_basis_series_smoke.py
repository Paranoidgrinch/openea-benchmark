#!/usr/bin/env python3
"""Real OH/OH- aug-cc-pV{D,T,Q}Z electronic-EA basis-series smoke.

Each basis point is produced by invoking the already validated
`attachment_oh_ea_smoke.py` end-to-end workflow.  This script then feeds only
the resulting electronic-EA intervals into the generic basis-convergence
layer.

The default convergence thresholds are validation-only.  They are explicit
CLI settings and must not be interpreted as production OpenEA defaults.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceSettings,
    EAIntervalEV,
    ElectronicEABasisPoint,
    assess_cardinal_convergence,
)


BASIS_SERIES = (
    ("aug-cc-pvdz", 2),
    ("aug-cc-pvtz", 3),
    ("aug-cc-pvqz", 4),
)


def _run_one(
    *,
    smoke_script: Path,
    basis: str,
    max_rounds: int,
    target_half_width_ev: float,
    neutral_center: float,
    anion_center: float,
    half_width: float,
) -> dict:
    cmd = [
        sys.executable,
        str(smoke_script),
        "--basis",
        basis,
        "--max-rounds",
        str(max_rounds),
        "--target-half-width-ev",
        str(target_half_width_ev),
        "--neutral-center",
        str(neutral_center),
        "--anion-center",
        str(anion_center),
        "--half-width",
        str(half_width),
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{basis}: OH electronic-EA smoke exited with code {proc.returncode}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{basis}: smoke output was not valid JSON"
        ) from exc
    return payload


def _point_from_payload(payload: dict, *, basis: str, cardinal: int):
    ea = payload["electronic_ea"]
    if ea["decision_status"] != "BOUND":
        raise RuntimeError(
            f"{basis}: electronic EA is {ea['decision_status']}, not BOUND"
        )
    values = (ea["lower_ev"], ea["central_ev"], ea["upper_ev"])
    if any(value is None for value in values):
        raise RuntimeError(f"{basis}: numeric EA interval is incomplete")

    return ElectronicEABasisPoint(
        basis_name=basis,
        cardinal_number=cardinal,
        augmentation_level=1,
        method_signature=(
            "CCSD(T)|adaptive-equilibrium-PEC|"
            "same-basis-molecule-and-fragments|electronic-only"
        ),
        ea=EAIntervalEV(
            lower_ev=float(ea["lower_ev"]),
            central_ev=float(ea["central_ev"]),
            upper_ev=float(ea["upper_ev"]),
        ),
        decision_status="BOUND",
        evidence_quality="CONVERGENCE_ESTIMATED",
        is_production_ea=False,
    )


def _assessment_payload(result):
    interval = result.expanded_highest_interval
    return {
        "status": result.status.value,
        "action": result.action.value,
        "augmentation_level": result.augmentation_level,
        "highest_cardinal": result.highest_cardinal,
        "latest_increment_bound_ev": result.latest_increment_bound_ev,
        "contraction_ratio": result.contraction_ratio,
        "residual_estimate_ev": result.residual_estimate_ev,
        "expanded_highest_interval": (
            None
            if interval is None
            else {
                "lower_ev": interval.lower_ev,
                "central_ev": interval.central_ev,
                "upper_ev": interval.upper_ev,
            }
        ),
        "evidence_quality": result.evidence_quality,
        "evidence": list(result.evidence),
        "rationale": result.rationale,
        "is_production_ea": result.is_production_ea,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--target-half-width-ev", type=float, default=0.05)
    parser.add_argument("--neutral-center", type=float, default=0.97)
    parser.add_argument("--anion-center", type=float, default=0.97)
    parser.add_argument("--half-width", type=float, default=0.02)
    parser.add_argument(
        "--cardinal-increment-target-ev",
        type=float,
        default=0.02,
        help="validation-only convergence threshold",
    )
    parser.add_argument(
        "--cardinal-contraction-ratio-max",
        type=float,
        default=0.75,
        help="validation-only contraction threshold",
    )
    args = parser.parse_args()

    smoke_script = Path(__file__).with_name("attachment_oh_ea_smoke.py")
    if not smoke_script.exists():
        raise SystemExit(f"missing smoke script: {smoke_script}")

    runs = []
    points = []
    for basis, cardinal in BASIS_SERIES:
        payload = _run_one(
            smoke_script=smoke_script,
            basis=basis,
            max_rounds=args.max_rounds,
            target_half_width_ev=args.target_half_width_ev,
            neutral_center=args.neutral_center,
            anion_center=args.anion_center,
            half_width=args.half_width,
        )
        point = _point_from_payload(
            payload,
            basis=basis,
            cardinal=cardinal,
        )
        points.append(point)
        runs.append(
            {
                "basis": basis,
                "cardinal_number": cardinal,
                "electronic_ea": payload["electronic_ea"],
                "neutral_equilibrium": payload["neutral_equilibrium"],
                "anion_equilibrium": payload["anion_equilibrium"],
                "anion_binding": payload["anion_binding"],
                "dissociation_channels": payload["dissociation_channels"],
                "validation_only": payload["validation_only"],
                "is_production_ea": payload["is_production_ea"],
            }
        )

    settings = BasisConvergenceSettings(
        cardinal_increment_target_ev=args.cardinal_increment_target_ev,
        cardinal_contraction_ratio_max=(
            args.cardinal_contraction_ratio_max
        ),
        diffuse_increment_target_ev=0.01,
        diffuse_contraction_ratio_max=0.60,
        force_double_augmentation=False,
    )
    assessment = assess_cardinal_convergence(
        tuple(points),
        augmentation_level=1,
        settings=settings,
    )

    output = {
        "species": "OH/OH-",
        "series": "aug-cc-pV{D,T,Q}Z",
        "validation_only": True,
        "method_signature": points[0].method_signature,
        "settings": {
            "cardinal_increment_target_ev": (
                args.cardinal_increment_target_ev
            ),
            "cardinal_contraction_ratio_max": (
                args.cardinal_contraction_ratio_max
            ),
        },
        "basis_runs": runs,
        "cardinal_convergence": _assessment_payload(assessment),
        "scientific_caveats": [
            "This is an electronic-EA validation series, not a production adiabatic EA.",
            "The convergence thresholds are explicit smoke-test settings, not universal OpenEA defaults.",
            "No DZ/TZ/QZ averaging is performed.",
            "A CLEARED cardinal result is convergence evidence, not yet a component-resolved CBS extrapolation.",
            "Diffuse convergence remains a separate diagnostic.",
            "ZPE and downstream high-accuracy corrections are not included.",
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
