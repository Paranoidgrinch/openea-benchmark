#!/usr/bin/env python3
"""Real numerical OH/OH- electronic-EA smoke test.

This is validation infrastructure, not a production OpenEA calculation.

It performs:
1. an adaptive Stage-3 CCSD(T) PEC loop for neutral OH,
2. an adaptive Stage-3 CCSD(T) PEC loop for OH-,
3. high-level atomic fragment calculations for both OH- dissociation channels,
4. Stage-3 -> attachment bridging,
5. equilibrium energy interval resolution,
6. dissociation-stability assessment,
7. electronic EA interval decision.

The default STO-3G basis and thresholds are intentionally cheap smoke-test
settings. The resulting number must not be interpreted as a recommended OH EA.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
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
from openea_benchmark.adaptive.stage3_loop import (
    Stage3LoopRetrySettings,
    run_stage3_refinement_loop,
)
from openea_benchmark.adaptive.stage3_refinement import Stage3RefinementSettings
from openea_benchmark.attachment.equilibrium import EquilibriumResolverSettings
from openea_benchmark.attachment.fragment_execution import (
    AtomicFragmentRequest,
    FragmentExecutionStatus,
    build_validation_dissociation_channel,
    run_atomic_fragment,
)
from openea_benchmark.attachment.workflow import evaluate_stage3_electronic_ea


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

EQUILIBRIUM_SETTINGS = EquilibriumResolverSettings(
    max_side_points=2,
    minimum_admissible_models=2,
    minimum_curvature_hartree_per_angstrom2=1.0e-3,
    max_model_geometry_spread_angstrom=0.01,
    max_model_energy_spread_hartree=5.0e-4,
    point_energy_tolerance_hartree=1.0e-8,
)


def source_checkpoint(
    *,
    path: Path,
    charge: int,
    spin_2s: int,
    center_r: float,
    basis: str,
):
    mol = gto.M(
        atom=f"O 0 0 0; H 0 0 {center_r}",
        basis=basis,
        charge=charge,
        spin=spin_2s,
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )
    mf = scf.RHF(mol) if spin_2s == 0 else scf.ROHF(mol)
    mf.chkfile = str(path)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(
            f"source SCF did not converge for q={charge}, 2S={spin_2s}"
        )


def make_request(
    *,
    job_id: str,
    charge: int,
    spin_2s: int,
    basis: str,
    center_r: float,
    source: Path,
    r: float,
    init: int,
    grid_index: int,
    root: str,
):
    return Stage3ExecutionRequest(
        request_id=f"{job_id}__init{init:02d}_{root}__g{grid_index:03d}",
        job_id=job_id,
        system="OH",
        atoms=("O", "H"),
        charge=charge,
        spin_2s=spin_2s,
        component_id=f"{job_id}__component",
        r_angstrom=r,
        basis=basis,
        methods=("CCSD", "CCSD(T)"),
        requested_reference="ROHF",
        scf_reference="RHF" if spin_2s == 0 else "ROHF",
        source_link_status="MULTIPLE_DFT_INITIALIZATIONS",
        source_root_id=root,
        source_checkpoint_path=str(source),
        source_origin_guess="OH_EA_SMOKE",
        grid_index=grid_index,
        initialization_index=init,
        dft_center_r_angstrom=center_r,
        dft_center_energy_hartree=0.0,
        requires_independent_state_identity_validation=True,
    )


def run_oh_state_loop(
    *,
    tmp: Path,
    label: str,
    charge: int,
    spin_2s: int,
    center_r: float,
    basis: str,
    half_width: float,
    max_rounds: int,
    execution: Stage3ExecutionSettings,
):
    source = tmp / f"{label}_source.chk"
    source_checkpoint(
        path=source,
        charge=charge,
        spin_2s=spin_2s,
        center_r=center_r,
        basis=basis,
    )

    grid = (center_r - half_width, center_r, center_r + half_width)
    requests = []
    job_id = f"oh_ea_smoke__{label}"
    for grid_index, r in enumerate(grid):
        requests.append(
            make_request(
                job_id=job_id,
                charge=charge,
                spin_2s=spin_2s,
                basis=basis,
                center_r=center_r,
                source=source,
                r=r,
                init=0,
                grid_index=grid_index,
                root=f"{label}_root_A",
            )
        )
        if grid_index == 1:
            requests.append(
                make_request(
                    job_id=job_id,
                    charge=charge,
                    spin_2s=spin_2s,
                    basis=basis,
                    center_r=center_r,
                    source=source,
                    r=r,
                    init=1,
                    grid_index=grid_index,
                    root=f"{label}_root_B",
                )
            )

    initial_results = [
        run_stage3_point(request, settings=execution)
        for request in requests
    ]
    if any(
        result.status is not PointExecutionStatus.COMPLETED
        for result in initial_results
    ):
        raise RuntimeError(
            json.dumps(
                [result.to_dict() for result in initial_results],
                indent=2,
                sort_keys=True,
            )
        )

    return run_stage3_refinement_loop(
        initial_requests=requests,
        initial_results=initial_results,
        refinement_settings=REFINEMENT_SETTINGS,
        identity_thresholds=IDENTITY_THRESHOLDS,
        branch_thresholds=BRANCH_THRESHOLDS,
        max_refinement_rounds=max_rounds,
        execution_settings=execution,
        retry_settings=Stage3LoopRetrySettings(
            max_retries_per_request=1,
            cycle_multiplier=2.0,
        ),
    )


def fragment_specs(basis: str):
    # Explicit validation-only atomic state choices:
    # O  : 3P -> 2S = 2
    # O- : 2P -> 2S = 1
    # H  : 2S -> 2S = 1
    # H- : 1S -> 2S = 0
    return (
        AtomicFragmentRequest(
            "O_minus_2P",
            "O",
            -1,
            1,
            basis,
            state_label="2P",
        ),
        AtomicFragmentRequest(
            "H_2S",
            "H",
            0,
            1,
            basis,
            state_label="2S",
        ),
        AtomicFragmentRequest(
            "O_3P",
            "O",
            0,
            2,
            basis,
            state_label="3P",
        ),
        AtomicFragmentRequest(
            "H_minus_1S",
            "H",
            -1,
            0,
            basis,
            state_label="1S",
        ),
    )


def interval_payload(interval):
    if interval is None:
        return None
    return {
        "lower_hartree": interval.lower_hartree,
        "upper_hartree": interval.upper_hartree,
    }


def equilibrium_payload(result):
    if result is None:
        return None
    return {
        "status": result.status.value,
        "geometry_interval_angstrom": result.geometry_interval_angstrom,
        "geometry_central_angstrom": result.geometry_central_angstrom,
        "energy_interval_hartree": interval_payload(
            result.energy_interval_hartree
        ),
        "energy_central_hartree": result.energy_central_hartree,
        "generated_model_count": result.generated_model_count,
        "rejected_model_count": result.rejected_model_count,
        "evidence_quality": result.evidence_quality,
        "evidence": list(result.evidence),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="aug-cc-pvdz")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--neutral-center", type=float, default=0.97)
    parser.add_argument("--anion-center", type=float, default=0.97)
    parser.add_argument("--half-width", type=float, default=0.02)
    parser.add_argument("--target-half-width-ev", type=float, default=0.05)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="openea_oh_ea_smoke_") as tmpdir:
        tmp = Path(tmpdir)
        execution = Stage3ExecutionSettings(
            scf_conv_tol=1.0e-9,
            scf_conv_tol_grad=1.0e-6,
            cc_conv_tol=1.0e-8,
            max_memory_mb=2000,
            verbose=0,
            artifact_dir=str(tmp / "high_level"),
        )

        neutral = run_oh_state_loop(
            tmp=tmp,
            label="neutral",
            charge=0,
            spin_2s=1,
            center_r=args.neutral_center,
            basis=args.basis,
            half_width=args.half_width,
            max_rounds=args.max_rounds,
            execution=execution,
        )
        anion = run_oh_state_loop(
            tmp=tmp,
            label="anion",
            charge=-1,
            spin_2s=0,
            center_r=args.anion_center,
            basis=args.basis,
            half_width=args.half_width,
            max_rounds=args.max_rounds,
            execution=execution,
        )

        fragment_results = [
            run_atomic_fragment(request, settings=execution)
            for request in fragment_specs(args.basis)
        ]
        if any(
            result.status is not FragmentExecutionStatus.COMPLETED
            for result in fragment_results
        ):
            print(
                json.dumps(
                    {
                        "fragment_failure": [
                            result.to_dict()
                            for result in fragment_results
                        ]
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            raise SystemExit(2)

        by_id = {result.fragment_id: result for result in fragment_results}
        channels = (
            build_validation_dissociation_channel(
                channel_id="O_minus_plus_H",
                fragment_a=by_id["O_minus_2P"],
                fragment_b=by_id["H_2S"],
            ),
            build_validation_dissociation_channel(
                channel_id="O_plus_H_minus",
                fragment_a=by_id["O_3P"],
                fragment_b=by_id["H_minus_1S"],
            ),
        )

        result = evaluate_stage3_electronic_ea(
            neutral_loop_result=neutral,
            anion_loop_result=anion,
            equilibrium_settings=EQUILIBRIUM_SETTINGS,
            anion_dissociation_channels=channels,
            target_half_width_ev=args.target_half_width_ev,
            neutral_state_id="OH_neutral_2Pi_validation",
            anion_state_id="OH_anion_1Sigma_validation",
            neutral_state_label="2Pi (validation label)",
            anion_state_label="1Sigma+ (validation label)",
        )

        payload = {
            "basis": args.basis,
            "validation_only": True,
            "neutral_stage3_status": neutral.status.value,
            "anion_stage3_status": anion.status.value,
            "neutral_minimum_candidates_r_angstrom": [
                x.r_angstrom
                for x in neutral.final_pec.minimum_scout.candidates
            ],
            "anion_minimum_candidates_r_angstrom": [
                x.r_angstrom
                for x in anion.final_pec.minimum_scout.candidates
            ],
            "neutral_equilibrium": equilibrium_payload(
                result.neutral_equilibrium
            ),
            "anion_equilibrium": equilibrium_payload(
                result.anion_equilibrium
            ),
            "fragments": [
                item.to_dict()
                for item in fragment_results
            ],
            "dissociation_channels": [
                {
                    "channel_id": channel.channel_id,
                    "fragment_a": channel.fragment_a,
                    "fragment_b": channel.fragment_b,
                    "energy_hartree": channel.asymptotic_energy_hartree,
                    "source": channel.source,
                }
                for channel in channels
            ],
            "anion_binding": (
                None
                if result.anion_binding is None
                else {
                    "status": result.anion_binding.status.value,
                    "evidence": list(result.anion_binding.evidence),
                    "energy_margin_hartree": (
                        result.anion_binding.energy_margin_hartree
                    ),
                }
            ),
            "electronic_ea": {
                "workflow_status": result.status.value,
                "decision_status": result.ea_decision.status.value,
                "precision_status": (
                    result.ea_decision.precision_status.value
                ),
                "lower_ev": result.ea_decision.ea_lower_ev,
                "central_ev": result.ea_decision.ea_central_ev,
                "upper_ev": result.ea_decision.ea_upper_ev,
                "half_width_ev": result.ea_decision.half_width_ev,
                "rationale": list(result.ea_decision.rationale),
            },
            "is_electronic_ea_only": result.is_electronic_ea_only,
            "includes_zpe": result.includes_zpe,
            "is_production_ea": result.is_production_ea,
            "ground_state_assigned": result.ground_state_assigned,
            "authorizes_pruning": result.authorizes_pruning,
            "scientific_caveats": [
                "aug-cc-pVDZ/default smoke settings are validation-only",
                "atomic fragment states are explicitly requested, not autonomously state-manifold resolved",
                "fragment thresholds are scalar same-level smoke evidence without production uncertainty intervals",
                "ZPE and downstream high-accuracy corrections are not included",
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))

        if neutral.status.value in (
            "EXECUTION_BLOCKED",
            "SCIENTIFICALLY_UNRESOLVED",
        ):
            raise SystemExit(3)
        if anion.status.value in (
            "EXECUTION_BLOCKED",
            "SCIENTIFICALLY_UNRESOLVED",
        ):
            raise SystemExit(3)
        if (
            result.is_production_ea
            or result.includes_zpe
            or result.ground_state_assigned
            or result.authorizes_pruning
        ):
            raise SystemExit(4)


if __name__ == "__main__":
    main()
