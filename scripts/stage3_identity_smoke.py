#!/usr/bin/env python3
"""Manual end-to-end Stage-3 HF identity/continuity smoke test.

Run on Artemis/Theia with the OpenEA venv activated and ``nice -n 19``.
The thresholds below are smoke-test policy only, not OpenEA production defaults.
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
    Stage3ExecutionRequest,
    Stage3ExecutionSettings,
    run_stage3_point,
)
from openea_benchmark.adaptive.stage3_identity import resolve_high_level_identity_and_pec


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


def case_spec(case: str):
    if case == "h2":
        return ("H", "H"), 0.74, 0, 0
    if case == "oh":
        return ("O", "H"), 0.97, 0, 1
    raise ValueError(case)


def source_checkpoint(case: str, path: Path):
    atoms, r, charge, spin = case_spec(case)
    mol = gto.M(
        atom=f"{atoms[0]} 0 0 0; {atoms[1]} 0 0 {r}",
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
    return atoms, r, charge, spin


def make_request(*, case, atoms, center, charge, spin, source, r, init, grid_index, root):
    return Stage3ExecutionRequest(
        request_id=f"identity_smoke__{case}__init{init:02d}_{root}__g{grid_index:03d}",
        job_id=f"identity_smoke__{case}__job",
        system=case.upper(),
        atoms=atoms,
        charge=charge,
        spin_2s=spin,
        component_id="identity_smoke_component",
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
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="openea_stage3_identity_") as tmp:
        tmp = Path(tmp)
        source = tmp / f"{args.case}_source.chk"
        artifacts = tmp / "high_level"
        atoms, center, charge, spin = source_checkpoint(args.case, source)

        grid = (center - 0.02, center, center + 0.02)
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

        settings = Stage3ExecutionSettings(
            scf_conv_tol=1.0e-9,
            scf_conv_tol_grad=1.0e-6,
            cc_conv_tol=1.0e-8,
            max_memory_mb=2000,
            verbose=0,
            artifact_dir=str(artifacts),
        )
        results = [run_stage3_point(req, settings=settings) for req in requests]
        if any(result.status.value != "COMPLETED" for result in results):
            print(json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True))
            raise SystemExit(2)

        resolution = resolve_high_level_identity_and_pec(
            requests=requests,
            results=results,
            identity_thresholds=IDENTITY_THRESHOLDS,
            branch_thresholds=BRANCH_THRESHOLDS,
        )
        payload = {
            "case": args.case,
            "n_requests": len(requests),
            "n_checkpoint_audits": len(resolution.checkpoint_audits),
            "initialization_review": resolution.initialization_review.status.value,
            "geometry_continuity_review": resolution.geometry_continuity_review.status.value,
            "initialization_relations": [x.relation.value for x in resolution.initialization_comparisons],
            "continuity_relations": [x.relation.value for x in resolution.continuity_comparisons],
            "continuity_min_overlaps": [
                min(x.alpha_occ_overlap_min, x.beta_occ_overlap_min)
                for x in resolution.continuity_comparisons
            ],
            "pec_status": resolution.pec.status.value,
            "minimum_status": resolution.pec.minimum_scout.status.value,
            "minimum_r_angstrom": (
                resolution.pec.minimum_scout.candidates[0].r_angstrom
                if len(resolution.pec.minimum_scout.candidates) == 1
                else None
            ),
            "is_production_ea": resolution.pec.is_production_ea,
            "ground_state_assigned": resolution.pec.ground_state_assigned,
            "authorizes_pruning": resolution.pec.authorizes_pruning,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))

        if resolution.initialization_review.status.value != "CLEARED":
            raise SystemExit(3)
        if resolution.geometry_continuity_review.status.value != "CLEARED":
            raise SystemExit(4)
        if resolution.pec.status.value != "READY_FOR_DISCRETE_MINIMUM_SCOUT":
            raise SystemExit(5)


if __name__ == "__main__":
    main()
