#!/usr/bin/env python3
"""Tiny manual PySCF smoke tests for OpenEA Stage-3 execution.

Run on Artemis/Theia with the project venv activated, preferably under
``nice -n 19``.  This is intentionally not part of the default pytest suite.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from pyscf import gto, scf

from openea_benchmark.adaptive.stage3_execution import (
    Stage3ExecutionRequest,
    Stage3ExecutionSettings,
    run_stage3_point,
)


def make_checkpoint(case: str, path: Path):
    if case == "h2":
        atoms = ("H", "H")
        r = 0.74
        charge = 0
        spin = 0
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="sto-3g", charge=charge, spin=spin, verbose=0)
        mf = scf.RHF(mol)
    elif case == "oh":
        atoms = ("O", "H")
        r = 0.97
        charge = 0
        spin = 1
        mol = gto.M(atom=f"O 0 0 0; H 0 0 {r}", basis="sto-3g", charge=charge, spin=spin, verbose=0)
        mf = scf.ROHF(mol)
    else:
        raise ValueError(case)
    mf.chkfile = str(path)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("smoke checkpoint SCF did not converge")
    return atoms, r, charge, spin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("h2", "oh"), default="h2")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="openea_stage3_smoke_") as tmp:
        chk = Path(tmp) / f"{args.case}.chk"
        atoms, r, charge, spin = make_checkpoint(args.case, chk)
        req = Stage3ExecutionRequest(
            request_id=f"smoke__{args.case}",
            job_id=f"smoke__{args.case}__job",
            system=args.case.upper(),
            atoms=atoms,
            charge=charge,
            spin_2s=spin,
            component_id="smoke_component",
            r_angstrom=r,
            basis="sto-3g",
            methods=("CCSD", "CCSD(T)"),
            requested_reference="ROHF",
            scf_reference="RHF" if spin == 0 else "ROHF",
            source_link_status="SINGLE_DFT_INITIALIZATION",
            source_root_id="smoke_root",
            source_checkpoint_path=str(chk),
            source_origin_guess="smoke",
            grid_index=0,
            initialization_index=0,
            dft_center_r_angstrom=r,
            dft_center_energy_hartree=0.0,
            requires_independent_state_identity_validation=True,
        )
        result = run_stage3_point(
            req,
            settings=Stage3ExecutionSettings(
                scf_conv_tol=1e-9,
                scf_conv_tol_grad=1e-6,
                cc_conv_tol=1e-8,
                max_memory_mb=2000,
                verbose=0,
            ),
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        if result.status.value != "COMPLETED":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
