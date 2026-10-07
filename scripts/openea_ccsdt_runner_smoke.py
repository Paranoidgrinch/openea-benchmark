#!/usr/bin/env python3
"""Tiny PySCF -> CCpy smoke test for the generic OpenEA CCSDT runner.

This is a SOFTWARE INTEGRATION TEST, not an electron-affinity calculation and
not validation evidence for OH.  It deliberately uses STO-3G and a single
nominal cardinal solely to verify the generic runner's real backend path:

* neutral OH ROHF source checkpoint;
* OH- RHF source checkpoint;
* explicit C1 reconstruction;
* frozen O(1s) correlation space (nfrozen=1);
* CCpy CCSD(T) and CCSDT for both states;
* formation of one DeltaT3 point;
* no CCSDTQ.

A single point is expected to finish with NEED_MORE_EVIDENCE / an unresolved
diagnostic because basis convergence is intentionally not tested here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

# Keep this tiny capability smoke deterministic and non-invasive.  This is not
# the production-parallelism policy.
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from pyscf import gto, scf

from openea_benchmark.adaptive.ccsdt_diagnostic_runner import (
    AdaptiveCCSDTDiagnosticStatus,
    CCSDTDiagnosticAuthorization,
    CCSDTDiagnosticBasisSpec,
    CCSDTDiagnosticExecutionSettings,
    CCSDTDiagnosticStateSpec,
    run_adaptive_ccsdt_diagnostic_series,
)


def make_source_checkpoint(path: Path, *, charge: int, spin_2s: int, r_angstrom: float) -> None:
    mol = gto.M(
        atom=[("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, r_angstrom))],
        basis="sto-3g",
        charge=charge,
        spin=spin_2s,
        unit="Angstrom",
        symmetry="C1",
        verbose=0,
        max_memory=4000,
    )
    mf = scf.RHF(mol) if spin_2s == 0 else scf.ROHF(mol)
    mf.chkfile = str(path)
    mf.conv_tol = 1.0e-10
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"Smoke-test source SCF did not converge: {path.name}")
    if not path.is_file():
        raise RuntimeError(f"Smoke-test source checkpoint was not written: {path}")


def main() -> None:
    with TemporaryDirectory(prefix="openea_ccsdt_smoke_") as tmp:
        root = Path(tmp)
        neutral_chk = root / "oh_neutral.chk"
        anion_chk = root / "oh_anion.chk"
        make_source_checkpoint(neutral_chk, charge=0, spin_2s=1, r_angstrom=0.97)
        make_source_checkpoint(anion_chk, charge=-1, spin_2s=0, r_angstrom=0.97)

        neutral = CCSDTDiagnosticStateSpec(
            role="neutral",
            system="OH_SMOKE",
            atoms=("O", "H"),
            charge=0,
            spin_2s=1,
            state_id="SMOKE_NEUTRAL_DOUBLET",
            r_angstrom=0.97,
            source_root_id="SMOKE_NEUTRAL_ROOT",
            source_checkpoint_path=str(neutral_chk),
            state_identity_validated=True,
            nfrozen_spatial_orbitals=1,
        )
        anion = CCSDTDiagnosticStateSpec(
            role="anion",
            system="OH_SMOKE",
            atoms=("O", "H"),
            charge=-1,
            spin_2s=0,
            state_id="SMOKE_ANION_SINGLET",
            r_angstrom=0.97,
            source_root_id="SMOKE_ANION_ROOT",
            source_checkpoint_path=str(anion_chk),
            state_identity_validated=True,
            nfrozen_spatial_orbitals=1,
        )
        basis = CCSDTDiagnosticBasisSpec(
            cardinal=2,
            family_id="SOFTWARE_SMOKE_STO3G",
            label="STO-3G (software smoke only)",
            basis_by_element={"O": "sto-3g", "H": "sto-3g"},
        )
        auth = CCSDTDiagnosticAuthorization(
            authorized=True,
            reason="Explicit one-cardinal software integration smoke for the generic PySCF/CCpy backend.",
            evidence_ids=("SOFTWARE_SMOKE_ONLY",),
            authorized_cardinals=(2,),
        )
        settings = CCSDTDiagnosticExecutionSettings(
            max_memory_mb=4000,
            require_internal_stability=False,
            require_rhf_external_stability=False,
        )
        result = run_adaptive_ccsdt_diagnostic_series(
            authorization=auth,
            neutral=neutral,
            anion=anion,
            basis_specs=(basis,),
            initial_cardinals=(2,),
            maximum_cardinal=2,
            execution_settings=settings,
            checkpoint_dir=root / "postcc",
        )

        if result.status is not AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_UNRESOLVED:
            raise RuntimeError(
                "Unexpected smoke-test terminal status; one cardinal should provide real backend evidence but remain convergence-unresolved: "
                f"{result.status.value} error={result.execution_error_type}:{result.execution_error_message}"
            )
        if result.assessment is None or result.assessment.status != "NEED_MORE_EVIDENCE":
            raise RuntimeError("Smoke test did not produce the expected one-cardinal PostCCAssessment")
        if len(result.evidence) != 1 or len(result.evidence[0].calculations) != 4:
            raise RuntimeError("Smoke test did not complete all four neutral/anion x CCSD(T)/CCSDT subpoints")
        if any(calc.result.status.value != "COMPLETED" for calc in result.evidence[0].calculations):
            raise RuntimeError("At least one CCpy smoke subcalculation did not complete")

        summary = {
            "status": "PASS",
            "scope": "SOFTWARE_INTEGRATION_SMOKE_ONLY_NOT_SCIENTIFIC_EA",
            "runner_status": result.status.value,
            "assessment_status": result.assessment.status,
            "assessment_action": result.assessment.action,
            "delta_t3_ev": result.evidence[0].point.delta_t3_ev,
            "subcalculations": [
                {
                    "role": calc.request.role,
                    "method": calc.request.method.value,
                    "ccpy_version": calc.result.ccpy_version,
                    "reference_energy_hartree": calc.result.reference_energy_hartree,
                    "total_energy_hartree": calc.result.total_energy_hartree,
                }
                for calc in result.evidence[0].calculations
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
