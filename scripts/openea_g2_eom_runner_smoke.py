#!/usr/bin/env python3
"""Tiny LiH/STO-3G PySCF G2 EA-EOM smoke; NOT a scientific EA/continuum test.

Runs RHF -> RCCSD -> EA-EOM for two raw roots and exercises the new generic
G2 runner/checkpoint logic. Does NOT assert that LiH- is bound, nor select a
physical anion state, nor mark D08 CLEARED.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pyscf import gto, scf

from openea_benchmark.adaptive.attachment_eom_runner import (
    G2EOMAuthorization,
    G2EOMBasis,
    G2EOMNeutralState,
    G2EOMSettings,
    G2EOMStatus,
    run_g2_eom_diagnostics,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="openea-g2-eom-smoke-") as temp:
        directory = Path(temp)
        mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", charge=0,
                    spin=0, symmetry=False, unit="Angstrom", verbose=0)
        mf = scf.RHF(mol)
        mf.chkfile = str(directory / "neutral.chk")
        mf.conv_tol = 1e-10
        mf.kernel()
        if not mf.converged or not Path(mf.chkfile).is_file():
            raise RuntimeError("LiH reference checkpoint could not be generated")
        neutral = G2EOMNeutralState(
            system="LiH_SOFTWARE_SMOKE", atoms=("Li", "H"), charge=0,
            spin_2s=0, state_id="NEUTRAL_SMOKE_RHF", r_angstrom=1.6,
            source_root_id="SOFTWARE_SMOKE_REFERENCE_ONLY",
            source_checkpoint_path=str(mf.chkfile),
            state_identity_validated=True,
            reference_character_validated=True, scf_reference="RHF",
        )
        auth = G2EOMAuthorization(True, "Software integration test only",
                                   ("SOFTWARE_SMOKE_AUTHORIZATION",), ("aug:0",))
        result = run_g2_eom_diagnostics(
            authorization=auth,
            neutral=neutral,
            basis_specs=(G2EOMBasis(0, "sto-3g software smoke", "SMOKE",
                                    {"Li": "sto-3g", "H": "sto-3g"}),),
            settings=G2EOMSettings(nroots=2),
            checkpoint_dir=directory / "g2-checkpoints",
        )
        if result.status is not G2EOMStatus.COMPLETE_ROOT_REVIEW_REQUIRED:
            raise RuntimeError(f"G2 smoke did not complete: {result.notes}")
        repeat = run_g2_eom_diagnostics(
            authorization=auth, neutral=neutral,
            basis_specs=(G2EOMBasis(0, "sto-3g software smoke", "SMOKE",
                                    {"Li": "sto-3g", "H": "sto-3g"}),),
            settings=G2EOMSettings(nroots=2),
            checkpoint_dir=directory / "g2-checkpoints",
        )
        if not repeat.subpoints[0].checkpoint_reused:
            raise RuntimeError("G2 smoke checkpoint resume did not work")
        print(json.dumps({
            "status": "PASS",
            "scope": "SOFTWARE_INTEGRATION_ONLY_NOT_SCIENTIFIC_EA",
            "runner_status": result.status.value,
            "backend_version": result.subpoints[0].result.pyscf_version,
            "root_omegas_hartree": [r.omega_hartree for r in result.subpoints[0].result.roots],
            "roots_automatically_state_validated": False,
            "checkpoint_resume": "PASS",
            "attachment_continuum_status": "UNRESOLVED_ROOT_AND_CONTINUUM_IDENTITY",
        }, indent=2))


if __name__ == "__main__":
    main()
