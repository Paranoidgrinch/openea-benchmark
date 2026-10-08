#!/usr/bin/env python3
"""Real neutral/anion PySCF CASSCF->CASCI->NEVPT2 integration, not scientific EA."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from pyscf import gto, scf

from openea_benchmark.adaptive.mr_casscf_nevpt2_runner import (
    MRPointAuthorization, MRPointRequest, MRPointSettings, MRPointStatus,
    run_mr_casscf_nevpt2_point,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="openea-mr-smoke-") as td:
        directory = Path(td)
        authorization = MRPointAuthorization(True, "Backend integration smoke only",
                                             ("SMOKE_ONLY_NOT_VALIDATED_MR",),
                                             max_fci_determinants=1000)
        settings = MRPointSettings(scf_conv_tol=1e-9, casscf_conv_tol=1e-7,
                                   casscf_max_macro=60, verbose=0)
        report = {}
        for role, charge, spin, na, nb, nroots in (
            ("neutral", 0, 0, 1, 1, 2),
            ("anion", -1, 1, 2, 1, 1),
        ):
            mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g",
                        charge=charge, spin=spin, symmetry=False,
                        unit="Angstrom", verbose=0)
            mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
            mf.chkfile = str(directory / (role + ".chk"))
            mf.conv_tol = 1e-9
            mf.kernel()
            if not mf.converged:
                raise RuntimeError(f"{role}: source HF did not converge")
            req = MRPointRequest(
                request_id="MR_SMOKE_" + role.upper(), system="LiH_SMOKE",
                role=role, atoms=("Li", "H"), charge=charge, spin_2s=spin,
                state_manifold_id="SMOKE_FIXED_SPIN", r_angstrom=1.6,
                basis_label="STO-3G_SMOKE", basis_by_element={"Li": "sto-3g", "H": "sto-3g"},
                source_root_id="SMOKE_HF_REFERENCE", source_checkpoint_path=mf.chkfile,
                active_orbital_indices=(1, 2), active_electrons_alpha=na,
                active_electrons_beta=nb, nroots=nroots,
                active_space_review_ids=("SMOKE_ACTIVE_SPACE_ONLY",),
                state_manifold_review_ids=("SMOKE_STATE_MANIFOLD_ONLY",),
            )
            result = run_mr_casscf_nevpt2_point(req, authorization, settings=settings)
            if result.status is not MRPointStatus.COMPLETE_REVIEW_REQUIRED:
                raise RuntimeError(f"{role}: {result.status.value}: {result.reason}")
            if len(result.roots) != nroots:
                raise RuntimeError(f"{role}: incorrect number of roots")
            report[role] = {
                "status": result.status.value,
                "roots": [dict(root=i.root_index, e_casci=i.casci_hartree,
                               e_nevpt2=i.sc_nevpt2_total_hartree,
                               s2=i.spin_square) for i in result.roots],
                "source_digest_recorded": bool(result.source_checkpoint_sha256),
                "signature_recorded": bool(result.result_signature),
            }
        print(json.dumps({
            "status": "PASS", "scope": "SOFTWARE_INTEGRATION_ONLY_NOT_SCIENTIFIC_EA",
            "state_selection_validated": False,
            "active_space_auto_selected": False,
            "mr_production_validated": False,
            "neutral_anion": report,
        }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
