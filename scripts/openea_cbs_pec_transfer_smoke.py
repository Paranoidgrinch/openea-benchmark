#!/usr/bin/env python3
"""Real PySCF HF two-basis, multi-R *numerical-transfer* smoke.

NOT a CBS extrapolation, NOT correlated CCSD(T), NOT a production electron
 affinity: only tests the data/geometry/matrix/transfer plumbing.
"""
from __future__ import annotations

import json
import sys

from pyscf import gto, scf

from openea_benchmark.adaptive.cbs_pec_transfer import (
    MatchedCorrectionPoint, TransferStatus, evaluate_pec_correction_transfer,
)


def main() -> None:
    points = []
    for role, charge, spin in (("neutral", 0, 0), ("anion", -1, 1)):
        for j, r in enumerate((1.5, 1.6, 1.7)):
            energies = []
            for basis in ("sto-3g", "6-31g"):
                mol = gto.M(
                    atom=f"Li 0 0 0; H 0 0 {r}", basis=basis,
                    charge=charge, spin=spin, unit="Angstrom",
                    symmetry=False, verbose=0,
                )
                mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
                mf.conv_tol = 1e-10
                mf.max_cycle = 150
                energy = float(mf.kernel())
                if not mf.converged:
                    raise RuntimeError(f"HF failed for {role} at {r} {basis}")
                energies.append(energy)
            points.append(MatchedCorrectionPoint(
                role=role, atoms=("Li", "H"), charge=charge, spin_2s=spin,
                state_id=f"{role}_HF_smoke", r_angstrom=r,
                lower_model_id="HF_STO3G", upper_model_id="HF_631G",
                lower_energy_hartree=energies[0], upper_energy_hartree=energies[1],
                lower_source_id=f"{role}_HF_STO3G_{j}",
                upper_source_id=f"{role}_HF_631G_{j}",
                state_identity_reviewed=True,  # explicitly smoke-only assumption
            ))
    assessment = evaluate_pec_correction_transfer(
        points=points, neutral_target_angstrom=1.58, anion_target_angstrom=1.63,
    )
    if assessment.status is not TransferStatus.CANDIDATE_REVIEW_REQUIRED:
        raise RuntimeError(f"Unexpected result: {assessment.status}")
    result = assessment.to_dict()
    result.update({
        "status": "PASS", "pyscf_version": __import__("pyscf").__version__,
        "scope": "HF_SOFTWARE_INTEGRATION_ONLY_NOT_CBS_OR_SCIENTIFIC_EA",
        "number_of_real_hf_calculations": 12,
        "correlated_cbs_convergence_validated": False,
        "state_identity_validated": False,
        "uncertainty_bounded": False,
    })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
