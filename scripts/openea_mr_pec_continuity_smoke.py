#!/usr/bin/env python3
"""Real PySCF AO-density/active-space PEC continuity integration smoke.

Small LiH/STO-3G neutral geometry pair, NOT a scientific PEC, state identity,
MR production validation, or an electron affinity.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
from pyscf import gto, scf

from openea_benchmark.adaptive.mr_casscf_nevpt2_runner import (
    MRPointAuthorization, MRPointRequest, MRPointSettings,
    MRPointStatus, run_mr_casscf_nevpt2_point,
)
from openea_benchmark.adaptive.mr_pec_continuity import (
    MRPECContinuityThresholds, MRPECContinuityStatus,
    assess_mr_pec_continuity,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="openea-mr-pec-smoke-") as tmp:
        output: list[tuple[MRPointRequest, object]] = []
        for k, r in enumerate((1.60, 1.63)):
            mol = gto.M(atom=[('Li',(0,0,0)),('H',(0,0,r))],
                        basis='sto-3g', charge=0, spin=0, unit='Angstrom',
                        symmetry=False, verbose=0)
            hf = scf.RHF(mol)
            hf.chkfile = str(Path(tmp) / f'hf_{k}.chk')
            hf.conv_tol = 1e-9
            hf.kernel()
            if not hf.converged:
                raise RuntimeError(f'{r}: RHF source not converged')
            req = MRPointRequest(
                request_id=f'MR_PEC_SMOKE_{k}', system='LiH_SMOKE',
                role='neutral', atoms=('Li','H'), charge=0, spin_2s=0,
                state_manifold_id='SINGLET_SMOKE', r_angstrom=r,
                basis_label='STO-3G_SMOKE',
                basis_by_element={'Li':'sto-3g','H':'sto-3g'},
                source_root_id=f'RHF_{k}', source_checkpoint_path=hf.chkfile,
                active_orbital_indices=(1,2), active_electrons_alpha=1,
                active_electrons_beta=1, nroots=2,
                active_space_review_ids=('SMOKE_NO_CAS_VALIDATION',),
                state_manifold_review_ids=('SMOKE_NO_STATE_VALIDATION',),
            )
            auth = MRPointAuthorization(
                True, 'Only this LiH/STO-3G software integration smoke',
                ('MR_PEC_SOFTWARE_SMOKE',), max_fci_determinants=100,
            )
            result = run_mr_casscf_nevpt2_point(
                req, auth, settings=MRPointSettings(verbose=0,
                                                     casscf_conv_tol=1e-7),
            )
            if result.status is not MRPointStatus.COMPLETE_REVIEW_REQUIRED:
                raise RuntimeError(f'{req.request_id}: {result.status}: {result.reason}')
            if result.active_mo_coeff_ao is None or any(
                root.active_rdm1 is None for root in result.roots
            ):
                raise RuntimeError('Real PySCF did not return MR 1RDM/AO fingerprints')
            for root in result.roots:
                gamma = np.asarray(root.active_rdm1)
                if abs(float(np.trace(gamma)) - 2.0) > 1e-5:
                    raise RuntimeError('MR active 1RDM electron count incorrect')
            output.append((req, result))
        comp = assess_mr_pec_continuity(
            output[0][0], output[0][1], output[1][0], output[1][1],
            thresholds=MRPECContinuityThresholds(
                max_delta_r_angstrom=0.1,
                min_active_subspace_singular_value=0.5,
                min_root_density_similarity=0.2,
                min_root_assignment_margin=0.0,
                max_active_orthonormality_error=1e-5,
            ),
        )
        if not comp.root_density_similarities or len(comp.active_subspace_singular_values) != 2:
            raise RuntimeError(f'Unable to evaluate cross-geometry AO/1RDM metrics: {comp.reason}')
        if comp.status not in (MRPECContinuityStatus.CANDIDATE_REVIEW_REQUIRED,
                               MRPECContinuityStatus.UNRESOLVED):
            raise RuntimeError('Unexpected MR continuity status')
        print(json.dumps({
            'status': 'PASS',
            'scope': 'SOFTWARE_INTEGRATION_ONLY_NOT_SCIENTIFIC_EA',
            'pyscf_version': __import__('pyscf').__version__,
            'ao_active_mos': 'PASS', 'per_root_casci_1rdm': 'PASS',
            'ao_cross_geometry_metric': 'PASS',
            'root_density_similarities': comp.root_density_similarities,
            'active_subspace_singular_values': comp.active_subspace_singular_values,
            'continuity_status': comp.status.value,
            'root_candidate_pairs': comp.root_candidates,
            'scientific_state_identity_cleared': False,
            'mr_production_validated': False,
        }, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
