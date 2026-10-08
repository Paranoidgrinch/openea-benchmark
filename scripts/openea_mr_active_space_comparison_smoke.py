#!/usr/bin/env python3
"""Two real PySCF CAS partitions on LiH/STO-3G (software smoke only).

No recommended CAS size, state assignment, accuracy claim, or EA is inferred.
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
from openea_benchmark.adaptive.mr_active_space_comparison import (
    MRActiveSpaceThresholds, assess_mr_active_space_pair,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='openea-mr-cas-comparison-') as tmp:
        mol = gto.M(
            atom=[('Li',(0,0,0)),('H',(0,0,1.60))],
            basis='sto-3g', charge=0, spin=0, unit='Angstrom',
            symmetry=False, verbose=0,
        )
        hf = scf.RHF(mol)
        hf.chkfile = str(Path(tmp) / 'source.chk')
        hf.conv_tol = 1e-9
        hf.kernel()
        if not hf.converged:
            raise RuntimeError('Source HF failed to converge')
        settings = MRPointSettings(verbose=0, casscf_conv_tol=1e-7)
        authorization = MRPointAuthorization(
            True, 'Two separately approved LiH/STO-3G software-smoke calculations',
            ('SOFTWARE_SMOKE_ONLY',), max_fci_determinants=100,
        )
        common = dict(
            system='LiH_SMOKE', role='neutral', atoms=('Li','H'), charge=0,
            spin_2s=0, state_manifold_id='TWO_SINGLET_ROOTS_SOFTWARE_ONLY',
            r_angstrom=1.60, basis_label='STO-3G_SMOKE',
            basis_by_element={'Li':'sto-3g','H':'sto-3g'},
            source_root_id='SAME_HF_SOURCE', source_checkpoint_path=hf.chkfile,
            active_electrons_alpha=1, active_electrons_beta=1, nroots=2,
            state_manifold_review_ids=('ROOT_IDENTITY_NOT_VALIDATED',),
        )
        requests = (
            MRPointRequest(
                request_id='CAS_2_2', active_orbital_indices=(1,2),
                active_space_review_ids=('CAS_2_2_NOT_VALIDATED',), **common),
            MRPointRequest(
                request_id='CAS_2_3', active_orbital_indices=(1,2,3),
                active_space_review_ids=('CAS_2_3_NOT_VALIDATED',), **common),
        )
        results = []
        for req in requests:
            out = run_mr_casscf_nevpt2_point(
                req, authorization, settings=settings,
            )
            if out.status is not MRPointStatus.COMPLETE_REVIEW_REQUIRED:
                raise RuntimeError(f'{req.request_id}: {out.status}: {out.reason}')
            if out.inactive_mo_coeff_ao is None or out.active_mo_coeff_ao is None:
                raise RuntimeError('Real PySCF did not provide active and inactive AO orbitals')
            if any(root.active_rdm1 is None for root in out.roots):
                raise RuntimeError('Real PySCF did not provide CASCI root 1RDMs')
            if np.asarray(out.inactive_mo_coeff_ao).shape[1] != 1:
                raise RuntimeError('LiH CAS(2,n) must have one doubly occupied inactive orbital')
            results.append(out)
        comparison = assess_mr_active_space_pair(
            requests[0], results[0], requests[1], results[1],
            thresholds=MRActiveSpaceThresholds(
                min_density_similarity=0.2,
                min_root_assignment_margin=0.0,
                max_orbital_metric_error=1e-5,
                min_ao_metric_eigenvalue=1e-10,
            ),
            left_settings=settings,
            right_settings=settings,
        )
        if not comparison.density_similarities or len(comparison.density_similarities) != 2:
            raise RuntimeError(f'Failed to evaluate real AO-density metric: {comparison.reason}')
        if any((comparison.root_identity_validated,
                comparison.active_space_converged,
                comparison.method_uncertainty_bounded,
                comparison.mr_production_validated)):
            raise RuntimeError('Software smoke must never certify active-space convergence')
        print(json.dumps({
            'status': 'PASS',
            'scope': 'SOFTWARE_INTEGRATION_ONLY_NOT_SCIENTIFIC_EA',
            'pyscf_version': __import__('pyscf').__version__,
            'cas_models': [
                {'request_id': req.request_id, 'ncas': req.ncas,
                 'ncore_orbitals': len(out.inactive_mo_coeff_ao[0]),
                 'nevpt2_energies_hartree': [r.sc_nevpt2_total_hartree for r in out.roots]}
                for req, out in zip(requests, results)
            ],
            'full_ao_one_particle_densities': 'PASS',
            'density_similarity_matrix': comparison.density_similarities,
            'comparison_status': comparison.status.value,
            'reason': comparison.reason,
            'candidate_root_shifts': [
                {'from_root': x.left_root, 'to_root': x.right_root,
                 'delta_nevpt2_hartree': x.sc_nevpt2_total_shift_hartree}
                for x in comparison.candidate_root_shifts
            ],
            'active_space_auto_selected': False,
            'root_identity_validated': False,
            'active_space_converged': False,
            'method_uncertainty_bounded': False,
            'mr_production_validated': False,
        }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
