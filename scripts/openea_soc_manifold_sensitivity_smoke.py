#!/usr/bin/env python3
"""Real FCI-SISO nested spin-manifold sensitivity smoke (software-only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pyscf import gto, scf

from openea_benchmark.adaptive.mr_casscf_nevpt2_runner import MRPointRequest
from openea_benchmark.adaptive.soc_fci_siso_runner import (
    SOCPointAuthorization, SOCPointRequest, SOCPointSettings, SOCSpinManifold,
)
from openea_benchmark.adaptive.soc_manifold_sensitivity import (
    SOCManifoldSensitivityStatus, run_soc_manifold_expansion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fci-siso-checkout', required=True)
    parser.add_argument('--neutral-only', action='store_true')
    args = parser.parse_args()
    checkout = str(Path(args.fci_siso_checkout).expanduser().resolve())
    results = []
    with TemporaryDirectory(prefix='openea-soc-manifold-') as tmp:
        for role, charge, spin, cas, nel, more_spin in (
            ('neutral', 0, 0, (1,2), (1,1), 2),
            ('anion', -1, 1, (1,2,3), (2,1), 3),
        ):
            if args.neutral_only and role == 'anion':
                continue
            mol = gto.M(atom=[('Li',(0,0,0)),('H',(0,0,1.6))],
                        basis='sto-3g',charge=charge,spin=spin,
                        symmetry=False,verbose=0,unit='Angstrom')
            mf = scf.RHF(mol) if spin==0 else scf.ROHF(mol)
            checkpoint = Path(tmp)/f'{role}_HF.chk'
            mf.chkfile=str(checkpoint)
            mf.conv_tol=1.e-9
            mf.kernel()
            if not mf.converged:
                raise RuntimeError(f'{role}: reference HF not converged')
            point=MRPointRequest(
                request_id=f'SOC_MANIFOLD_{role.upper()}',
                system='LiH_SOFTWARE_SMOKE',role=role,atoms=('Li','H'),
                charge=charge,spin_2s=spin,state_manifold_id='UNVALIDATED_SMOKE',
                r_angstrom=1.6,basis_label='STO3G_SOFTWARE',
                basis_by_element={'Li':'sto-3g','H':'sto-3g'},
                source_root_id=f'{role}_HF',source_checkpoint_path=str(checkpoint),
                active_orbital_indices=cas,active_electrons_alpha=nel[0],
                active_electrons_beta=nel[1],nroots=1,
                active_space_review_ids=('UNVALIDATED_CAS',),
                state_manifold_review_ids=('UNVALIDATED_MANIFOLD',),
            )
            small=SOCPointRequest(point,(SOCSpinManifold(spin,1),),spin,
                                   ('UNVALIDATED_GROUND',))
            expanded=SOCPointRequest(point,(SOCSpinManifold(spin,1),
                                            SOCSpinManifold(more_spin,1)),spin,
                                     ('UNVALIDATED_GROUND',))
            a=SOCPointAuthorization(True,'Independent cost authority: restricted manifold',
                                    ('SOFTWARE_SMOKE_SMALL',),
                                    max_fci_determinants=100,max_spin_orbit_states=16)
            b=SOCPointAuthorization(True,'Independent cost authority: expanded manifold',
                                    ('SOFTWARE_SMOKE_LARGE',),
                                    max_fci_determinants=100,max_spin_orbit_states=16)
            comparison=run_soc_manifold_expansion(small,expanded,a,b,
                point_settings=SOCPointSettings(fci_siso_checkout=checkout))
            if comparison.status is not SOCManifoldSensitivityStatus.CANDIDATE_REVIEW_REQUIRED:
                raise RuntimeError(f'{role}: {comparison.status.value}: {comparison.reason}')
            if comparison.soc_uncertainty_bounded or comparison.production_soc_validated:
                raise RuntimeError('Software smoke cannot close SOC uncertainty')
            results.append({
                'role':role,
                'restricted_spin_2s':[spin],
                'expanded_spin_2s':[spin,more_spin],
                'comparison_status':comparison.status.value,
                'retained_spin_free_max_drift_hartree':comparison.retained_spin_free_max_drift_hartree,
                'restricted_soc_shift_hartree':comparison.base_soc_shift_hartree,
                'expanded_soc_shift_hartree':comparison.expanded_soc_shift_hartree,
                'observed_shift_change_ev':comparison.observed_shift_change_ev,
            })
    print(json.dumps({
        'status':'PASS',
        'scope':'SOFTWARE_INTEGRATION_ONLY_NOT_SCIENTIFIC_SOC_OR_EA',
        'pyscf_version':__import__('pyscf').__version__,
        'results':results,
        'spin_free_state_identity_cleared':False,
        'spin_manifold_complete':False,
        'soc_uncertainty_bounded':False,
        'production_soc_validated':False,
    },indent=2,sort_keys=True))


if __name__=='__main__':
    main()
