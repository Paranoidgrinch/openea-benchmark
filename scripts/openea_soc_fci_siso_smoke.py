#!/usr/bin/env python3
"""REAL PySCF + pinned FCI-SISO software integration, not a scientific EA.

The first smoke performs two CASCI spin sectors each for LiH and LiH- in STO-3G.
This is *not* a general SOC accuracy benchmark or active-space validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pyscf import gto, scf

from openea_benchmark.adaptive.mr_casscf_nevpt2_runner import MRPointRequest
from openea_benchmark.adaptive.soc_fci_siso_runner import (
    SOCPointAuthorization, SOCPointRequest, SOCPointSettings, SOCPointStatus,
    SOCSpinManifold, assess_soc_ea_pair, run_soc_fci_siso_point,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fci-siso-checkout', required=True,
                        help='Pinned hczhai/fci-siso checkout at e0f1031...')
    parser.add_argument('--neutral-only', action='store_true',
                        help='First verify the simplest real spin-manifold calculation')
    args = parser.parse_args()
    checkout = Path(args.fci_siso_checkout).expanduser().resolve()
    results = []
    requests = []
    with TemporaryDirectory(prefix='openea-soc-fci-siso-') as d:
        for role, charge, spin, active, na, nb, sectors in (
            ('neutral', 0, 0, (1,2), 1, 1, ((0,1),(2,1))),
            ('anion', -1, 1, (1,2,3), 2, 1, ((1,1),(3,1))),
        ):
            if role=='anion' and args.neutral_only:
                continue
            mol = gto.M(atom=[('Li',(0,0,0)),('H',(0,0,1.60))],
                        unit='Angstrom', basis='sto-3g',charge=charge,
                        spin=spin, symmetry=False, verbose=0)
            mf=scf.RHF(mol) if spin==0 else scf.ROHF(mol)
            chk=Path(d)/f'{role}_hf.chk'
            mf.chkfile=str(chk)
            mf.conv_tol=1e-9
            mf.kernel()
            if not mf.converged:
                raise RuntimeError(f'{role}: HF source failed')
            point = MRPointRequest(
                request_id=f'SOC_SMOKE_{role.upper()}', system='LiH_SMOKE',
                role=role, atoms=('Li','H'), charge=charge, spin_2s=spin,
                state_manifold_id='SOFTWARE_ONLY_SPIN_MANIFOLDS',
                r_angstrom=1.60, basis_label='STO3G_SOFTWARE',
                basis_by_element={'Li':'sto-3g','H':'sto-3g'},
                source_root_id=f'{role}_HF', source_checkpoint_path=str(chk),
                active_orbital_indices=active, active_electrons_alpha=na,
                active_electrons_beta=nb, nroots=1,
                active_space_review_ids=('SMOKE_UNVALIDATED_ACTIVE_SPACE',),
                state_manifold_review_ids=('SMOKE_UNVALIDATED_STATES',),
            )
            request=SOCPointRequest(
                point, tuple(SOCSpinManifold(s,n) for s,n in sectors), spin,
                ('SMOKE_GROUND_STATE_NOT_SCIENTIFICALLY_VALIDATED',),
            )
            authorization=SOCPointAuthorization(
                True, 'Small explicitly authorized SOC software smoke',
                ('SOFTWARE_ONLY',), max_fci_determinants=100,
                max_spin_orbit_states=16,
            )
            result=run_soc_fci_siso_point(
                request, authorization, settings=SOCPointSettings(
                    fci_siso_checkout=str(checkout)),
            )
            if result.status is not SOCPointStatus.COMPLETE_REVIEW_REQUIRED:
                raise RuntimeError(f'{role}: {result.status.value}: {result.reason}')
            if result.production_soc_validated or result.soc_uncertainty_bounded:
                raise RuntimeError('Software smoke cannot certify SOC accuracy')
            requests.append(request)
            results.append(result)
        candidate = (assess_soc_ea_pair(requests[0],results[0],requests[1],results[1])
                     if len(results)==2 else None)
        if candidate and candidate.status != 'CANDIDATE_REVIEW_REQUIRED':
            raise RuntimeError(f'SOC pair: {candidate.status}: {candidate.reason}')
        print(json.dumps({
            'status': 'PASS',
            'scope': 'SOFTWARE_INTEGRATION_ONLY_NOT_SCIENTIFIC_EA',
            'pyscf_version': __import__('pyscf').__version__,
            'fci_siso_revision_pinned': True,
            'point_diagnostics': [
                {'role':r.point.role, 'status':v.status.value,
                 'spin_manifolds':[(m.spin_2s,m.nroots) for m in r.spin_manifolds],
                 'spin_free_energies_hartree':[x.energy_hartree for x in v.spin_free_roots],
                 'soc_energies_hartree':list(v.soc_energies_hartree),
                 'delta_soc_hartree':v.delta_soc_hartree,
                 'backend_sha256':v.fci_siso_source_sha256}
                for r,v in zip(requests, results)
            ],
            'soc_ea_candidate_ev':candidate.delta_ea_soc_ev if candidate else None,
            'ground_state_identity_validated':False,
            'spin_manifold_complete':False,
            'soc_uncertainty_bounded':False,
            'production_soc_validated':False,
        }, indent=2, sort_keys=True))


if __name__=='__main__':
    main()
