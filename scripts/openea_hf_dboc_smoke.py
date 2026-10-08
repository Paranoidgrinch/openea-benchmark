#!/usr/bin/env python3
"""Real PySCF RHF/ROHF finite-difference DBOC integration smoke (not EA)."""
from __future__ import annotations
import json
from dataclasses import replace

from openea_benchmark.adaptive.dboc_finite_difference import (
    DBOCStatus, HFDBOCRequest, HFDBOCSettings,
    run_hf_dboc_point, assess_hf_dboc_pair,
)


def main() -> None:
    # Nuclear masses, NOT default isotope-averaged atomic masses.
    # This is a software integration smoke, not a publication-quality mass table.
    spec = dict(
        atoms=('Li','H'), r_angstrom=1.60,
        basis_by_element={'Li':'sto-3g', 'H':'sto-3g'},
        state_id='LiH_smoke_hf_reference',
        nuclear_masses_amu=(7.0143579,1.007276466621),
        nuclear_mass_source='explicit approximate Li-7 nucleus/H-1 proton mass, smoke only',
        state_identity_reviewed=True,  # smoke-only assumption, NOT production G1
    )
    neutral_request = HFDBOCRequest(request_id='DBOC_HF_LIH_NEUTRAL', role='neutral',
                                    charge=0, spin_2s=0, scf_reference='RHF', **spec)
    anion_request = HFDBOCRequest(request_id='DBOC_HF_LIH_ANION', role='anion',
                                  charge=-1, spin_2s=1, scf_reference='ROHF', **spec)
    settings = HFDBOCSettings()
    neutral = run_hf_dboc_point(neutral_request, settings)
    anion = run_hf_dboc_point(anion_request, settings)
    if neutral.status is not DBOCStatus.COMPLETE_REVIEW_REQUIRED or anion.status is not DBOCStatus.COMPLETE_REVIEW_REQUIRED:
        raise RuntimeError('DBOC HF backend failed: neutral=' + neutral.reason + '; anion=' + anion.reason)
    pair = assess_hf_dboc_pair(neutral_request, anion_request, neutral, anion)
    assert pair.status is DBOCStatus.COMPLETE_REVIEW_REQUIRED
    payload = {
        'status': 'PASS',
        'scope': 'HF_DBOC_NUMERICAL_INTEGRATION_ONLY_NOT_CORRELATED_SCIENTIFIC_EA',
        'pyscf_version': __import__('pyscf').__version__,
        'neutral_dboc_hartree': neutral.dboc_by_step_hartree,
        'anion_dboc_hartree': anion.dboc_by_step_hartree,
        'dboc_ea_candidate_hartree': pair.delta_ea_hartree_candidate,
        'zero_step_ea_candidate_hartree': pair.zero_step_ea_candidate_hartree,
        'zero_step_observed_shift_ev': pair.zero_step_observed_shift_ev,
        'observed_step_sensitivity_ev': pair.observed_step_sensitivity_ev,
        'minimum_occupied_determinant_fidelity': min(a.fidelity for r in (neutral,anion) for a in r.axes),
        'number_of_real_scf_calculations': neutral.scf_count + anion.scf_count,
        'isotopic_nuclear_masses_explicit': True,
        'correlated_dboc_included': False,
        'nonadiabatic_remainder_bounded': False,
        'production_dboc_validated': False,
        'uncertainty_bounded': False,
    }
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
