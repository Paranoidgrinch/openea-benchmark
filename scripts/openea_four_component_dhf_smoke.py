#!/usr/bin/env python3
"""Actual PySCF 4c-DHF Coulomb/Gaunt/Breit integration smoke; NOT production EA."""
from __future__ import annotations
import argparse
import json

from openea_benchmark.adaptive.four_component_two_electron_probe import (
    FourComponentPointRequest, FourComponentSettings, FourComponentStatus,
    assess_four_component_pair, run_four_component_point,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--neutral-only', action='store_true',
                   help='Check the even-electron 4c implementation before the odd-electron anion')
    args = p.parse_args()
    base = dict(atoms=('Li', 'H'), r_angstrom=1.60,
                basis_by_element={'Li': 'sto-3g', 'H': 'sto-3g'},
                state_identity_reviewed=True)  # Explicit smoke assumption, NOT validated G1.
    neutral_req = FourComponentPointRequest(
        request_id='FOURC_LIH_NEUTRAL', role='neutral', charge=0, spin_2s=0,
        state_id='LiH_SINGLET_SMOKE', source_state_review_id='SMOKE_NEUTRAL_ASSUMPTION', **base)
    anion_req = FourComponentPointRequest(
        request_id='FOURC_LIH_ANION', role='anion', charge=-1, spin_2s=1,
        state_id='LiH_ANION_DOUBLET_SMOKE', source_state_review_id='SMOKE_ANION_ASSUMPTION', **base)
    settings = FourComponentSettings(authorized=True, max_calculations=3,
                                      max_memory_mb=3500, max_cycle=150)
    n = run_four_component_point(neutral_req, settings)
    if n.status is not FourComponentStatus.COMPLETE_REVIEW_REQUIRED:
        raise RuntimeError(f'Neutral 4c-DHF failed: {n.reason}')
    result = dict(
        status='PASS', scope='DHF_4C_OPERATOR_SENSITIVITY_ONLY_NOT_PHYSICAL_EA_CORRECTION',
        neutral=n.to_dict(), three_hamiltonians_per_charge=True,
        two_electron_x2c_picture_change_computed=False,
        scalar_relativistic_remainder_validated=False, soc_double_counting_excluded=False,
        correlated_correction_included=False, production_correction_validated=False,
        uncertainty_bounded=False,
    )
    if not args.neutral_only:
        a = run_four_component_point(anion_req, settings)
        if a.status is not FourComponentStatus.COMPLETE_REVIEW_REQUIRED:
            raise RuntimeError(f'Anion 4c-DHF failed: {a.reason}')
        pair = assess_four_component_pair(neutral_req, anion_req, n, a)
        if pair.status is not FourComponentStatus.COMPLETE_REVIEW_REQUIRED:
            raise RuntimeError('4c-DHF neutral/anion comparison failed')
        result['anion'] = a.to_dict()
        result['pair'] = pair.to_dict()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
