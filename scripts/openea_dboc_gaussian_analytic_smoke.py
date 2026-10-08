#!/usr/bin/env python3
"""Independent one-primitive Gaussian AO-metric benchmark for HF-DBOC numerics.

This tests the cross-center overlap / factor-of-four / zero-step extrapolation
against an exact normalized one-electron Gaussian, NOT molecular DBOC physics.
"""
from __future__ import annotations

import json
from math import exp

import numpy as np
from pyscf import gto

from openea_benchmark.adaptive.dboc_finite_difference import (
    BOHR_TO_ANGSTROM, determinant_fidelity, quantum_metric_from_fidelity,
    zero_step_richardson,
)


def main():
    alpha = 0.7  # Bohr^-2, uncontracted s Gaussian exponent
    hs = (0.008, 0.004)
    measured = []
    exact_finite_step = []
    for h in hs:
        xyz = h * BOHR_TO_ANGSTROM
        kwargs = dict(unit='Angstrom', charge=0, spin=1,
                      basis={'H': [[0, (alpha, 1.0)]]}, verbose=0)
        left = gto.M(atom=[('H', (-xyz, 0.0, 0.0))], **kwargs)
        right = gto.M(atom=[('H', (+xyz, 0.0, 0.0))], **kwargs)
        cross_ao = gto.intor_cross('int1e_ovlp', left, right)
        one = np.ones((1, 1))
        empty = np.zeros((1, 0))
        fidelity = determinant_fidelity(one, empty, one, empty, cross_ao)
        analytic_fidelity = exp(-4.0 * alpha * h * h)
        if abs(fidelity-analytic_fidelity) > 1e-11:
            raise AssertionError(f'Cross-AO Gaussian fidelity disagrees at step {h}')
        measured.append(quantum_metric_from_fidelity(fidelity, h))
        exact_finite_step.append(-np.expm1(-4.0*alpha*h*h)/(4.0*h*h))
    if not np.allclose(measured, exact_finite_step, rtol=1e-8, atol=1e-8):
        raise AssertionError('Numerical HF quantum metric disagrees with closed-form finite-step value')
    extrap = zero_step_richardson(hs, tuple(measured))
    if abs(extrap-alpha) > 1e-8:
        raise AssertionError('Zero-step Gaussian metric does not approach alpha')
    print(json.dumps({
        'status': 'PASS',
        'scope': 'ONE_ELECTRON_GAUSSIAN_AO_METRIC_ONLY_NOT_MOLECULAR_DBOC_VALIDATION',
        'pyscf_version': __import__('pyscf').__version__,
        'analytic_zero_step_metric_inverse_bohr2': alpha,
        'measured_finite_step_metric_inverse_bohr2': measured,
        'richardson_zero_step_metric_inverse_bohr2': extrap,
        'full_molecular_dboc_validated': False,
        'correlated_dboc_included': False,
        'center_of_mass_treatment_validated': False,
        'scientific_uncertainty_bounded': False,
    }, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
