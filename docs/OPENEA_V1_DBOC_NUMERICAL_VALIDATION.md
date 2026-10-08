# OpenEA v1 — Patch 25: HF DBOC numerical cross-validation (not physical certification)

## Motivation and result

Patch 24 passed the real PySCF neutral/anion smoke, but **two stable nuclear steps
only verify finite-step reproducibility**, not the physical accuracy of the
Born–Oppenheimer diagonal correction. Patch 25 validates the *quantum-metric
arithmetic* against an independent, analytically solvable Gaussian orbital,
and records a leading-order Richardson zero-step estimate without promoting
it to a certified error bar or production result.

## Closed-form one-electron Gaussian test

For a normalized primitive Cartesian s Gaussian of exponent alpha (Bohr^-2)
whose center is displaced by -h and +h along x,

    <phi(-h)|phi(+h)> = exp(-2 alpha h^2)
    F = |<phi(-h)|phi(+h)>|^2 = exp(-4 alpha h^2)
    g(h) = [1-F] / [4 h^2]  -> alpha   (h -> 0)

The new Artemis executable tests these formulae using actual PySCF
`gto.intor_cross` AO integrals, then the same determinant-fidelity and
quantum-metric functions used in the Patch-24 molecular runner. This is an
independent **factor, unit and AO-cross-center implementation check**.
It is not an independently validated molecular DBOC reference value. In
particular, a center translation of an orbital in lab coordinates must **not**
be confused with a demonstration of physical center-of-mass/recoil separation.

## Diagnostics, NOT certified uncertainties

Given two nuclear displacement steps h1 > h2, with g(h)=g0+c*h^2+O(h^4),

    g0_est = (h1^2*g(h2)-h2^2*g(h1))/(h1^2-h2^2)

The same expression is applied to HF DBOC energies (linearity in the metric)
for both charge states. Outputs retain both the original smaller-step
candidate and the Richardson estimate. Their difference is labeled
`zero_step_observed_shift_ev`: it is **not** a guaranteed uncertainty bound.
Unstable/negative zero-step intercepts are rejected rather than silently
corrected. Source request hashes and SCF settings hashes are now matched before joining neutral
and anion outcomes, preventing geometry or state changes with reused IDs.

## Still out of scope

- Correlation corrections to HF-DBOC (CCSD/CCSD(T)/MR), and whether a
  correlated DBOC is necessary for the molecule and target tolerance.
- Nonadiabatic couplings, isotope-dependent rovibrational effects and the
  correct center-of-mass/translational convention for a publication-quality
  DBOC, independently benchmarked against molecular reference data.
- Two-electron relativistic picture change; SFX2C1E does not supply it.

No G3 evidence review or terminal EA status is changed by this patch.

## Artemis smoke

```bash
cd /srv/storage/homes/analysis/dschmid/photodetachment/openea-benchmark
source .venv/bin/activate
python -m pytest -q tests/test_dboc_numerical_validation_v1.py tests/test_dboc_finite_difference_v1.py
nice -n 19 python scripts/openea_dboc_gaussian_analytic_smoke.py
nice -n 19 python scripts/openea_hf_dboc_smoke.py
```
