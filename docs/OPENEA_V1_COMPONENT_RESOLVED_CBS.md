# OpenEA v1 — Component-resolved CBS

This layer consumes completed QZ/5Z molecular component evidence and performs
no new electronic-structure calculation.

## Why components are separated

The SCF and correlation energies do not converge at the same rate.  In v1,
OpenEA therefore extrapolates:

1. SCF
2. CCSD correlation
3. perturbative triples `(T)`

independently for neutral and anion, and only then forms the electron
affinity.

## Primary Q/5 model

The primary `aug-cc-pV{Q,5}Z` inverse-power exponents are:

- SCF: 8.7042
- CCSD correlation: 3.2711
- `(T)`: 3.6018

The CBS form is

    E(X) = E(CBS) + A X^(-alpha)

and is solved analytically from X=4 and X=5.

## Sensitivity evidence

The primary CBS value is not averaged with alternate models.  Alternate
literature-motivated exponents and an independent combined-correlation
inverse-cubic extrapolation are used only to estimate model sensitivity.

If the maximum deviation from the primary CBS EA exceeds the configured
threshold, the layer fails closed with `NEED_MORE_EVIDENCE`.

## Diffuse correction

The cardinal series uses singly augmented QZ/5Z evidence.  The additional
diffuse correction is retained separately as

    Delta_diff = EA[d-aug-cc-pV5Z] - EA[aug-cc-pV5Z]

evaluated at the same fixed reference geometries.

The convergence-based diffuse residual estimate remains a separate
uncertainty contribution.

## Scope

The current historical OH result is an all-electron CCSD(T) calculation performed with valence-oriented aug-cc-pVnZ basis sets. It is retained as a diagnostic, not as the frozen-core production baseline. A separate frozen-core QZ/5Z/d-aug-5Z evidence run is required before a core-valence correction can be added.  It does
not yet include:

- core-valence correlation
- scalar relativity
- spin-orbit coupling
- post-CCSD(T) correlation
- zero-point / nuclear-motion effects

It is therefore not a production adiabatic electron affinity.
