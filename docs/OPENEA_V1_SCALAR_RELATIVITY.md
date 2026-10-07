# OpenEA v1 — Scalar relativity

For the OH validation case, OpenEA evaluates the scalar-relativistic
correction as a same-basis Hamiltonian difference

    Delta_SR(X) = EA_SFX2C1E(X) - EA_NR(X)

at fixed, already validated neutral and anion geometries.

## Hamiltonian

The relativistic member uses PySCF's spin-free one-electron X2C
(`sfx2c1e`).  This is a scalar-relativistic correction.  Spin-orbit
coupling is deliberately excluded and is handled later as a separate
OpenEA correction.

PySCF propagates the SFX2C decoration into subsequent post-SCF methods.
Its SCF-object conversion machinery also transfers the X2C decoration
when the open-shell ROHF reference is converted to UHF for UCCSD(T).

## Orbital basis

For first-row oxygen the correct relativistically recontracted
core-valence correlation-consistent family is `aug-cc-pCVXZ-DK`, not
`aug-cc-pwCVXZ-DK`.

OH therefore uses

- O: `aug-cc-pCVTZ/QZ/5Z-DK`
- H: `aug-cc-pVTZ/QZ/5Z-DK`

The weighted-core `aug-cc-pwCVXZ-DK` family does not cover first-row O;
its relevant coverage begins in the transition-metal region.

The NR and SFX2C1E members at a given cardinal number use the exact same
DK-recontracted orbital basis.  Thus the finite-basis difference is a
Hamiltonian correction and does not mix two different orbital bases.

Using a DK-recontracted basis with the X2C Hamiltonian is intentional:
the basis is scalar-relativistically recontracted and available over the
required O/H cardinal sequence.  OpenEA records this explicitly rather
than calling it an X2C-specific recontraction.

## Convergence policy

All requested TZ/QZ/5Z basis assignments are preflighted before any
expensive CCSD(T) point starts.

For each cardinal X:

1. nonrelativistic all-electron CCSD(T) neutral and anion;
2. SFX2C1E all-electron CCSD(T) neutral and anion;
3. form `Delta_SR(X)`.

The highest directly calculated correction is accepted when its change
from the preceding cardinal is at most the configured threshold
(default 0.5 meV).  Otherwise OpenEA requests the next cardinal up to
5Z and fails closed if convergence is not established.

No scalar-relativistic extrapolation is assumed in v1.

The output remains an electronic intermediate, not a production
adiabatic electron affinity.

## Generic production execution

The OH driver documented above is retained as validation provenance.  Generic
OpenEA production execution is implemented separately in
`adaptive/scalar_relativity_runner.py`; see
`OPENEA_V1_GENERIC_SCALAR_RELATIVITY_RUNNER.md`.  The generic runner requires
explicit validated state provenance and an explicit per-element
relativistically suitable basis policy and does not inherit the OH O/H basis
mapping as a universal rule.
