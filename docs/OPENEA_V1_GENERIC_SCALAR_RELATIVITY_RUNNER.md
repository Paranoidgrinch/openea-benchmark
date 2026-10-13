# OpenEA v1 — Generic scalar-relativity runner

The generic OpenEA single-reference scalar-relativity runner evaluates

```text
Delta_SR(X) = EA_SFX2C1E(X) - EA_NR(X)
```

using matched all-electron CCSD(T) calculations at already validated neutral
and anion states.

## Scientific contract

For a given cardinal number, the NR and SFX2C1E members must have the same:

- neutral/anion state identity and provenance;
- geometry;
- charge and spin sector;
- orbital basis assignment;
- all-electron correlation space;
- SCF/CC convergence settings;
- CCSD(T) treatment.

Only the one-electron Hamiltonian changes.  The SFX2C1E member uses PySCF's
spin-free one-electron X2C transformation.

The caller must provide an explicit per-element basis policy and certify that
it is scalar-relativistically appropriate/recontracted.  The runner does not
infer `-DK`, X2C, or other relativistic basis families from element symbols or
from a molecule name.  ECP/core-replacement models are outside this v1 runner.

The complete supplied basis series is preflighted before expensive points are
started.  Missing or unavailable basis assignments therefore fail closed.

## Adaptive convergence

The default initial series is X=3,4.  The central scalar correction is the
highest directly calculated value.  If the change between the latest two
cardinals exceeds the active threshold, the existing scalar-relativity
assessment may request X=5.  OpenEA only performs that point when an explicit
X=5 basis policy was supplied; it never invents the next basis.

No scalar-relativity CBS extrapolation is assumed in v1.

## Checkpoint/resume

Each of the four subcalculations at a cardinal number

```text
neutral NR
neutral SFX2C1E
anion   NR
anion   SFX2C1E
```

has an independent signed checkpoint record.  The signature includes state,
geometry, basis mapping, Hamiltonian and numerical settings.  Stale or
incompatible evidence is rejected rather than silently reused.

## Scope limits

A cleared generic scalar-relativity run establishes only the converged
**spin-free one-electron X2C correction**.  It does not include or close:

- spin-orbit coupling;
- two-electron relativistic terms;
- X2C picture-change remainder beyond the implemented one-electron treatment;
- nuclear motion;
- a complete production electron affinity.

Accordingly the result always records
`requires_scalar_relativistic_remainder_assessment = true`.  The separate
OpenEA `SCALAR_RELATIVITY_REMAINDER` evidence component must still be bounded
or otherwise resolved before G3d can close when it is physically relevant.

The historical `scripts/openea_oh_scalar_relativity.py` remains an OH
validation driver.  Its O/H basis choices are validation evidence, not a
universal element-to-basis rule.

## Patch 48b: NR member in the nonrelativistic partner basis

Using the Douglas-Kroll recontracted basis for the NR member was wrong: its
contraction is made for the relativistic Hamiltonian.  MgH NR SCF in cc-pVXZ-DK
rose with X (-200.0963 / -200.0944 / -200.0926 Eh, 60 mEh above the cc-pVXZ
value) and ClO's Delta_SR jumped by 0.024 eV from QZ to 5Z.  Now SFX2C1E uses
aug-cc-pVXZ-DK and NR its partner aug-cc-pVXZ (same primitives, NR contraction),
the usual practice; MgH NR SCF now falls with X (-200.1552 / -200.1565 / -200.1571).
`OPENEA_SR_BASIS_RULE_V2`; earlier SR points are recomputed.
