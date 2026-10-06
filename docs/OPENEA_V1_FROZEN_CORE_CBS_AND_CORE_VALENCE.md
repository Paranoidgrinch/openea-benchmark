# OpenEA v1 — Corrected frozen-core CBS and core-valence correction

## Corrected frozen-core CBS

The six-point frozen-core evidence set is now sufficient to rebuild the
valence CBS baseline:

- neutral/anion aug-cc-pVQZ
- neutral/anion aug-cc-pV5Z
- neutral/anion d-aug-cc-pV5Z

SCF, CCSD correlation, and `(T)` are extrapolated independently from QZ/5Z.
The frozen-core d-aug-minus-aug 5Z shift is applied as a separate diffuse
correction.

The older all-electron diffuse-axis residual may be retained only as
`INDIRECTLY_ESTIMATED` uncertainty evidence. It is not direct frozen-core
convergence evidence.

## Core-valence correction

After the frozen-core baseline is established, the additive core-valence
correction is defined in a single common core-valence basis:

    Delta_CV(X) = EA_AE(X) - EA_FC(X)

The basis family is `aug-cc-pwCVXZ`, which is specifically designed for
core-valence correlation.

The initial evidence is TZ/QZ. The highest-cardinal value is the central
correction and the latest change is the convergence bound. If TZ/QZ differs
by more than the configured target, 5Z is requested automatically.

No arbitrary CV extrapolation formula is imposed in v1.


## Mixed basis rule for OH

The `aug-cc-pwCVXZ` family is not defined for H because hydrogen has no
inner core shell.  OH therefore uses a cardinally matched mixed basis:

- O: `aug-cc-pwCVTZ/QZ/5Z`
- H: `aug-cc-pVTZ/QZ/5Z`

The all-electron and frozen-core calculations at a given cardinal number use
the exact same mixed orbital basis.  Only the CC correlation space differs.
Thus `Delta_CV = EA_AE - EA_FC` remains a clean core-valence correction.

OpenEA persists both a human-readable mixed-basis label and the explicit
element-to-basis mapping as provenance.
