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
