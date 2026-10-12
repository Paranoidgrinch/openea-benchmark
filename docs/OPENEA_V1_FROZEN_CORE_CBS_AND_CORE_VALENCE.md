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

The production runner does not hard-code one basis family.  It requires an
explicit, cardinally matched element-to-basis policy whose suitability has
already been established for the system.  Correlation-consistent core-valence
families such as `aug-cc-pwCVXZ` are one important policy where available.

The initial evidence is normally TZ/QZ. The highest-cardinal value is the
central correction and the latest change is the convergence bound. If TZ/QZ
differs by more than the configured target, the assessment requests the next
cardinal only when an explicit basis policy for that cardinal exists.

No arbitrary CV extrapolation formula is imposed in v1, and an ECP/core-
replacement model is not silently treated as an all-electron CV correction.


## OH validation policy (not a generic OpenEA rule)

For the existing OH validation calculation, H has no inner core shell.  The
explicit validation policy therefore uses a cardinally matched mixed basis:

- O: `aug-cc-pwCVTZ/QZ/5Z`
- H: `aug-cc-pVTZ/QZ/5Z`

The all-electron and frozen-core calculations at a given cardinal number use
the exact same mixed orbital basis.  Only the CC correlation space differs.
Thus `Delta_CV = EA_AE - EA_FC` remains a clean core-valence correction.

OpenEA persists both a human-readable mixed-basis label and the explicit
element-to-basis mapping as provenance.


## Generic execution layer

`openea_benchmark.adaptive.core_valence_runner` now executes the matched AE/FC
series for arbitrary validated diatomic state specifications and explicit
basis policies.  It preserves neutral/anion state provenance, supports
subpoint checkpoint/resume, and feeds the resulting cardinal points into the
existing `assess_core_valence()` convergence policy.  See
`OPENEA_V1_GENERIC_CORE_VALENCE_RUNNER.md`.

## Patch 46: automated core-valence layer

`core_valence_stage.py` runs Delta_CV in every automated run after the CBS layer
(`--no-cv` skips it):

- States: the selected neutral and anion, at their def2-TZVPPD equilibrium, seeded
  from the nearest high-level checkpoint; symmetry block (point group, irrep
  occupation) and the seed-continuous reference branch carried as in the CBS layer.
- Basis rule (`core_valence_basis.py`, per element, molecule-independent): no core
  electrons (H, He) -> aug-cc-pVXZ; otherwise the first family available at every
  needed cardinal among aug-cc-pwCVXZ, cc-pwCVXZ, aug-cc-pCVXZ, cc-pCVXZ (PySCF
  library, then the local Basis Set Exchange package); a non-aug family gets the
  diffuse primitives that aug-cc-pVXZ adds to cc-pVXZ.  Coverage today: H-Cl,
  Ga-Br (via cc-pCVXZ + diffuse); K and Ca have none -> `POLICY_BLOCKED_NO_CV_BASIS`.
- "Core" = the noble-gas core of `valence_frozen_core_orbitals`, i.e. exactly what
  the frozen-core CBS left uncorrelated.
- Convergence: `assess_core_valence` (T,Q first; X=5 if the T->Q change exceeds
  2 meV and a 5Z basis exists for all elements; central = highest X, bound = last
  change; no extrapolation).
- Assembly: EA_0 = EA_CBS(FC) + Delta_ZPE + Delta_CV only when the layer is
  `CORE_VALENCE_CLEARED`; the bound widens the interval.  Otherwise the CV status is
  recorded and the value stays frozen-core (`includes_core_valence: false`).
