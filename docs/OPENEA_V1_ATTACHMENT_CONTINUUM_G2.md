# OpenEA v1 — G2 electron attachment and continuum validity

## Purpose

G2 is not identical to molecular fragmentation stability.

A finite Gaussian orbital basis discretizes the one-electron continuum.  It is
therefore possible to obtain a variationally optimized anion-like solution and
a molecular minimum even when the extra electron is not physically bound in
the complete-basis limit.

OpenEA must consequently distinguish two questions:

1. is the anionic molecular state below the relevant molecular dissociation
   channels?;
2. is the extra electron itself a physically bound attachment rather than a
   finite-basis continuum pseudostate?

Only after both questions are resolved can G2 proceed toward
`PHYSICALLY_BOUND_ANION`.

## Canonical D08 evidence

Patch 11 introduces `adaptive/attachment_continuum.py` as the typed evidence
contract for the second question.

The canonical evidence streams are:

- **attachment-character review** — whether the state is valence bound,
  diffuse bound, near threshold, continuum-like, or unresolved;
- **direct Delta-CC diffuse convergence** — the independently computed
  neutral-minus-anion energy difference must be stable with respect to diffuse
  augmentation;
- **state-resolved EA-EOM-CCSD diffuse series** — an independent electron-
  attachment channel must remain consistently bound or non-binding as the
  diffuse space is enlarged;
- **stabilization / continuum scan** — mandatory for `DIFFUSE_BOUND`,
  `NEAR_THRESHOLD`, and `CONTINUUM_LIKE` cases.

A positive finite-basis Delta-CC EA alone never closes D08.

## EA-EOM sign convention

PySCF reports the EA-EOM charged-excitation eigenvalue as

```text
omega_EA = E(N+1) - E(N)
```

whereas OpenEA stores electron affinity as

```text
EA = E(N) - E(N+1)
```

so

```text
EA_OpenEA = -omega_EA
```

The helper `pyscf_eom_eigenvalue_to_attachment_ea_ev()` makes that conversion
explicit so the sign cannot be silently inverted by downstream code.

## Diffuse-series policy

The EA-EOM evidence layer requires at least three consecutive augmentation
levels before it may declare a converged bound/non-binding result.  The latest
increment must contract under an explicit numerical policy, and a conservative
residual estimate is propagated around the highest-level result.

This is a convergence estimate, not a statistical confidence interval and not
by itself a complete continuum proof.

State identity must be explicitly cleared at every point.  A root switch in an
EA-EOM diffuse series makes the attachment evidence unresolved.

## Stabilization policy

For diffuse, threshold, or continuum-like attachment, OpenEA additionally
requires an explicit stabilization/continuum scan.  Patch 11 defines the
assessment contract for a series of attachment energies under controlled
scaling of diffuse basis exponents.

Patch 11 does **not** yet generate those scaled basis sets.  This separation is
intentional: the scientific acceptance criteria are fixed before the execution
backend is implemented.

A stabilization assessment requires:

- at least three distinct scale factors;
- state/root identity continuity across the scan;
- energy stationarity within an explicit policy tolerance;
- a conservative stabilization envelope that lies wholly on one side of the
  electron-detachment threshold.

## Bound path

For a state already independently reviewed as `VALENCE_BOUND`, D08 may close
without a stabilization scan if:

- attachment character is cleared;
- direct Delta-CC diffuse convergence is cleared;
- the state-resolved EA-EOM diffuse series is converged and strictly bound.

For `DIFFUSE_BOUND` or `NEAR_THRESHOLD`, stabilization evidence is mandatory.

## Unbound path

OpenEA may conclude `NO_BOUND_ATTACHMENT` only when the state is reviewed as
`NEAR_THRESHOLD` or `CONTINUUM_LIKE` and both:

- the converged EA-EOM attachment channel is non-binding; and
- the stabilization scan is stable and non-binding.

A state previously reviewed as `VALENCE_BOUND` or `DIFFUSE_BOUND` that conflicts
with non-binding EA-EOM evidence is **UNRESOLVED**, not silently reclassified.

## Integration with final G2

`physical_validity_from_binding()` now accepts the typed
`AttachmentContinuumAssessment` while retaining legacy `Review` input for
source compatibility.

This enables an important terminal path:

```text
molecular minimum below dissociation
        +
D08 = NO_BOUND_ATTACHMENT
        ->
NO_PHYSICALLY_BOUND_ANION
        ->
UNBOUND
```

Thus a finite-basis molecular minimum cannot force a false `BOUND` result.

## What remains for the next patch

Patch 11 is an evidence-contract patch.  The main missing execution capability
is a generic D08 runner that can:

1. run neutral-reference EA-EOM-CCSD for RCCSD/UCCSD where scientifically
   applicable;
2. retain and track the target EA-EOM root across diffuse augmentation;
3. generate controlled diffuse-exponent stabilization points;
4. checkpoint/resume each subcalculation;
5. emit the typed records defined here without deciding the final scientific
   status itself.

Until that runner is validated, D08 may be supplied by external/validation
calculations but must remain unresolved when the required evidence is absent.

## Patch 12 execution bridge

`adaptive/attachment_eom_runner.py` now supplies a generic RCCSD/UCCSD
neutral-reference EA-EOM executor with explicitly permitted diffuse augmentation
points, reviewed uncontracted-exponent stabilization points, checkpoint/resume,
and *raw* EOM roots. These results **do not** automatically close D08.
Cross-basis/spin-sector root identity and stronger continuum review remain
separate scientific prerequisites. See `OPENEA_V1_GENERIC_G2_EOM_DIAGNOSTICS.md`.
