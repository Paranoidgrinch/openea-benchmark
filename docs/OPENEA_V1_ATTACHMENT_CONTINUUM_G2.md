# OpenEA v1 — G2 physical boundness (simplified, risk-triggered policy)

## Canonical purpose

G2 asks whether the relevant anion ground state represents a physically bound
molecular anion. Molecular dissociation stability and electron attachment
validity are separate. G1 state completeness and the final anion `J=0,v=0`
binding check remain mandatory. A positive electronic EA in one Gaussian basis
is **not** an electron-boundness proof.

## Default: ordinary localized valence attachment

For a state **independently reviewed as** `VALENCE_BOUND`, the typed
`ValenceAttachmentEvidence` plus the existing direct `Delta-CC` diffuse
convergence and attachment-character review can clear the electron-attachment
part of G2 **without EA-EOM, exponent-scaling, CAP or scattering**.

The evidence object must carry:

1. A **strictly positive** vertical-detachment-energy **interval** including
   the relevant numerical/basis/model residual, with explicit molecular formula,
   geometry (angstrom) and identified neutral/anion state IDs, plus a named electron-detachment
   threshold and concrete calculation sources.
2. Separately **CLEARED** review of **valence localization / compactness** of
   the added electron. The mere existence of a Gaussian orbital does not
   establish locality; the review must assess the actual density/attachment
   character and basis dependence.
3. **CLEARED** SCF/reference stability and **CLEARED** state continuity.
4. A **CLEARED** direct diffuse-convergence assessment and independently
   **CLEARED** valence character.

There is deliberately **no hard-coded EA threshold**. Intervals straddling
zero, a failed density/stability/state review, or insufficient diffuse
convergence block this route. The object is a scientific evidence interface,
**not** an automatic density-analysis implementation: external reviews are
not magically validated by Python.

**Conflicting existing evidence is never suppressed**: a converged nonbinding
EA-EOM result or independent nonbinding continuum result makes G2 `UNRESOLVED`.
Unresolved scientific evidence from an already-run diagnostic must be
reconciled. Uncomputed optional diagnostics are **not** missing production
terms and are not assigned `0 ± 0`.

## Exception: genuinely diffuse, threshold or continuum-like states

For `DIFFUSE_BOUND`, `NEAR_THRESHOLD`, `CONTINUUM_LIKE` or unresolved
attachment character, G2 **does not automatically schedule a specific
method**. It returns `NEED_MORE_EVIDENCE`/`UNRESOLVED` and asks the planner
to select diagnostics appropriate to the physical regime, uncertainty and
cost. Existing EOM/root-continuity/stabilization and external continuum
modules are **optional, separately justified diagnostic capabilities**.

The previously implemented EOM + independently reviewed stabilization route
remains available if already calculated and applicable. A flat stabilization
curve alone still proves neither binding nor nonbinding; CAP resonance data
for a single root cannot prove absence of any other bound ground state.
A robust global `NO_BOUND_ATTACHMENT` continues to require resolved G1/state
completeness and appropriately scoped evidence, rather than one negative
finite-basis energy or one resonance.

## Connection to the scientific resolver

`assess_attachment_continuum(...)` returns a typed
`AttachmentContinuumAssessment`. `physical_validity_from_binding(...)`
accepts this typed result and combines it with the molecular fragmentation
assessment. **An arbitrary legacy `Review(CLEARED, ...)` is now diagnostic
provenance only and cannot close attachment G2.** Final G2 for a bound
molecular anion additionally requires the reviewed `J=0,v=0` nuclear binding
check. G2 is never an adiabatic EA calculation by itself.

## Methods and scope

- CCSD(T) + basis/diffuse convergence: normal production electronic EA.
- EOM-EA-CCSD and AO root continuity: optional state-sensitive attachment
  diagnostics; not the mandatory source of a valence EA.
- Diffuse-exponent stabilization: optional numerical diagnostic; plateau does
  not certify continuum exclusion.
- CAP-EOM / scattering: conditional specialist research diagnostics, **not
  an automatic OpenEA-v1 dependency**. The committed CAP evidence structures
  may remain archived/available; they do not enter the default path.
- CCSDT: a separate **correlation-reliability diagnostic**, not a replacement
  for G2 nor a universal production step.

See `OPENEA_V1_POST_G2_METHOD_GAPS.md` for the remaining planned production
methods and closure work.
