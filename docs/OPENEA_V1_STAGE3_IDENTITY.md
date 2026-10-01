# OpenEA v1 — Stage-3 high-level electronic identity

This layer closes the gap between the Stage-3 RHF/ROHF point executor and the
conservative high-level PEC assembler.

## Purpose

Stage-2 state identity is based on density fingerprints and occupied-subspace
continuity, but its checkpoint loaders are tailored to RKS/UKS scout roots.
Stage 3 uses RHF/ROHF references.  `adaptive/stage3_identity.py` therefore acts
as an evidence adapter: it extracts the corresponding spin-resolved HF density
and occupied spaces and then delegates the scientific classifications to the
existing OpenEA `compare_states` and `classify_branch_metrics` policies.

No production thresholds are defined here.  `IdentityThresholds` and
`BranchThresholds` are mandatory inputs to the top-level resolver.

## Same-geometry initialization review

For every completed Stage-3 request the high-level HF checkpoint is audited
against the request/result record (atoms, charge, spin, geometry, SCF energy,
electron count and spin population).  RHF/ROHF spatial orbitals are converted
to alpha/beta density matrices.  Löwdin-orthonormal densities are converted to
the existing `StateFingerprint` representation and compared with
`compare_states`.

A geometry with multiple initializations is `CLEARED` only when every pair is
classified `SAME_STATE`.  `DISTINCT_STATE`, `AMBIGUOUS`, missing checkpoints or
a failed audit produce `UNRESOLVED`.

## Geometry continuity review

Only after initialization identity is cleared is one deterministic canonical
request per geometry followed across the grid.  Adjacent RHF/ROHF occupied
alpha/beta spaces are compared with cross-geometry AO overlaps.  Principal
singular values are passed to the existing `classify_branch_metrics` policy.
Every adjacent edge must be `CONTINUOUS`; `AMBIGUOUS` and `DISCONTINUOUS` both
fail closed.

## PEC gate

`resolve_high_level_identity_and_pec()` passes both independently generated
reviews into `assemble_high_level_pec()`.  The existing PEC layer therefore
remains the gatekeeper for a discrete high-level minimum.  This layer does not
fit a continuous PEC, assign a ground state, calculate an EA, or authorize
state pruning.

## Smoke test

`scripts/stage3_identity_smoke.py` runs small STO-3G H2 or OH examples with
three geometries and a duplicate center initialization.  Its thresholds are
explicitly validation-only and are not workflow defaults.
