# OpenEA v1: Stage-3 high-level PEC assembly

This increment turns completed Stage-3 single-point records into a conservative
sampled high-level local PEC. It does **not** compute an electron affinity and
does **not** assign a ground state.

## Key rules

- Similar energies are not state identity evidence.
- Independent Stage-2 initializations remain distinct until a separate
  electronic-state identity review clears them as equivalent.
- Cleared duplicate initializations are not averaged; one deterministic
  numerical duplicate is retained and the full energy spread is recorded.
- Geometry-to-geometry continuity is an independent prerequisite.
- Only after continuity is cleared may a strict discrete local-minimum scout
  be evaluated.
- The scout never interpolates or claims an equilibrium geometry.
- Multiple local-minimum candidates are retained.

## High-level checkpoint persistence

`Stage3ExecutionSettings.artifact_dir` can now request persistence of the
reconverged high-level HF checkpoint. `Stage3PointResult` records the actual
checkpoint path when the file exists. These files are intended for a later
orbital/subspace identity bridge using the repository's existing checkpoint
fingerprint and continuity machinery.
