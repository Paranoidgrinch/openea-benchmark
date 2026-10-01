# Stage-3 adaptive PEC refinement overlay

Adds `stage3_refinement.py` and its contract tests. This is an additive overlay;
it does not modify existing source modules.

It converts an identity-cleared high-level PEC into conservative next-geometry
proposals and, when requested, executable Stage-3 refinement requests seeded by
persisted high-level HF checkpoints.
