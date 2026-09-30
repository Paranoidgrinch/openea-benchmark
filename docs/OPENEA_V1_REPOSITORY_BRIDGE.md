# OpenEA v1 repository bridge — discovery evidence to adaptive diagnostics

This additive bridge is the first direct integration between the existing
OpenEA DFT/state-discovery infrastructure and the v1 adaptive decision layer.
It is deliberately read-only.

## Existing scientific contracts respected

The current repository already separates numerical SCF lifecycle from physical
interpretation. `SCFRootRecord` explicitly states that its status does not
encode spin purity, state identity, ground-state assignment, or BOUND/UNBOUND.
Likewise, `BranchGraph` is a continuity graph inside one fixed charge/spin/
method context and does not assign a ground state.

The bridge preserves those contracts.

## What is connected now

- `SCFRootRecord -> D01` as spin-contamination warning evidence.
- `SCFRootRecord -> D02` through the existing stability adapter.
- `BranchGraph -> D04` as branch-continuity / state-competition evidence.
- Explicit higher-level reviews -> G1 state completeness.

## Deliberate fail-closed rules

1. Low spin contamination cannot clear D01 by itself.
2. Internal SCF stability with unavailable external stability remains unresolved.
3. An internally unambiguous BranchGraph cannot clear D04 unless an explicit
   higher-level candidate-competition review is supplied.
4. No local graph can clear G1. G1 requires separate clearance of state-search
   coverage, candidate competition, and PEC/asymptotic coverage.
5. No chemistry threshold is frozen here. The D01 warning threshold is a
   caller-supplied development policy and is not a universal SR/MR boundary.

## Next integration seam

The next bridge should consume the existing local-PEC / manifold planning
objects and derive structured D09 evidence plus candidate energy intervals.
Only after that should the adaptive planner begin issuing high-level correlated
jobs.
