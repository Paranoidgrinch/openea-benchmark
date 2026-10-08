# OpenEA v1 — D12 nuclear-motion orchestration

## Purpose

`adaptive.nuclear_motion_orchestration` closes the control loop between the generic diatomic nuclear-motion solver and the existing Stage-3 high-level PEC machinery.

The layer handles two D12 closure actions:

```text
REFINE_NUCLEAR_PEC
ASSESS_NUCLEAR_PEC_MODEL_CONVERGENCE
```

It does not change the nuclear Hamiltonian and it does not introduce a new electronic-structure method.

## REFINE_NUCLEAR_PEC

A `NuclearMotionAssessment` may request more electronic PEC evidence because:

- the electronic minimum lies at a sampled boundary;
- the radial wavefunction lacks adequate lower-/upper-R boundary clearance;
- the finite-domain test is sensitive to the current PEC range;
- the vibrational level is too sensitive to interpolation on the current electronic grid.

The orchestration layer translates only those diagnostics into new Stage-3 geometries:

- requested lower/upper domain extensions use the explicit `Stage3RefinementSettings.extension_step_angstrom`;
- interpolation sensitivity triggers midpoint bisection of the existing PEC intervals;
- no potential point outside the calculated electronic PEC is extrapolated;
- every new electronic point is represented by a normal `Stage3ExecutionRequest`.

The existing Stage-3 high-level checkpoint is used as the electronic continuation seed. After the batch completes, the full request/result set is sent back through the canonical Stage-3 same-geometry identity and geometry-continuity resolver. A numerically completed batch whose identity does not re-clear is not accepted as a refined nuclear PEC.

Element-specific `basis_by_element` assignments are preserved exactly when refinement requests are built.

## Checkpoint requirement

Real production refinement requires `Stage3ExecutionSettings.artifact_dir` so that each new high-level HF reference is retained as a checkpoint for the mandatory identity/continuity review.

The orchestrator fails before expensive work when that provenance cannot be retained. Test/alternative backends may inject their own resolver and checkpoint semantics explicitly.

## Electronic PEC-model convergence

Numerical convergence of the radial equation is distinct from convergence of `DeltaZPE` with respect to the electronic PEC model.

The comparison level is represented by `NuclearPECModelLevel` and must provide:

- an explicit model ID;
- explicit orbital-basis provenance, including `basis_by_element` when relevant;
- the validated Stage-3 `CCSD` / `CCSD(T)` method tuple;
- explicit authorization evidence and rationale.

OpenEA does **not** choose the comparison basis/cardinal automatically in this runner. The upstream scientific planner/precision controller must authorize it.

The comparison PEC uses the same geometries as the primary PEC and is initialized from the canonical primary high-level checkpoints. When the orbital basis changes, production execution requires explicit checkpoint projection (`checkpoint_project=True`).

The resulting comparison PEC is again subjected to the canonical Stage-3 identity/continuity review.

## Cross-model electronic-state identity

Two PECs can each be internally continuous while still describing different electronic states. Therefore matching `state_id` labels and similar energies are not enough to interpret their `DeltaZPE` difference as model sensitivity.

Before `derive_nuclear_motion_model_evidence(...)` is allowed to close the model bound, a separate cross-model state-identity evidence record must be supplied.

Without it, the result is:

```text
CROSS_MODEL_IDENTITY_REVIEW_REQUIRED
```

and no model uncertainty is created.

This is deliberately fail-closed.

## Reuse

If an explicitly authorized comparison neutral/anion PEC pair already exists, it can be supplied as existing comparison contexts. The runner reuses it and performs no new electronic calculation.

This is important because OpenEA optimizes information gain per computational cost rather than repeating already available evidence.

## Relationship to the final D12 correction

Once the primary and comparison nuclear solves are numerically cleared and cross-model state identity is evidenced, the observed `DeltaZPE` shift is converted into a conservative `NuclearMotionModelEvidence` bound.

The primary PEC is then solved again with that model evidence to obtain the final bounded Born–Oppenheimer nuclear-motion correction.

This still does not close the separate `ADIABATIC_NUCLEAR_REMAINDER` obligation for DBOC/non-adiabatic effects.
