# OpenEA v1 — Production closure execution routing

## Purpose

The production-evidence bridge can identify the scientifically highest-priority
unresolved action.  This layer decides whether that action is actually
executable by the generic OpenEA core.

It does **not** make scientific decisions and it does not invent missing run
context.

## Current generic execution capabilities

| Capability | Generic execution status | Notes |
|---|---|---|
| Cardinal convergence | runner available | `adaptive_cardinal_runner.run_adaptive_cardinal_series`; requires explicitly bound molecule/state/evaluator context |
| Diffuse convergence | runner available | `adaptive_diffuse_runner.run_adaptive_diffuse_series`; requires explicitly bound context |
| CBS model closure | assessment only | CBS extrapolation exists, but arbitrary missing-evidence jobs are not yet routed universally |
| Core-valence | assessment only | generic assessment exists; current execution script is OH-specific and is not a universal production runner |
| Scalar relativity | assessment only | generic assessment exists; current execution script is OH-specific |
| CCSDT / DeltaT3 | assessment only | diagnostic semantics are generic, current execution script remains validation-specific |
| Fixed-geometry transfer | not implemented as universal adapter | Stage-3 machinery exists but no production closure adapter yet |
| Scalar-relativistic remainder | not implemented | must remain open/bounded externally until implemented |
| SOC | not implemented | explicit capability gap |
| Nuclear motion | not implemented | explicit capability gap |

## Priority rule

The dispatcher never skips an earlier closure action to run a later one.

For example:

```text
REASSESS_REFERENCE_CHARACTER
COMPUTE_NEXT_CARDINAL
ASSESS_SOC
```

must stop at reference-character reassessment.  Running the cardinal job first
would spend compute under a method whose validity is currently open.

## Explicit runtime binding

A generic runner being present is not enough to execute a job.  The runtime
must supply an `ExecutionAdapter` with the exact authorized runner ID and the
molecule/state/checkpoint/evaluator context.

This prevents the execution layer from fabricating geometry, state, basis,
charge, spin, or checkpoint provenance.

## Fail-closed rules

* OH validation scripts are not promoted to universal runners.
* CCSDTQ/T4/higher-rank automatic actions are policy-blocked.
* Review/repair actions remain diagnostic/review work, not blind calculations.
* Missing SOC/nuclear-motion support remains a visible capability gap.
