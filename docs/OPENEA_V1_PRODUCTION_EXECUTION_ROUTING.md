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
| Core-valence | runner available | `core_valence_runner.run_adaptive_core_valence_series`; requires validated state provenance and explicit per-element CV basis policy |
| Scalar relativity | runner available | `scalar_relativity_runner.run_adaptive_scalar_relativity_series`; matched all-electron NR/SFX2C1E CCSD(T), explicit relativistically suitable basis policy |
| CCSDT / DeltaT3 | runner available | `ccsdt_diagnostic_runner.run_adaptive_ccsdt_diagnostic_series`; explicitly authorized diagnostic only, with validated states, frozen-core definition, basis policy, source checkpoints and authorized cardinals |
| Fixed-geometry transfer | not implemented as universal adapter | Stage-3 machinery exists but no production closure adapter yet |
| Scalar-relativistic remainder | not implemented | must remain open/bounded externally until implemented |
| SOC | not implemented | explicit capability gap |
| Nuclear motion | radial solver available | `nuclear_motion.run_diatomic_nuclear_motion`; explicit masses + identity-cleared PECs + numerical settings required |
| Nuclear PEC refinement | runner available | `nuclear_motion_orchestration.run_nuclear_pec_refinement`; solver-directed Stage-3 range/density refinement with identity/continuity re-review |
| Nuclear PEC model convergence | runner available | `nuclear_motion_orchestration.run_nuclear_pec_model_convergence`; explicitly authorized second PEC level or reuse; cross-model state identity required before forming a model bound |
| Beyond-BO nuclear remainder | not implemented | DBOC/non-adiabatic residual must be bounded or explicitly reviewed; the J=0 BO solver does not set it to zero |

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

* OH validation scripts are not promoted to universal runners; generic CV/SR/CCSDT-diagnostic execution is implemented in separate molecule-independent modules.
* CCSDTQ/T4/higher-rank automatic actions are policy-blocked.
* Review/repair actions remain diagnostic/review work, not blind calculations.
* SOC remains a visible capability gap. Nuclear radial solving, solver-directed Stage-3 PEC refinement, and explicitly authorized PEC-model comparison are generic. DBOC/non-adiabatic effects remain a separate explicit `ADIABATIC_NUCLEAR_REMAINDER` obligation.
