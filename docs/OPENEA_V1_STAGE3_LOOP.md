# OpenEA v1 — Autonomous Stage-3 PEC refinement loop

This layer closes the control loop between four existing components:

1. Stage-3 point execution,
2. high-level HF electronic-state identity/continuity resolution,
3. guarded high-level PEC assembly,
4. adaptive geometry refinement.

The loop is intentionally **not** an EA engine. It cannot assign a ground state,
compute a production electron affinity, or prune competing electronic states.

## Terminal states

- `CONVERGED`: the explicit sampled-bracket-width target is met after the latest
  identity and continuity review.
- `SCIENTIFICALLY_UNRESOLVED`: identity/continuity or geometric evidence is not
  sufficient to schedule another defensible refinement step.
- `EXECUTION_BLOCKED`: at least one newly scheduled Stage-3 point failed
  numerically.
- `ROUND_LIMIT_REACHED`: the explicit calculation-batch budget was exhausted
  while refinement was still scientifically actionable. This is **not**
  convergence.

## Loop semantics

For each evaluation:

`requests/results -> HF identity + continuity -> guarded PEC -> refinement plan`

If more points are needed:

`refinement plan -> new requests -> CCSD(T) execution -> append provenance -> repeat`

Interior refinement points continue independently from both neighboring
high-level HF checkpoints. A new geometry therefore cannot silently inherit a
single branch simply because that branch has the lower energy.

## Threshold policy

No production values are defined in this module for state identity, branch
continuity, geometry refinement, or duplicate-energy tolerance. Those remain
explicit protocol inputs. The smoke script contains validation-only values.
