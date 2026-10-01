# OpenEA Stage-3 numerical retry overlay

This overlay hardens the autonomous Stage-3 PEC loop against isolated numerical
execution failures without weakening scientific validity gates.

## Changes

- bounded per-request retry policy in `stage3_loop.py`;
- retries only `ERROR`, `SCF_NOT_CONVERGED`, and `CCSD_NOT_CONVERGED`;
- `SCF_UNSTABLE` is explicitly forbidden from automatic retry;
- retries increase SCF/CC iteration budgets only; convergence tolerances and
  stability requirements are unchanged;
- every attempt is recorded in the loop-round provenance;
- persistent failure remains `EXECUTION_BLOCKED`;
- smoke output now includes attempt history and final failed-result diagnostics.

No retry result assigns a ground state, computes a production EA, or authorizes
state pruning.
