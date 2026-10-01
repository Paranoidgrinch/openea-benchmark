# OpenEA v1 — Adaptive Stage-3 Bridge

This bridge connects the adaptive D04/D09 decision layer to the existing
`stage3_plan.build_stage3_plan()` implementation without replacing its
checkpoint provenance, local geometry grid, method/basis configuration, or
job identity logic.

## Contract fixes included

The previous PEC-selection bridge had only been exercised with synthetic
`groups`/`sectors` inputs. The real state-selection report uses
`charge_groups = {neutral, anion}` and stores retained Stage-3 seed values as
`r_candidate_angstrom` and `energy_hartree`. This patch teaches the bridge to
consume that real contract directly. Scout values remain provenance/scheduling
information only; no production uncertainty interval is manufactured.

## Release logic

The new `build_adaptive_stage3_plan()` composes:

1. existing state-selection report -> D04/D09 evidence,
2. D04/D09 evidence -> adaptive actions and complete Stage-3 seed queue,
3. existing `build_stage3_plan()` -> actual Stage-3 job payloads,
4. one-to-one contract validation and scientific release/defer decision.

Possible release states:

- `RELEASED`: high-accuracy comparison is the current action and every seed
  maps uniquely to a Stage-3 job with non-UNRESOLVED checkpoint provenance.
- `DEFERRED_PREREQUISITE`: a preceding scientific prerequisite (for example
  asymptotic PEC resolution) must be completed first.
- `BLOCKED_PROVENANCE`: one or more required Stage-3 jobs have unresolved
  checkpoint provenance.
- `BLOCKED_CONTRACT`: adaptive seeds and legacy Stage-3 jobs are not in a
  one-to-one relation, or another planning invariant is violated.
- `NOT_REQUESTED`: D04 currently does not request a high-accuracy candidate
  comparison (for example open PECs require extension first, or higher-level
  competition is already cleared).

## Scientific invariants

- DFT/scout splittings never authorize pruning.
- Every relevant bracketed candidate is retained.
- DFT/scout energies do not become high-accuracy uncertainty intervals.
- Local minimum bracketing does not clear D09 asymptotic coverage.
- Stage-2 checkpoint provenance remains owned by `stage3_plan.py`.
- A Legacy Stage-3 job absent from the adaptive seed queue blocks the bridge;
  it cannot silently bypass adaptive decision making.
- Experimental validation metadata are not used for planning.

## Tests

`tests/test_adaptive_stage3_bridge_v1.py` exercises the real `charge_groups`
contract, direct retained-seed fields, prerequisite deferral, complete
candidate release, no-pruning ordering, unresolved checkpoint provenance,
open-PEC behavior, bidirectional seed/job contract checking, and validation
metadata independence.
