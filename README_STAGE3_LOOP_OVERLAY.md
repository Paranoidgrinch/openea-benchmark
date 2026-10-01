# OpenEA Stage-3 autonomous refinement-loop overlay

Adds:

- `src/openea_benchmark/adaptive/stage3_loop.py`
- `tests/test_adaptive_stage3_loop_v1.py`
- `scripts/stage3_loop_smoke.py`
- `docs/OPENEA_V1_STAGE3_LOOP.md`

This overlay is additive and does not overwrite an existing source module.

The loop repeatedly executes new high-level points, re-runs electronic identity
and geometry continuity resolution, rebuilds the guarded high-level PEC, and
asks the adaptive refiner for the next geometry batch. It terminates with an
explicit status and never interprets numerical completion as an EA or a
pruning decision.
