# OpenEA v1 Stage-3 bridge overlay

Files in this overlay:

- `src/openea_benchmark/adaptive/pec_selection_bridge.py` — updates the bridge
  to consume the real `charge_groups` state-selection schema and direct
  `r_candidate_angstrom` / `energy_hartree` seed fields.
- `src/openea_benchmark/adaptive/stage3_bridge.py` — composes adaptive planning
  with the existing `build_stage3_plan()` and releases/defer/blocks jobs
  conservatively.
- `tests/test_adaptive_stage3_bridge_v1.py` — real-contract integration tests.
- `docs/OPENEA_V1_STAGE3_BRIDGE.md` — scientific/technical contract.

No quantum-chemistry calculations are launched by this overlay.
