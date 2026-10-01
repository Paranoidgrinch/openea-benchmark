# OpenEA v1 Stage-3 PEC overlay

Adds conservative aggregation of Stage-3 point results into a sampled
high-level local PEC and optional persistence of the reconverged high-level HF
checkpoint.

Files:
- `src/openea_benchmark/adaptive/stage3_execution.py` (updates existing file)
- `src/openea_benchmark/adaptive/stage3_pec.py`
- `tests/test_adaptive_stage3_pec_v1.py`
- `docs/OPENEA_V1_STAGE3_PEC.md`

Important: this overlay **updates an existing file**. Apply it with overwrite
(`unzip -o`), not `unzip -n`.
