# OpenEA Stage-3 bounded naming overlay

This overlay hardens adaptive Stage-3 provenance against recursive identifier
and filesystem-name growth.

## Changes

- Adds `adaptive/stage3_naming.py` as the single source of truth for bounded,
  deterministic Stage-3 refinement IDs and HF checkpoint basenames.
- Refinement request IDs retain human-readable round/geometry/seed indices but
  replace recursive parent-ID embedding with a stable SHA-256 digest.
- Full parent provenance remains in `source_root_id` and
  `source_checkpoint_path`.
- HF checkpoint basenames are independently bounded even when an upstream
  request ID is unexpectedly long.
- `Errno 36 / File name too long` and other clear contract/filesystem errors are
  classified as non-retryable by the Stage-3 loop. Numerical non-convergence
  retry behaviour is unchanged.

No scientific thresholds, energies, identity gates, or pruning rules are
changed by this overlay.
