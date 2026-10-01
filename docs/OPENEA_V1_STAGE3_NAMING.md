# Stage-3 identifier and artifact naming contract

Adaptive refinement must not encode its complete ancestry recursively into the
next request identifier. Provenance and identifiers serve different purposes:

- `source_root_id` and `source_checkpoint_path` retain the full explicit parent
  provenance required for auditing.
- `request_id` is a stable bounded key derived from job, refinement round,
  geometry index, seed index, target geometry, and a digest of the complete
  parent identity.
- HF checkpoint filenames are independently bounded and include a digest of the
  full request ID.

This prevents path-length failures after several adaptive refinement rounds
without discarding any scientific provenance.

The retry layer also treats deterministic contract/filesystem failures such as
`File name too long`, missing files, permission failures, assertion failures,
and invalid arguments as non-retryable. Increasing SCF/CC iteration limits
cannot resolve those failures. Scientific `SCF_UNSTABLE` remains non-retryable
as before.
