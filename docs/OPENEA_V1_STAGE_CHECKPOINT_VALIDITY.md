# OpenEA v1 — Scientific checkpoint validity

A serialized result is not automatically reusable scientific evidence.

The d-aug OH validation exposed an important bug: a failed atomic fragment
(`SCF_NOT_CONVERGED`) had been serialized as a stage checkpoint.  On the next
run, the workflow loaded that failure and skipped the newly implemented SCF
retry policy entirely.

The stage checkpoint contract now distinguishes persistence from reuse:

- successful molecular PEC stages remain reusable;
- a fragment checkpoint is reusable only when its status is `COMPLETED`;
- failed or obsolete fragment checkpoints are automatically invalidated;
- failed results are still preserved in status/child JSON for auditability;
- only a newly completed fragment is written back as a reusable checkpoint.

This means solver improvements can actually be exercised on rerun while
expensive successful upstream stages remain cached.
