# Stage-3 numerical recovery policy

An adaptive high-level PEC can request multiple independent continuations at a
new geometry. A single numerical failure must not be confused with either
scientific non-existence of a state or successful PEC convergence.

The Stage-3 loop therefore permits a bounded retry only for numerical statuses:
`ERROR`, `SCF_NOT_CONVERGED`, and `CCSD_NOT_CONVERGED`. The retry uses the same
request provenance, basis, reference, convergence tolerances, and stability
requirements. Only SCF and CC iteration budgets may be increased.

`SCF_UNSTABLE` is not retryable by this policy because stability is a physical
validity gate rather than an iteration-budget failure. Persistent numerical
failure remains `EXECUTION_BLOCKED` and is reported with request/error evidence.

This policy does not allow a successful sibling initialization at the same R to
silently replace a failed independent continuation. Every requested
initialization must either complete or remain explicitly unresolved/blocked.
