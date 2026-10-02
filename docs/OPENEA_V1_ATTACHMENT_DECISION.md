# OpenEA Attachment Layer v0.3 — EA Decision Engine

This layer closes the first conservative electronic-EA decision path while keeping distinct gates separate.

For neutral interval `[L_N, U_N]` and anion interval `[L_A, U_A]`:

`EA interval = [L_N - U_A, U_N - L_A]`

Policy:
- pairing not `VALID` -> `UNRESOLVED`
- anion physically `UNBOUND` -> `UNBOUND`
- anion binding unresolved -> `UNRESOLVED`
- missing energy interval -> `UNRESOLVED`
- EA upper bound <= 0 -> `UNBOUND`
- EA interval touches/crosses zero -> `UNRESOLVED`
- EA lower bound > 0 and physical gates cleared -> `BOUND`

Precision is reported separately as `TARGET_MET`, `TARGET_NOT_MET`, or `NOT_ASSESSED`.
Intervals are conservative model/evidence intervals, not frequentist confidence intervals.
No experimental EA reference is used by this layer.
