# OpenEA v1 — Matched-model PEC correction transfer (Patch 23)

This is a **working numerical correction-transfer evaluator**, not a new quantum
chemistry method, proof of CBS convergence, or automatic production-EA closure.
It consumes completed, state-matched pairs of correlated electronic energies
from two explicitly authorized models at *each* of several bond lengths,
then evaluates their difference at independently determined neutral/anion
reference geometries.

For each charge/state, define `delta(R) = E_upper(R) - E_lower(R)`.
Within the sampled interval, OpenEA forms a piecewise linear transfer
and an independent local three-point quadratic sensitivity diagnostic.
The reported electronic-EA candidate is `delta_N(R_N) - delta_A(R_A)`;
the difference between linear and quadratic interpolation and the observed
interior leave-one-out deviations are **sensitivity diagnostics**, never
certified upper bounds.  Sampling span, state/model/provenance matching and
non-extrapolation are hard requirements.

`MatchedCorrectionPoint` must contain state-review evidence and distinct
source IDs; `matched_stage3_point` can adapt real `Stage3PointResult` energies.
Validation of both electronic levels and state continuity remains an upstream
responsibility.  The evaluator never fills missing energies, never assigns a
zero omitted term, never auto-selects models and always returns
`CANDIDATE_REVIEW_REQUIRED` (or rejects invalid inputs). It does not
replace the existing component-resolved CBS or Stage-3 PEC engines.

**Remaining gaps:** automatic choice/requesting of multi-R refinement points,
robust method-model extrapolation uncertainty, actual DBOC and beyond-BO
nuclear corrections, two-electron scalar/picture-change relativistic remainder.
SOC and MR runners exist as research/validation branches; neither is
universally certified for production across arbitrary diatomics.
