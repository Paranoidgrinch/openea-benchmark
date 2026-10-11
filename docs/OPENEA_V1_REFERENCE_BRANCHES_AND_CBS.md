# OpenEA v1 — ROHF reference branches (Patch 38) and automated CBS layer (Patch 39)

## Patch 38 — two ROHF reference branches in one symmetry block

Wave-0 two-atom AlO (2026-10-11): the X2Sigma+ CCSD(T) grid (def2-TZVPPD) jumped
between 1.595 and 1.62 Å.  Inside the fixed irrep occupation the ROHF equations
have two solutions (different Al->O charge transfer).  The scout seed converges to
one; from 1.62 Å on an internal instability exists and following it reached the
other solution, lower at SCF level (8-10 mEh) but *higher* at CCSD(T) (~2 mEh).
The job mixed two curves and the continuity review rejected it (fail-closed, correct).

`reference_branches.py`, wired into `run_benchmark_acquisition`
(`AcquisitionSettings.reference_branch_splitting`, default on):

- A job whose completed points mix followed and unfollowed references is split into
  `<job>__refS` (SEED_CONTINUOUS: scout seed, the converged reference is kept even if
  internally unstable; recorded as `INTERNAL_INSTABILITY_RECORDED_SEED_BRANCH_KEPT`)
  and `<job>__refF` (STABILITY_FOLLOWED: starts at the followed points and walks
  outward, each point seeded from the neighbouring high-level reference).
- Both are computed at every grid point and reviewed as separate continuous curves;
  the parent is `SUPERSEDED_BY_REFERENCE_BRANCHES`; branches identical within 1e-6 Eh
  at every point are merged.
- Electronic-EA stage: branches of one state do not compete in D04 with each other;
  the lower CCSD(T) equilibrium is used and the offset of the other branch is
  recorded as `reference_sensitivity_ev` and widens the EA interval on both sides.
  A second reference solution is itself a hint of multireference character.
- `Stage3ExecutionRequest.reference_branch` is carried into refinement/D12 requests.
- v2 rerun (2026-10-11, scout bisection moved the grid): every AlO X2Sigma+ point
  followed an instability, but to *different* solutions (SCF -316.7651/-316.7654/
  -316.7696/-316.7606/-316.7573 Eh), so nothing was "mixed" and the job stayed
  rejected.  Jobs where every point followed and the continuity review is not
  CLEARED are now split as well; the followed branch starts from the single
  lowest-SCF followed point and continues neighbour to neighbour.  Seed-branch
  points run concurrently, followed-branch points in order.

Real check (AlO X2Sigma+, R = 1.62 Å, seed branch): CCSD(T) -317.264295 Eh, smooth
with the unfollowed points (followed branch: -317.262164 Eh).

## Patch 39 — frozen-core valence CBS layer

`cbs_stage.py`, run after the electronic-EA stage (`AcquisitionSettings.cbs_stage`;
CLI on by default in `symmetry-auto`, `--no-cbs`, `--cbs-memory-mb`; also
`scripts/openea_electronic_ea_from_acquisition.py --cbs`).

- States: the selected neutral/anion plus every other resolved candidate of the same
  charge within 0.5 eV (single-basis), excluding reference-branch siblings.
- Frozen-core CCSD(T) at each state's own def2-TZVPPD equilibrium bond length in
  aug-cc-pVQZ and aug-cc-pV5Z (aug-cc-pV(X+d)Z for Al-Ar), plus d-aug at 5Z for the
  selected pair.  d-aug comes from PySCF where available, otherwise from the standard
  even-tempered construction (one more primitive a1²/a2 per angular momentum; it
  reproduces PySCF's d-aug-cc-pV{Q,5}Z for H, C, N, O exactly).
- Each point is seeded from the state's nearest high-level reference (projected, same
  irrep occupation) and keeps that reference (no following), so the basis series stays
  on one branch.
- Extrapolation: existing `component_resolved_cbs.resolve_cbs` (SCF/CCSD/(T) separate
  Q/5 exponents, sensitivity model, combined-correlation check, separate d-aug-minus-aug
  diffuse correction, NEED_MORE_EVIDENCE if model sensitivity exceeds 0.005 eV).
- Competitors are re-ranked with their primary CBS energy; any competitor not above
  the selected state gives `UNRESOLVED_STATE_ORDER_AT_CBS`.
- Each state's basis series runs in order (states concurrently on the point pool); a
  point that fails from the projected TZ reference is retried once from the previous
  completed basis of the same state with doubled cycles (`reseeded` in cbs.json).
  Seen for the NO and O2 anions in d-aug-cc-pV5Z.
- Not a production EA: no core-valence, scalar relativity, SOC, post-CCSD(T), ZPE,
  anion-binding layers; `verdict = NOT_ISSUED_NO_CV_SR_ZPE_BINDING_LAYERS`.
