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

## Patch 40 — binding, multireference screening and branch sensitivity at CBS

Findings of the v2 rerun (2026-10-11): N2- came out `CLEARED` although d-aug moved its
EA by 0.50 eV (all other systems <= 0.01 eV); SiC was 0.14 eV above experiment with
no warning; the AlO reference-branch offset (0.06 eV) was not in the CBS interval.

- **Branch sensitivity:** `reference_sensitivity_ev` of the electronic-EA record widens
  `ea_cbs_lower/upper_ev` on both sides (`reference_sensitivity_source =
  SINGLE_BASIS_REFERENCE_BRANCH_OFFSET`; the sibling branch is not recomputed at CBS).
- **Binding:** interval > 0 -> BOUND, < 0 -> UNBOUND (`ANION_UNBOUND_AT_CBS`,
  `ea_cbs_ev = None`, raw value kept as `ea_cbs_raw_ev`), else
  `NEED_MORE_EVIDENCE_BINDING`.  A diffuse correction above 0.05 eV means the anion
  follows the basis into the detachment continuum: negative EA -> UNBOUND, positive ->
  `NEED_MORE_EVIDENCE_BINDING`.
- **Multireference screening:** one extra frozen-core CCSD(T) point per selected
  state at its def2-TZVPPD equilibrium (`<job>__CBS__MRDIAG_STAGE3_BASIS`) records
  T1, D1 (closed shell only), max |T2| and the (T)/CCSD-correlation fraction.  Open
  shell T1 = sqrt((|t1a|^2+|t1b|^2)/(2 N_corr)), identical to PySCF's closed-shell
  value for a closed shell run as UHF.  Screening values (literature, not calibrated
  here): T1 > 0.02 closed / 0.044 open shell, D1 > 0.05, max |T2| > 0.1.
  Reported as `multireference_screening` / `multireference_exceeded`, **not** a
  status: on the v2 set (2026-10-12) they fire for CN (open-shell T1 0.057), ClO
  (0.047), NO (max T2 0.102) and AlO- (T1 0.030, D1 0.061), whose CBS EAs agree with
  experiment within 0.04 eV, while SiC (+0.14 eV; T1 0.056/0.051) is not separated
  from them.  The (T)/CCSD fraction is the only value that separates SiC (0.065
  neutral, 0.081 anion vs <= 0.061 elsewhere); one case is no basis for a threshold.
  A real multireference check (e.g. CASSCF natural occupations) is still missing.
- Status precedence: state order > unbound > binding undetermined > CBS model status
  (`cbs_model_status`).
- The point-cache key keeps the Patch-39 policy fields only, so new thresholds do not
  invalidate computed points.

## Patch 41 — valence frozen core (MgH)

PySCF's `CCSD.set_frozen()` freezes only the n-2 shell for Li/Be, Na/Mg and K/Ca
(Mg: 1s only).  The "frozen-core" CBS points of MgH therefore correlated Mg 2s2p in
aug-cc-pVxZ, which has no core-correlating functions: the Mg-atom correlation
energy went -0.053 / -0.065 / -0.097 Eh for T/Q/5Z, the MgH valence correlation
jumped by 40 mEh from QZ to 5Z, and EA_CBS came out 0.744 eV (MRCI+Q/CV-5Z 0.88 eV
incl. ZPE, experiment 0.90 +- 0.05 eV, Buytendyk et al. 2014).  With the Mg 2s2p
shell frozen the Mg correlation is -0.03375 / -0.03393 / -0.03399 Eh.

`valence_frozen_core_orbitals` (stage3_execution) freezes the previous noble-gas core
(plus 3d for Ga-Kr); identical to PySCF for B-Ar, so only systems with Li/Be, Na/Mg,
K/Ca or Ga-Kr change, and only their CBS point cache key gets the
`frozen_core_rule` entry.  Transition metals keep PySCF's table.
Separate check (aug-cc-pCVxZ on Mg): correlating Mg 2s2p properly changes the MgH
EA by only +0.000 (TZ) / +0.001 eV (QZ).

## Patch 42 — automated zero-point layer and adiabatic EA

`vibrational_stage.py`, run after the CBS layer (`AcquisitionSettings.zpe_stage`; CLI
on by default in `symmetry-auto`, `--no-zpe`).  Writes `nuclear_motion/nuclear_motion.json`
and `adiabatic_ea.json`:  EA_0 = EA_e(CBS) + ZPE(neutral) - ZPE(anion).

- Tool: the existing J=0 radial solver (`adaptive/nuclear_motion.py`) on the
  identity-cleared def2-TZVPPD CCSD(T) curves of the selected states (refinement
  loops handed over by the electronic-EA stage via `loops_out`).  Masses: most
  abundant isotopes (PySCF table).
- Diagnosis -> tool: the equilibrium search leaves only about +-0.05 A of curve;
  the solver reports insufficient boundary clearance / finite-domain sensitivity
  and the existing D12 orchestration (`run_nuclear_pec_refinement`) adds points
  outward (0.025 A steps = Stage-3 grid spacing, seeded from the boundary
  checkpoint, re-reviewed for identity/continuity) until the solve clears.  With
  0.04 A steps OH needed 9 rounds (0.56-1.38 A), but NO/O2/ClO failed continuity
  (minimum occupied overlap 0.946 < 0.95, AMBIGUOUS): the continuity thresholds are
  calibrated for the grid spacing.  An extension failure records the point statuses
  and the reviews in `failed_extension`.
  Because the domain check trims a fraction of the total span, both sides must
  grow; extending only the outer side was tried and does not converge.
- Model sensitivity: the curve is recomputed at the same geometries in
  def2-QZVPPD (`run_nuclear_pec_model_convergence`).  The cross-model identity
  evidence D12 demands is generated from explicit checks: no instability
  following, equilibrium bond lengths within 0.05 A (MgH-: 1.883 vs 1.858 A, a
  flat anion curve; 0.02 A rejected it), and no kink in the QZ-minus-TZ
  energy offset (each point within 0.5 mEh of the line through its neighbours; OH:
  0.16 mEh, while the offset itself varies by 6 mEh over the range).
- Assembly: interval = CBS interval shifted by Delta_ZPE and widened by its bound;
  BOUND only if the whole interval is positive and the anion v=0 lies below its
  dissociation channels; an anion unbound at CBS is not ZPE-corrected.
- OH (2026-10-12): ZPE 0.2289 / 0.2290 eV, Delta_ZPE -0.0001 +- 0.0007 eV,
  EA_0 = 1.836 [1.832, 1.840] eV, CLEARED.
- Binding check at the comparison level uses fragment asymptotes computed in the
  comparison basis (`comparison_fragments/`).  Before, the QZ curve was compared
  with TZ asymptotes; the basis shift of the total energy then looked like a
  model uncertainty of the binding margin (ClO: anion binding UNRESOLVED although
  the anion lies 1.3 eV below Cl- + O).

## Patch 43: single-reference collapse, SCF rescue

- Collapse detector (electronic-EA stage): a CCSD(T) point with max|T2| >= 1 (the
  reference determinant no longer dominates) or |(T)| >= 0.25 |E_corr(CCSD)| is not
  an energy of the labelled state.  Reference set: healthy states <= 0.34 / 0.14;
  C2 second 1Sigmag+ (3sigma_g^2 instead of 2sigma_u^2) 1.5-1.8 / 0.35-0.44, its
  CCSD(T) energy 5.8 eV *below* the true X state.
  Such a candidate gets `SINGLE_REFERENCE_COLLAPSED` and no energy floor.  It is
  excluded from the competition only if its reference determinant (lowest SCF
  energy) lies > 3 eV above that of the lowest candidate
  (`multireference_only_excluded`); otherwise it blocks with
  `MULTIREFERENCE_METHOD_REQUIRED`.  The thresholds only classify points and are
  not part of the point-cache signature.
- SCF rescue: a DIIS SCF that does not converge is continued by second-order SCF
  (`mf.newton()`, same occupations and symmetry block) from its last orbitals,
  accepted at orbital gradient 1e-5 (SOSCF stalls at a numerical floor ~1e-6 with
  diffuse bases while the energy is stationary to 1e-13; energy error ~ gradient^2).
  The point carries the warning `SCF_CONVERGED_BY_SOSCF_RESCUE`; identity and
  continuity reviews decide whether it is on the same branch.  Cached
  SCF_NOT_CONVERGED points without the rescue tag are recomputed once.
  AlO X2Sigma+ (refS) at 1.7075 A: DIIS stalls at -316.7537, SOSCF -316.7712 Eh.
