# OpenEA v1 — G2 independent continuum discrimination (Phase C, Patch 15)

## Scientific purpose and boundary

An EA-EOM eigenvalue in a finite Gaussian basis is not an asymptotic electron-boundness proof. Neither is a stationary energy under diffuse exponent scaling. Patch 14 correctly requires independently reviewed continuum evidence; Patch 15 **also prevents a free-form `Review(CLEARED, ("independent",))` from silently fulfilling that requirement**.

Patch 15 introduces a **typed out-of-band `IndependentContinuumDossier`**. A dossier is a researcher's explicit scientific attestation supported by an actual continuum-capable method, NOT an automatically generated result. Its method, root/symmetry scope, neutral parent, geometry, detachment threshold, separate raw data, state identity and methodological review all remain explicit. `assess_stabilization_series` requires a matching dossier to return `STABLE_BOUND` or `STABLE_UNBOUND`. Old free-form `continuum_discrimination_review` is accepted only for backward-compatible unresolved diagnostics; it can **never** close G2 by itself.

For this v1 contract, a **CAP-EOM-EA_CCSD** dossier may attest an identified resonance only; CAP alone cannot certify a square-integrable bound state or exhaust a sector. A dossier describing a **single temporary-anion resonance** can NEVER certify that no bound anion ground state exists. A `NO_BOUND_STATES_IN_REVIEWED_SECTOR` finding requires scope `ALL_RELEVANT_STATES`, a cleared completeness review, a named symmetry/electronic sector and an independently sourced inventory of all relevant states. A single-root resonance produces at most root-specific evidence.

Even a well-formed dossier represents a **reviewed scientific assertion**; the schema cannot verify the underlying physics. Users must cite actual raw calculations/publications and document the method's regime, uncertainties and applicability. Do not mint the dossier's CLEARED reviews from an EOM source ID, root proposal ID, CAP trajectory report or exponent stabilization profile.

## Precomputed CAP-EOM trajectories

New module: `adaptive/attachment_continuum_independent.py`. A `CAPPoint` stores external **first-order-deperturbed** complex attachment energies, after subtraction of the neutral + detached-electron channel, at an explicitly specified absorber strength `eta` and basis identifier:

- `position_ev = Re(E_anion - E_neutral - E_free_electron)` (positive is **above** detachment threshold)
- `imag_energy_ev = Im(E_anion - E_neutral)`
- `width_ev = -2 * imag_energy_ev >= 0` (the resonance width)

The latter is a diagnostic resonance width, **not** a statistical uncertainty in the electron affinity. A CAP scan should normally be optimized with respect to CAP strength and scrutinized for CAP onset/box extent, first-order deperturbation, left/right-state consistency, root tracking, basis convergence and independent threshold uncertainty. The current module does not calculate CAP complex eigenvalues, select the optimal CAP strength or certify a continuum pole.

`assess_cap_trajectory` requires at least three distinct eta points in at least two explicitly distinct basis series; it checks first-order deperturbation, external threshold/root/method review, observed position/width spans and whether the chosen method-specific, *policy-configurable* width threshold is resolved. It returns **only** `RESONANCE_CANDIDATE_REVIEW_REQUIRED`, `SUBTHRESHOLD_CANDIDATE_REVIEW_REQUIRED`, `INCONSISTENT` or `INSUFFICIENT_EVIDENCE`. Every output has `ReviewStatus.UNRESOLVED`, `boundness_decision=UNRESOLVED`, and a deterministic `G2_CAP_TRAJECTORY:<sha256>` evidence identifier. The thresholds in `CAPTrajectorySettings` are analysis controls, **not** universal chemical binding thresholds or calibrated confidence intervals.

A subthreshold CAP point with vanishing imaginary energy **cannot establish a discrete L2 bound state** in isolation. Equally, a finite width of one named metastable root does not prove there is no different bound ground-state root.

## Matched provenance requirements

`StabilizationPoint` now carries optional physical provenance (`system`, `neutral_state_id`, `source_root_id`, `r_angstrom`). The G2 EOM adapter fills these fields from the *actual* neutral source request for both scaled points and the unscaled factor-1 parent. A stabilization assessment can only consume a continuum dossier when **all** points have provenance and it matches the dossier. Imported legacy data without source provenance remains unresolved.

The `IndependentContinuumDossier.finding` has exactly three supported statements:

- `BOUND_IDENTIFIED_STATE` at `IDENTIFIED_ROOT` scope: can support a *matched positive* stabilized attachment series after out-of-band method/threshold/root review.
- `RESONANT_IDENTIFIED_STATE` at `IDENTIFIED_ROOT` scope: **never** supports a global `UNBOUND` claim, even if its EOM position is positive above threshold.
- `NO_BOUND_STATES_IN_REVIEWED_SECTOR` at `ALL_RELEVANT_STATES` scope: only with independently reviewed completeness and state inventory; may support a *matched negative* stabilized attachment series.

Contradictory signs, root/geometry/source mismatches, recycled diagnostic IDs, unreviewed methodology, incomplete sector inventories and free-form `CLEARED` review texts all leave G2 unresolved (or fail input validation). The existing Scientific Resolver, D12, G3 and production methods are unchanged.

## Methods and validation roadmap

1. Validate at least one external CAP-EOM/complex-energy implementation against known temporary anions and verify first-order deperturbed width/position convergence with respect to eta, onset and basis. Do not assume the stock PySCF `eaccsd()` includes CAP.
2. For scattering (e.g. molecular electron scattering / R-matrix), establish the open-channel threshold, symmetry blocks, bound-state search and completeness of relevant electronic states before issuing `NO_BOUND_STATES_IN_REVIEWED_SECTOR`.
3. For weak/dipole-bound anions, separately assess long-range potential, diffuse basis and nonperturbative continuum effects. Bound and resonant diagnostics may require different methods.
4. Only after numerical/experimental reference validation and exhaustive target-state identity work should an independent dossier be marked `CLEARED` and consumed by G2.

### Literature

- Zuev et al., *J. Chem. Phys.* **141**, 024102 (2014), DOI: 10.1063/1.4885056 — CAP within EOM-CC methods, resonance widths, CAP parameter and basis dependence.
- Jagau et al., *J. Phys. Chem. Lett.* **5** (2014), DOI: 10.1021/jz501515j — molecular-anion resonance PECs, deperturbed CAP-EOM EA and bound/resonance crossings.
- Jagau, *J. Chem. Phys.* **148**, 024104 (2018), DOI: 10.1063/1.5006374 — electron correlation effects on positions and widths of temporary anions.

Patch 15 contains **no** CAP electronic-structure backend and should not be represented as closing the remaining independent-continuum G2 physics gap.
