# OpenEA v1 — Automated electronic-EA stage after Stage-3 (Patch 31)

## Problem found by the wave-0 benchmark (2026-10-09)

`benchmark-auto` ended after Stage-3 point acquisition with a hard-coded `UNRESOLVED`.
Refinement loop, equilibrium resolver, atomic fragments, dissociation check and the
EA interval decision existed, but were only chained by OH-specific scripts with
hand-written fragment states.

## What `electronic_ea_stage.py` does

For one system, after acquisition (enabled by default in `benchmark-auto`;
`--no-electronic-ea` to skip; also runnable on an existing acquisition with
`scripts/openea_electronic_ea_from_acquisition.py`):

1. **Refinement.** Every Stage-3 job whose acquisition review cleared initialization
   identity and geometry continuity with a bracketed sampled minimum is passed to the
   canonical `run_stage3_refinement_loop` (completed points only). The loop uses the same
   identity semantics as the acquisition review, including the axial SO(2) orbit
   equivalence for Pi/Delta initializations. New points are cached in
   `electronic_ea/refinement_points/` (resumable).
2. **Equilibrium.** Converged loops are bridged (`bridge_stage3_loop_to_attachment`) and
   `resolve_stage3_equilibrium` yields an energy interval per state.
3. **Automated high-level D04.** Per charge, the lowest resolved equilibrium interval is
   selected only if it lies below the lower bound of every other resolved candidate.
   A candidate without a resolved interval blocks, unless its lowest sampled CCSD(T)
   point lies more than `competitor_exclusion_margin_ev` (1.0 eV) above the selected
   upper bound; such exclusions are listed explicitly (`excluded_by_margin`).
4. **Atomic fragments.** For each element, charge 0 and -1, every 2S of correct parity up
   to 6 is computed at the Stage-3 CCSD(T)/basis level via `run_atomic_fragment`; the
   lowest completed state per (element, charge) defines the anion dissociation channels
   A-+B and A+B-. No atomic term symbol is assumed.
5. **Decision.** `evaluate_stage3_electronic_ea` (pairing, equilibrium, anion dissociation
   stability, EA interval) returns BOUND / UNBOUND / UNRESOLVED.

Output: `<acquisition>/electronic_ea/electronic_ea.json` and an `electronic_ea` summary in
`auto_state.json`. The overall `outcome` stays `UNRESOLVED` and `ea_computed=false`:
this is a single-basis electronic EA without CBS, core-valence, relativity, SOC or
ZPE, and the identity thresholds are still the weekend pilot values.

## Policy values

All values in `ElectronicEAStagePolicy` are explicit pilot values (refinement step
0.025 Å, bracket target 0.021 Å, equilibrium resolver as in the OH validation driver,
1.0 eV competitor margin). None is a validated production threshold.

## First real run (MgH, def2-TZVPPD, v2 acquisition)

Neutral X and MgH- singlet refined, equilibrium resolved (R = 1.734 / 1.880 Å), neutral
competition CLEARED, fragments for Mg/Mg-/H/H- over all spins computed. The anion
triplet refinement was SCIENTIFICALLY_UNRESOLVED because the loop was first wired
without the axial-symmetry identity policy; fixed by passing the same resolver as the
acquisition review.

After the fix, on the v3 acquisition (open-PEC extension applied, Stage-3 RELEASED), all
four MgH candidates converged and resolved; D04 CLEARED for both charges (neutral X 2S=1,
anion singlet), anion 2.1 eV below the lowest dissociation channel (Mg + H-), status
**BOUND**, electronic EA = 0.757 eV (def2-TZVPPD, single basis, no CBS/ZPE). This is the
first EA produced by the workflow from the two-atom manifest entry without manual steps.
