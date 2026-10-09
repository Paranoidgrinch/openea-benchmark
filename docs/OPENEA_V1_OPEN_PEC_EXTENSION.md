# OpenEA v1 — D09 executor: open-PEC extension and DFT fragment asymptotes (Patch 30)

## Problem found by the wave-0 benchmark (2026-10-09)

Every wave-0 system ended with `next_actions = [EXTEND_OPEN_PEC, EXTEND_OPEN_PEC]` and
`stage3_release = NOT_REQUESTED`. Scout components whose sampled energy still decreased
at the edge of the bounded DFT grid kept D04/D09 `CONFIRMED` forever; nothing executed
the planner's `EXTEND_OPEN_PEC` action and no asymptote evidence existed for D09.

## What `open_pec_extension.py` does

1. **Continuation** of each open component along its decreasing direction at the scout
   DFT level (functional/basis from the scout). Each geometry is seeded from the previous
   geometry's checkpoint (adiabatic continuation of the same state); internal
   instabilities are followed. Outward steps grow from 0.05 to 1.0 Å.
2. **Fragment asymptotes** at the same DFT level: every element, charge 0 and -1, every
   2S of correct parity up to `max_atomic_2s` (default 6), three initial guesses, lowest
   energy kept. No atomic ground term is assumed. A molecular sector's asymptote is the
   lowest channel whose atomic spins can couple to its 2S (|S_A-S_B| <= S <= S_A+S_B).
3. **Classification** (sampled evidence, never pruning):
   - `MINIMUM_BRACKETED`: strict discrete minimum found -> merged into the scout
     component (points, manifolds, representative roots with the new checkpoints) and
     re-classified with the canonical `scout_local_pec_minimum`; becomes an ordinary
     Stage-3 seed.
   - `DISSOCIATIVE_TO_ASYMPTOTE`: energy flattens (< 10 meV per step beyond 2.5 Å) onto
     the computed compatible asymptote (within 0.25 eV) -> status
     `dissociative_to_asymptote`, recorded under `asymptote_resolved_components`; never a seed.
   - `UNRESOLVED`: no plateau, asymptote mismatch, SCF failure, <S^2> jump > 0.3
     (possible state switch), short-R limit, step budget exhausted.
4. **Automated D09 review** (`asymptote_review_from_extension`): `CLEARED` only when no
   state-selection component remains unresolved and a compatible asymptote exists for
   every scouted sector. The rationale is labelled `AUTOMATED_SCOUT_LEVEL_ASYMPTOTE_REVIEW`.
   D04 (higher-level competition) is not touched: with D09 cleared the planner requests
   the high-accuracy comparison and the Stage-3 bridge releases the candidate jobs.

The scout `summary.json` is never modified; the merge happens in memory and the
evidence lives in `<system>/open_pec_extension.json` (+ `open_pec_checkpoints/`),
resumable per atomic state and per component.

## Wave-0 prototype result

In a scratch prototype (same algorithm, before the module's S^2/asymptote checks) all 23
open components across the 9 systems reached a minimum or a plateau: 4 plateaus within
0.00–0.20 eV of their computed asymptote (OH 2S=3, OH- 2S=4, MgH 2S=3, MgH- 2S=4), the
rest real minima at larger R (0.8–14 eV above the lowest bracketed candidate of their
charge group; e.g. CN- quintet, ClO- triplet, SiC quintet). Two pieces (N2- doublet,
SiC singlet) continued into the minimum of an already bracketed component of the same
sector; they are kept as separate seeds (possible duplicates are cheaper than a missed
state). Sectors without any resolved scout PEC (ClO- 2S=4, N2- 2S=5) are not touched by
this executor and keep D09 open.

## CLI

`benchmark-auto` runs the extension by default after the scout
(`--no-open-pec-extension` to skip; `--auto-dry-run` only reuses existing evidence).
Because the plan changes (more seeds, released jobs), use a new `--acquisition-dir`.
