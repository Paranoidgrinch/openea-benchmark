# OpenEA v1 — PEC / State-Selection Bridge

This bridge connects the existing conservative `state_selection` / minimum-scout
semantics to adaptive diagnostics D04 (state competition) and D09 (PEC/asymptote
coverage).

## Scientific contract

1. A sampled DFT/scout minimum is provenance and scheduling evidence.
2. `bracketed_single_minimum` means a unique *discrete sampled* local minimum is
   bracketed; it is not an equilibrium-geometry proof and not a rigorous bound
   on the continuous PEC minimum.
3. `delta_from_lowest_bracketed_ev` may rank scout candidates but is not a
   high-accuracy state-energy uncertainty interval.
4. Therefore the bridge always emits `production_interval_ev=None` for scout
   candidates.
5. D04 clears only after an independent higher-level candidate-competition
   review.
6. D09 clears only after every relevant local scout component is bracketed and
   an independent PEC/asymptote review is cleared.
7. G1 is composed from independent state-search, D04 and D09 reviews; no single
   PEC/DFT result can manufacture state completeness.

This is intentionally fail-closed and read-only. It does not change existing
PEC construction, minimum scouting, state selection, Stage-3 planning or
checkpoints.
