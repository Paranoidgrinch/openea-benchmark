# OpenEA v1 — Adaptive Planner v1

This layer translates reviewed D04/D09 evidence into deterministic scientific
next actions. It is intentionally not a job executor.

## Current contract

- Existing DFT/scout state splittings are **ordering hints only**.
- Every usable bracketed candidate is retained in the Stage-3 seed queue.
- No scout-derived action or seed may authorize state pruning.
- Missing bracketed-minimum provenance fails closed and triggers an audit.
- D09 prerequisite work (open PECs / asymptotes) is ordered before D04
  high-accuracy candidate comparison when both are open.
- A high-accuracy comparison request with no reproducible seed is converted to
  a state-selection completion action rather than an empty calculation.

## Why this is separate from `stage3_plan.py`

The existing Stage-3 planner already knows how to create calculation seeds from
conservative state-selection output. The adaptive layer first decides *why* a
calculation is scientifically needed and *which candidates must remain alive*.
A subsequent thin adapter can translate `Stage3SeedCandidate` records into the
existing Stage-3 planner contract without rewriting its chemistry/provenance
logic.
