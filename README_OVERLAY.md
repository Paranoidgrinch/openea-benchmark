# OpenEA v1 starter overlay – 2026-09-30

A non-destructive additive integration package for the PUBLIC repository
`Paranoidgrinch/openea-benchmark`.

Contains:
- docs/OPENEA_V1_ARCHITECTURE.md: consolidated science/method/gating contract
- docs/OPENEA_V1_MIGRATION.md: current repo observations + safe install steps
- src/openea_benchmark/adaptive/: separate immutable decision/evidence model
- tests/test_adaptive_decision_v1.py: synthetic zero-chemistry tests

This is **not a finished molecular EA solver**. It deliberately does not infer
G1/G2/G3 from raw SCF convergence; actual providers and validated production
methods are separate next increments. No checkpoint, result or old source file
is modified by the overlay. All error intervals are evidence-dependent
estimates, not rigorous certification.
