# Contributing to OpenEA

OpenEA is a general workflow for electron affinities of diatomic molecules: two atoms in,
EA₀ with an interval (or `UNBOUND` / `UNRESOLVED`) out. It is under active scientific
development.

## Setup

    git clone <your-fork>
    cd openea-benchmark
    python -m venv .venv
    .venv/bin/pip install -e .[dev]
    .venv/bin/pytest

Spin–orbit and multistate layers additionally need OpenMolcas (`pymolcas` on `PATH`);
post-CCSD(T) needs CCpy.

## Development rules

- Develop on a feature branch; keep commits focused on one layer.
- Before changing code, name the layer you change (see the pipeline table in `README.md`)
  and change one scientific assumption at a time, so that it is clear why an EA changed.
- No molecule-specific special cases: a fix for one molecule must be a general rule
  (state discovery, continuity, reference choice, basis convergence, …) that holds for
  diatomics in general.
- Every layer fails closed: if it cannot prove its part, it reports a status instead of
  a number.
- Add a unit test for every rule; tests must run without quantum-chemistry jobs.

## Validation philosophy

Experimental electron affinities are validation data only. They must never be used to
select a root or state, tune a threshold, pick the result closer to experiment, or add
an empirical correction. A stable anion is never assumed: if the anion is not bound,
the result is `UNBOUND`.

## What belongs in Git

Source code, tests, documentation and small validation scripts. Calculation output,
checkpoints, integral and scratch files, logs, archives and virtual environments stay
outside Git (`.gitignore`).
