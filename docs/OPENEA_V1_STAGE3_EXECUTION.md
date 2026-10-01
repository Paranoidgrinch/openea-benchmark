# OpenEA v1 — Stage-3 single-point execution

This layer is the first OpenEA-v1 component that may perform a high-level
electronic-structure calculation.  It remains downstream of the adaptive
scientific decision/authorization path.

## Boundary

`build_stage3_execution_requests()` accepts an existing Stage-3 plan plus an
**explicit allow-list of authorized job IDs**.  Merely appearing in the legacy
Stage-3 plan is not execution authorization.

Each authorized Stage-3 job expands into

`job × retained DFT initialization × local-grid geometry`.

Thus multiple Stage-2 electronic initializations are not silently collapsed.
They remain independent high-level reference checks.

## Reference handling

* `2S = 0`: RHF -> RCCSD -> RCCSD(T).
* `2S > 0`: ROHF is reconverged and checked for internal stability.  Before CC,
  the ROHF solution is converted to a UHF representation and alpha/beta
  occupied and virtual subspaces are canonicalized explicitly.  CC then runs
  as UCCSD/UCCSD(T).
* ROHF external stability is recorded as unavailable rather than guessed.
* RHF external stability is checked by default.

The explicit open-shell semicanonicalization is intentional.  PySCF converts a
ROHF CCSD request to UCCSD, and non-iterative triples are sensitive to the
orbital representation if the open-shell reference is not semicanonical.

## Checkpoints

Stage-2 checkpoint orbitals are initial guesses only.  A new HF calculation is
performed at every Stage-3 geometry.  PySCF's checkpoint initial-guess routine
is called with `project=None` by default, so PySCF decides whether basis
projection is required.  The source root/path remains in every result record.

## Result semantics

`Stage3PointResult` records SCF convergence/stability, actual reference path,
PySCF/CC class, CCSD convergence, CCSD energy, triples correction and CCSD(T)
energy.  The ordinary PySCF T1 diagnostic is recorded only for the restricted
closed-shell path; it is not silently reinterpreted for UCCSD.

A completed point is **not**:

* an equilibrium geometry,
* a production PEC minimum,
* a ground-state assignment,
* an electron affinity,
* permission to prune another state.

Those decisions belong to later evidence/uncertainty layers.

## Manual smoke tests

The default unit tests do not launch quantum-chemistry jobs.  After integration
run the tiny H2 and OH smoke cases manually on Artemis/Theia:

```bash
nice -n 19 python scripts/stage3_execution_smoke.py --case h2
nice -n 19 python scripts/stage3_execution_smoke.py --case oh
```

H2 exercises RHF/RCCSD(T).  OH exercises ROHF -> semicanonical UHF ->
UCCSD(T).
