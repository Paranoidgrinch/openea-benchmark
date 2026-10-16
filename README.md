# OpenEA Benchmark

**Goal:** give the workflow two atoms, get back the adiabatic electron affinity (EA₀) of the
diatomic molecule they form, or `UNBOUND` if the anion is not bound. There are no manual
choices in between. The workflow finds the electronic states, decides which methods and
corrections it needs, computes them, and reports an interval with a status. No reference data
enter the calculation; experimental EAs are only used afterwards to check the result.

```bash
python -m openea_benchmark.workflow --atoms O H --output runs/OH
```

All quantum chemistry runs in open-source software: PySCF, CCpy (CCSDT/CCSDT(Q)) and
OpenMolcas (spin–orbit, multistate CASPT2).

## Pipeline

| # | Layer | What it decides | Module |
|---|-------|-----------------|--------|
| 1 | Symmetry scout | Candidate states of neutral and anion from Wigner–Witmer rules, atomic terms and DFT scans per symmetry block | `symmetry_scout`, `state_candidates`, `symmetry_states` |
| 2 | Fragment asymptotes / open curves | Dissociation limits; excludes states that lie too high | `open_pec_extension` |
| 3 | Stage-3 CCSD(T) curves (def2-TZVPPD) | High-level curves with identity, continuity and stability checks | `adaptive/stage3_*`, `benchmark_auto` |
| 4 | Reference branches | Splits a curve that jumps between ROHF solutions; their offset becomes an uncertainty | `reference_branches` |
| 5 | Electronic EA | Minima, ground-state competition, anion binding against the fragments | `electronic_ea_stage` |
| 6 | CBS limit | Frozen-core CCSD(T) aug-cc-pV{Q,5}Z + d-aug check; re-ranks competitors; detects continuum anions | `cbs_stage` |
| 7 | Core-valence | All-electron minus frozen-core CCSD(T) | `core_valence_stage` |
| 8 | Scalar relativity | SFX2C-1e minus non-relativistic CCSD(T) | `scalar_relativity_stage` |
| 9 | Spin–orbit | First-order CASSCF/AMFI; multistate SO-XMS-CASPT2 beyond Kr | `spin_orbit_stage`, `spin_orbit_multistate` |
| 10 | Post-CCSD(T) | CCSDT and CCSDT(Q) increments | `post_ccsd_t_stage` |
| 11 | Multireference competitors | SA-CASSCF/NEVPT2 for states CCSD(T) cannot describe; MR term gaps | `mr_competitors`, `mr_term_gap`, `mr_nuclear_curve` |
| 12 | Zero-point vibration | Vibrational problem on the curve → EA₀ = EA_e + ΔZPE | `vibrational_stage` |
| 13 | Final verdict | Combines all layers into `EA` with interval, `UNBOUND` or `UNRESOLVED` | `final_resolver` |

Every layer fails closed: if it cannot prove its part, the status says so (`UNRESOLVED`,
`NEED_MORE_EVIDENCE_*`) instead of returning a number.

## Elements

| Route | Elements | Hamiltonian / basis |
|---|---|---|
| All-electron | H – Kr | non-relativistic + SR/SO corrections, Dunning series |
| Pseudopotential | Y – Xe, Hf – Rn | cc-pVnZ-PP series with ECP |
| All-electron X2C | Rb, Sr, Cs, Ba, La – Lu, Fr – Lr | spin-free X2C, cc-pVnZ-X2C series |
| Not supported | Z > 103 | fails closed with `ELEMENT_NOT_SUPPORTED` |

Atomic fragments of d/f elements get configuration control (every low configuration is
targeted, CCSD(T) on each) and a multistate layer (SO-XMS-CASPT2) for the fragment ground level.

## Results (reference set, fully automatic)

| Molecule | EA₀ OpenEA (eV) | Experiment (eV) | Δ (eV) |
|---|---|---|---|
| OH  | 1.828 (1.806–1.851) | 1.828 | 0.000 |
| CN  | 3.861 (3.850–3.872) | 3.862 | −0.001 |
| NO  | 0.022 (0.010–0.034) | 0.026 | −0.004 |
| O₂  | 0.444 (0.430–0.459) | 0.448 | −0.004 |
| ClO | 2.279 (2.260–2.299) | 2.276 | +0.003 |
| AlO | 2.613 (2.539–2.688) | 2.60  | +0.01 |
| SiC | 2.374 (2.352–2.396) | 2.396 | −0.022 |
| C₂  | 3.286 (3.267–3.306) | 3.269 | +0.017 |
| MgH | 0.877 (0.872–0.881) | 0.90 ± 0.05 | −0.02 |
| N₂  | `UNBOUND` | unbound | ✓ |

Running: TiO (3d), Ag₂ (pseudopotential), La atom (configuration/multistate layer).

## Repository

- `src/openea_benchmark/workflow.py`: entry point.
- `src/openea_benchmark/`: one module per layer (table above).
- `src/openea_benchmark/adaptive/`: Stage-3 execution, identity checks, nuclear motion.
- `src/openea_benchmark/attachment/`: atomic fragments, CBS extrapolation, EA assembly.
- `scripts/`: memory watchdog, validation scripts.
- `docs/`: design notes per layer, see `docs/README.md`.
- `tests/`: unit tests (`pytest`), no quantum-chemistry jobs.

Calculation output and caches are never committed (see `.gitignore`). The state before the
October 2026 cleanup (old DFT-scout pipeline, evidence/planning layers, FeH pilots) is
tagged `archive/pre-cleanup-2026-10-16`.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e .[dev]
.venv/bin/pytest
```

License: GPL-3.0-only.
