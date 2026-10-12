# OpenEA Benchmark

**Goal:** give the workflow two atoms, get back the adiabatic electron affinity (EA) of the
diatomic molecule they form, or `UNBOUND` if the anion is not bound. No manual
choices in between. The workflow finds the electronic states, decides which methods and
corrections it needs, computes them, and reports an interval with a status. Experimental
EAs are used only to check the result afterwards, never during the calculation.

```bash
python -m openea_benchmark.workflow --mode symmetry-auto --atoms O H --output runs/OH
```

All quantum chemistry runs in open-source software: PySCF, plus CCpy for higher-order
coupled cluster.

## Pipeline

| # | Layer | What it decides | Status |
|---|-------|-----------------|--------|
| 1 | Symmetry scout | Candidate states of neutral and anion, from Wigner–Witmer rules and DFT scans per symmetry block | ✅ |
| 2 | Fragment asymptotes / open curves | Dissociation limits; excludes states that lie too high | ✅ |
| 3 | Stage-3 CCSD(T) curves (def2-TZVPPD) | High-level energies along the bond, with identity and continuity checks | ✅ |
| 4 | Reference branches | Splits a curve that jumps between two ROHF solutions; the offset between them becomes an uncertainty | ✅ |
| 5 | Electronic EA | Locates each minimum, picks the ground states from the competition between candidates, checks anion binding against fragments | ✅ |
| 6 | CBS limit | Frozen-core CCSD(T) in aug-cc-pV{Q,5}Z plus a d-aug check; re-ranks competing states; detects a continuum (unbound) anion | ✅ |
| 7 | Zero-point vibration | Solves the vibrational problem on the curve and extends the curve where the solver asks for it; checks against a QZ curve; outputs EA₀ = EA_CBS + ΔZPE | ✅ new |
| 8 | Multireference diagnosis → post-CCSD(T) | CASSCF / amplitude diagnostics decide whether CCSDT(Q)-type corrections are needed | 🚧 in progress |
| 9 | Core-valence, scalar relativity, spin–orbit | Small corrections, often 0.01–0.05 eV | ⏳ modules exist, not yet in the automated run |

Every layer fails closed: if it cannot prove its part, the status says so
(`NEED_MORE_EVIDENCE_*`, `UNRESOLVED_*`) instead of returning a number. No final verdict
is issued until layers 8 and 9 are wired in.

## Where we are (2026-10-12)

Reference set: nine diatomics, each run fully automatically from the two atoms
(frozen-core CCSD(T)/CBS, electronic, without ZPE/SOC):

| Molecule | States (neutral / anion) | EA_CBS (eV) | Experiment (eV) | Δ (eV) |
|---|---|---|---|---|
| OH  | X²Π / X¹Σ⁺   | 1.836 | 1.828 | +0.01 |
| CN  | X²Σ⁺ / X¹Σ⁺  | 3.887 | 3.862 | +0.03 |
| NO  | X²Π / X³Σ⁻   | 0.012 | 0.026 | −0.01 |
| O₂  | X³Σg⁻ / X²Πg | 0.418 | 0.448 | −0.03 |
| ClO | X²Π / X¹Σ⁺   | 2.320 | 2.276 | +0.04 |
| AlO | X²Σ⁺ / X¹Σ⁺  | 2.629 (2.57–2.69, two ROHF branches) | 2.60 | +0.03 |
| SiC | X³Π / X²Σ⁺   | 2.406 | 2.396 | +0.01 |
| MgH | X²Σ⁺ / X¹Σ⁺  | 0.847 | 0.90 ± 0.05 | −0.05 |
| N₂  | X¹Σg⁺ / X²Πg | `UNBOUND` | unbound | ✓ |

Experimental values: SiC from Yan et al., *J. Chem. Phys.* 162, 144308 (2025); MgH from
Buytendyk et al. (2014); the others are the standard photodetachment values.

All nine come out within 0.05 eV of experiment, with the correct ground states and the
correct sign. Adiabatic values (EA₀, including ZPE) are being computed now. Next:
multireference diagnosis, with C₂ as the strong-multireference test case.

## Repository

- `src/openea_benchmark/workflow.py`: entry point (`--mode symmetry-auto`).
- `src/openea_benchmark/`: one module per layer, for example `symmetry_scout.py`,
  `electronic_ea_stage.py`, `cbs_stage.py` and `vibrational_stage.py`.
- `src/openea_benchmark/adaptive/`: Stage-3 execution, identity checks, nuclear motion.
- `docs/`: design notes per layer. `OPENEA_V1_REFERENCE_BRANCHES_AND_CBS.md` covers
  layers 4, 6 and 7.
- `tests/`: unit tests (`pytest`), run without quantum-chemistry jobs.
- Older FeH-centred material (`pilots/`, `benchmarks/`, `README_*_OVERLAY.md`) is kept
  for provenance only.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e .[dev]
.venv/bin/pytest
```

License: GPL-3.0-only.
