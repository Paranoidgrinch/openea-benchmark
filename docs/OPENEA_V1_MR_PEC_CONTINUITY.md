# OpenEA v1 — MR active-space and root continuity along a PEC (Phase C / Patch 19)

## Purpose

Given **completed, explicitly authorized** CASSCF → multi-root CASCI → SC-NEVPT2 single-geometry calculations, propose the correspondence of the *same charge/spin manifold* across adjacent geometries. The patch **does not** implement active-space auto-selection, resolve ground-state completeness, certify electronic identity, create an MR PEC or a production electron affinity, or change `MRProductionCapability`.

The MR method runner now captures two missing state-specific objects:

1. Columns of the optimized **active CASSCF/CASCI MOs in the AO basis** (`MRPointResult.active_mo_coeff_ao`).
2. The **spin-summed active-orbital CASCI one-particle RDM for each root** (`MRRootEnergy.active_rdm1`).

These are diagnostics, **not** a full many-electron overlap, biorthogonal transition density, Dyson orbital, or a proof of root identity. Distinct correlated states can have identical spin-summed 1RDMs and must remain **UNRESOLVED**.

## Intergeometry comparison

For geometries A and B, compute the exact cross-basis AO overlap `S_AB` with `pyscf.gto.intor_cross('int1e_ovlp', mol_A, mol_B)` constructed from the **declared elements, positions, and basis_by_element**. Verify `C_Aᵀ S_AA C_A = I`, `C_Bᵀ S_BB C_B = I` and the minimum singular value of the projected active-space overlap `X = C_Aᵀ S_AB C_B`.

For root `i` in A and `j` in B, the spin-summed 1RDM similarity is

```
score(i,j) = Tr[ gamma_A(i) X gamma_B(j) X.T ] /
             sqrt( Tr[gamma_A(i)^2] Tr[gamma_B(j)^2] )
```

All matrices are real spin-free orbitals in this first MR implementation. The comparison is invariant to orbital sign and consistent active-orbital basis rotations, but it is **not** a many-electron state overlap. Root indices/energies do not determine identity. Candidate pairing requires a reciprocal best match with explicit similarity, separation-margin, active-subspace and geometry-step thresholds; ties/conflicts remain `UNRESOLVED`.

Incompatible charge, electron count, target spin, manifold, CAS dimensions, incomplete CASCI roots, missing checkpoint/result signatures, missing RDM/orbital data, or invalid AO metrics **fail closed**. Missing inputs are never replaced with guessed orbitals or energy-only matches.

## Review and production policy

`CANDIDATE_REVIEW_REQUIRED` is **not** an approved electronic-state continuity review. Both `scientific_state_identity_cleared` and `mr_production_validated` are hard-coded `False`; there is no automatic G1/G2/G3 closure and no MR–SR energy addition. A subsequent scientific review needs separate root symmetry/spin/configuration, active-space sensitivity, asymptotic state coverage, and possibly higher-order wavefunction overlap / transition densities. Thresholds must be set and justified per validation domain — **no universal magic cutoff**.

## Artemis integration

Run after applying Patch 19 (which requires already-validated Patch 18 + sort_mo hotfix):

```
nice -n 19 python scripts/openea_mr_pec_continuity_smoke.py
```

This makes two real LiH/STO-3G neutral MR points at 1.60 and 1.63 Å, checks CASCI 1RDMs and AO cross metrics, and prints `SOFTWARE_INTEGRATION_ONLY_NOT_SCIENTIFIC_EA`. It is deliberately neither an EA benchmark nor evidence that the active space or states are scientifically valid.
