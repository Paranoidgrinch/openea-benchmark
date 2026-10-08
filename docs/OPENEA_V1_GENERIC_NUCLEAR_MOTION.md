# OpenEA v1 — Generic diatomic nuclear-motion layer

## Scope

`openea_benchmark.adaptive.nuclear_motion` implements the OpenEA-v1 D12 nuclear-motion layer for diatomic molecules.

It is a **J=0 Born–Oppenheimer vibrational solver**. It consumes already state-resolved, continuity-cleared electronic PECs and returns the zero-point contribution

```text
Delta_nuc = ZPE(neutral) - ZPE(anion)
```

needed to convert the electronic adiabatic EA into the 0 K adiabatic quantity `EA0`.

The layer does not assign electronic states, does not run PySCF/CCpy, and does not infer isotope masses from the formula.

## Radial equation

For each electronic state OpenEA solves

```text
[-1/(2 mu) d^2/dR^2 + V(R)] u_0(R) = E_v0 u_0(R)
```

in atomic units with `J=0` and explicit reduced nuclear mass `mu`.

The reported zero-point energy is

```text
ZPE = E_v0 - min(V)
```

and the electron-affinity correction is

```text
Delta_nuc = ZPE_N - ZPE_A
```

No rotational, DBOC, non-adiabatic, or SOC term is included here. The J=0 rotational ground state needs no rotational excitation correction, but DBOC/non-adiabatic effects remain a separate physical remainder and are not silently absorbed into DeltaZPE.

## Explicit isotope/mass contract

`DiatomicMassSpecification` requires both masses in u plus provenance evidence IDs.

OpenEA does not silently choose natural-abundance masses or the most abundant isotope. This keeps isotope-specific EA work reproducible and prevents hidden mass-model changes.

## PEC contract

`NuclearMotionPEC` requires:

- explicit neutral/anion role;
- explicit state ID;
- strictly ordered positive bond lengths;
- finite absolute electronic energies;
- one explicit electronic method and basis provenance;
- identity status `CLEARED`;
- geometry-continuity status `CLEARED`;
- concrete evidence IDs.

`nuclear_motion_pec_from_high_level(...)` binds an existing identity-cleared Stage-3 `HighLevelPEC` into this contract. It refuses incomplete, mixed-method, or non-cleared Stage-3 PECs.

Neutral and anion nuclear-motion PECs must use the same electronic method and orbital-basis policy. A mixed electronic model is not silently used to form `DeltaZPE`.

## Potential representation and numerical checks

The primary potential is a shape-preserving PCHIP interpolation of the sampled high-level PEC.

An independent natural-cubic spline is used as an interpolation-sensitivity cross-check.

The J=0 finite-difference Hamiltonian is solved on nested uniform radial grids. The lowest eigenvalue is obtained from the symmetric-tridiagonal Sturm sequence rather than by constructing a large dense matrix.

A state can be `CLEARED` only when all configured checks pass:

1. coarse/fine radial-grid convergence;
2. PCHIP/natural-cubic interpolation sensitivity;
3. finite-domain sensitivity from a trimmed-domain solve;
4. adequate potential clearance at both radial boundaries;
5. an interior physical PEC minimum.

If the finite box or interpolation is not adequate, the result is fail-closed and requests PEC refinement rather than treating a box state as a molecular vibrational state.

## PEC-model convergence is separate from numerical convergence

A numerically precise solve on one electronic PEC is not sufficient to close D12.

`NuclearMotionModelEvidence` therefore supplies an explicit external bound on the electronic-PEC model sensitivity of `DeltaZPE`, for example from a cardinal/method comparison.

Without this bound the result is

```text
MODEL_CONVERGENCE_REQUIRED
```

with action

```text
ASSESS_NUCLEAR_PEC_MODEL_CONVERGENCE
```

even if both radial eigenproblems are numerically converged.

The final nuclear-motion uncertainty is conservatively formed from

```text
neutral numerical bound
+ anion numerical bound
+ DeltaZPE PEC-model bound
```

without assuming statistical independence.

## Vibrational binding of the anion

When dissociation channels are supplied, the anion absolute `v=0` interval is compared with the lowest supplied asymptote.

This matters for shallow anions: an electronic minimum can lie below dissociation while the `v=0` level lies above it.

For final G2 use, `NuclearMotionModelEvidence` must also provide a bound on the model sensitivity of the anion `v=0` dissociation margin. Without it, vibrational binding remains `UNRESOLVED`.

`physical_validity_with_nuclear_motion(...)` combines this result with the pre-existing electronic/attachment G2 evidence:

- vibrationally bound `v=0` -> final bound-anion G2 may close;
- vibrationally unbound `v=0` -> `NO_PHYSICALLY_BOUND_ANION`;
- unresolved `v=0` binding -> G2 remains unresolved.

Electronic unboundness remains an allowed early terminal result and does not require a nuclear solve.

## Production-evidence bridge

`nuclear_motion_evidence(...)` translates a cleared D12 result into the `NUCLEAR_MOTION` additive uncertainty component used by the G3 error budget.

A missing or unconverged nuclear-motion result never becomes `0 +/- 0`.

Likewise, a cleared Born-Oppenheimer `NUCLEAR_MOTION` component does **not** silently zero DBOC/non-adiabatic effects. OpenEA carries a separate `ADIABATIC_NUCLEAR_REMAINDER` physical-correction obligation. It must be bounded explicitly or reviewed as physically negligible before G3d can close.

## Stage-3 orchestration

The radial solver and its electronic-PEC closure actions are now connected to generic execution:

```text
SOLVE_NUCLEAR_MOTION
REFINE_NUCLEAR_MOTION_GRID
REFINE_NUCLEAR_PEC
ASSESS_NUCLEAR_PEC_MODEL_CONVERGENCE
```

`REFINE_NUCLEAR_PEC` uses the existing Stage-3 refinement/request/execution/identity machinery. It extends only solver-indicated boundaries and bisects existing intervals when interpolation sensitivity demands denser electronic sampling; it does not extrapolate uncalculated potential points.

`ASSESS_NUCLEAR_PEC_MODEL_CONVERGENCE` can generate or reuse an explicitly authorized second Stage-3 PEC level. The comparison basis/model is never selected implicitly by D12. A separate cross-model electronic-state-identity evidence record is required before the observed `DeltaZPE` shift can become a model-sensitivity bound. See `OPENEA_V1_NUCLEAR_MOTION_ORCHESTRATION.md`.
