# OpenEA v1 — Canonical scientific resolution

`openea_benchmark.adaptive.scientific_resolution` is the single terminal orchestration layer for the OpenEA-v1 scientific verdict.

It does not run quantum chemistry. It consumes separately reviewed evidence.

## Inputs

- molecule identifier;
- G1 state-completeness review;
- canonical G2 `PhysicalValidityAssessment`;
- optional G3 `ProductionEvidenceBundle`;
- requested numerical half-width;
- any critical unresolved scientific questions.

## G2 contract

`PhysicalValidityStatus` has three states:

```text
PHYSICALLY_BOUND_ANION
NO_PHYSICALLY_BOUND_ANION
UNRESOLVED
```

A fragmentation-binding result is not automatically promoted to a final cleared G2. When an anion electronic minimum is below molecular dissociation channels, a separately cleared electron attachment / continuum-validity review is still required, and the anion `J=0, v=0` level must subsequently be shown to remain bound below dissociation.

`PhysicalValidityAssessment` therefore tracks whether nuclear binding has been resolved. A bound electronic/attachment candidate with pending D12 evidence is an intermediate state: it may proceed through production work, but its G2 review is not final.

`physical_validity_with_nuclear_motion(...)` folds the reviewed anion vibrational-binding result into G2. If the electronic well exists but the `v=0` level lies above dissociation, the correct physical result becomes `NO_PHYSICALLY_BOUND_ANION`.

If the anion is already demonstrably not below a relevant electronic fragmentation limit, that evidence is sufficient to establish `NO_PHYSICALLY_BOUND_ANION` without a nuclear solve.

## Early UNBOUND

`NO_PHYSICALLY_BOUND_ANION` plus cleared G1 terminates as `UNBOUND` without requiring a numerical negative EA or G3 production refinement.

## Final BOUND

`BOUND` requires the physical-boundness path plus closed G3 evidence and a closed adiabatic EA0 interval.

For a molecular final EA0, final G2 must include resolved anion `v=0` binding and `NUCLEAR_MOTION` must be a `CLEARED`, bounded `ZPE_N-ZPE_A` correction. `NOT_APPLICABLE` is not accepted as a shortcut for this requirement. Beyond-Born-Oppenheimer DBOC/non-adiabatic effects are tracked separately as `ADIABATIC_NUCLEAR_REMAINDER` and must likewise be bounded or explicitly reviewed before G3d can close.

## Conflict policy

If G2 establishes a physically bound anion but a closed EA0 interval is wholly non-positive, the result is `UNRESOLVED` with a G2/G3 conflict reason. OpenEA must not silently privilege one inconsistent evidence stream.

## Precision

Precision target attainment remains independent:

```text
BOUND + TARGET_MET
BOUND + TARGET_NOT_MET
```

are both valid scientific `BOUND` outcomes.
