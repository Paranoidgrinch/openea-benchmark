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

A fragmentation-binding result is not automatically promoted to `PHYSICALLY_BOUND_ANION`. When an anion minimum is below molecular dissociation channels, a separately cleared electron attachment / continuum-validity review is still required.

If the anion is demonstrably not below a relevant fragmentation limit, that evidence is sufficient to establish `NO_PHYSICALLY_BOUND_ANION`.

## Early UNBOUND

`NO_PHYSICALLY_BOUND_ANION` plus cleared G1 terminates as `UNBOUND` without requiring a numerical negative EA or G3 production refinement.

## Final BOUND

`BOUND` requires the physical-boundness path plus closed G3 evidence and a closed adiabatic EA0 interval.

For a molecular final EA0, `NUCLEAR_MOTION` must be a `CLEARED`, bounded correction. `NOT_APPLICABLE` is not accepted as a shortcut for this requirement.

## Conflict policy

If G2 establishes a physically bound anion but a closed EA0 interval is wholly non-positive, the result is `UNRESOLVED` with a G2/G3 conflict reason. OpenEA must not silently privilege one inconsistent evidence stream.

## Precision

Precision target attainment remains independent:

```text
BOUND + TARGET_MET
BOUND + TARGET_NOT_MET
```

are both valid scientific `BOUND` outcomes.
