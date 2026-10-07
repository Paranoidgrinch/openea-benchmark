# OpenEA v1 — reference-character and precision contracts

## Reference-character gate

The canonical gate outcome is one of:

- `SAFE_SINGLE_REFERENCE`
- `BORDERLINE`
- `MULTIREFERENCE_RISK`
- `UNRESOLVED` when required gate evidence is incomplete or internally invalid

The gate consumes reviewed diagnostic evidence.  It does not infer
multireference character from one raw scalar threshold.  D03 is the composite
multireference review and may be marked `CONFIRMED` only after the required
multi-indicator scientific assessment.

`SAFE_SINGLE_REFERENCE` can enter the CCSD(T) production branch directly.
`BORDERLINE` remains a single-reference candidate but is **not production-ready**
until an explicit expanded-reference-diagnostics review is `CLEARED`; after
that clearance it may enter CCSD(T) only with an enlarged reference-character
uncertainty contribution. `MULTIREFERENCE_RISK` blocks single-reference
production and enters the MR interface.  The method/basis advisor does not
select a production MR method on its own.

## Method roles

Every new method-facing contract can declare one of:

- `PRODUCTION`
- `DIAGNOSTIC`
- `REFINEMENT`
- `VALIDATION`

Validation-only methods are excluded from automatic precision-refinement
recommendations.

## Error budget and precision controller

`ErrorBudget` keeps the baseline uncertainty and physical/numerical correction
intervals separate.  Unknown components remain unknown and block cost
optimization; they are never converted to zero.

`PrecisionTargetAssessment` is independent of the scientific terminal status.
Missing a requested numerical half-width does not by itself imply
`UNRESOLVED`.

When the target is not met, `plan_precision_refinement`:

1. requires a closed error budget;
2. identifies the largest documented half-width contribution;
3. considers only scientifically valid calculations that address that term;
4. excludes `VALIDATION` calculations from automatic recommendation;
5. ranks eligible actions by expected uncertainty reduction per relative cost;
6. returns a recommendation only — it never executes a quantum-chemistry job.
