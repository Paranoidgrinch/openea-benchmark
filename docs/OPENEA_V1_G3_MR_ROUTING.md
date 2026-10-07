# OpenEA v1 — G3a–G3e and multireference routing

This document records the Phase-B Patch-02 architecture contract. It does not
add a new electronic-structure production method.

## Refined energy-reliability gate

The legacy top-level `G3_ENERGY_RELIABILITY` remains accepted for checkpoint
and source compatibility. New production logic should use the component-resolved
`EnergyReliabilityGateSet`:

- `G3A_REFERENCE_METHOD_VALIDITY`
- `G3B_BASIS_DIFFUSE_CONVERGENCE`
- `G3C_CORRELATION_RELIABILITY`
- `G3D_PHYSICAL_CORRECTIONS`
- `G3E_UNCERTAINTY_CLOSURE`

When component-resolved G3 evidence is present it is authoritative. A legacy
composite `G3=CLEARED` cannot hide an open G3a–G3e subgate.

`NOT_APPLICABLE` closes a subgate only when the `Review` contains the required
physical rationale. `CONFIRMED` is a warning/problem status and never counts as
gate clearance.

## G3a — reference-method validity

`SAFE_SINGLE_REFERENCE` clears G3a for the single-reference production path.

`BORDERLINE` does **not** clear G3a by itself. A separate expanded-diagnostics
review must be `CLEARED`. Even then the production route records the requirement
to enlarge the reference-character uncertainty contribution.

`MULTIREFERENCE_RISK` never clears the single-reference G3a gate. It can
clear G3a only after the MR branch supplies an explicitly validated production
capability for the intended state/PEC/EA role.

## G3b — basis and diffuse convergence

Both the cardinal and diffuse convergence assessments must be independently
`CLEARED`. A missing assessment does not mean a zero residual.

## G3c — correlation reliability

Correlation reliability is supplied as reviewed evidence. A confirmed
post-CCSD(T) warning keeps G3c open and sends the workflow back to method/reference
validity rather than requesting automatic CCSDTQ.

## G3d — physical corrections

The caller declares the physically relevant correction inventory (for example
core-valence, scalar relativity, SOC and nuclear motion). Every declared item
must be either `CLEARED` or explicitly `NOT_APPLICABLE` with a physical
rationale. An empty inventory is unresolved, not `0 +/- 0` missing physics.

## G3e — uncertainty closure

The canonical `ErrorBudget` closes G3e only when every included component has
an explicit correction interval. Any `UNKNOWN` component keeps G3e open. For a
`BORDERLINE` reference, G3e additionally requires explicit reviewed evidence
that the enlarged reference-character uncertainty has actually been represented;
a numerically closed budget alone is insufficient.

## Planner binding

`plan_production_route` is the method-family routing contract after the
Reference Character Gate:

```text
SAFE_SINGLE_REFERENCE
    -> READY_SINGLE_REFERENCE / CCSD(T)

BORDERLINE
    -> expanded diagnostics not cleared: DIAGNOSTICS_REQUIRED
    -> expanded diagnostics cleared: READY_SINGLE_REFERENCE / CCSD(T)
       + ENLARGE_REFERENCE_CHARACTER_UNCERTAINTY

MULTIREFERENCE_RISK
    -> validated MR production capability exists: READY_MULTIREFERENCE
    -> otherwise: TERMINAL_UNRESOLVED
                  reason = MULTIREFERENCE_METHOD_REQUIRED

UNRESOLVED
    -> DIAGNOSTICS_REQUIRED
       action = RESOLVE_REFERENCE_CHARACTER
```

A `READY_*` route only authorizes entry to the corresponding production branch;
it is not itself an EA or a final scientific decision.

## MR capability contract

The presence of CASSCF, NEVPT2, OpenMolcas, or any other MR-capable software is
not sufficient to authorize MR production. `MRProductionCapability` must be
explicitly `VALIDATED_AVAILABLE` for the intended state/PEC/EA role and must
provide method-family provenance.

OpenEA-v1 otherwise fails closed:

```text
scientific_status = UNRESOLVED
reason = MULTIREFERENCE_METHOD_REQUIRED
```

This prevents both automatic high-rank single-reference escalation and an
equally unjustified automatic switch to an arbitrary multireference method.
