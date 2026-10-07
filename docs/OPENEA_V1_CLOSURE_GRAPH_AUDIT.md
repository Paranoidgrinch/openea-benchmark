# OpenEA v1 — Closure graph audit after generic CCSDT diagnostics

## Scope

This audit records the production-closure state after the architecture reset and after the generic execution work for cardinal/diffuse convergence, core-valence, scalar relativity, and CCSDT triples diagnostics.

It is an architecture/status document, not a new chemistry model.

## What is now generic and executable

| Layer | Status | Notes |
|---|---|---|
| State/manifold discovery | reusable core | Stage-2 / state-identity / manifold logic remains authoritative upstream |
| Stage-3 CCSD(T) PEC execution | generic | checkpoint-aware, state identity remains separately reviewed |
| Cardinal convergence | generic runner | runtime context must be explicitly bound |
| Diffuse convergence | generic runner | runtime context must be explicitly bound |
| Core-valence | generic runner | matched AE/FC, explicit CV basis policy, no ECP masquerading as AE |
| Scalar relativity | generic runner | matched NR/SFX2C1E, all-electron, one-electron X2C only |
| CCSDT | generic diagnostic runner | explicitly authorized DeltaT3 evidence only; no automatic CCSDTQ |
| G3a--G3e | implemented | fail-closed evidence aggregation |
| Error budget / precision controller | implemented | unknown terms remain unknown |
| Nuclear motion / D12 | generic radial solver | J=0 anharmonic v=0, explicit isotope masses, numerical/domain/interpolation checks; PEC-model bound required |

## Remaining capability gaps

### G1 / G2

G1 state completeness has a repository bridge, but G2 physical validity still depends on explicitly reviewed attachment/continuum evidence in addition to molecular dissociation binding. A minimum below a molecular fragmentation asymptote is not sufficient by itself to exclude a finite-basis continuum artifact.

The new scientific-resolution layer therefore treats G2 as an explicit `PhysicalValidityAssessment` and refuses to manufacture a bound-anion conclusion from fragmentation binding alone.

### G3b residual execution gaps

Cardinal and diffuse runners are generic, but two residual closure actions still lack a general job adapter:

- arbitrary CBS-model repair/closure;
- fixed-geometry/full-PEC transfer bounding when the needed evidence is absent.

These do not require a new electronic-structure method, but they still need explicit orchestration.

### G3d physical corrections

Core-valence and one-electron scalar relativity are now generically executable.

Still open:

- scalar-relativity two-electron / picture-change remainder;
- spin-orbit coupling;
- DBOC/non-adiabatic remainder beyond the J=0 Born-Oppenheimer vibrational solve.

Nuclear motion now has a generic J=0 radial solver and a production-evidence bridge. Remaining D12 orchestration work is explicit high-level electronic PEC extension/model-convergence generation when the solver requests more PEC evidence. A cleared DeltaZPE does not erase omitted beyond-BO nuclear physics: `ADIABATIC_NUCLEAR_REMAINDER` remains explicit until bounded or reviewed.

SOC is not yet suitable for promotion from the FeH pilot into the universal production graph. The current pilot is system-specific and should remain validation/research evidence until a method/capability policy is validated across representative states.

Nuclear motion is different: for a final molecular adiabatic EA0 it is universally relevant. It may be small, but it cannot be declared `NOT_APPLICABLE` merely to close G3d.

### Multireference branch

The MR branch contract is present and fails closed. There is still no universal validated MR production implementation. A molecule that requires MR treatment remains `UNRESOLVED` unless an explicitly validated MR capability is supplied.

## Final scientific decision gap found by the audit

Before this patch, OpenEA had:

- G1 evidence;
- G2-related attachment/binding objects;
- G3a--G3e production evidence;
- a generic interval sign classifier;

but no single authoritative orchestration layer joining these objects into the final production decision.

That creates two risks:

1. an intermediate electronic Stage-3 EA could be mistaken for a final adiabatic EA0;
2. different modules could independently emit `BOUND` / `UNBOUND` semantics without the same gate requirements.

`adaptive/scientific_resolution.py` closes this architecture gap.

## Canonical terminal logic

### Early UNBOUND

If:

- G1 state completeness is cleared; and
- G2 robustly establishes `NO_PHYSICALLY_BOUND_ANION`;

OpenEA returns:

```text
UNBOUND
```

without requiring a precise negative EA or downstream CCSD(T)/CBS physical corrections.

### BOUND path

If G2 establishes a physically bound anion candidate, the final result requires:

- G1 cleared;
- G2 cleared;
- G3a--G3e closed;
- no pending production closure actions;
- a closed additive EA uncertainty budget;
- nuclear-motion correction explicitly resolved as a bounded contribution;
- final EA0 interval strictly above zero.

A missed requested precision target remains `BOUND / TARGET_NOT_MET`.

### Scientific conflict

If G2 claims a physically bound anion while the independently closed adiabatic EA0 interval is non-positive, OpenEA returns `UNRESOLVED` rather than silently choosing one evidence stream.

## Recommended next implementation priority

The generic nuclear-motion solver is now implemented. The immediate next orchestration target should be the remaining D12 closure loop: connect `REFINE_NUCLEAR_PEC` and `ASSESS_NUCLEAR_PEC_MODEL_CONVERGENCE` to the existing Stage-3 high-level PEC machinery without inventing extrapolated points or automatic method escalation.

After that closure loop is wired, the remaining priorities should be re-audited between:

- G2 attachment/continuum execution;
- SOC relevance gate + validated SOC production path;
- scalar-relativity remainder policy;
- beyond-BO DBOC/non-adiabatic remainder policy;
- CBS geometry-transfer closure;
- MR production implementation.
