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
| Nuclear motion / D12 | generic radial solver + Stage-3 closure orchestration | J=0 anharmonic v=0, explicit isotope masses, solver-directed electronic PEC refinement, explicitly authorized second-level PEC comparison; PEC-model bound required |

## Remaining capability gaps

### G1 / G2

G1 state completeness has a repository bridge.  G2 now also has a canonical typed attachment/continuum evidence contract in `adaptive/attachment_continuum.py`.  Molecular dissociation binding and electron binding remain separate questions: a minimum below a molecular fragmentation asymptote is not sufficient by itself to exclude a finite-basis continuum artifact.

D08 now requires reviewed attachment character, direct Delta-CC diffuse convergence, and an independent state-resolved EA-EOM diffuse series.  Diffuse, near-threshold, and continuum-like cases additionally require stabilization/continuum evidence.  The execution runner that generates the EA-EOM and scaled-diffuse stabilization evidence is still a capability gap.

The scientific-resolution layer accepts the typed D08 assessment and can now terminate `NO_PHYSICALLY_BOUND_ANION` when molecular binding exists but independent attachment/continuum evidence excludes a bound electron.

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

Nuclear motion now has a generic J=0 radial solver, production-evidence bridge, solver-directed Stage-3 PEC range/density refinement, and explicit second-level PEC model comparison/reuse. A cross-model state-identity record is still mandatory before the DeltaZPE difference is interpreted as model sensitivity. A cleared DeltaZPE does not erase omitted beyond-BO nuclear physics: `ADIABATIC_NUCLEAR_REMAINDER` remains explicit until bounded or reviewed.

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

The D12 closure loop is now connected to Stage-3 without extrapolated electronic points or automatic method escalation. The next architecture audit should prioritize the remaining *scientific* blockers rather than add another execution wrapper by default. In particular:

- G2 EA-EOM/stabilization execution runner (the evidence contract is now implemented);
- SOC relevance gate followed by a validated SOC production path where required;
- scalar-relativity two-electron/picture-change remainder policy;
- beyond-BO DBOC/non-adiabatic remainder policy;
- CBS geometry-transfer closure;
- MR production implementation.

Cross-model state identity for D12 remains an explicit evidence requirement. The current orchestration can generate/reuse the comparison PEC, but it will not convert that comparison into a model uncertainty until such identity evidence is supplied.
