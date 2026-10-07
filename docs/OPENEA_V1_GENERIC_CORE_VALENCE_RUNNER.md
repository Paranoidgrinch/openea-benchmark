# OpenEA v1 — Generic core-valence production runner

## Purpose

OpenEA now has a molecule-independent execution layer for the additive
core-valence correction on the validated single-reference branch.

The correction remains

```text
Delta_CV(X) = EA_AE(X) - EA_FC(X)
```

where `AE` and `FC` are evaluated in the **same orbital basis at the same
molecular-state geometry**.  Only the CC correlation space changes.

This runner is a production-component runner.  Its result is not a complete
production electron affinity and it never assigns a ground state or authorizes
state pruning.

## State contract

Execution requires explicit neutral and anion state specifications containing:

- molecular system and atom ordering;
- charge and `2S`;
- stable state identifier;
- the state-specific reference geometry;
- source-root identifier;
- source checkpoint path;
- an affirmative record that state identity has already been validated.

The generic runner does not assume the OH sectors `neutral=(0,doublet)` and
`anion=(-1,singlet)`.  It only requires that the supplied anion contains one
additional electron relative to the supplied neutral.

The source checkpoint is an initial guess.  High-level HF is reconverged by the
existing Stage-3 point executor for every requested CV calculation.

## Basis-policy contract

The runner deliberately does **not** infer a core-valence basis from an element
symbol.  The caller supplies a cardinal-indexed `CoreValenceBasisSpec` with:

- a declared basis-family identifier;
- a human-readable mixed-basis label;
- an explicit `element -> basis` mapping;
- an explicit electron model.

All cardinal points in one convergence series must declare the same basis
family.  The basis mapping must cover exactly the molecular elements.

OpenEA v1 currently accepts only `electron_model = ALL_ELECTRON` in this generic
CV runner.  An ECP/core-replacement calculation is not silently treated as an
all-electron core-valence correction; such a policy fails closed until a
dedicated treatment exists.

This design supports legitimate mixed bases.  For example, an OH validation
policy may use `aug-cc-pwCVXZ` on O and the cardinal-matched `aug-cc-pVXZ` on H,
but that is a supplied validation policy rather than a generic OpenEA rule.

## Matched AE/FC execution

For each cardinal X the runner executes exactly four CCSD(T) points:

```text
neutral / all-electron
neutral / frozen-core
anion   / all-electron
anion   / frozen-core
```

For a given role and cardinal, AE and FC share:

- state identity and state sector;
- geometry;
- orbital basis and element-wise assignment;
- source-root/checkpoint provenance;
- SCF/CC numerical settings;
- nonrelativistic Hamiltonian.

The frozen-core member differs only through
`Stage3ExecutionSettings.frozen_core=True`, which invokes the established PySCF
`CCSD.set_frozen()` chemical-core policy.  Each calculation records the
correlation-space label and the frozen-core policy in CV provenance.

Scalar relativity is prohibited inside this runner because `Delta_CV` and
`Delta_scalar-rel` are independent OpenEA corrections.

## Adaptive convergence

The existing `assess_core_valence()` policy remains authoritative.

The default series begins with TZ/QZ.  The highest-cardinal correction is the
central correction and the latest cardinal change is the convergence bound.
If required, the assessment requests the next cardinal.  With three points,
the increment must also contract according to the configured threshold.

If the requested next cardinal has no explicitly supplied basis policy, the
runner returns `POLICY_BLOCKED`; it never guesses a basis.

If the maximum cardinal is reached without convergence, the result is
`CORE_VALENCE_UNRESOLVED` rather than a fabricated correction.

## Checkpoint / resume

Optional subpoint checkpoints are stored separately for each

```text
X / role / correlation-space
```

combination.  The checkpoint signature binds the complete Stage-3 request and
the relevant numerical/correlation-space settings.  A changed geometry, state,
basis assignment, source provenance, or convergence setting therefore refuses
reuse rather than silently importing stale evidence.

Completed compatible subpoints are reused individually, so a partially
completed cardinal can resume without repeating successful expensive jobs.

## Production execution routing

`CORE_VALENCE` closure actions now map to the generic runner capability
`run_adaptive_core_valence_series`.

The capability still has disposition `NEEDS_BOUND_CONTEXT`: the dispatcher does
not fabricate state identities, geometries, checkpoints, or basis policies.
Those must be bound by the workflow before execution.

Scalar relativity, CCSDT, SOC, and nuclear motion remain separate capabilities
and are not implied by the existence of this CV runner.
