# OpenEA v1 — Generic CCSDT triples-reliability diagnostic runner

## Purpose

`adaptive/ccsdt_diagnostic_runner.py` provides the molecule-independent
execution layer for the optional OpenEA-v1 CCSDT diagnostic.

It computes, for already resolved neutral and anion states,

```text
Delta_T3(X) = EA_CCSDT(X) - EA_CCSD(T)(X)
```

and passes the resulting cardinal points to `assess_post_ccsd_t`.

This runner is **DIAGNOSTIC**.  It is not a mandatory production rung and it
never authorizes CCSDTQ, CCSDTQP, FCI, or another higher-rank escalation.

## Explicit authorization

The presence of CCpy is not permission to spend CCSDT resources.

Every run receives a `CCSDTDiagnosticAuthorization` containing:

- an explicit authorized/not-authorized decision;
- the reason;
- evidence IDs supporting the decision;
- the cardinal numbers authorized for **new computation**.

A compatible completed checkpoint may be reused as existing evidence even if
that cardinal is not newly authorized.  If a required subpoint is absent and
its cardinal is not authorized for new work, execution fails closed.

This preserves the Precision Controller's role: the runner executes a decision;
it does not decide that another expensive CCSDT point is worthwhile.

## State contract

Neutral and anion are supplied independently as resolved state objects with:

- molecular system and atom ordering;
- charge and `2S`;
- state ID;
- fixed reference geometry;
- source-root and source-checkpoint provenance;
- explicit confirmation that state identity was validated;
- explicit number of frozen spatial core orbitals.

The neutral and anion must use the same frozen-core definition.  The runner
never infers the number of core orbitals from the element symbols.

## Basis contract

Each cardinal uses an explicit `basis_by_element` map and one declared basis
family.  OpenEA does not infer `aug-cc-pVXZ`, `cc-pCVXZ`, ECPs, or another
family from the molecule.

The v1 CCpy diagnostic supports only **all-electron orbital bases** with an
explicit frozen-core *correlation* space.  ECP/core-replacement post-CC
semantics are therefore rejected rather than silently treated as equivalent.

All requested bases are preflighted before expensive coupled-cluster work.

## Matched CCSD(T)/CCSDT semantics

For each state and cardinal, CCSD(T) and CCSDT are constructed from the same:

- charge and spin sector;
- fixed geometry;
- basis assignment;
- source checkpoint and projection policy;
- RHF/ROHF reference policy;
- SCF convergence settings;
- frozen-core count;
- nonrelativistic Hamiltonian;
- explicit C1 symmetry metadata.

The independently reconstructed SCF reference energies must agree within the
configured tolerance.  If they do not, DeltaT3 is rejected rather than formed
from mismatched references.

## Why explicit C1?

The validated CCpy/PySCF bridge expects molecular symmetry metadata.  OpenEA
therefore constructs the reference with explicit `symmetry="C1"`.

C1 is used only to provide nonrestrictive symmetry metadata to CCpy.  OpenEA
v1 does not exploit a higher molecular point group in this diagnostic.

## Backend

The default backend is PySCF + CCpy:

- RHF for closed-shell states;
- ROHF for open-shell states;
- `Driver.from_pyscf(..., nfrozen=...)`;
- CCpy `ccsd` + `ccsd(t)` for CCSD(T);
- CCpy `ccsdt` for iterative triples.

CCpy is imported lazily.  Unit tests can inject a method runner without
requiring CCpy or executing electronic-structure calculations.

## Real backend smoke test

`scripts/openea_ccsdt_runner_smoke.py` is an explicit software-integration
smoke for the default PySCF/CCpy backend.  It uses a deliberately tiny
OH/OH- STO-3G calculation and a single nominal cardinal.  Its output is not a
scientific electron affinity and must never enter validation or production
evidence.  The expected terminal scientific-diagnostic state is
`TRIPLES_RELIABILITY_UNRESOLVED` / `NEED_MORE_EVIDENCE`, because one basis
point cannot establish DeltaT3 convergence.  PASS means only that all four
neutral/anion x CCSD(T)/CCSDT backend subcalculations executed through the
generic runner.

## Checkpoint/resume

Each individual subpoint is checkpointed using collision-free method tokens:

```text
CCSD(T) -> CCSD_pT
CCSDT   -> CCSDT
```

The checkpoint signature includes the complete request and numerical settings.
A stale or incompatible checkpoint is rejected.  Completed subpoints are
reused independently, so an interrupted series does not repeat already
validated expensive calculations.

## Interpretation

After at least two cardinal points are available, the existing canonical
assessment applies:

- small and basis-stable DeltaT3 -> `CLEARED`, optional DeltaT3 correction;
- large and/or unstable DeltaT3 -> `POST_CC_WARNING` ->
  `REASSESS_REFERENCE_CHARACTER`;
- insufficient evidence -> request another T3 diagnostic point, which still
  requires explicit cardinal authorization before new computation.

No uncomputed higher-order contribution is assigned `0 +/- 0`.

## Separation from other physics

The runner is fixed to:

```text
method role        = DIAGNOSTIC
correlation space  = frozen-core valence
Hamiltonian        = nonrelativistic
scalar relativity  = excluded
SOC                = excluded
```

Core-valence, scalar relativity, SOC, nuclear motion, and any validated MR
production path remain separate OpenEA components.
