# OpenEA v1 — CCSDT triples-reliability diagnostic

## Canonical role

CCSDT is a **DIAGNOSTIC**, not a mandatory production rung.  It tests the
reliability of the perturbative triples approximation in CCSD(T):

    Delta_T3(X) = EA[CCSDT](X) - EA[CCSD(T)](X)

The comparison uses matched geometry, basis, frozen-core definition,
Hamiltonian, and reference semantics.

## Decision policy

A CCSDT diagnostic may be authorized when correlation uncertainty is relevant
to the active error budget, when reference-character evidence is borderline,
or during explicit validation/benchmark work.

For the active tolerance `triples_target_ev`:

- **small and basis-stable Delta_T3**: the diagnostic is `CLEARED`; the latest
  Delta_T3 may be used as an optional post-CCSD(T) correction and the latest
  cardinal change provides a residual convergence bound;
- **only one cardinal available**: request one further CCSDT control point;
- **large or basis-unstable Delta_T3**: return `POST_CC_WARNING` and
  `REASSESS_REFERENCE_CHARACTER`.

A large or unstable Delta_T3 does **not** request CCSDTQ.

## Generic execution

`adaptive/ccsdt_diagnostic_runner.py` is the molecule-independent execution
layer.  It requires resolved neutral/anion state provenance, an explicit
all-electron orbital-basis policy, an explicit frozen-core orbital count and
upstream authorization for every cardinal that may require new CCSDT work.
Compatible completed checkpoints can be reused without reauthorizing their
compute cost.

The default backend uses the already validated PySCF/CCpy C1 interface.  C1
is metadata-only and imposes no nontrivial spatial-symmetry restriction.
CCSD(T) and CCSDT reference energies must match within a configured tolerance
before DeltaT3 is formed.

The OH script remains a validation/research driver; it is not the generic
production-facing execution path.

## CCSDTQ and legacy data

CCSDTQ is not part of the automatic OpenEA-v1 production graph.  Existing
CCSDTQ fields remain readable in `PostCCPoint` solely so historical validation
records and checkpoints retain provenance.  Such values are not added to the
production correction by `assess_post_ccsd_t`.

The OH helper script can run a DZ CCSDTQ point only through the explicit
`--validation-include-ccsdtq-dz` opt-in.  That point is validation/research
evidence and cannot trigger further coupled-cluster escalation.

## Separation from other physics

The diagnostic is a frozen-core, nonrelativistic valence-correlation test.
Core-valence, scalar relativity, SOC, and nuclear motion remain independent
error-budget components and must not be hidden inside this term.
