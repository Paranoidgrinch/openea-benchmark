# OpenEA-Benchmark

OpenEA-Benchmark is a research project for developing a fully open-source,
adaptive, accuracy-driven workflow for high-accuracy adiabatic electron
affinities of diatomic molecules.

The project is currently in the scientific-method and software-capability
validation phase.

## Scientific target

For a bound diatomic anion, the primary target quantity is

\[
EA_0 =
E_{v=0}(\mathrm{neutral})
-
E_{v=0}(\mathrm{anion}).
\]

The intended workflow combines, when scientifically justified:

- conservative electronic-state discovery and state following;
- coupled-cluster baseline calculations;
- diffuse-basis and CBS convergence;
- higher-order correlation corrections;
- core-valence correlation;
- scalar-relativistic corrections;
- spin-orbit corrections;
- multireference escalation;
- explicit fragment/asymptote validation;
- nuclear-motion treatment;
- calibrated uncertainty estimates.

The final production hierarchy is **not yet frozen**.

## Principles

- Fully open-source production path.
- No molecule-specific rescue rules.
- Validation and prediction datasets remain separate.
- State identity, numerical convergence, electronic-structure adequacy,
  fragment identity, bound/unbound classification, nuclear motion, and
  uncertainty remain separate scientific layers.
- Expensive methods are added adaptively only when justified.
- Prediction calculations begin only after method and uncertainty-policy
  freeze.

## Relationship to DiatomicEA v0.9

The predecessor workflow is maintained separately:

https://github.com/Paranoidgrinch/diatomic-ea-workflow

OpenEA-Benchmark may selectively reuse validated v0.9 infrastructure for
state discovery, SCF Stability, state identity, branch following, Fine-QC,
checkpointing, and related scientific-policy logic.

The predecessor repository is not overwritten.

## Current verified software capability

Capability Gate 001 established an open-source higher-order
single-reference stack based on:

- PySCF 2.14.0
- CCpy 0.0.5
- CCpy source commit
  `cf72da4865e054f7a935e2df27e920cf5e335435`

Verified on Artemis:

- RHF CCSDT
- RHF CCSDTQ
- UHF CCSDT
- UHF CCSDTQ with explicitly nonzero T4 amplitudes

See:

`provenance/CAPABILITY_GATE_001.md`

## Status

The project is currently validating software capabilities and scientific
method components.

No production electron-affinity method is frozen yet.

## License

GPL-3.0.
