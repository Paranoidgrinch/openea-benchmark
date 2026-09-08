# OpenEA-Benchmark

OpenEA-Benchmark is a research project for developing a fully open-source,
adaptive, accuracy-driven workflow for high-accuracy adiabatic electron
affinities of diatomic molecules.

The project is currently in the scientific-method and software-capability
validation phase.

No production electron-affinity method is frozen yet.

## Scientific target

For a bound diatomic anion, the primary target quantity is

\[
EA_0 =
E_{v=0}(\mathrm{neutral})
-
E_{v=0}(\mathrm{anion}).
\]

The workflow is intended to distinguish explicitly between electronic
minimum-to-minimum affinities, nuclear-motion corrections, relativistic and
spin-orbit contributions, and the final adiabatic electron affinity.

For systems close to threshold, the workflow must also distinguish robustly
between bound anions, unbound anions, unresolved near-threshold cases, and
systems requiring a resonance or metastable-state treatment.

## Design principles

- Fully open-source production path.
- Accuracy-driven rather than fixed-cost method hierarchy.
- Conservative electronic-state discovery and state following.
- No molecule-specific rescue rules.
- Explicit diffuse-basis and near-threshold checks for anions.
- Validation and prediction datasets remain separate.
- Expensive methods are added adaptively only when justified.
- Single-reference and multireference diagnostics are treated explicitly.
- Fragment and dissociation-channel identity is validated rather than assumed.
- Electronic structure, nuclear motion, relativity, spin-orbit effects, and
  uncertainty remain separate scientific layers.
- Prediction calculations begin only after the method, escalation criteria,
  quality-control rules, and uncertainty policy have been frozen.

## Intended scientific hierarchy

The final hierarchy is still under development, but the project is evaluating
a workflow containing, where scientifically justified:

1. electronic-state discovery and state continuity;
2. single-reference correlated baseline calculations;
3. diffuse-basis and complete-basis-set convergence;
4. higher-order coupled-cluster corrections;
5. core-valence correlation;
6. scalar-relativistic corrections;
7. multireference escalation;
8. spin-orbit corrections;
9. explicit fragment and asymptote validation;
10. nuclear-motion treatment;
11. calibrated uncertainty estimation.

The purpose of the adaptive hierarchy is not to apply the most expensive
method to every molecule. It is to escalate only when diagnostics and
convergence evidence show that a simpler level is insufficient.

## Verified software capabilities

Software capability is established through explicit capability gates before a
method is admitted to scientific workflow development.

### Capability Gate 001 — higher-order coupled cluster

An open-source single-reference stack based on PySCF and CCpy has been
runtime-validated for:

- RHF CCSDT;
- RHF CCSDTQ;
- UHF CCSDT;
- UHF CCSDTQ with explicitly nonzero quadruple-excitation amplitudes.

This establishes that an open-source route to explicit higher-order
coupled-cluster corrections is technically available.

See
[`provenance/CAPABILITY_GATE_001.md`](provenance/CAPABILITY_GATE_001.md).

### Capability Gate 002 — multireference / SOC software baseline

An OpenMolcas v26.06 stack using a separately verified OpenBLAS 0.3.34 ILP64
build has been compiled, validated, and installed successfully.

The official OpenMolcas default verification suite produced:

| Result | Count |
|---|---:|
| OK | 516 |
| Skipped optional tests | 6 |
| Failed | 0 |
| Failed critical tests | 0 |

All six skipped tests were traced to deliberately disabled optional
interfaces.

The installed stack provides the executables required to investigate:

- CASSCF/RASSCF;
- CASPT2;
- multistate correlation treatments;
- RASSI state interaction;
- spin-orbit coupling.

This gate establishes **software capability only**. It does not yet validate
the corresponding OpenEA scientific production policies.

See
[`provenance/CAPABILITY_GATE_002.md`](provenance/CAPABILITY_GATE_002.md).

## Current scientific status

The project has now established open-source software baselines for both:

- explicit higher-order single-reference coupled cluster; and
- multireference / spin-orbit workflow development.

The next development stage is molecule-level scientific validation of the
multireference chain

\[
\mathrm{CASSCF/RASSCF}
\rightarrow
\mathrm{CASPT2}
\rightarrow
\mathrm{RASSI/SOC}.
\]

Important production choices remain deliberately unresolved, including active
space selection, state averaging, intruder-state handling, single-reference
versus multireference escalation, relativistic and spin-orbit correction
strategies, basis-set convergence, and uncertainty assignment.

## Relationship to DiatomicEA v0.9

The predecessor workflow is maintained separately:

https://github.com/Paranoidgrinch/diatomic-ea-workflow

OpenEA-Benchmark may selectively reuse validated concepts and infrastructure
from that workflow, including state discovery, SCF stability analysis, state
identity, branch following, fine quality control, checkpointing, and related
scientific-policy logic.

The predecessor repository is not overwritten.

## Reproducibility

Capability-gate documents under [`provenance/`](provenance/) record exact
software versions, validation evidence, limitations, and unresolved scientific
questions.

These records distinguish verified software functionality from scientific
method assumptions.

## License

GPL-3.0.
