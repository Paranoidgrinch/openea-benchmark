<!-- OPENEA_GENERAL_SCOPE -->

# OpenEA Benchmark

**OpenEA is not a FeH-specific project.**

The goal of this repository is to develop a reproducible, open-source,
high-accuracy workflow for calculating electron affinities of
diatomic molecules.

The intended result is a workflow that can be applied to a broad range
of neutral/anion diatomic pairs without relying on experimental
electron affinities during the calculation and without introducing
molecule-specific rescue rules whenever a difficult system is
encountered.

## Why does FeH currently appear so often?

FeH / FeH- is the current main stress-test and validation system.
It is not the target molecule of the project.

FeH was deliberately selected because it combines many of the
difficulties that a general diatomic electron-affinity workflow must
eventually handle:

- open-shell neutral and anion states
- several low-lying electronic states
- transition-metal 3d correlation
- significant multireference character
- strong dynamic correlation
- difficult single-reference diagnostics
- basis-set sensitivity
- scalar-relativistic effects
- spin-orbit coupling
- state tracking along potential-energy curves
- different electronic structures of neutral and anion

In other words, FeH is being used because it is an unusually demanding
diatomic benchmark.

A workflow that can handle FeH reliably is a much stronger starting
point for less pathological diatomics than a workflow developed only
on simple systems.

The FeH development has progressed from strongly inaccurate early
results to a high-level value close to the experimental electron
affinity. This makes FeH useful for identifying, separating and
testing individual sources of error.

Experimental FeH data are used only for validation.

They must not be used to choose hidden roots, tune thresholds,
select a preferred method simply because it agrees with experiment,
or introduce FeH-specific corrections into the general workflow.

## Intended workflow

The general workflow under development contains roughly the following
layers:

1. electronic-state discovery
2. determinant identity checks
3. electronic-manifold construction
4. state continuity along bond length
5. potential-energy-curve construction
6. high-level electronic correlation
7. single-reference / multireference diagnostics
8. basis-set convergence
9. scalar-relativistic corrections where needed
10. high-level geometry refinement
11. zero-point vibrational corrections
12. spin-orbit corrections where needed
13. uncertainty and reliability assessment
14. final EA or UNBOUND decision

If an anion is confidently unbound, the scientifically relevant result
is simply:

    UNBOUND

A precise negative electron affinity is not required.

## Development philosophy

The workflow should remain:

- general rather than molecule-specific
- open source
- reproducible
- state-aware
- conservative when electronic identity is ambiguous
- explicit about uncertainty
- capable of detecting unbound anions
- independent of experimental values during prediction

FeH-specific development calculations live under:

    pilots/feh_high_accuracy/

They are validation and development material for the general workflow,
not the final architecture itself.

Useful project documents:

- docs/WORKFLOW_SCOPE.md
- CONTRIBUTING.md
- pilots/feh_high_accuracy/STATUS.md

---

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

> **OpenEA-v1 role note:** this gate establishes software capability only.
> The current production architecture uses CCSDT only as an optional
> triples-reliability diagnostic. CCSDTQ remains validation/research capability
> and is not an automatic production escalation step.

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

### Capability Gate 003 — molecule-level multireference / SOC chain

Molecule-level scientific capability has now been demonstrated on transparent
reference systems using the verified open-source OpenMolcas stack.

The validation chain includes:

- an official Ge RASSCF/MS-CASPT2/RASSI-SOC regression test;
- an official LiF diatomic state-averaged RASSCF and multistate-CASPT2 test;
- an independent OH X2Pi DKH2-CASSCF/AMFI/RASSI-SOC test;
- an OH state-specific CASPT2/EJOB/RASSI-SOC integration test.

For OH, the two spin-free Pi components remained degenerate, RASSI produced
two Kramers pairs with Omega = 3/2 below Omega = 1/2, and the electronic
spin-orbit splitting was `135.5065 cm^-1`.

Supplying state-specific CASPT2 diagonal energies through EJOB preserved both
the Pi-state degeneracy and the spin-orbit structure.

This establishes scientific **capability**, not a frozen production method.
Active-space selection, state averaging, CASPT2 variant selection,
intruder-state handling, relativistic and SOC correction policies, and
uncertainty assignment remain unresolved.

See
[`provenance/CAPABILITY_GATE_003.md`](provenance/CAPABILITY_GATE_003.md).

## DFT scout-method validation

The project has now completed a first empirical validation series for the
inexpensive DFT state-discovery layer.

The gates tested:

- functional availability and semantics;
- SCF stability;
- basis-set sensitivity;
- transition-metal spin-state ordering;
- multiple SCF roots;
- stability canonicalization;
- occupied-space projection between basis sets;
- broken-symmetry behavior;
- generic SCF rescue.

The current provisional development panel retains `r2SCAN`, `r2SCANh`, and
`PBE0`.

`ma-def2-TZVPP` is the leading broadly available Tier-0 scout-basis candidate,
with `aug-pcseg-2` retained as an independent 3d cross-check and
`def2-QZVPPD` as an anchor/refinement basis.

These choices are not yet frozen production policy.

The validation also established that SCF convergence alone is insufficient:
state continuity, stability, spin contamination, and root identity must remain
explicit workflow diagnostics.

See [`provenance/DFT_SCOUT_VALIDATION_001.md`](provenance/DFT_SCOUT_VALIDATION_001.md).

## Current scientific status

The project has now established:

- an open-source higher-order single-reference coupled-cluster capability;
- a verified OpenMolcas multireference / spin-orbit software baseline;
- molecule-level CASSCF, CASPT2, RASSI, and SOC scientific capability;
- an empirically tested DFT scout strategy for state discovery and basis/root
  continuity.

The next development stage is **DFT-0C**, the first tested implementation of
the DFT state-discovery and local-PEC workflow.

DFT-0C will initially cover candidate spin discovery, multiple generic SCF
roots, stability canonicalization, root deduplication, state/branch identity,
local PEC construction, geometry scouting, and preliminary delta-SCF
electron-affinity diagnostics on a small validation subset.

Important production decisions remain deliberately unresolved, including the
final DFT panel and basis policy, single-reference versus multireference
escalation logic, active-space policy, state averaging, correlated basis-set
convergence, core-valence and relativistic corrections, SOC treatment for
electron affinities, PEC correction strategies, nuclear motion, and uncertainty
assignment.

No prediction calculations begin until those policies and their validation
criteria have been defined and tested.

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
