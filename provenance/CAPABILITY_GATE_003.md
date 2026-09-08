# Capability Gate 003 — Molecule-Level Multireference / SOC Scientific Capability

**Date:** 2026-09-08
**Status:** PASS

## Purpose

Determine whether the fully open-source OpenMolcas baseline established in
Capability Gate 002 can perform the actual scientific building blocks needed
for the OpenEA multireference and spin-orbit branches on small, transparent
reference systems.

This gate tests scientific capability.

It does **not** freeze a production electron-affinity method.

## Software baseline

The calculations used the already verified Capability Gate 002 stack:

- OpenMolcas `v26.06`
- OpenMolcas commit
  `8355057f32d65706a35996b5ab07cac2962bb728`
- OpenBLAS `v0.3.34` ILP64
- OpenBLAS commit
  `e0166008be8e466242aa76b2ff75ce3f0fbf574a`

The OpenMolcas source tree and the OpenEA working tree remained clean during
the scientific capability calculations.

## Evidence chain

Capability Gate 003 deliberately combines three different kinds of evidence:

1. an official upstream atomic regression test for the complete
   multireference / perturbation / spin-orbit chain;
2. an official upstream diatomic multistate-CASPT2 regression test;
3. an independent OpenEA diatomic spin-orbit capability test on OH.

The individual tests establish different properties and are not substitutes
for one another.

## 003A — official Ge RASSCF / MS-CASPT2 / RASSI-SO regression

OpenMolcas official test:

`standard:003`

was executed from the installed OpenMolcas v26.06 tree.

The upstream test describes the system as a Ge atom and explicitly exercises:

- RASSCF;
- MS-CASPT2;
- RASSI;
- spin-orbit interaction.

It contains separate triplet and singlet RASSCF manifolds:

- 3 triplet roots;
- 6 singlet roots.

The embedded numerical checks comprise:

- 9 RASSCF energies;
- 9 CASPT2 energies;
- 9 MS-CASPT2 energies;
- 9 spin-free RASSI energies;
- 15 spin-orbit energies.

The official verification completed with:

- test result: `standard:003 OK`;
- exit code: 0;
- failed critical tests: 0.

The numerical values reproduced the embedded upstream references within the
test tolerances.

### 003A conclusion

The installed open-source stack can execute a numerically referenced

\[
\mathrm{RASSCF}
\rightarrow
\mathrm{MS\!-\!CASPT2}
\rightarrow
\mathrm{RASSI/SOC}
\]

chain.

This is an upstream software/scientific regression result on an atom, not yet
a diatomic OpenEA validation.

## 003B — official LiF diatomic multistate-CASPT2 regression

OpenMolcas official test:

`standard:043`

was executed independently.

The upstream test describes:

- molecule: LiF near dissociation;
- internuclear separation: 8.5 bohr;
- symmetry: C2v;
- active electrons: 6;
- active orbitals: 6;
- two singlet RASSCF roots.

It exercises several multistate CASPT2 formulations.

The embedded reference set contains:

- 2 RASSCF energies;
- 8 CASPT2 energies;
- 8 multistate-CASPT2 energies.

Thus 18 energy entries were checked numerically.

Result:

- test result: `standard:043 OK`;
- verification exit code: 0;
- failed critical tests: 0;
- entries outside tolerance: 0;
- largest observed reference deviation:
  approximately `3.71e-10 Eh`.

The different multistate formulations produced materially different
two-state gaps.

### 003B conclusion

OpenMolcas can perform state-averaged RASSCF and multiple multistate-CASPT2
variants on a genuine diatomic molecule in a difficult near-dissociation
regime.

The variation among multistate formulations is also direct evidence that
OpenEA must not treat "CASPT2" as a single interchangeable production method.

No production CASPT2 variant is selected by this gate.

## 003C — independent OH X2Pi CASSCF / RASSI-SOC gate

A dedicated OpenEA capability test was constructed for neutral OH at

`R = 0.969660 Angstrom`.

### Basis and relativistic setup

The calculation used:

- `ANO-RCC-VDZP`;
- C2v symmetry;
- molecular axis along z;
- explicit AMFI integral generation.

The installed basis resolved to:

- O: `3s2p1d`;
- H: `2s1p`.

With ANO-RCC, the installed OpenMolcas setup generated a scalar
Douglas-Kroll-Hess Hamiltonian with:

- DKH order of Hamiltonian: 2.

The scientific model used for this gate is therefore described as:

\[
\mathrm{DKH2\!-\!CASSCF}
+
\mathrm{AMFI/RASSI\!-\!SOC}.
\]

### Symmetry mapping

The runtime C2v character table established:

- symmetry 1 = a1;
- symmetry 2 = b1;
- symmetry 3 = b2;
- symmetry 4 = a2.

The two Abelian components of the linear OH X2Pi state were therefore treated
as separate B1 and B2 states.

### Active space

A full-valence:

\[
\mathrm{CAS}(7,5)
\]

was used with:

- 7 active electrons;
- active-orbital distribution `(3,1,1,0)`;
- one inactive a1 orbital;
- doublet spin.

This active space is a transparent validation choice for OH.

It is **not** an OpenEA automatic active-space-selection policy.

### Independent B1/B2 CASSCF calculations

The B1 and B2 calculations were independently initialized from the same
GUESSORB orbital set.

Both converged after 10 RASSCF macro iterations.

Printed final energies:

- B1: `-75.49201258 Eh`;
- B2: `-75.49201258 Eh`.

When read by RASSI at higher precision:

- B1: `-75.49201258322125 Eh`;
- B2: `-75.49201258322120 Eh`.

Difference:

\[
5.68\times10^{-14}\ E_h.
\]

The natural occupations showed the expected symmetry-partner exchange:

- one Pi component has the singly occupied b1 orbital;
- the other has the singly occupied b2 orbital;
- the remaining active occupations are symmetry-equivalent.

### RASSI spin-orbit result

RASSI received the two doublet spin-free states and AMFI spin-orbit
integrals.

It produced four spin-orbit states:

- states 1 and 2: exactly degenerate;
- states 3 and 4: exactly degenerate.

The two Kramers pairs were:

- lower pair: Omega = 3/2;
- upper pair: Omega = 1/2.

The relevant printed SOC matrix-element magnitude was:

`67.753 cm^-1`.

The resulting electronic splitting was:

\[
\Delta_\mathrm{SO}
=
135.506487\ \mathrm{cm^{-1}}
=
0.016800663\ \mathrm{eV}.
\]

This has the correct qualitative level structure and the expected physical
scale for the OH X2Pi ground state.

The external spectroscopic value is not used as a hard numerical gate because
the present calculation is a single-geometry electronic result rather than a
fully rovibronic observable.

### 003C conclusion

The open-source OpenMolcas stack can construct the two components of a
linear diatomic Pi state, preserve their spin-free degeneracy, couple them
through AMFI/RASSI SOC, preserve Kramers degeneracy, and produce the correct
Omega ordering.

## 003D — OH SS-CASPT2 / EJOB / RASSI-SOC integration

A final integration calculation inserted dynamic-correlation energies into
the OH spin-orbit chain.

For each of the B1 and B2 components:

- state-specific CASPT2 was used;
- one root was treated;
- IPEA was explicitly set to `0.25`;
- the O 1s core was explicitly frozen;
- CASPT2 convergence threshold was `1.0e-9`.

### CASPT2 diagnostics

B1:

- reference energy: `-75.4920125832 Eh`;
- E2: approximately `-0.1355187197 Eh`;
- total energy: `-75.6275313030 Eh`;
- residual norm: `6e-10`;
- reference weight: `0.96285`.

B2:

- reference energy: `-75.4920125832 Eh`;
- E2: approximately `-0.1355187195 Eh`;
- total energy: `-75.6275313027 Eh`;
- residual norm: `6e-10`;
- reference weight: `0.96285`.

In the printed denominator diagnostic:

- no obvious intruder-state pattern was observed;
- the smallest displayed denominator was approximately `2.11 Eh`;
- the largest displayed absolute first-order coefficient was approximately
  `0.054`.

These observations apply only to this small capability test.

They do not define general OpenEA intruder-state thresholds.

### Degeneracy after CASPT2

The two CASPT2-derived RASSI diagonal energies were:

- `-75.62753130295540 Eh`;
- `-75.62753130267214 Eh`.

Difference:

\[
2.83\times10^{-10}\ E_h
\]

or approximately:

\[
0.000062\ \mathrm{cm^{-1}}.
\]

The degeneracy is therefore preserved to much better than any physically
relevant scale for this capability test.

### EJOB behavior

RASSI was run with:

`EJOB`

so that the spin-free diagonal energies were read from the CASPT2 JOBMIX
files.

OpenMolcas warned that EJOB was used while an effective Hamiltonian was
available and that possible additional interactions would be ignored.

For this specific gate this does not invalidate the result:

- each JOBMIX contains only one state;
- the two states belong to different C2v irreducible representations;
- no relevant same-symmetry multistate off-diagonal CASPT2 interaction is
  being discarded.

This interpretation is specific to the present one-state-per-JOBMIX test.

A general production policy for MS-CASPT2 effective Hamiltonians remains
unresolved.

### Correlated SOC result

After CASPT2/EJOB:

- lower Kramers pair internal splitting: `0`;
- upper Kramers pair internal splitting: `0`;
- Omega ordering remained 3/2 below 1/2;
- SOC splitting remained:

\[
135.506487\ \mathrm{cm^{-1}}
=
0.016800663\ \mathrm{eV}.
\]

The unchanged splitting is expected here because the two CASPT2 diagonal
corrections are essentially identical while the AMFI SOC matrix elements
remain those of the CASSCF state basis.

### 003D conclusion

The complete open-source data path

\[
\mathrm{CASSCF}
\rightarrow
\mathrm{CASPT2}
\rightarrow
\mathrm{JOBMIX/EJOB}
\rightarrow
\mathrm{RASSI/SOC}
\]

is operational on a real diatomic molecule.

## Numerical warnings

The OpenMolcas runs emitted GNU Fortran IEEE status notes including underflow,
denormal, invalid, and divide-by-zero flags in some modules.

For the accepted gates:

- all relevant modules returned `_RC_ALL_IS_WELL_`;
- no NaN or Infinity values were found;
- no OpenMolcas runtime-error markers were found;
- the calculations ended with `Happy landing!`.

These status notes are recorded rather than silently ignored.

They did not coincide with numerical failure in the present tests.

## Capability conclusion

### scientific-capability-verified

Capability Gate 003 establishes that the fully open-source OpenMolcas backend
can support development of an OpenEA multireference / relativistic /
spin-orbit branch containing:

- CASSCF/RASSCF;
- state averaging;
- state-specific CASPT2;
- multiple multistate-CASPT2 variants;
- RASSI state interaction;
- DKH2 scalar-relativistic treatment with ANO-RCC;
- AMFI spin-orbit matrix elements;
- CASPT2 diagonal energies supplied to RASSI through EJOB;
- diatomic Pi-state symmetry handling;
- Kramers-degenerate spin-orbit states.

This is sufficient to proceed from software-capability testing to formal
OpenEA scientific-method specification.

### still unresolved

Capability Gate 003 does **not** establish:

- an automatic or molecule-independent active-space-selection algorithm;
- the production state-averaging policy;
- the production CASPT2 variant;
- IPEA-shift policy;
- real- or imaginary-level-shift policy;
- general intruder-state detection or rescue thresholds;
- how multistate effective Hamiltonians are passed to RASSI in production;
- whether CASPT2 is sufficiently accurate for a given molecule;
- the SR-to-MR escalation policy;
- basis-set convergence of MR corrections;
- core-valence treatment in the MR branch;
- scalar-relativistic correction strategy for final electron affinities;
- SOC correction strategy for neutral-minus-anion electron affinities;
- PEC sampling rules for MR or SOC corrections;
- active-space and state continuity along a PEC;
- uncertainty assignment for MR, scalar-relativistic, or SOC corrections;
- computational scaling limits for transition-metal and heavy-element targets.

No production electron-affinity method is frozen by Capability Gate 003.

The next phase is formal scientific-method specification and validation-matrix
design before implementation of production workflow logic.
