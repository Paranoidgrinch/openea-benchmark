# OpenEA v1 — CCSDT diagnostic CCpy/PySCF C1 interface

This document records the validated symmetry-interface requirement used by
the generic CCSDT triples-reliability runner.

The production-facing quantity is

    Delta_T3(X) = EA[CCSDT] - EA[CCSD(T)]

with matched fixed geometry, orbital basis, explicitly declared frozen-core
space, nonrelativistic Hamiltonian and reference semantics.  CCSDTQ is not an
automatic production step; any historical CCSDTQ evidence is validation-only.

The generic runner lives in
`adaptive/ccsdt_diagnostic_runner.py`; the OH script is retained as a
validation/research driver.

## PySCF / CCpy symmetry interface

CCpy's `Driver.from_pyscf()` interface labels molecular-orbital symmetries
using PySCF's `mol.irrep_name` and `mol.symm_orb`.  These attributes are not
initialized when a PySCF molecule is constructed with `symmetry=False`.

OpenEA therefore constructs the PySCF reference for this layer with explicit

    symmetry="C1"

rather than disabling symmetry entirely.

C1 is intentionally chosen instead of exploiting the full linear OH point
group.  It supplies the single `A` irrep and symmetry-adapted metadata that
CCpy requires while imposing no nontrivial spatial-symmetry restriction on
the wavefunction or correlated excitation space.

The runner verifies the C1 metadata before any production coupled-cluster
iteration and fails closed if the PySCF/CCpy symmetry contract is not met.
