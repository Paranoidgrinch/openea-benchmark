# OpenEA v1 — post-CCSD(T) correlation

The post-CCSD(T) layer resolves higher-order valence correlation into two
physically distinct increments:

    Delta_T3(X) = EA[CCSDT]  - EA[CCSD(T)]
    Delta_T4(X) = EA[CCSDTQ] - EA[CCSDT]

All energies entering a difference use the same:

- fixed neutral/anion reference geometry;
- aug-cc-pVXZ orbital basis;
- frozen-core definition;
- nonrelativistic Hamiltonian;
- CCpy reference wavefunction.

The correction is therefore additive to the frozen-core valence CBS
baseline. Core-valence and scalar-relativistic corrections remain separate.

## Basis policy

Higher-order coupled-cluster corrections converge faster with basis size than
the dominant CCSD(T) valence energy and are much more expensive. OpenEA v1
therefore uses:

- aug-cc-pVDZ and aug-cc-pVTZ for `T3-(T)`;
- full CCSDTQ at aug-cc-pVDZ for connected quadruples;
- aug-cc-pVTZ CCSDTQ only if the DZ quadruples increment is too large to
  bound conservatively.

No post-CCSD(T) extrapolation formula is imposed.

## Uncertainty policy

`T3-(T)` is accepted when the DZ->TZ change is below its configured target.

For connected quadruples, if the directly computed DZ correction is already
small, its full absolute magnitude is retained as a conservative basis
uncertainty. It is not silently set to zero.

If the DZ quadruples term is larger than the threshold, TZ CCSDTQ is
requested and the DZ->TZ change becomes the evidence-based convergence bound.

The layer fails closed if the computed evidence does not satisfy its targets.

## Implementation

CCpy is used because it provides CCSD(T), CCSDT, and CCSDTQ implementations
for RHF/ROHF/UHF references. For OH one O(1s) spatial orbital is frozen.

This layer is nonrelativistic. Scalar relativity and SOC are additive
corrections handled elsewhere in OpenEA.


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
