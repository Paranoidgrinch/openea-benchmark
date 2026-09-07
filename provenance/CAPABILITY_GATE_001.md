# Capability Gate 001 — Open-Source Higher-Order Coupled Cluster

**Date:** 2026-09-07
**Host:** `artemis.ikp.uni-koeln.de`
**Status:** PASS

## Purpose

Determine whether a fully open-source software stack can support the
higher-order single-reference coupled-cluster hierarchy required for
focal-point electron-affinity development, including genuine open-shell
CCSDTQ.

This gate establishes software capability only.

It does not freeze:

- production basis sets;
- CBS extrapolation formulas;
- frozen-core policy;
- core-valence treatment;
- higher-order correction basis sizes;
- ROHF/UHF production-reference policy;
- multireference escalation criteria;
- relativistic or spin-orbit policy;
- uncertainty methodology.

## Environment

- Linux x86_64
- Python 3.12.14
- PySCF 2.14.0
- NumPy 2.5.3
- SciPy 1.18.1
- CCpy 0.0.5
- CCpy commit:
  `cf72da4865e054f7a935e2df27e920cf5e335435`
- GCC 10.3.1
- GFortran 10.3.1
- Meson 1.6.0
- Ninja 1.13.2
- OpenBLAS 0.3.15
- `pip check`: PASS

CCpy was built from the pinned source commit using its Meson build system.

The small capability calculations used:

- `nice -n 19`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

The one-thread configuration belongs only to these controlled smoke tests.
It is not the OpenEA production-parallelism policy.

## PySCF capability

Directly verified with PySCF 2.14.0:

| Reference | CCSD(T) | CCSDT | CCSDTQ |
|---|---:|---:|---:|
| RHF | PASS | PASS | PASS |
| UHF | PASS | PASS | not implemented |
| ROHF input | routed to UHF | routed to UHF | not implemented |

PySCF therefore provides an open-source path through UCCSDT but not
UCCSDTQ in version 2.14.0.

## PySCF to CCpy interface validation

### LiH / RHF / STO-3G

PySCF reference energy:

`-7.861864769809 Eh`

CCpy-imported reference-energy difference:

`-5.329070518201e-15 Eh`

Result: **PASS**

### OH / UHF / STO-3G

PySCF reference energy:

`-74.362669194767 Eh`

CCpy-imported reference-energy difference:

`0.000000000000e+00 Eh`

Result: **PASS**

For a PySCF UHF reference the tested interface was:

`Driver.from_pyscf(..., uhf=True)`

The trivial `C1` point group was supplied so that the orbital-symmetry
metadata expected by the CCpy PySCF interface existed without imposing
nontrivial molecular symmetry.

## Closed-shell higher-order cross-validation

### LiH / RHF / STO-3G

PySCF CCSDT:

`-7.882324241481 Eh`

CCpy CCSDT:

`-7.882324241825 Eh`

Difference:

`-3.433244799567e-10 Eh`

PySCF CCSDTQ:

`-7.882324378526 Eh`

CCpy CCSDTQ:

`-7.882324378884 Eh`

Difference:

`-3.579954110933e-10 Eh`

CCpy higher-order difference:

\[
E_{\mathrm{CCSDTQ}}-E_{\mathrm{CCSDT}}
=
-1.370597120243\times10^{-7}\ E_h
\]

Result: **PASS**

## Open-shell CCSDT cross-validation

### OH / UHF / 6-31G

UHF reference energy:

`-75.363168249576 Eh`

Spin expectation:

\[
\langle S^2\rangle = 0.75377418
\]

PySCF UCCSDT:

`-75.462646385531 Eh`

CCpy CCSDT:

`-75.462646384429 Eh`

Difference:

`+1.101653879232e-09 Eh`

Result: **PASS**

## Genuine open-shell CCSDTQ validation

### OH / UHF / 6-31G

CCpy CCSDT:

`-75.462646384429 Eh`

CCpy CCSDTQ:

`-75.462864797141 Eh`

Therefore:

\[
\Delta T_4
=
E_{\mathrm{CCSDTQ}}-E_{\mathrm{CCSDT}}
=
-2.184127122860\times10^{-4}\ E_h
\]

The CCSDTQ solver converged successfully.

Explicit T4 amplitude-block norms:

| block | norm |
|---|---:|
| `aaaa` | `1.570400549136e-03` |
| `aaab` | `7.115849966328e-03` |
| `aabb` | `7.371898262736e-03` |
| `abbb` | `3.134751373888e-03` |
| `bbbb` | `1.391269028646e-05` |

Largest T4 block norm:

`7.371898262736e-03`

Thus this was a genuine nonzero open-shell quadruple-excitation
calculation rather than a successful method dispatch with trivial T4
amplitudes.

Result:

**UHF CCSDTQ VERIFIED WITH NONZERO T4**

## Capability conclusion

### software-verified

The combined PySCF + CCpy open-source stack provides:

- RHF CCSD(T)
- UHF CCSD(T)
- RHF CCSDT
- UHF CCSDT
- RHF CCSDTQ
- UHF CCSDTQ

The single-reference focal-point architecture can therefore continue to be
evaluated through explicit quadruple excitations without requiring
proprietary software.

### still unresolved

Scientific questions remaining include:

- CCSD(T)/CBS baseline definition;
- basis convergence of delta-T3 and delta-T4;
- diffuse augmentation policy;
- frozen-core versus core-valence policy;
- ROHF versus UHF production-reference policy;
- single-reference adequacy diagnostics;
- realistic memory and scaling limits;
- scalar-relativistic treatment;
- spin-orbit treatment;
- multireference production branch;
- nuclear-motion treatment;
- calibrated uncertainty model.

No production method is frozen by this capability gate.
