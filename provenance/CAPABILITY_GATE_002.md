# Capability Gate 002 — Open-Source Multireference / SOC Software Baseline

**Date:** 2026-09-08
**Host:** `artemis.ikp.uni-koeln.de`
**Status:** PASS

## Purpose

Determine whether a fully open-source OpenMolcas-based software stack can be
built, linked, validated, and installed reproducibly on Artemis as a candidate
backend for the multireference and spin-orbit branches of OpenEA-Benchmark.

This gate establishes software and installation capability only.

It does **not** yet establish that the OpenEA scientific production branch for:

- state-averaged CASSCF/RASSCF;
- CASPT2;
- multistate CASPT2;
- RASSI state interaction;
- spin-orbit coupling;
- active-space selection;
- multireference escalation;
- relativistic electron-affinity corrections;

has been scientifically validated.

Those capabilities require separate molecule-level gates.

## Source versions

### OpenMolcas

- version/tag: `v26.06`
- commit: `8355057f32d65706a35996b5ab07cac2962bb728`
- license: LGPL-2.1
- source tree after build/tests: clean

### OpenBLAS

- version/tag: `v0.3.34`
- commit: `e0166008be8e466242aa76b2ff75ce3f0fbf574a`
- built separately for the OpenMolcas ILP64 requirement
- source tree after build/tests: clean

### Bundled Libxc

OpenMolcas v26.06 selected its built-in Libxc source:

- release: 7.0.0
- configured OpenMolcas pin: `7bd5bb41`
- actual fetched commit: `7bd5bb41415968db94c499a2f093309c9a2dcf53`

### Bundled libwignernj

OpenMolcas v26.06 used its bundled libwignernj fallback:

- tag: `v0.6.0`
- actual fetched commit: `0c907ee4e726343e333be1d4c8663172da95cbe8`

## Environment

- Linux x86_64
- 80 logical CPUs
- GCC 10.3.1
- GFortran 10.3.1
- CMake 3.20.1
- Python 3.12.14
- lxml 6.1.3

## OpenBLAS ILP64 gate

The system OpenBLAS 0.3.15 was explicitly rejected for this OpenMolcas build.

Its installed headers did not define:

`OPENBLAS_USE64BITINT`

and the compile-time ILP64 test failed as expected.

A pinned OpenBLAS 0.3.34 build was therefore created with:

- `BINARY=64`
- `INTERFACE64=1`
- `DYNAMIC_ARCH=1`
- `USE_THREAD=1`
- `NUM_THREADS=80`
- `NO_AFFINITY=1`
- `CC=gcc`
- `FC=gfortran`

The resulting runtime configuration reported:

`OpenBLAS 0.3.34 USE64BITINT DYNAMIC_ARCH NO_AFFINITY SkylakeX MAX_THREADS=80`

Independent ABI validation established:

- `OPENBLAS_USE64BITINT` present;
- `sizeof(blasint) = 8`;
- C ABI test: PASS;
- Fortran build with `-fdefault-integer-8`: PASS;
- DGEMM numerical test: PASS;
- DSYEV numerical test: PASS.

Result:

**OPENBLAS 0.3.34 ILP64 VERIFIED**

## OpenMolcas configuration

OpenMolcas was configured as an isolated Release build with:

- `LINALG=OpenBLAS`
- custom `OPENBLASROOT` pointing to the verified OpenBLAS 0.3.34 ILP64 build
- `MPI=OFF`
- `GA=OFF`
- `OPENMP=OFF`
- `HDF5=OFF`
- `TOOLS=OFF`
- `INSTALL_TESTS=ON`
- `BUILD_TESTING=ON`
- `DMRG=OFF`
- `NEVPT2=OFF`
- `BLOCK=OFF`
- `CHEMPS2=OFF`
- `DICE=OFF`
- `MSYM=OFF`
- `NECI=OFF`
- `WFA=OFF`
- `EFPLIB=OFF`

OpenMolcas itself verified during CMake configuration:

- address mode: 64 bit;
- `OPENBLAS_USE64BITINT`: found;
- Fortran LAPACK `dsyev`: found;
- compiler integer model: `-fdefault-integer-8`.

## Build result

The configured OpenMolcas v26.06 source was built with:

- `nice -n 19`;
- 24 parallel compiler jobs;
- BLAS/OpenMP thread counts constrained during the build.

Build result:

**PASS**

Required executables were produced:

- `gateway.exe`
- `seward.exe`
- `rasscf.exe`
- `caspt2.exe`
- `rassi.exe`

Dynamic-link inspection confirmed that RASSCF, CASPT2, and RASSI resolve
`libopenblas.so.0` from the custom OpenBLAS 0.3.34 ILP64 installation rather
than the system OpenBLAS.

## Input validation

The Python environment was supplemented with:

`lxml==6.1.3`

and:

`python -m pip check`

returned no broken requirements.

Official OpenMolcas test 000 was run both:

- with explicit input validation;
- as a normal numerical verification test.

Both runs passed with exit code 0.

## Full official OpenMolcas verification suite

The OpenMolcas v26.06 virtual `.default` verification group contained:

**522 tests**

The suite was executed with:

- 24 simultaneous independent verification jobs;
- `nice -n 19`;
- `OPENBLAS_NUM_THREADS=1`;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`;
- `MOLCAS_NPROCS=1`;
- `MOLCAS_THREADS=1`;
- `MOLCAS_VALIDATE=YES`.

Results:

| Result | Count |
|---|---:|
| OK | 516 |
| Skipped | 6 |
| Failed | 0 |
| Failed critical tests | 0 |

### Explained skips

The six skipped tests were:

- `additional:926` — HDF5-dependent stochastic/M7 test;
- `additional:929` — HDF5-dependent stochastic/M7 test;
- `grayzone:834` — Gromacs interface test;
- `grayzone:851` — Dice interface test;
- `grayzone:853` — Dice interface test;
- `grayzone:854` — Dice interface test.

These skips correspond to optional capabilities deliberately disabled in this
build.

No unexplained skip remained.

Result:

**OPENMOLCAS DEFAULT VERIFICATION: PASS**

## Installed-tree verification

The already-tested build was installed without reconfiguration into the
isolated OpenEA external-software tree.

Installation result:

**PASS**

The installed tree contains:

- `gateway.exe`
- `seward.exe`
- `rasscf.exe`
- `caspt2.exe`
- `rassi.exe`

The installed executables continue to resolve the verified custom
OpenBLAS ILP64 library.

Official test `standard:000` was then executed from the installed tree.

Result:

- test: `standard:000 OK`
- failed critical tests: 0
- verification exit code: 0

## Capability conclusion

### software-verified

A fully open-source OpenMolcas software baseline is operational on Artemis.

The verified stack provides installed executables required to investigate:

- CASSCF/RASSCF;
- CASPT2;
- RASSI;
- spin-orbit state interaction.

The software stack is therefore suitable for proceeding to dedicated
scientific capability tests without requiring proprietary software.

### still unresolved

This gate does not establish:

- an OpenEA active-space selection policy;
- state-averaging rules;
- root/state continuity policy for the MR backend;
- CASPT2 production settings;
- intruder-state handling policy;
- MS-CASPT2 or XMS-like production policy;
- MR diagnostic thresholds;
- scalar-relativistic production treatment;
- SOC correction strategy for electron affinities;
- basis-set convergence of MR corrections;
- MR/SR consistency criteria;
- uncertainty assignment for MR or SOC corrections;
- computational scaling limits for production molecules.

The next gate must therefore test the actual scientific chain:

\[
\mathrm{CASSCF/RASSCF}
\rightarrow
\mathrm{CASPT2}
\rightarrow
\mathrm{RASSI/SOC}
\]

on small, transparent reference systems.

No production electron-affinity method is frozen by Capability Gate 002.
