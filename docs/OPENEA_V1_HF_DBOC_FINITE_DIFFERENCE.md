# OpenEA v1 — Patch 24: executable HF DBOC finite-difference pilot

## Why this exists

OpenEA's J=0 vibrational solver adds nuclear zero-point motion on Born–Oppenheimer (BO) PECs. This **does not** evaluate the diagonal Born–Oppenheimer correction (DBOC), which modifies the electronic BO PEC, nor off-diagonal nonadiabatic state coupling. The residual `ADIABATIC_NUCLEAR_REMAINDER` remains open.

PySCF 2.14 does **not** expose a readily usable built-in electronic DBOC energy method. This patch implements a real but intentionally limited **Hartree–Fock determinant-based** finite-difference estimator using converged displaced SCF solutions and cross-geometry AO overlap matrices. No additional dependency is installed.

## Physical definition and implementation

For a real, normalized electronic state, the nuclear diagonal correction is the mass-weighted diagonal quantum metric:

\[ E_{\rm DBOC}(R)=\sum_{A\alpha} {1\over 2M_A}\left(\langle \partial_{A\alpha}\Psi|\partial_{A\alpha}\Psi\rangle - |\langle\Psi|\partial_{A\alpha}\Psi\rangle|^2\right). \]

For each nuclear Cartesian coordinate \(q_{A\alpha}\), compute the electronic *determinant* overlap between independently SCF-optimized states at \(q-h\) and \(q+h\), with moving AO centers:

\[ F(h)=|\langle D(q-h)|D(q+h)\rangle|^2,\quad g_{A\alpha}(h)={1-F(h)\over4h^2}. \]

Occupied spin-orbital overlaps are computed from **actual cross-geometry AO integrals** `pyscf.gto.intor_cross('int1e_ovlp', mol_minus, mol_plus)`; alpha and beta determinant overlaps multiply. This correctly accounts for distinct RHF, ROHF and UHF occupations (only the requested HF reference is used). Different orbital *phases* and occupied rotations do not change the determinant fidelity. All six nuclear Cartesian components (two atoms x three axes) are included; nuclear masses and their source must be provided explicitly. The chosen convention is lab-frame, Cartesian, clamped-nuclei electronic derivatives. Center-of-mass/translation treatment needs separate benchmarking before scientific use.

Two explicitly specified step sizes yield **observed step sensitivity only**. The estimator is positive for each species; the candidate *EA* contribution obeys `DBOC_neutral - DBOC_anion` (same nuclear isotopologue). No step difference is promoted to a bound. SCF root changes/low overlap, incomplete convergence, model mismatches and insufficient compute budget block output.

## Limitations and validation status

- **Only HF determinants**, not CCSD(T), correlated DBOC, MR-DBOC, nor a reliable correction to an existing CCSD(T)/NEVPT2 energy without a correlation-error study.
- **No off-diagonal nonadiabatic corrections**, mass-polarization treatment or certification of a full beyond-BO molecular Hamiltonian.
- Requires *scientifically reviewed* electronic state and explicitly sourced **nuclear**, not mean atomic, isotope masses. Smoke marks review true solely to exercise software; it is not production evidence.
- Sensitive to SCF reference/stability, orbital basis, finite-difference step, isotope and the geometric coordinate convention; must be benchmarked against an independently validated correlated DBOC before production.
- Outputs `COMPLETE_REVIEW_REQUIRED` or `UNRESOLVED`; never closes the OpenEA nuclear remainder, G3d, uncertainty budget or scientific EA on its own.

### Artemis tests

```bash
cd /srv/storage/homes/analysis/dschmid/photodetachment/openea-benchmark
source .venv/bin/activate
python -m pytest -q tests/test_dboc_finite_difference_v1.py
nice -n 19 python scripts/openea_hf_dboc_smoke.py
```

### Suggested science validation before any upgrade

Compare independently against HF DBOC values for H2, LiH or BH using the **same** isotope masses, electronic reference, geometry, Cartesian convention and basis. Then test basis and step convergence for neutral and anion separately; evaluate correlated DBOC-minus-HF DBOC and the beyond-diagonal nonadiabatic remainder. If an adequate open-source correlated method is unavailable, keep these terms unresolved or establish a defensible bound by other independently validated evidence.
