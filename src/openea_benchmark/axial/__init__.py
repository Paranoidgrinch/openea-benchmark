"""Lz-adapted (axial) orbitals for diatomics and atoms on the z axis (Route B, 2026-11-24).

Real orbitals represent a Hund component (maximal M_L at maximal M_S) as one determinant
only for some occupations: pi^1, pi^2 3Sigma-, f^0,1,6,7,8,13,14 ...  1Delta (pi+^2) and
open f^2-5, f^9-12 need orbitals that are eigenfunctions of Lz, R(rho, z) e^{i m phi}.
Integrals over such orbitals are real (the phi integration conserves m), so SCF and
coupled cluster run in real arithmetic; only the permutational symmetry drops from 8- to
4-fold.  CCpy accepts these integrals unchanged (CCSD/CCSDT of one determinant written
in real and in axial orbitals agree to 3e-16 Eh).

basis: axial AO basis and integrals; scf: ROHF with a fixed occupation per m block;
cc: CCpy coupled cluster on the axial determinant (ROHF and semicanonical UHF form).
"""
