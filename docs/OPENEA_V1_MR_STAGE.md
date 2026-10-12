# OpenEA v1 - automated multireference layer (Patches 50, 51)

Two general tasks, both with full-valence CASSCF (ns, np of each atom; 1s for
H/He; noble-gas cores inactive) and strongly contracted NEVPT2 in def2-TZVPPD,
C2v symmetry blocks (homonuclear D2h labels mapped onto C2v).  Neither changes
the electronic EA, which stays CCSD(T)/CBS.

## Patch 50: multireference nuclear curve (`mr_nuclear_curve.py`)

Used by the zero-point layer only after both single-reference routes failed:
the selected CCSD(T) curve could not be extended and the reference-branch
fallback (Patch 45) did not help either.  AlO X2Sigma+ is the motivating case: the
seed ROHF branch ends at 1.69 A, the followed branch changes character at 1.46 A.

- Neutral and anion both get an MR curve (the nuclear solver requires one method).
- Each curve: state's spin and C2v block, lowest root, walk outward from the TZ
  equilibrium in 0.025 A steps, seeded from the neighbour (projected CASSCF
  orbitals; a fresh SCF start if that does not converge), until 0.6 eV above the
  minimum on both sides.  Every point runs a converged SCF first: NEVPT2 reads the
  SCF object, and an unconverged placeholder gave Hartree-sized errors.
- Continuity: each NEVPT2 energy within 0.5 mEh of the cubic through its four
  nearest neighbours; active natural occupations change by < 0.1 per step.
- Model term: relative harmonic-frequency difference to the selected CCSD(T)
  curve (quartic fit within +-0.08 A, second derivative at the fitted minimum),
  times that role's ZPE, added to the Delta_ZPE half-width.

## Patch 51: MR-only competitor states (`mr_competitors.py`)

The scout lists Wigner-Witmer states no single high-spin determinant represents
(1Delta, open-shell singlets, Sigma- from pi^2, Phi, Gamma).  They were an explicit
gap in every run.  Now:

- Each lies in a (charge, 2S+1, C2v irrep) block: Sigma+ -> A1, Sigma- -> A2,
  Delta/Gamma -> A2 (A1 also holds Sigma+), Pi/Phi -> B1.
- Per block: two-root SA-CASSCF + per-root NEVPT2 on an R window (-0.15 ... +0.25 A
  around the selected state's equilibrium).  The lowest root is a lower bound for
  every state in that block.  In the selected state's own block the second root is
  the bound.
- Excluded when the bound lies > 0.3 eV above the selected state at the same
  level; otherwise `UNRESOLVED_MR_COMPETITOR`.
- Robustness: a stalled 2-root average is retried with 3 roots, then 1 root (a
  block can hold fewer states than requested).  If CASSCF diverges entirely, CASCI
  on the SCF orbitals + NEVPT2 is used; that has no lower-bound property, so such a
  block needs > 1.0 eV.
- The run state lists only the candidates that remain open.
