# OpenEA v1 — Symmetry-labelled state candidates from two atoms (Patch 33, redesign step 2)

`state_candidates.py` turns two element symbols into the list of molecular electronic
states that the workflow must cover, each with exact diatomic labels (2S+1, Lambda,
+/- for Sigma, g/u for like nuclei). No atomic or molecular reference data are read.

1. **Atomic configurations**: Madelung aufbau for the ground configuration; single
   promotions ns->np, ns->(n-1)d and (n-1)d->ns are added as alternatives, so the
   lowest configuration of e.g. transition metals is not guessed.
2. **Atomic terms**: Russell-Saunders terms (2S+1, L, parity) by exact microstate counting
   over all open subshells.
3. **Channels**: neutral molecule A+B, A+ + B-, A- + B+; anion A- + B, A + B-.
4. **Wigner-Witmer**: spin coupling |S_A-S_B|..S_A+S_B; Lambda from m_A + m_B; Sigma +/-
   from sigma_v|m_A,m_B> = w_A w_B (-1)^(L_A+L_B) |-m_A,-m_B>; g/u for identical atoms in
   identical states from the inversion = (-1)^N_atom (-1)^(2S_atom - S) x orbital swap;
   unlike states of like atoms give each state as g and u.

Validated against Herzberg's correlation tables: O(3P)+H(2S), C(3P)+N(4S), O(3P)+O(3P),
N(4S)+N(4S), F(2P)+F(2P), O(3P)+O(1D).

The candidate set says *which* symmetries exist; which one is the ground state remains a
computational question. Example sizes (charge 0 / -1): OH 7/6 symmetries (2S {1,3}/{0,2}),
O2 29/18, AlO 19/20, FeH 31/32.

Next redesign steps: compute per symmetry block (PySCF symmetry=True, Dooh/Coov irreps),
follow only symmetry-preserving instabilities, track PECs per block.

## Redesign step 3 — symmetry-block SCF and symmetry scout (Patches 34/35)

`symmetry_states.py` runs SCF in the abelian subgroups C2v (heteronuclear) / D2h
(homonuclear). In PySCF's real Coov/Dooh irreps the x/y components of a pi shell are
coupled, so a Pi state with pi_x^2 pi_y^1 has no stationary point there (OH X2Pi did not
converge); in C2v/D2h both components are independent. |m| of every orbital is measured
from <Lz^2> (int1e_cg_irxp), labels follow from the occupied orbitals: at most one unpaired
x/y component overall, else the determinant mixes Lambda and is reported as MR-only.
Symmetry-preserving instabilities are followed, symmetry-breaking ones only recorded.
x/y partner occupations of the same Lambda state are kept once.

`symmetry_scout.py` (script `scripts/openea_symmetry_scout.py A B --output DIR`): from two
element symbols, Wigner-Witmer labels -> symmetric reference at the covalent-radius start
geometry -> single-determinant occupations per label -> seeded R scan per block (0.05 Å
steps near the start, growing to 0.5 Å) until a bracketed minimum or a large-R plateau.
A bracket left coarse by the grown steps is bisected (seeded from the minimum point, at
most `max_bracket_refinements`) until both neighbours lie within one fine step.
A label change along a scan is never accepted.

First real runs (PBE0/def2-TZVPPD):
- OH: X2Pi R=0.97 Å, E=-75.688729 (identical to the old scout), OH- X1Sigma+ R=0.97 Å;
  2Delta/2Sigma- and OH- 1Delta/1Pi reported MR-only.
- O2: X3Sigma_g- R=1.17 Å with no symmetry-breaking instability at any scan point (the
  state that broke the old Stage-3 continuity), O2- X2Pi_g R=1.32 Å; a1Delta_g, b1Sigma_g+
  etc. reported MR-only.

## Redesign step 3c — two-atom input end to end (Patch 36)

`python -m openea_benchmark.workflow --mode symmetry-auto --atoms A B --output DIR`

symmetry scout -> DFT fragment asymptotes (open-PEC executor) -> legacy summary export
(`to_legacy_summary`) + synthesized manifest (`manifest_for_two_atoms`) -> benchmark-auto
acquisition -> electronic EA stage. No YAML system entry, spin list or start geometry.

- Stage-3 requests carry `point_group`, `irrep_nelec`, `symmetry_label`; the ROHF/RHF
  reference is solved in the same symmetry block as the scout, also in refinement and
  nuclear-motion requests, so state identity is fixed by symmetry from DFT to CCSD(T).
- Dissociation-limit bound: the ground state of a charge lies at or below its lowest
  dissociation limit. A block whose lowest sampled scout energy (scanned to large R)
  is above the lowest computed fragment limit of its charge by more than 1.0 eV is
  `above_lowest_dissociation_limit` (resolved, not a Stage-3 seed, evidence recorded).
- Labels without single-determinant representation are listed as
  `mr_only_candidates_not_computed` in auto_state (explicit open gap).

First run, O + H: Stage-3 RELEASED, 20/20 points, automated D04 cleared (X2Pi / X1Sigma+),
electronic EA 1.7044 eV (def2-TZVPPD) - identical to the YAML-driven route with half of the
CCSD(T) points; 14 high-lying blocks excluded by the dissociation-limit bound.

## Patch 37 — coarse scout brackets and unbracketed high-level curves (wave-0 ClO)

ClO- 3Pi lies far from the covalent start; the grown steps bracketed its DFT minimum only
between 2.24 and 2.87 Å, the CCSD(T) grid at 2.44-2.54 Å was still downhill, and the
competitor (0.95 eV above X1Sigma+) blocked D04 under the 1 eV margin for unresolved
candidates. Fixes: (1) the scout bisects coarse brackets to one fine step; (2) the
electronic-EA stage also refines curves whose high-level minimum is not bracketed
(NO_BRACKETED_MINIMUM) - the canonical loop extends towards the lower energy - with
`max_refinement_rounds` 12.

## Redesign step 4 — Lambda and g/u of multireference roots (Patch 53)

`mr_competitors.py` bounds MR-only Wigner-Witmer candidates by SA-CASSCF/SC-NEVPT2 roots
of their (charge, spin, C2v irrep) block. A C2v block mixes Lambda values (B1: Pi, Phi;
A2: Sigma-, Delta, Gamma) and, for like atoms, g and u. The lowest root can therefore be a
state Stage 3 already computes: C2 X1Sigma_g+ had its 3Phi candidates blocked by a3Pi_u
(0.17 eV < 0.3 eV margin), final verdict UNRESOLVED.

Each of the 6 lowest CASCI roots is now classified before NEVPT2 (which rotates the
CASCI object to a reordered semicanonical basis):
- Lambda = round(sqrt(<Lz^2>)) over the active orbitals (`int1e_cg_irxp`), accepted only
  within 0.25 and only if allowed in the block (A1/A2 even, B1/B2 odd; a Pi/Phi mixture
  with <Lz^2> near 4 in B1 stays unassigned);
- like atoms: g/u = <inversion> of the CI vector from the D2h labels of the active
  orbitals (accepted within 0.1 of +-1).
A candidate's bound is the lowest root of its Lambda (and parity), unassigned roots
included; if none of the computed roots qualifies, the highest computed root. Caches carry
`root_classification` and older block caches are recomputed (~10 s per point).

C2 (def2-TZVPPD): 3B1 roots a3Pi_u, d3Pi_g (2.4 eV), first possible 3Phi root 5.9 eV
above X -> excluded. C2 cc-pVDZ checks: X1Sigma_g+ (A1: Lambda 0, g), B1Delta_g (A1 and A2:
Lambda 2, g), C2- A2Pi_u (B1: Lambda 1, u).
