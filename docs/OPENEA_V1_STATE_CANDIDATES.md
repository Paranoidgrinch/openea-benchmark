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

## Redesign step 5 — MR term-gap route for MR-only states (Patch 54)

Until now an MR-only candidate (1Delta, 1Sigma-, open-shell singlets, ...) that the lower
bound of step 4 could not exclude kept the run UNRESOLVED. Such a state can be the ground
state of the neutral or the anion (likely for transition-metal diatomics), so the
workflow must compute it. `mr_term_gap.py`:

- gap0 = E0(MR state) - E0(selected SR state of the same charge), both followed as *the*
  root of their Lambda (and g/u) in their own (spin, C2v) block
  (`mr_competitors.block_energies`), never as the lowest root of a block;
- each minimum: downhill walk on a 0.025 A grid from the competitor window, quartic
  through five points; Delta ZPE harmonic at the two minima;
- basis: gap at the TZ minima recomputed in def2-QZVPPD (central value; difference =
  basis term); method term `method_uncertainty_ev` (pilot, see validation below);
- E0(MR state) = E0(SR state, CCSD(T)/CBS + all corrections) + gap0. Per charge the
  ground-state offset min(0, gap0) has the interval [min(0, lo), min(0, hi)], so an
  overlapping gap widens the EA instead of forcing a choice;
- corrections computed for the SR state (CV, SR, post-CCSD(T)) are transferred with
  their full size as extra uncertainty when an MR state can be the ground state; an MR
  state with S > 0 and Lambda > 0 that can be the ground state blocks (first-order
  spin-orbit not computed);
- final resolver: an open MR competitor block no longer blocks when the term-gap stage is
  CLEARED. Only main-group atoms: `full_valence_space` has no d shells yet.

Orbital start (block cache version 3, also for the competitor bounds): NH 1Delta landed at
two R on a CASSCF solution with split pi orbitals (<Lz^2> 3.955 instead of 4, NEVPT2
10 mEh too high, a kink in the curve). Each block now also starts from the ROHF two spins
higher; among converged solutions, one whose lowest root has pure Lambda
(|sqrt(<Lz^2>) - Lambda| < 0.005) wins, then the lower CASCI root.

Validation (`scripts/openea_mr_term_gap_validation.py`, runs/mr_term_gap_validation;
measured T0 after Huber & Herzberg, used only for this comparison; start R = sum of
covalent radii):

| state | gap0 (eV) | measured T0 | error | half-width |
|---|---|---|---|---|
| O2 a1Delta_g | 0.953 | 0.977 | -0.024 | 0.159 |
| O2 b1Sigma_g+ | 1.582 | 1.627 | -0.045 | 0.162 |
| NH a1Delta | 1.491 | 1.558 | -0.067 | 0.181 |
| NF a1Delta | 1.360 | 1.418 | -0.058 | 0.194 |
| SO a1Delta | 0.765 | 0.728 | +0.037 | 0.171 |
| S2 a1Delta_g | 0.574 | 0.545 | +0.029 | 0.180 |

Largest error 0.067 eV, below the pilot method term of 0.15 eV; every measured value lies
inside its interval. Basis shifts TZ -> QZ: -0.007 to -0.043 eV. The seven-point check
flags the S2 1Delta kink (ZPE fit spread 0.014 eV).

## Patch 55 — robust MR blocks, 3d metals, spin-orbit for any Lambda

- **Block solver (cache version 4):** when no CASSCF of a block converges, the
  converged single-root CASSCF orbitals of another block of the same charge and spin
  are used (ClO- 1A2: NEVPT2 on SCF orbitals had put 1Delta 8.3 eV *below* X1Sigma+, so
  the run stayed UNRESOLVED; now 5.6 eV above, A1/A2 components within 0.25 eV). Roots
  whose NEVPT2 correction deviates > 0.1 Eh from the block median are intruders: CASCI
  energy + median correction, flagged.
- **3d transition metals:** valence frozen core [Ar] for Sc-Zn (PySCF froze only [Ne]);
  full-valence CAS 4s 3d for the metal (FeO: CAS(14,10), Cr2: CAS(12,12)). 4d and
  heavier fail closed (def2 ECPs).
- **Spin-orbit for any Lambda:** in C2 both components of a Lambda > 0 state share one
  symmetry; a state average over the lowest roots and RASSI |<i|Lz|j>| = Lambda
  identify the pair, a second RASSI couples it. TiO X3Delta 199 cm-1 (measured 4A =
  202), NO 129 (123), ClO 271 (320). MR states with S > 0 and Lambda > 0 get their
  spin-orbit shift in the term gap instead of blocking.
- Tested and not adopted for the term gap (O2 a1Delta_g, measured Te 0.982 eV):
  MRCI+Q/aug-cc-pVTZ 1.030; DEA-EOMCC(3p-1h)/aug-cc-pVTZ 0.792; with a (4p-2h)
  cc-pVDZ increment 1.067. NEVPT2/QZ gives 0.958; it stays the method.

## Patch 55b — fixes found in the validation reruns

- Block cache v4 replaced "intruder" roots by the median NEVPT2 correction. High
  roots (2-3 eV up) legitimately carry larger corrections, so the median moved and
  the physical low roots were replaced: NH a1Delta came out at -1.2 eV. Reverted
  (v5): the deviation from the lowest root's correction is only recorded.
- v6: when CASSCF mixes pi_g and pi_u (both b1 in C2v), the D2h orbital labels fail
  and g/u was unknown (O2 a1Delta_g lost at 1.225 A). The parity of each root now
  comes from the CI vector under the inversion operator in the active-orbital basis
  (`root_parity_general`, AO inversion: swap atoms, (-1)^l).

## MR-only adiabatic EA (checked, not adopted)

A fallback EA from SC-NEVPT2 curves alone (full valence, def2-TZVPPD minima,
def2-QZVPPD, harmonic ZPE; `scripts/openea_mr_ea_validation.py`) against the nine
measured EAs of the benchmark set: OH 1.483 (1.828), CN 3.509 (3.862), NO -0.217
(0.026), O2 0.101 (0.448), MgH 0.531 (0.90), SiC 2.154 (2.396), C2 3.005 (3.269),
AlO 2.170 (2.60), ClO 1.899 (2.276) eV: mean error -0.33, range -0.24 to -0.43 eV.
Full-valence NEVPT2 misses most of the extra electron's dynamic correlation; the
same-charge term gaps (errors <= 0.07 eV) do not suffer from this. Strong MR
character of the SR states therefore stays in the CC hierarchy (CCSDT, CCSDTQ
increments of the post-CCSD(T) layer, whose size is the uncertainty).

Term-gap validation with cache v6 (ten measured T0): O2 a 0.954 (0.977), O2 b 1.582
(1.627), NH a 1.491 (1.558), NF a 1.360 (1.418), SO a 0.764 (0.728), S2 a 0.574
(0.545), NH b 2.643 (2.629), NF b 2.346 (2.341), SO b 1.271 (1.303), S2 b 1.024
(0.990) eV: mean -0.011, RMS 0.039, max 0.067 eV. The method term is now 0.10 eV
(1.5 x the largest error), replacing the pilot 0.15 eV.
