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
