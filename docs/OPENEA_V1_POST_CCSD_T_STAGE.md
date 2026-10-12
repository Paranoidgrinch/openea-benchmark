# OpenEA v1 - automated post-CCSD(T) layer (Patch 49)

`post_ccsd_t_stage.py` (`--no-post-cc` skips it), selected states at their TZ
equilibrium, frozen noble-gas core, CCpy for every method:

    Delta_T = EA[CCSDT]  - EA[CCSD(T)]   aug-cc-pVDZ and aug-cc-pVTZ
    Delta_Q = EA[CCSDTQ] - EA[CCSDT]     first affordable of aug-cc-pVDZ, cc-pVDZ

- Reference: ROHF/RHF converged in the state's symmetry block (seeded from its
  high-level checkpoint), then carried into C1 for CCpy; the C1 energy must equal
  the symmetry-block energy (1e-7 Eh).
- Central/bound: Delta_T(aTZ), |Delta_T(aTZ) - Delta_T(aDZ)|; Delta_Q, 50 % of
  |Delta_Q| (pilot).
- CCSDTQ storage ~ (o v)^4; estimate (MB) per basis, limit 120 GB.  aug-cc-pVDZ
  fits only hydrides (OH 16 GB, MgH 1.4 GB); CN, NO, O2, ClO, SiC, C2, AlO, N2
  need 120-540 GB there and 11-38 GB in cc-pVDZ.
- Triples and quadruples partly cancel: the layer enters EA_0 only when both are
  available (`CLEARED`); otherwise `QUADRUPLES_NOT_AFFORDABLE` with Delta_T as
  evidence.
- CR-CC(2,4) is not used (OH study: its DZ->TZ transfer disagreed with exact
  CCSDTQ by 0.1 eV).
- Caveat: cc-pVDZ has no diffuse functions; the anion is poorly described there
  (OH- is unbound in cc-pVDZ CCSD(T)).  The quadruples increment is a correlation
  difference, but this is why the bound is 50 %.
