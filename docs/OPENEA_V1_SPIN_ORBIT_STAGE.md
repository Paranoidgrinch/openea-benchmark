# OpenEA v1 - first-order spin-orbit layer (Patch 47)

Experimental EAs connect the lowest spin-orbit levels of anion and neutral; all
CCSD(T)/CBS energies are spin-free.  `spin_orbit_stage.py` adds

    Delta_EA_SO = Delta_SO(neutral) - Delta_SO(anion),
    Delta_SO    = E(lowest SO level) - E(spin-free)

for the selected states of every automated run (`--no-soc` skips it).

- Sigma states and singlets: no first-order shift (`FIRST_ORDER_ZERO`); the
  second-order shift is not computed (recorded).
- Pi states: OpenMolcas, ANO-RCC-VTZP, AMFI mean-field SO integrals, full-valence
  CASSCF (ns, np of each atom, 1s for H/He, noble-gas cores inactive) for each
  Lambda component (C2v b1, b2), RASSI spin-orbit interaction of the two.
  The two spin-free component energies must be degenerate.
- d-valence atoms, Delta/Phi states: `SOC_ACTIVE_SPACE_NOT_DERIVED` (fail closed).
- Bound: 30 % of |Delta_SO| per state (pilot).  Calibration against measured
  splittings, not used in the computation:

  | state | computed (cm-1) | experiment (cm-1) |
  |---|---|---|
  | OH X2Pi | 133 | 139 |
  | NO X2Pi | 126 | 123 |
  | O2- X2Pig | 153 | ~160 |
  | ClO X2Pi | 244 | 320 |

  A minimal open-shell CAS gave ClO 196 cm-1; the ROHF-like pi* is too
  O-localized.  The ClO remainder needs dynamic correlation (MRCI-level).
- Assembly: included only when CLEARED; the bound widens the EA_0 interval.
