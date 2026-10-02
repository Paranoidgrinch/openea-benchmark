# OpenEA Stage-3 → Attachment Bridge v0.1

This bridge converts a successfully converged Stage-3 high-level PEC loop into
the generic attachment-layer `ElectronicState` and `PECBranch` records.

Required gates:
- Stage-3 loop status `CONVERGED`
- terminal refinement action `BRACKET_TARGET_MET`
- final PEC `READY_FOR_DISCRETE_MINIMUM_SCOUT`
- same-geometry initialization identity `CLEARED`
- geometry continuity `CLEARED`
- exactly one `BRACKETED_SINGLE_MINIMUM`
- all final PEC points accepted

Scientific boundary:
Stage 3 supplies a *discrete sampled minimum* and a converged geometry bracket.
It does not yet supply a defensible two-sided energy interval for the true
continuous equilibrium minimum. The bridge therefore records the discrete
minimum and explicitly leaves `energy_interval_hartree=None`.

No Stage-3 energy is silently promoted into a production adiabatic EA.
The next layer must resolve equilibrium geometry/energy and its uncertainty.
