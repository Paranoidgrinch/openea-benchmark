# OpenEA v1 — NumPy 2.4 / PySCF Newton compatibility

The real OH d-aug-cc-pV5Z validation reached the newly enabled atomic-fragment
Newton/CIAH retry and failed with:

    AttributeError: module 'numpy.linalg' has no attribute 'linalg'

NumPy 2.4 removed the deprecated module alias `numpy.linalg.linalg`; the
implementation module is now private as `numpy.linalg._linalg` and public
linear-algebra functions live under `numpy.linalg`.

OpenEA does not downgrade NumPy globally.  Instead, only during the PySCF
Newton/CIAH call, it temporarily restores the removed legacy module alias and
points it to NumPy's `_linalg` implementation module.  The alias is removed
immediately after the Newton call.

The retry now also explicitly seeds PySCF Newton with the orbitals and
occupancies produced by the failed primary ROHF attempt.

If Newton raises for any reason, OpenEA preserves:

- the primary SCF energy;
- `PRIMARY_DIIS` in the solver path;
- `CIAH_NEWTON_FROM_PRIMARY_ORBITALS`;
- whether the NumPy compatibility alias was required;
- `scf_newton_attempted=True`;
- the concrete exception and NumPy version.

This is a numerical compatibility layer only.  Charge, spin, basis, reference
state, HF equations and downstream CCSD(T) method are unchanged.
