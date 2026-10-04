# OpenEA v1 — Atomic fragment SCF retry policy

The d-aug-cc-pV5Z OH validation completed both expensive molecular PECs before
`O-(2P)` returned `SCF_NOT_CONVERGED`. The fragment runner now keeps the same
RHF/ROHF problem and convergence tolerances but, after an exhausted primary
DIIS run, continues from those orbitals with PySCF's second-order CIAH/Newton
optimizer. This is a numerical retry, not a change of electronic-structure
method.

The result records the primary/Newton solver path and convergence status.
Failure after Newton remains `SCF_NOT_CONVERGED`; it is never accepted silently.
Existing stage checkpoints mean the d-aug neutral and anion PECs are reused on
retry, so only the failed atomic-fragment path continues.

`openea_status.py` now also reports total elapsed time and the age of the current
step, which makes long detached runs easier to interpret.
