# OpenEA v1 — Atomic two-electron RHF external-stability policy

The real OH/OH- smoke test exposed a specific H- reference issue:

- H- is a closed-shell two-electron atom.
- PySCF reports the converged RHF solution as externally unstable with respect
  to the RHF -> UHF relaxation channel.
- Internal RHF stability is nevertheless cleared.

OpenEA must not disable RHF external-stability checking globally. Instead,
`fragment_execution.py` now contains a deliberately narrow exception:

    atomic
    AND closed shell
    AND exactly two electrons
    AND internal stability cleared

Only for that case may a failed RHF external-stability diagnostic be retained
as evidence while the CCSD calculation proceeds.

The result records:

- `external_stable = false`
- `external_instability_waived = true`
- `external_instability_waiver_reason =
  ATOMIC_CLOSED_SHELL_TWO_ELECTRON_CCSD_COMPLETE_EXCITATION_SPACE`
- `two_electron_ccsd_exact_space = true`

The waiver does not apply to:
- molecules,
- open-shell fragments,
- atoms with more or fewer than two electrons.

CCSD spans the complete excitation rank for a two-electron system in a fixed
one-particle basis. This is why the special case can be treated differently
without weakening the molecular Stage-3 SCF-stability contract.

This remains validation-only fragment evidence:
`is_production_threshold = false`.

The OH/OH- smoke default basis is also changed from STO-3G to aug-cc-pVDZ,
because STO-3G does not provide a meaningful virtual/diffuse space for the
anion fragment calculations.
