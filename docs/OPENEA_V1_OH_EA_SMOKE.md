# OpenEA v1 — Real OH/OH- Electronic-EA Smoke Test

This validation layer supplies the missing numerical dissociation-threshold
evidence needed for a real end-to-end OH/OH- smoke run.

## Atomic fragment execution

`attachment/fragment_execution.py` implements an auditable *atomic* CCSD(T)
runner using the same Stage-3 numerical settings:

- closed shell: RHF -> RCCSD(T)
- open shell: ROHF -> semicanonical UHF -> UCCSD(T)
- internal HF stability required
- RHF external stability checked when requested

The requested atomic charge/spin/state is explicit input. v0.1 does **not**
perform an autonomous atomic state-manifold search, so the resulting threshold
is labelled validation-only and cannot be promoted to production evidence.

For OH-, the smoke script evaluates both explicitly requested channels:

    OH- -> O-(2P) + H(2S)
    OH- -> O(3P)  + H-(1S)

The lower computed scalar threshold is used automatically by the existing
binding analyzer.

## End-to-end smoke

`scripts/attachment_oh_ea_smoke.py` runs:

1. adaptive Stage-3 CCSD(T) PEC for OH neutral,
2. adaptive Stage-3 CCSD(T) PEC for OH-,
3. the four atomic fragment calculations,
4. equilibrium-energy resolution for both molecular PECs,
5. anion dissociation assessment,
6. electronic EA interval decision.

Default `sto-3g` and all thresholds are validation-only. The result explicitly
remains:

- `is_electronic_ea_only = true`
- `includes_zpe = false`
- `is_production_ea = false`
- `ground_state_assigned = false`
- `authorizes_pruning = false`

The smoke succeeds when the numerical path executes consistently. It does not
require the cheap-basis OH EA to match experiment and does not convert a
validation number into a scientific benchmark.
