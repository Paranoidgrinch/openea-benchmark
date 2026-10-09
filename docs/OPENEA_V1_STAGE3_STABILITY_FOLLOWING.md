# OpenEA v1 — Stage-3 SCF stability following and partial PEC reviews (Patch 29)

## Problem found by the wave-0 benchmark (2026-10-09)

`SCF_UNSTABLE` was terminal in Stage-3: any internal HF instability or any RHF->UHF
external instability discarded the point, and a single failed point blocked the
identity/continuity review of the whole state. 6 of 9 wave-0 systems lost states this
way, including the ground states of O2, O2-, AlO, MgH- and (via one point) N2.
For MgH the missing singlet anion produced a wrong-sign provisional EA.

## D02: internal instability following

`Stage3ExecutionSettings.stability_following_max_steps` (CLI `--stability-following-steps`,
default 4 in `benchmark-auto`, 0 = legacy) restarts the SCF along PySCF's internal
instability direction **within the same RHF/ROHF space** until internally stable.
This is the architecture's D02 escalation "reoptimize unstable solution". A
non-converged start is also analyzed when following is enabled.

The followed solution is a lower HF stationary point and may be a *different*
electronic state than the scout seed. It is therefore not trusted: it goes through
the unchanged Stage-3 initialization-identity and geometry-continuity review.
Recorded per point: `stability_following_steps`, `stability_following_lowering_hartree`,
warning `INTERNAL_INSTABILITY_FOLLOWED`. Still unstable after the budget -> `SCF_UNSTABLE`.

Wave-0 prototype: AlO X, O2 X, O2-, SiC singlet, CN- triplet all became internally
stable after 1–2 steps (lowering 0.4–17 mEh).

## D03 evidence: RHF -> UHF external instability

A closed-shell singlet is **not** switched to a spin-contaminated UHF reference for
CCSD(T). With `rhf_external_instability_policy="RECORD_AND_CONTINUE"` (CLI
`--rhf-external-instability`, default in `benchmark-auto`; legacy `FAIL`):

- RHF -> RCCSD(T) proceeds on the (internally stable) RHF reference;
- a diagnostic broken-symmetry UHF is converged along the external instability and
  `rhf_external_uhf_lowering_hartree = E_RHF - E_UHF` and `rhf_external_uhf_s2` are recorded;
- warning `RHF_EXTERNAL_INSTABILITY_RECORDED` is attached.

No threshold classifies these numbers here; they are magnitude evidence for the
reference-character gate. Wave-0 prototype values span the full range:
N2 (R=1.15 Å) 0.1 mEh / <S^2>=0.07, MgH- 3 mEh / 0.46, NO- singlet 52 mEh / 1.03,
SiC singlet 60 mEh / 1.30 — the last two are genuine open-shell singlets that
need the MR route, the first is benign.

`auto_state.json` carries `reference_warning_summary` per job (counts and maxima).

## Partial PEC reviews

`benchmark-auto` no longer waits forever for a failed point. Once nothing is pending,
the identity/continuity review runs on the completed subset, lists every
`excluded_failed_points` entry and is labelled `DIAGNOSTIC_ONLY_PARTIAL` with
`complete_point_coverage=false`. A failed initialization is never silently replaced
by a successful sibling (Stage-3 retry contract); the partial label keeps that visible.

## Compatibility

Defaults of the library settings are unchanged (fail-closed). The new settings enter
the point signature, so a re-run with the new CLI defaults must use a fresh
`--acquisition-dir` (e.g. `benchmark_auto_v2`); old records stay intact.
`dump_input=False` stops PySCF from copying the calling script into every log.
