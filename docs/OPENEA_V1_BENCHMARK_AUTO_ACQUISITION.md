# OpenEA v1: Automated benchmark evidence acquisition (Patch 27)

## Problem corrected

The old `python -m openea_benchmark.workflow` CLI stopped at `dft-scout`.
Stage-2 state selection, adaptive D04/D09 evaluation, Stage-3 job planning,
CCSD(T) point execution, and high-level PEC-identity review existed but required
separate manual calls.  Patch 27 connects those **existing implementations**.
It is the first orchestration increment, **not** a scientifically complete
adiabatic-EA production workflow.

## One-command invocation

Run within the established OpenEA `.venv` (PySCF 2.14.0) with reduced HPC
scheduling priority (`nice -n 19`):

```bash
nice -n 19 python -m openea_benchmark.workflow \
  /srv/storage/homes/analysis/dschmid/photodetachment/openea_diverse_ea_validation_v1/benchmark/wave0_legacy_dft_scout_compat.yaml \
  --mode benchmark-auto --system OH \
  --output /srv/storage/homes/analysis/dschmid/photodetachment/openea_diverse_wave0_scout \
  --resume --threads 4 --max-stage3-points 100
```

`--mode benchmark-auto` performs/reuses the DFT scout, composes Stage-2 state
selection and the canonical adaptive Stage-3 bridge, and writes complete plan,
review, and numerical-execution records. It executes all reproducibly linked
bracketed PEC seeds as **provisional data-acquisition points**, even if the
D04/D09 ground-state/large-R prerequisites are still open. It DOES NOT change
the canonical bridge's release status or claim the Stage-3 comparison has been
scientifically released.  All candidates are retained; no energetic threshold
prunes high-spin/open PECs. For each point, the canonical bounded numerical
retry policy is used.  On return the controller attempts the existing Stage-3
identity/PEC review with **explicit weekend pilot thresholds** and labels every
such review diagnostic-only.

Two controls:
- `--no-provisional-stage3` restricts execution to requests formally released
  by the existing D04/D09 bridge (OH will remain deferred).
- `--auto-dry-run` runs planning without expensive Stage-3 calculations.

`--max-stage3-points N` bounds the number of **new geometry/initialization
requests** started per invocation. The command can be rerun unchanged to resume
pending points. It does not count retries as new requests; retry policy remains
bounded independently. `--threads` controls the PySCF thread count and
`--stage3-memory-mb` controls single-point memory. Use one process per run
output directory, not concurrent invocations in the same directory.

## Persisted files

In `<output>/OH/benchmark_auto/`:

- `state_selection_report.json`: all bracketed and still-open Stage-2 components.
- `stage3_plan.json`: all canonical Stage-3 jobs and checkpoint provenance.
- `points/<SHA256(request_id)>.json`: atomic, resumable result with source
  checkpoint content hash, request/settings signature, SCF/CC energies/status,
  numerical attempt history, and explicit provisional-vs-released purpose.
- `stage3_pec_reviews.json`: automatic high-level initialization/continuity
  review for completed jobs; **diagnostic thresholds only**.
- `auto_state.json`: machine-readable status of completed, failed, pending, and
  blocked jobs, competing/open states and their **observed, non-bounding** DFT
  gaps, next actions and the unclosed scientific gates.

Files are saved atomically. Resumption refuses conflicting manifest/scout
identities and refuses stale results if source checkpoint contents change. A
missing source checkpoint blocks the affected job, not the entire benchmark.
Errors are recorded per point, not silently called convergence.

## Scientific boundaries (not optional)

- A bracketed DFT minimum is not a global ground-state assignment.
- Unbracketed PECs and the asymptote prerequisite remain active independent of
  observed positive DFT energy gaps. In particular OH high-spin open branches
  **cannot be discarded** from a +5.7/+18.1/+9.7 eV sampled gap.
- Stage-3 points collected provisionally cannot clear D04/D09 or become a
  certified production EA. The current `auto_state.json` outcome must remain
  `UNRESOLVED`, with `ea_computed=false`.
- The automatic high-level identity/continuity check uses the *existing weekend
  pilot thresholds*, not validated molecule-independent production thresholds.
- SOC, MR, nuclear motion/ZPE, CBS, state competition closure, attachment,
  error-budget and bound/unbound resolution are not automatically connected
  end-to-end by Patch 27. Those remain explicit subsequent integration tasks.
- No experimental reference EA, calibration target or validation outcome enters
  the computational route. External experimental comparison occurs only after
  a defensible prediction is independently frozen.

## Validation

Unit tests exercise open high-spin sectors, provisional scope, no-pruning,
point budgeting, retries, resume, source-checkpoint hashing and stale-manifest
rejection. Genuine PySCF end-to-end validation is required on Artemis after
applying the patch. Local import-only PySCF stubs do not constitute a backend
smoke test.
