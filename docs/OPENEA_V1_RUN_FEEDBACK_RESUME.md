# OpenEA v1 — Persistent progress, resume, and detached runs

Long calculations must remain observable and recoverable.

`ProgressReporter` writes the current step, completed steps, following steps, details, and terminal status both to stderr and to an atomically updated `status.json`. Scientific stdout remains parseable JSON.

Completed finite-basis EA points are checkpointed immediately. A rerun with the same run ID/configuration reuses them; a strict signature prevents reuse across incompatible configurations.

Use `scripts/openea_nohup.sh NAME -- COMMAND ...` to launch under `nohup nice -n 19`. The launcher writes `runs/<RUN_ID>/console.log`, PID and command metadata, then tails the log automatically. Closing SSH stops only the tail; the calculation continues. Reconnect with `scripts/openea_tail.sh <RUN_ID>` and inspect structured progress with `python scripts/openea_status.py <RUN_ID>`.

Diffuse runs that may require d-aug perform a Basis Set Exchange preflight before expensive calculations. PySCF itself falls back to the optional `basis-set-exchange` package for unknown basis names, so missing support is detected before the expensive cc-pV5Z/aug-cc-pV5Z work begins.
