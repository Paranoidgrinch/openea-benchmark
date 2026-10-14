#!/bin/bash
# Guarded batch: memory watchdog + several run_one.sh runs in one process namespace.
# Start it detached with start_detached.sh (setsid + nohup), not in a terminal.
# Usage: run_batch.sh C:C N:O Si:C O:H:benchmark_auto_p40   (from anywhere)
# An optional third field selects the acquisition subdirectory (run_one_dir.sh).
# Memory gate in each run (--mem-limit-frac 0.70 via run_one.sh) and
# scripts/openea_mem_watchdog.py (pause >= 70 %, resume < 65 %, kill >= 80 % or
# after 5 min above 70 % despite pausing).
# A run that fails (e.g. a worker killed by the watchdog) is resumed from the
# point cache, at most MAX_ATTEMPTS times.
cd "$(dirname "$0")/../.."
HERE=runs/two_atom_v2_20261011
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
# Own PySCF scratch per batch, removed at the end: killed calculations leave their
# integral files behind (123 GB on 2026-10-12, disk quota exceeded).
mkdir -p runs/scratch
export PYSCF_TMPDIR="$(mktemp -d "$PWD/runs/scratch/batch_$(date +%Y%m%d_%H%M%S)_XXXX")"
nohup nice -n 19 .venv/bin/python scripts/openea_mem_watchdog.py --log runs/mem_watchdog.log &
WATCHDOG=$!
trap 'kill -TERM $WATCHDOG 2>/dev/null; rm -rf "$PYSCF_TMPDIR"' EXIT
for pair in "$@"; do
  (
    IFS=: read -r a b acq <<< "$pair"
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
      echo "[BATCH] $a$b ${acq:-benchmark_auto} attempt $attempt $(date '+%F %T')" >> "$HERE/$a$b.log"
      if [ -n "$acq" ]; then "$HERE/run_one_dir.sh" "$a" "$b" "$acq"; else "$HERE/run_one.sh" "$a" "$b"; fi
      tail -1 "$HERE/$a$b.log" | grep -qx "EXIT 0" && break
    done
  ) &
done
wait $(jobs -p | grep -vx "$WATCHDOG")
