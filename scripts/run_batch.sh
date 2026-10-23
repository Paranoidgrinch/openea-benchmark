#!/bin/bash
# Guarded batch: memory watchdog + several molecules in one process group.
# Start it with start_detached.sh, not in a terminal.
# Usage: scripts/run_batch.sh RUN_NAME A:B [A:B ...]      e.g.  run_batch.sh verify_v1 O:H Ti:O
# Memory: gate in every calculation (--mem-limit-frac 0.70) and the watchdog
# scripts/openea_mem_watchdog.py (pause >= 70 %, resume < 65 %, kill >= 80 %).
# A molecule whose run fails (e.g. a worker killed by the watchdog) is resumed from its
# point cache, at most MAX_ATTEMPTS times, RETRY_WAIT seconds after the failure (a full
# disk on 2026-10-23 used up all three attempts of several molecules within minutes).
cd "$(dirname "$0")/.."
RUN="$1"; shift
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RETRY_WAIT="${RETRY_WAIT:-900}"
SCRATCH_ROOT="${OPENEA_PYSCF_SCRATCH:-/scratch/$USER/openea_pyscf}"
mkdir -p "runs/$RUN" "$SCRATCH_ROOT"
# Own PySCF scratch per batch, removed at the end: killed calculations leave their
# integral files behind (123 GB on 2026-10-12). On /scratch, not in the home quota
# (111 GB of batch scratch filled the home quota on 2026-10-22/23 -> crashes).
export PYSCF_TMPDIR="$(mktemp -d "$SCRATCH_ROOT/batch_$(date +%Y%m%d_%H%M%S)_XXXX")"
nohup nice -n 19 .venv/bin/python scripts/openea_mem_watchdog.py --log runs/mem_watchdog.log &
WATCHDOG=$!
trap 'kill -TERM $WATCHDOG 2>/dev/null; rm -rf "$PYSCF_TMPDIR"' EXIT
for pair in "$@"; do
  (
    IFS=: read -r a b <<< "$pair"
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
      echo "[BATCH] $a$b attempt $attempt $(date '+%F %T')" >> "runs/$RUN/$a$b.log"
      scripts/run_one.sh "$RUN" "$a" "$b"
      tail -1 "runs/$RUN/$a$b.log" | grep -qx "EXIT 0" && break
      [ "$attempt" -lt "$MAX_ATTEMPTS" ] && sleep "$RETRY_WAIT"
    done
  ) &
done
wait $(jobs -p | grep -vx "$WATCHDOG")
