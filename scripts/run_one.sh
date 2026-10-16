#!/usr/bin/env bash
# One molecule: two atoms in, EA out.  Called by run_batch.sh.
# Usage: scripts/run_one.sh RUN_NAME A B        (from the repository root)
# Log: runs/RUN_NAME/AB.log; results: $OPENEA_OUTPUT/RUN_NAME/AB/ (default ../openea_results).
# All runs on the machine share one core-slot budget and one memory gate (runs/core_slots).
set -u
RUN="$1"; A="$2"; B="$3"
THREADS="${THREADS:-10}"; CORE_SLOTS="${CORE_SLOTS:-8}"
OUTPUT="${OPENEA_OUTPUT:-../openea_results}/$RUN"
LOG="runs/$RUN/$A$B.log"
mkdir -p "runs/$RUN" "$OUTPUT"
# PySCF integral scratch on the project file system (5Z points write >15 GB each).
export PYSCF_TMPDIR="${PYSCF_TMPDIR:-$PWD/runs/scratch}" && mkdir -p "$PYSCF_TMPDIR"
export PYTHONUNBUFFERED=1  # logs show progress immediately
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
nice -n 19 .venv/bin/python -m openea_benchmark.workflow --atoms "$A" "$B" \
  --output "$OUTPUT" --threads "$THREADS" --cbs-memory-mb "${CBS_MEMORY_MB:-48000}" \
  --point-workers "$CORE_SLOTS" --core-slots "$CORE_SLOTS" --core-slots-dir runs/core_slots \
  --mem-limit-frac "${MEM_LIMIT_FRAC:-0.70}" >> "$LOG" 2>&1
echo "EXIT $?" >> "$LOG"
