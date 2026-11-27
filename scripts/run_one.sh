#!/usr/bin/env bash
# One molecule: two atoms in, EA out.  Called by run_batch.sh.
# Usage: scripts/run_one.sh RUN_NAME A B        (from the repository root)
# Log: runs/RUN_NAME/AB.log; results: $OPENEA_OUTPUT/RUN_NAME/AB/ (default ../openea_results).
# All runs on the machine share one core-slot budget and one memory gate (runs/core_slots).
# OPENEA_EXTRA_ARGS: extra workflow flags, e.g. "--auto-dry-run --no-electronic-ea" (scout only).
# Q_BASIS: basis of the quadruples correction, cc-pvdz or aug-cc-pvdz (aug where it fits, else
# cc-pvdz); required for full runs, e.g.  Q_BASIS=cc-pvdz scripts/start_detached.sh RUN A:B
set -u
RUN="$1"; A="$2"; B="$3"
# CORE_SLOTS caps concurrent tasks machine-wide; the CPU gate (point_pool) admits a slot only
# while the measured busy cores leave room for its THREADS, so 32 slots fill the cores when tasks are
# partly serial and back off when other users compute.  POINT_WORKERS = parallel points per molecule.
THREADS="${THREADS:-10}"; CORE_SLOTS="${CORE_SLOTS:-32}"; POINT_WORKERS="${POINT_WORKERS:-8}"
OUTPUT="${OPENEA_OUTPUT:-../openea_results}/$RUN"
LOG="runs/$RUN/$A$B.log"
mkdir -p "runs/$RUN" "$OUTPUT"
# PySCF integral scratch on /scratch, not in the home quota (5Z points write >15 GB each).
export PYSCF_TMPDIR="${PYSCF_TMPDIR:-/scratch/$USER/openea_pyscf}" && mkdir -p "$PYSCF_TMPDIR"
export PYTHONUNBUFFERED=1  # logs show progress immediately
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
nice -n 19 .venv/bin/python -m openea_benchmark.workflow --atoms "$A" "$B" \
  --output "$OUTPUT" --threads "$THREADS" --cbs-memory-mb "${CBS_MEMORY_MB:-48000}" \
  --point-workers "$POINT_WORKERS" --core-slots "$CORE_SLOTS" --core-slots-dir runs/core_slots \
  --mem-limit-frac "${MEM_LIMIT_FRAC:-0.70}" ${Q_BASIS:+--quadruples-basis "$Q_BASIS"} \
  ${OPENEA_EXTRA_ARGS:-} >> "$LOG" 2>&1
echo "EXIT $?" >> "$LOG"
