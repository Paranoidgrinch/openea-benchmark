#!/bin/bash
# Detached start of run_batch.sh: survives closing the terminal or the SSH session.
# Usage: Q_BASIS=cc-pvdz|aug-cc-pvdz scripts/start_detached.sh RUN_NAME A:B [A:B ...]
#   Q_BASIS = basis of the quadruples correction Delta_Q (required unless OPENEA_EXTRA_ARGS
#   switches the electronic EA or the post-CC layer off): cc-pvdz (small) or aug-cc-pvdz
#   (where it fits in memory, else cc-pvdz).
# Start it from a normal login shell (a sandboxed agent shell kills its child processes).
cd "$(dirname "$0")/.."
RUN="$1"
case " ${OPENEA_EXTRA_ARGS:-} " in
  *" --no-electronic-ea "*|*" --no-post-cc "*|*" --auto-dry-run "*) ;;
  *) case "${Q_BASIS:-}" in
       cc-pvdz|aug-cc-pvdz) echo "quadruples basis: $Q_BASIS" ;;
       *) echo "not started: set Q_BASIS=cc-pvdz or Q_BASIS=aug-cc-pvdz (basis of the quadruples correction)"; exit 1 ;;
     esac ;;
esac
mkdir -p "runs/$RUN"
# Free scratch space, no leftovers, no broken result files (scripts/preflight.py --fix repairs).
.venv/bin/python scripts/preflight.py "$RUN" || { echo "not started: preflight failed"; exit 1; }
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="runs/$RUN/batch_$STAMP.console.log"
setsid nohup scripts/run_batch.sh "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "runs/$RUN/batch_$STAMP.pid"
echo "run_batch.sh $* detached: pid $PID (process group $PID)"
echo "  logs     : runs/$RUN/<AB>.log"
echo "  results  : ${OPENEA_OUTPUT:-../openea_results}/$RUN/<AB>/"
echo "  stop all : kill -TERM -- -$PID"
