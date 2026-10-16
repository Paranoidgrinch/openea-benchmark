#!/bin/bash
# Detached start of run_batch.sh: survives closing the terminal or the SSH session.
# Usage: scripts/start_detached.sh RUN_NAME A:B [A:B ...]
# Start it from a normal login shell (a sandboxed agent shell kills its child processes).
cd "$(dirname "$0")/.."
RUN="$1"
mkdir -p "runs/$RUN"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="runs/$RUN/batch_$STAMP.console.log"
setsid nohup scripts/run_batch.sh "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "runs/$RUN/batch_$STAMP.pid"
echo "run_batch.sh $* detached: pid $PID (process group $PID)"
echo "  logs     : runs/$RUN/<AB>.log"
echo "  results  : ${OPENEA_OUTPUT:-../openea_results}/$RUN/<AB>/"
echo "  stop all : kill -TERM -- -$PID"
