#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 || "$2" != "--" ]]; then echo "Usage: $0 RUN_NAME -- COMMAND [ARGS...]" >&2; exit 2; fi
RUN_NAME="$1"; shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; RUN_ID="${OPENEA_RUN_ID:-${RUN_NAME}_${STAMP}}"; RUN_ROOT="${OPENEA_RUN_ROOT:-$ROOT/runs}"; RUN_DIR="$RUN_ROOT/$RUN_ID"; mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"; PIDFILE="$RUN_DIR/pid"; COMMAND_FILE="$RUN_DIR/command.txt"
printf '%q ' "$@" > "$COMMAND_FILE"; printf '\n' >> "$COMMAND_FILE"
export OPENEA_RUN_ID="$RUN_ID"; export OPENEA_RUN_ROOT="$RUN_ROOT"
nohup nice -n 19 "$@" >"$LOG" 2>&1 < /dev/null &
PID=$!; echo "$PID" > "$PIDFILE"; mkdir -p "$RUN_ROOT"; ln -sfn "$RUN_DIR" "$RUN_ROOT/latest"
echo "OpenEA detached run started"; echo "  run_id : $RUN_ID"; echo "  pid    : $PID"; echo "  log    : $LOG"; echo "  status : $RUN_DIR/status.json"; echo
echo "The calculation survives SSH/terminal closure."; echo "Ctrl-C stops only this live tail, not the calculation."; echo "Reconnect later with: ./scripts/openea_tail.sh $RUN_ID"; echo
tail -n 40 -F "$LOG"
