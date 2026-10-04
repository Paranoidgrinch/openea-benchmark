#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
RUN_DIR="$ROOT/runs/${1:-latest}"
if [[ ! -e "$RUN_DIR/console.log" ]]; then echo "No OpenEA console log found at $RUN_DIR/console.log" >&2; exit 2; fi
echo "Tailing $RUN_DIR/console.log"; echo "Ctrl-C stops the tail only."; tail -n 60 -F "$RUN_DIR/console.log"
