#!/bin/bash
# run_one.sh with a selectable acquisition subdirectory: run_one_dir.sh A B ACQ_DIR
cd "$(dirname "$0")/../.."
THREADS="${THREADS:-10}"; CORE_SLOTS="${CORE_SLOTS:-8}"
export PYSCF_TMPDIR="${PYSCF_TMPDIR:-$PWD/runs/scratch}" && mkdir -p "$PYSCF_TMPDIR"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
nice -n 19 .venv/bin/python -m openea_benchmark.workflow --mode symmetry-auto --atoms "$1" "$2" \
  --output ../openea_two_atom_v2 --threads "$THREADS" --cbs-memory-mb 48000 \
  --point-workers "$CORE_SLOTS" --core-slots "$CORE_SLOTS" --core-slots-dir runs/core_slots --mem-limit-frac "${MEM_LIMIT_FRAC:-0.70}" \
  --acquisition-dir "$3" "${@:4}" >> "runs/two_atom_v2_20261011/$1$2.log" 2>&1
echo "EXIT $?" >> "runs/two_atom_v2_20261011/$1$2.log"
