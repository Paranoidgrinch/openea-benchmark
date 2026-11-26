#!/bin/bash
# Rebuild main's CCpy (external/ccpy, meson-python editable install) with OpenMP.
# Upstream builds the Fortran without -fopenmp, so every !$omp loop runs serially; four files
# carry OMP directives that do not compile and are not used by OpenEA (CC(P;Q) corrections,
# IP/EA-EOM), their directives are turned into comments.  Measured (ccpy_dev copy, 2026-11-25):
# CC(P) CCSDT CN aDZ 7.3 -> 3.7 s/iteration, aTZ 276 -> 94 s; energies unchanged.
# RUN ONLY WHILE NO WORKFLOW BATCH IS RUNNING: the editable install rebuilds on import.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../openea-benchmark" && pwd)
C=$ROOT/external/ccpy
VENV=$ROOT/.venv
if pgrep -u "$USER" -f "openea_benchmark.workflow" > /dev/null; then
  echo "workflow processes are running: stop the batches first"; exit 1
fi
cd "$C"
[ -f ccpy/lib/meson.build.pre_openmp ] || cp -a ccpy/lib/meson.build ccpy/lib/meson.build.pre_openmp
"$VENV/bin/python" - <<'PY'
p = "ccpy/lib/meson.build"
s = open(p).read()
s = s.replace("fortran_args: '-ffree-line-length-512',\n)", "fortran_args: ['-ffree-line-length-512', '-fopenmp'],\n)", 1)
if "link_args: ['-fopenmp']" not in s:
    s = s.replace("  link_with:[fortran_lib, sort_lib],\n", "  link_with:[fortran_lib, sort_lib],\n  link_args: ['-fopenmp'],\n", 1)
open(p, "w").write(s)
PY
for f in ccp3_full_correction ccp3_full_correction_high_mem eaccp3_correction ipeomt_p_loops; do
  [ -f ccpy/lib/fortran/$f.f90.pre_openmp ] || cp -a ccpy/lib/fortran/$f.f90 ccpy/lib/fortran/$f.f90.pre_openmp
  sed -i 's/!\$omp/!-omp-disabled/I' ccpy/lib/fortran/$f.f90
done
grep -c "fopenmp" ccpy/lib/meson.build
export PATH=$VENV/bin:/opt/rh/gcc-toolset-10/root/usr/bin:$PATH
cd build/cp312 && ninja
"$VENV/bin/python" -c "import ccpy, ccpy.lib.core; print('ccpy imports with OpenMP build')"
