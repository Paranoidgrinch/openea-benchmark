"""One thread budget for PySCF's OpenMP pool and numpy's bundled OpenBLAS.

``pyscf.lib.num_threads`` only limits OpenMP.  numpy's OpenBLAS (used by
``lib.dot``/einsum inside CCSD) keeps its own pool, by default up to 64 threads per
process; several concurrent runs then oversubscribe the machine many times over.
"""
from __future__ import annotations

import ctypes
import glob
import os

_SETTERS = ("scipy_openblas_set_num_threads64_", "openblas_set_num_threads64_", "openblas_set_num_threads")


def _numpy_openblas_libs() -> list[str]:
    import numpy
    root = os.path.dirname(os.path.dirname(numpy.__file__))
    return sorted(glob.glob(os.path.join(root, "numpy.libs", "*openblas*"))
                  + glob.glob(os.path.join(root, "scipy_openblas64*", "lib", "*openblas*.so*")))


def set_compute_threads(n: int) -> int:
    """Limit OpenMP and numpy's OpenBLAS to ``n`` threads; returns the number of BLAS pools set."""
    from pyscf import lib
    n = max(1, int(n))
    lib.num_threads(n)
    hit = 0
    for path in _numpy_openblas_libs():
        try:
            handle = ctypes.CDLL(path)
        except OSError:
            continue
        for name in _SETTERS:
            setter = getattr(handle, name, None)
            if setter is not None:
                setter(ctypes.c_int(n))
                hit += 1
                break
    return hit
