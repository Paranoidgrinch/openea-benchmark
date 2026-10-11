"""numpy's OpenBLAS pool follows the requested thread budget."""
import ctypes

from openea_benchmark.compute_threads import _numpy_openblas_libs, set_compute_threads


def test_numpy_openblas_limited():
    assert set_compute_threads(3) >= 1
    from pyscf import lib
    assert lib.num_threads() == 3
    handle = ctypes.CDLL(_numpy_openblas_libs()[0])
    getter = next(getattr(handle, n) for n in ("scipy_openblas_get_num_threads64_", "openblas_get_num_threads64_",
                                                "openblas_get_num_threads") if hasattr(handle, n))
    assert getter() == 3
