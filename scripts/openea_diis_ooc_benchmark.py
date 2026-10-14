"""In-core vs out-of-core (GPFS scratch) DIIS for CCpy CCSDTQ: time per iteration and peak RAM.

Usage (from openea-benchmark/, nice 19, OMP threads as in production):
    nice -n 19 .venv/bin/python scripts/openea_diis_ooc_benchmark.py --mode incore --out runs/diis_bench/incore.json
    nice -n 19 .venv/bin/python scripts/openea_diis_ooc_benchmark.py --mode ooc    --out runs/diis_bench/ooc.json

Both modes run the identical CC iterations (same reference, same DIIS size), so the
energy after N iterations must agree; only storage of the DIIS vectors differs.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("incore", "ooc"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--atoms", default="C N")
    ap.add_argument("--r", type=float, default=1.1718)
    ap.add_argument("--spin", type=int, default=1)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--method", default="ccsdtq")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--cached", action="store_true", help="DIIS with cached overlap matrix (production class)")
    args = ap.parse_args()

    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver
    from openea_benchmark.post_ccsd_t_stage import DIIS_SIZE, cached_diis_class, cc_scratch_dir
    import ccpy.drivers.solvers as solvers
    from openea_benchmark.adaptive.stage3_execution import valence_frozen_core_orbitals

    a, b = args.atoms.split()
    mol = gto.M(atom=f"{a} 0 0 0; {b} 0 0 {args.r}", basis=args.basis, charge=args.charge, spin=args.spin,
                unit="Angstrom", symmetry="C1", verbose=0)
    mf = scf.ROHF(mol) if args.spin else scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    nfz = valence_frozen_core_orbitals(mol)

    if args.cached:
        solvers.DIIS = cached_diis_class()
    diis_cls = solvers.DIIS
    # Time spent inside DIIS push/extrapolate (the part that touches the disk).
    diis_seconds = {"push": 0.0, "extrapolate": 0.0}
    for name in diis_seconds:
        original = getattr(diis_cls, name)

        def timed(self, *a_, _orig=original, _name=name, **k):
            t = time.perf_counter()
            try:
                return _orig(self, *a_, **k)
            finally:
                diis_seconds[_name] += time.perf_counter() - t
        setattr(diis_cls, name, timed)

    driver = Driver.from_pyscf(mf, nfrozen=nfz)
    driver.options.update(energy_convergence=1e-14, amp_convergence=1e-14,
                          maximum_iterations=args.iterations, diis_size=DIIS_SIZE,
                          diis_out_of_core=(args.mode == "ooc"))
    t0 = time.perf_counter()
    if args.mode == "ooc":
        with cc_scratch_dir() as path:
            scratch = str(path)
            driver.run_cc(method=args.method)
    else:
        scratch = None
        driver.run_cc(method=args.method)
    wall = time.perf_counter() - t0
    record = {"mode": args.mode, "cached_overlap": args.cached, "system": args.atoms, "basis": args.basis, "method": args.method,
              "iterations": args.iterations, "diis_size": DIIS_SIZE, "nfrozen": nfz,
              "cc_wall_seconds": wall, "seconds_per_iteration": wall / args.iterations,
              "diis_seconds": diis_seconds, "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
              "correlation_energy": float(driver.correlation_energy), "scratch_dir": scratch,
              "scratch_removed": scratch is None or not Path(scratch).exists()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
