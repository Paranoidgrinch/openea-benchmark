"""Check the MR term-gap route (Patch 54) against measured term energies.

The reference values are used only for this comparison, never in a workflow run.
T0 values (v=0 to v=0) after Huber & Herzberg, Constants of Diatomic Molecules.
Start bond length: sum of covalent radii (no experimental geometry enters).

Usage: OMP_NUM_THREADS=4 nice -n 19 .venv/bin/python scripts/openea_mr_term_gap_validation.py OUTDIR
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

from openea_benchmark.mr_term_gap import MRTermGapPolicy, _spec_from_label, term_gap

CASES = [  # atoms, reference (SR) state, MR state, measured T0 in eV
    (("O", "O"), "3Sigmag-", "1Deltag", 0.977),
    (("O", "O"), "3Sigmag-", "1Sigmag+", 1.627),
    (("N", "H"), "3Sigma-", "1Delta", 1.558),
    (("N", "F"), "3Sigma-", "1Delta", 1.418),
    (("S", "O"), "3Sigma-", "1Delta", 0.728),
    (("S", "S"), "3Sigmag-", "1Deltag", 0.545),
]


def run(case, outdir):
    from pyscf.data import radii
    atoms, ref, state, t0 = case
    r0 = round(sum(radii.COVALENT[__import__("pyscf").gto.charge(a)] for a in atoms) * 0.529177210903 / 0.025) * 0.025
    res = term_gap(atoms=atoms, charge=0, reference=_spec_from_label(ref), state=_spec_from_label(state),
                   r_start=r0, policy=MRTermGapPolicy(max_memory_mb=12000),
                   cache_dir=Path(outdir) / "blocks" / "".join(atoms))
    res["measured_t0_ev"] = t0
    if res.get("status") == "COMPLETED":
        res["error_ev"] = res["gap0_ev"] - t0
        res["inside_interval"] = res["gap0_lower_ev"] <= t0 <= res["gap0_upper_ev"]
    print(json.dumps({k: res.get(k) for k in ("status", "gap_e_tz_ev", "basis_shift_ev", "delta_zpe_ev",
                                                "gap0_ev", "measured_t0_ev", "error_ev")} | {"case": f"{atoms} {state}"}),
          flush=True)
    return res


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    with Pool(len(CASES)) as pool:
        results = pool.starmap(run, [(c, str(out)) for c in CASES])
    (out / "validation.json").write_text(json.dumps(results, indent=1, default=str))
    print("case                      gap0    T0(exp)  error")
    for c, r in zip(CASES, results):
        g = r.get("gap0_ev")
        print(f"{''.join(c[0]):3s} {c[2]:10s} {r['status']:12s} "
              + (f"{g:7.3f}  {c[3]:7.3f}  {g - c[3]:+.3f}" if g is not None else ""))
