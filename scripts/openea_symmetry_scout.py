"""Run the symmetry-block DFT scout for two atoms and print the labelled minima.

    nice -n 19 python scripts/openea_symmetry_scout.py O H --output ../openea_symmetry_scout_test/OH
"""
import argparse
import json

from openea_benchmark.symmetry_scout import run_symmetry_scout


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("atom_a"); p.add_argument("atom_b")
    p.add_argument("--output", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--r0", type=float, default=None)
    a = p.parse_args()
    from pyscf import lib
    lib.num_threads(a.threads)
    rec = run_symmetry_scout(a.atom_a, a.atom_b, output_dir=a.output, r0=a.r0)
    for q, e in rec["charges"].items():
        print("charge", q, "MR-only/outside window:", [m["label"] for m in e["plan"]["mr_only_or_outside_window"]])
        blocks = sorted(e["blocks"].values(), key=lambda b: (b.get("minimum") or {}).get("energy_hartree") or 0.0)
        for b in blocks:
            m = b.get("minimum") or {}
            print(f"  {b['label']:10s} {b['status']:26s} Rmin={m.get('r_angstrom')} E={m.get('energy_hartree')} "
                  f"sides={b.get('side_status')} symbreak_pts={b.get('symmetry_breaking_instability_points')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
