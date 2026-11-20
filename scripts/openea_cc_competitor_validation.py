"""Check the CC competitor route (EOM-CCSD block energies) against measured term energies.

Term energy = lowest root of a (2S+1, C2v irrep) block minus the CCSD energy of the
3Sigma- ground state, both at the ground state's measured R_e (vertical; the measured T0
differ from it by the small R_e shift and Delta ZPE), compared with the lowest measured
state of that block (A1 singlets hold 1Sigma+ and one component of 1Delta: 1Delta).
T0 after Huber & Herzberg.
verify_v2 molecules (O2, ...) are left out (blind set).

Usage: OMP_NUM_THREADS=8 nice -n 19 .venv/bin/python scripts/openea_cc_competitor_validation.py
"""
from __future__ import annotations

import json

from openea_benchmark.cc_competitors import CCCompetitorPolicy, block_roots
from openea_benchmark.electronic_ea_stage import HARTREE_TO_EV

CASES = [  # atoms, R_e of X 3Sigma- (A), lowest state of the block, (2S+1, C2v irrep), its T0 eV
    (("N", "H"), 1.0362, "1Delta (A2)", (1, "A2"), 1.558),
    (("N", "H"), 1.0362, "1Delta (A1)", (1, "A1"), 1.558),
    (("S", "O"), 1.4811, "1Delta (A2)", (1, "A2"), 0.728),
    (("S", "O"), 1.4811, "1Delta (A1)", (1, "A1"), 0.728),
    (("S", "S"), 1.8892, "1Deltag (A2)", (1, "A2"), 0.545),
    (("S", "S"), 1.8892, "1Deltag (A1)", (1, "A1"), 0.545),
]


def main() -> None:
    policy = CCCompetitorPolicy()
    out = []
    for atoms, r, label, (mult, irrep), t0 in CASES:
        ground = block_roots(atoms, 0, 3, "A2", r, policy, {"two_s": 2, "irrep_nelec": None}, None)
        e_x = min(x["e_hartree"] for x in ground if x["method"] == "CCSD")
        roots = block_roots(atoms, 0, mult, irrep, r, policy, {"two_s": mult - 1, "irrep_nelec": None},
                            {"two_s": mult + 1, "irrep_nelec": None})
        gap = min(x["e_hartree"] - e_x for x in roots) * HARTREE_TO_EV
        out.append({"case": f"{''.join(atoms)} {label}", "cc_ev": gap, "t0_ev": t0, "error_ev": gap - t0})
        print(json.dumps(out[-1]), flush=True)
    print("case                cc      T0(exp)  error")
    for x in out:
        print(f"{x['case']:18s} {x['cc_ev']:7.3f} {x['t0_ev']:7.3f} {x['error_ev']:+7.3f}")


if __name__ == "__main__":
    main()
