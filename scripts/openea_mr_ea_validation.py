"""Accuracy of an adiabatic EA from SC-NEVPT2 curves alone (MR EA fallback check).

Neutral and anion ground states are followed as their Lambda/(g/u) roots
(``mr_term_gap.state_minimum`` on ``mr_competitors.block_energies``), def2-TZVPPD
minima, def2-QZVPPD single points there, harmonic ZPE.  Measured EAs are used only
for this comparison.  Spin-orbit is not included (small for these states).

Usage: OMP_NUM_THREADS=4 nice -n 19 .venv/bin/python scripts/openea_mr_ea_validation.py OUTDIR
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

from openea_benchmark.electronic_ea_stage import HARTREE_TO_EV
from openea_benchmark.mr_competitors import block_energies
from openea_benchmark.mr_term_gap import MRTermGapPolicy, _spec_from_label, harmonic_zpe_ev, matched_energy, state_minimum
from openea_benchmark.vibrational_stage import main_isotopologue_masses

CASES = [  # atoms, neutral, anion, measured EA (eV)
    (("O", "H"), "2Pi", "1Sigma+", 1.828),
    (("C", "N"), "2Sigma+", "1Sigma+", 3.862),
    (("N", "O"), "2Pi", "3Sigma-", 0.026),
    (("O", "O"), "3Sigmag-", "2Pig", 0.448),
    (("Mg", "H"), "2Sigma+", "1Sigma+", 0.90),
    (("Si", "C"), "3Pi", "2Sigma+", 2.396),
    (("C", "C"), "1Sigmag+", "2Sigmag+", 3.269),
    (("Al", "O"), "2Sigma+", "1Sigma+", 2.60),
    (("Cl", "O"), "2Pi", "1Sigma+", 2.276),
]


def state(atoms, charge, label, r0, policy, cache):
    spec = _spec_from_label(label)

    def energy_at(basis):
        def f(r):
            rec = block_energies(atoms, charge, spec["mult"], spec["irrep"], round(r, 4), policy.block_policy(basis),
                                 Path(cache) / basis.lower() / f"q{charge}_m{spec['mult']}_{spec['irrep']}_r{r:.4f}.json")
            return matched_energy(rec, spec["lambda"], spec["parity"])
        return f
    m = state_minimum(energy_at(policy.basis), r0, policy)
    if m["status"] != "COMPLETED":
        return m
    big = energy_at(policy.comparison_basis)(m["r_grid"])
    tz = energy_at(policy.basis)(m["r_grid"])
    masses = main_isotopologue_masses(atoms)
    return {**m, "qz_shift_hartree": None if big is None else big - tz,
            "zpe_ev": harmonic_zpe_ev(m["k_hartree_per_a2"], (masses.mass_a_u, masses.mass_b_u))}


def run(case, outdir):
    from pyscf import gto
    from pyscf.data import radii
    atoms, neutral, anion, ea_exp = case
    r0 = round(sum(radii.COVALENT[gto.charge(a)] for a in atoms) * 0.529177210903 / 0.025) * 0.025
    policy = MRTermGapPolicy(max_memory_mb=12000)
    cache = Path(outdir) / "blocks" / "".join(atoms)
    n, a = state(atoms, 0, neutral, r0, policy, cache), state(atoms, -1, anion, r0, policy, cache)
    res = {"case": "".join(atoms), "neutral": n, "anion": a, "measured_ea_ev": ea_exp}
    if n.get("status") == "COMPLETED" and a.get("status") == "COMPLETED" and None not in (
            n["qz_shift_hartree"], a["qz_shift_hartree"]):
        tz = (n["e_min_hartree"] - a["e_min_hartree"]) * HARTREE_TO_EV
        qz = tz + (n["qz_shift_hartree"] - a["qz_shift_hartree"]) * HARTREE_TO_EV
        res.update(ea_tz_ev=tz, ea_qz_ev=qz, ea0_ev=qz + n["zpe_ev"] - a["zpe_ev"])
        res["error_ev"] = res["ea0_ev"] - ea_exp
    print(json.dumps({k: res.get(k) for k in ("case", "ea_tz_ev", "ea_qz_ev", "ea0_ev", "measured_ea_ev",
                                              "error_ev")} | {"n": n.get("status"), "a": a.get("status")}),
          flush=True)
    return res


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    with Pool(len(CASES)) as pool:
        results = pool.starmap(run, [(c, str(out)) for c in CASES])
    (out / "validation.json").write_text(json.dumps(results, indent=1, default=str))
    print("case   EA0(NEVPT2)  measured  error")
    for r in results:
        if r.get("ea0_ev") is not None:
            print(f"{r['case']:5s} {r['ea0_ev']:9.3f} {r['measured_ea_ev']:9.3f} {r['error_ev']:+.3f}")
        else:
            print(f"{r['case']:5s} failed: {r['neutral'].get('status')} / {r['anion'].get('status')}")
