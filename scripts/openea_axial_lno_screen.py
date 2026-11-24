"""f-configuration screen of LnO or LnO- at one bond length: every Hund high-spin f
determinant (up to the m mirror) for the two f counts of the ionic picture, ranked by
CCSD(T) energy (the UO pilot showed HF ordering is not reliable across f counts).
Each determinant: free SCF following the ion's 4f and 6s; if that does not converge or
loses an f electron to 5d, frozen-f SCF (f orbitals of the free ion, CC relaxes them).

Usage: openea_axial_lno_screen.py OUTDIR ELEMENT CHARGE [R_ANGSTROM] [X] [threads]
Writes OUTDIR/ELEMENT/screen_q{CHARGE}_{X}Z_R{R}.jsonl (one line per determinant, resumable).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from openea_benchmark.axial.ln_oxide import XE_CORE, Z, f_configurations, oxide_point

H2EV = 27.211386245988


def candidates(element: str, charge: int):
    """Ln(2-q)+ ... : k valence electrons beyond [Xe] on Ln; f^(k-ns) 6s^ns for ns = 0..2."""
    k = Z[element] - XE_CORE - (2 + charge)
    out = []
    for ns in (0, 1, 2):
        nf = k - ns
        if 0 <= nf <= 14 and (ns < 2 or charge < 0) and not (ns == 0 and charge < 0):
            out += f_configurations(nf, ns)
    return out


def main(argv):
    outdir, el, charge = Path(argv[1]), argv[2], int(argv[3])
    r = float(argv[4]) if len(argv) > 4 else 1.82
    x = argv[5] if len(argv) > 5 else "D"
    if len(argv) > 6:
        from pyscf import lib
        lib.num_threads(int(argv[6]))
    path = outdir / el / f"screen_q{charge}_{x}Z_R{r:.3f}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["label"] for line in path.read_text().splitlines()} if path.is_file() else set()
    for cfg in candidates(el, charge):
        if cfg.label() in done:
            continue
        t = time.time()
        p = oxide_point(el, cfg, r, x, max_cycle=150)
        if not p.converged or min(p.f_purity.values(), default=1.0) < 0.8:
            # free SCF slid 4f -> 5d or did not converge: f orbitals of the free ion, frozen
            p = oxide_point(el, cfg, r, x, freeze_f=True)
        rec = {"label": cfg.label(), "config": asdict(cfg), "charge": p.charge, "r": r, "basis": x,
               "scf_model": p.scf_model,
               "e_scf": p.e_scf, "converged": p.converged, "s2": p.s2, "f_purity": p.f_purity,
               "correlation": p.correlation, "ion_e_scf": p.ion_e_scf, "seconds": time.time() - t}
        with path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{el} q={charge} {cfg.label():40s} {p.scf_model} conv {p.converged} <S2> {p.s2:.3f} f {p.f_purity} "
              f"SCF {p.e_scf:.8f} CCSD(T) {p.e_scf + p.correlation.get('ccsd(t)', float('nan')):.8f} "
              f"({time.time() - t:.0f} s)", flush=True)
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    ok = [q for q in recs if q["converged"] and "ccsd(t)" in q["correlation"]]
    e = {q["label"]: q["e_scf"] + q["correlation"]["ccsd(t)"] for q in ok}
    e0 = min(e.values())
    es = {q["label"]: q["e_scf"] for q in ok}
    s0 = min(es.values())
    print(f"\n{el} q={charge} R={r} {x}Z ranking (eV above the lowest CCSD(T); SCF rank in brackets):")
    model = {q["label"]: q.get("scf_model", "free") for q in ok}
    for lab in sorted(e, key=e.get):
        print(f"  {lab:40s} {(e[lab] - e0) * H2EV:7.3f}   [SCF {(es[lab] - s0) * H2EV:7.3f}] {model[lab]}")
    # frozen-f SCF is an upper bound: TbO 4f8 6s aDZ frozen vs free: SCF +0.11 eV, CCSD(T) +0.02 eV


if __name__ == "__main__":
    main(sys.argv)
