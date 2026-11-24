"""Adiabatic EA of a lanthanide monoxide from axial determinants (Route B, development).

Per charge (neutral, anion) the lowest determinant of the f-configuration screen
(openea_axial_lno_screen.py, CCSD(T)/aDZ at one R; only converged CC):
  1. CCSD(T)/aTZ downhill walk in 0.02 A steps, quartic through five points: R_e, harmonic ZPE
  2. at R_e: CCSD(T)/aQZ, correlation T/Q-extrapolated (X^-3), SCF from QZ
  3. Delta_T = CCSDT - CCSD(T), aDZ
  4. Delta_CV = CCSD(T) with Ln 4s4p4d + O 1s correlated - valence, aug-cc-pwCVTZ(-X2C)
  5. Delta_SO = E[X2C-1e 2c] - E[scalar part], lowest Omega, GHF + GCCSD(T),
     uncontracted Ln aDZ basis, virtual spinors below 5 Eh
EA_0 = E_0(neutral) - E_0(anion) with every term.  Measured EAs are not used here.

Usage: openea_axial_lno_ea.py OUTDIR ELEMENT [threads] [--steps tz,qz,dt,cv,so]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from openea_benchmark.axial.ln_oxide import LnConfig, oxide_point, soc_increment

H2EV = 27.211386245988
AMU_TO_ME = 1822.888486209
ANG_TO_BOHR = 1.0 / 0.529177210903
CM1_PER_HARTREE = 219474.6313632
STEP = 0.02
R_SCREEN = 1.82
SO_VIRTUAL_CUTOFF = 5.0


def lowest_config(outdir: Path, el: str, charge: int) -> tuple[LnConfig, dict]:
    path = outdir / el / f"screen_q{charge}_DZ_R{R_SCREEN:.3f}.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    ok = [q for q in recs if q["converged"] and q.get("cc_status") == "CONVERGED"]
    best = min(ok, key=lambda q: q["e_scf"] + q["correlation"]["ccsd(t)"])
    e0 = best["e_scf"] + best["correlation"]["ccsd(t)"]
    gaps = sorted(((q["e_scf"] + q["correlation"]["ccsd(t)"] - e0) * H2EV, q["label"]) for q in ok if q is not best)
    c = best["config"]
    cfg = LnConfig(tuple(c["f_alpha"]), tuple(c["f_beta"]), c["s_alpha"], c["s_beta"])
    return cfg, {"label": best["label"], "scf_model": best.get("scf_model", "free"), "next_gaps_ev": gaps[:3],
                 "n_screened": len(recs), "n_converged": len(ok)}


def point(el, cfg, r, x, methods, cache: Path, freeze_f: bool, cv: bool = False) -> dict:
    key = cache / f"{el}O_{cfg.label().replace(' ', '_')}_{'cv' if cv else ''}{x}_{r:.4f}_{'+'.join(methods)}.json"
    if key.is_file():
        return json.loads(key.read_text())
    t0 = time.time()
    p = oxide_point(el, cfg, r, x, methods, freeze_f=freeze_f, cv=cv)
    if not p.converged or p.cc_status != "CONVERGED":
        raise RuntimeError(f"{el}O {cfg.label()} {x} R={r}: SCF {p.converged}, CC {p.cc_status}")
    rec = {"scf": p.e_scf, **p.correlation, "s2": p.s2, "f_purity": p.f_purity, "scf_model": p.scf_model,
           "cc_diagnostics": p.cc_diagnostics, "seconds": time.time() - t0}
    key.write_text(json.dumps(rec))
    print(f"  {el}O q={p.charge} {cfg.label()} {'cv' if cv else ''}{x} R={r:.4f} E_SCF={p.e_scf:.8f} "
          + " ".join(f"{m}={v:.8f}" for m, v in p.correlation.items()) + f" ({rec['seconds']:.0f} s)", flush=True)
    return rec


def minimum(el, cfg, cache, freeze_f, masses):
    e = lambda rr: (lambda p: p["scf"] + p["ccsd(t)"])(point(el, cfg, round(rr, 4), "T", ("ccsd(t)",), cache, freeze_f))
    pts = {round(R_SCREEN + d * STEP, 4): None for d in (-2, -1, 0, 1, 2)}
    for _ in range(30):
        for rr in pts:
            if pts[rr] is None:
                pts[rr] = e(rr)
        rs = sorted(pts)
        i = min(range(len(rs)), key=lambda j: pts[rs[j]])
        if 2 <= i <= len(rs) - 3:
            break
        pts[round(rs[0] - STEP if i < 2 else rs[-1] + STEP, 4)] = None
    rs = sorted(pts)
    i = min(range(len(rs)), key=lambda j: pts[rs[j]])
    five = rs[i - 2:i + 3]
    c = np.polyfit(np.array(five) - five[2], [pts[x] for x in five], 4)
    roots = [z.real for z in np.roots(np.polyder(c)) if abs(z.imag) < 1e-12 and abs(z.real) <= 2 * STEP]
    dr = min(roots, key=lambda z: np.polyval(c, z))
    k2 = np.polyval(np.polyder(c, 2), dr)
    mu = masses[0] * masses[1] / sum(masses) * AMU_TO_ME
    omega = np.sqrt(k2 / ANG_TO_BOHR ** 2 / mu)
    return {"r_e": float(five[2] + dr), "e_tz": float(np.polyval(c, dr)), "omega_cm1": float(omega * CM1_PER_HARTREE),
            "zpe_ev": float(0.5 * omega * H2EV), "r_grid": rs}


def main(argv):
    outdir, el = Path(argv[1]), argv[2]
    if len(argv) > 3 and not argv[3].startswith("--"):
        from pyscf import lib
        lib.num_threads(int(argv[3]))
    steps = {"tz", "qz", "dt", "cv", "so"}
    if "--steps" in argv:
        steps = set(argv[argv.index("--steps") + 1].split(","))
    from pyscf.data.elements import MASSES, charge
    masses = (MASSES[charge(el)], MASSES[charge("O")])
    cache = outdir / el / "ea_points"
    cache.mkdir(parents=True, exist_ok=True)
    out = {"element": el, "steps": sorted(steps)}
    for q in (0, -1):
        cfg, info = lowest_config(outdir, el, q)
        freeze_f = info["scf_model"] == "frozen_f"
        print(f"{el}O q={q}: {info['label']} ({info['scf_model']}), next configurations {info['next_gaps_ev']}", flush=True)
        st = {"config": info}
        mn = minimum(el, cfg, cache, freeze_f, masses)
        r = round(mn["r_e"], 4)
        st.update(mn)
        tz = point(el, cfg, r, "T", ("ccsd(t)",), cache, freeze_f)
        st["e_tz"] = tz["scf"] + tz["ccsd(t)"]
        if "qz" in steps:
            qz = point(el, cfg, r, "Q", ("ccsd(t)",), cache, freeze_f)
            st["e_cbs"] = qz["scf"] + (4 ** 3 * qz["ccsd(t)"] - 3 ** 3 * tz["ccsd(t)"]) / (4 ** 3 - 3 ** 3)
        if "dt" in steps:
            dz = point(el, cfg, r, "D", ("ccsd(t)", "ccsdt"), cache, freeze_f)
            st["delta_t"] = dz["ccsdt"] - dz["ccsd(t)"]
        if "cv" in steps:
            cvp = point(el, cfg, r, "T", ("ccsd(t)",), cache, freeze_f, cv=True)
            st["delta_cv"] = cvp["cv_ccsd(t)"] - cvp["val_ccsd(t)"]
        if "so" in steps:
            so = soc_increment(el, cfg, r, "D", freeze_f=freeze_f, virtual_cutoff=SO_VIRTUAL_CUTOFF)
            if so["status"] != "COMPLETED":
                raise RuntimeError(f"SO increment q={q}: {so['status']}")
            st["delta_so"] = so["delta_so_hartree"]["ccsd(t)"]
            st["delta_so_hf"] = so["delta_so_hartree"]["hf"]
        out[str(q)] = st
        (outdir / el / "ea.json").write_text(json.dumps(out, indent=1))
    n, a = out["0"], out["-1"]
    terms = {"ea_e_tz": (n["e_tz"] - a["e_tz"]) * H2EV}
    if "e_cbs" in n:
        terms["ea_e_cbs"] = (n["e_cbs"] - a["e_cbs"]) * H2EV
    for k in ("delta_t", "delta_cv", "delta_so"):
        if k in n:
            terms[k] = (n[k] - a[k]) * H2EV
    terms["delta_zpe"] = n["zpe_ev"] - a["zpe_ev"]
    base = terms.get("ea_e_cbs", terms["ea_e_tz"])
    terms["ea0"] = base + sum(terms.get(k, 0.0) for k in ("delta_t", "delta_cv", "delta_so", "delta_zpe"))
    out["ea_terms_ev"] = terms
    (outdir / el / "ea.json").write_text(json.dumps(out, indent=1))
    print(f"{el}O EA terms (eV): " + " ".join(f"{k} {v:+.4f}" for k, v in terms.items()), flush=True)


if __name__ == "__main__":
    main(sys.argv)
