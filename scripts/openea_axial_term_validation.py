"""1Delta term energies as single axial determinants (pi+^2) with coupled cluster.

The 1Delta state (M_L = 2 component pi+^2) is one closed-shell determinant in Lz-adapted
orbitals; real orbitals need two.  Here it is computed like any single-reference state and
compared with the measured T0 (Huber & Herzberg, the values of
scripts/openea_mr_term_gap_validation.py; used only for this comparison).  The X 3Sigma-
reference is the high-spin determinant pi+ pi- (ROHF, semicanonical CCSD(T) as Stage 3).

Per state: CCSD(T)/aug-cc-pVTZ curve (0.02 A steps from the covalent-radii sum, quartic
through five points: R_e, harmonic ZPE); at each minimum CCSD(T)/aug-cc-pVQZ (correlation
T/Q-extrapolated with X^-3, SCF from QZ) and CCSDT - CCSD(T) in aug-cc-pVDZ.
T0 = T_e(CBS) + Delta_T + Delta ZPE.  O2 is left out (verify_v2 molecule).

Usage: OMP_NUM_THREADS=8 nice -n 19 .venv/bin/python scripts/openea_axial_term_validation.py OUTDIR [CASE ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

H2EV = 27.211386245988
AMU_TO_ME = 1822.888486209
ANG_TO_BOHR = 1.0 / 0.529177210903
CM1_PER_HARTREE = 219474.6313632
STEP = 0.02
# atoms, measured T0 (eV), sigma orbitals, doubly occupied pi per m (k): X = pi+ pi- open, a = pi+^2
CASES = {"NH": (("N", "H"), 1.558, 3, 0), "NF": (("N", "F"), 1.418, 5, 1),
         "SO": (("S", "O"), 0.728, 7, 2), "SS": (("S", "S"), 0.545, 9, 3)}
TIGHT_D = {"Al", "Si", "P", "S", "Cl", "Ar"}


def basis_of(atoms, x):
    return {a: (f"aug-cc-pv({x}+d)z" if a in TIGHT_D else f"aug-cc-pv{x}z") for a in atoms}


def occupation(n_sigma, k, state):
    if state == "X3Sigma-":
        return 2, {0: (n_sigma, n_sigma), 1: (k + 1, k), -1: (k + 1, k)}
    return 0, {0: (n_sigma, n_sigma), 1: (k + 1, k + 1), -1: (k, k)}


def point(atoms, r, x, state, n_sigma, k, methods, cache: Path) -> dict:
    key = cache / f"{''.join(atoms)}_{state}_{x}_{r:.4f}_{'_'.join(methods)}.json"
    if key.is_file():
        return json.loads(key.read_text())
    from pyscf import gto
    from openea_benchmark.adaptive.stage3_execution import valence_frozen_core_orbitals
    from openea_benchmark.axial.basis import axial_integrals
    from openea_benchmark.axial.cc import axial_cc
    from openea_benchmark.axial.scf import axial_rohf, minao_density
    spin, occ = occupation(n_sigma, k, state)
    mol = gto.M(atom=[(atoms[0], (0, 0, 0)), (atoms[1], (0, 0, r))], basis=basis_of(atoms, x), spin=spin,
                unit="Angstrom", verbose=0, max_memory=200000)
    t0 = time.time()
    ints = axial_integrals(mol)
    scf = axial_rohf(ints, occ, minao_density(mol, ints))
    if not scf.converged:
        raise RuntimeError(f"{atoms} {state} {x} {r}: SCF not converged")
    corr = axial_cc(ints, scf, mol.nelectron, valence_frozen_core_orbitals(mol), methods)
    rec = {"scf": scf.e_tot, **corr, "nao": mol.nao_nr(), "seconds": time.time() - t0}
    key.write_text(json.dumps(rec))
    print(f"  {''.join(atoms)} {state} {x} R={r:.4f} E_SCF={scf.e_tot:.8f} "
          + " ".join(f"{m}={v:.8f}" for m, v in corr.items()) + f" ({rec['seconds']:.0f} s)", flush=True)
    return rec


def minimum(atoms, state, n_sigma, k, cache):
    """Downhill walk at CCSD(T)/aTZ, quartic through the five points around the lowest."""
    from pyscf.data import radii, elements
    r = round(sum(radii.COVALENT[elements.charge(a)] for a in atoms) * 0.529177210903 / STEP) * STEP
    e = lambda rr: (lambda p: p["scf"] + p["ccsd(t)"])(point(atoms, round(rr, 4), "t", state, n_sigma, k,
                                                               ("ccsd(t)",), cache))
    pts = {round(r + d * STEP, 4): None for d in (-2, -1, 0, 1, 2)}
    for _ in range(40):
        for rr in pts:
            if pts[rr] is None:
                pts[rr] = e(rr)
        rs = sorted(pts)
        i = min(range(len(rs)), key=lambda j: pts[rs[j]])
        if 2 <= i <= len(rs) - 3:
            break
        nxt = rs[0] - STEP if i < 2 else rs[-1] + STEP
        pts[round(nxt, 4)] = None
    rs = sorted(pts)
    i = min(range(len(rs)), key=lambda j: pts[rs[j]])
    five = rs[i - 2:i + 3]
    c = np.polyfit(np.array(five) - five[2], [pts[x] for x in five], 4)
    roots = [z.real for z in np.roots(np.polyder(c)) if abs(z.imag) < 1e-12 and abs(z.real) <= 2 * STEP]
    dr = min(roots, key=lambda z: np.polyval(c, z))
    k2 = np.polyval(np.polyder(c, 2), dr)                       # Hartree / A^2
    from pyscf.data.elements import MASSES, charge
    mu = np.prod([MASSES[charge(a)] for a in atoms]) / sum(MASSES[charge(a)] for a in atoms) * AMU_TO_ME
    omega = np.sqrt(k2 / ANG_TO_BOHR ** 2 / mu)                 # Hartree
    return {"r_e": five[2] + dr, "e_tz": float(np.polyval(c, dr)), "omega_cm1": omega * CM1_PER_HARTREE,
            "zpe_ev": 0.5 * omega * H2EV, "r_grid": rs}


def run_case(name, cache):
    atoms, t0_exp, n_sigma, k = CASES[name]
    out = {"case": name, "measured_t0_ev": t0_exp}
    for state in ("X3Sigma-", "a1Delta"):
        mn = minimum(atoms, state, n_sigma, k, cache)
        r = round(mn["r_e"], 4)
        tz = point(atoms, r, "t", state, n_sigma, k, ("ccsd(t)",), cache)
        qz = point(atoms, r, "q", state, n_sigma, k, ("ccsd(t)",), cache)
        dz = point(atoms, r, "d", state, n_sigma, k, ("ccsd(t)", "ccsdt"), cache)
        cbs_corr = (4 ** 3 * qz["ccsd(t)"] - 3 ** 3 * tz["ccsd(t)"]) / (4 ** 3 - 3 ** 3)
        out[state] = {**mn, "e_tz_at_re": tz["scf"] + tz["ccsd(t)"], "e_qz": qz["scf"] + qz["ccsd(t)"],
                      "e_cbs": qz["scf"] + cbs_corr, "delta_t": dz["ccsdt"] - dz["ccsd(t)"]}
    x, a = out["X3Sigma-"], out["a1Delta"]
    out["te_tz_ev"] = (a["e_tz_at_re"] - x["e_tz_at_re"]) * H2EV
    out["te_qz_ev"] = (a["e_qz"] - x["e_qz"]) * H2EV
    out["te_cbs_ev"] = (a["e_cbs"] - x["e_cbs"]) * H2EV
    out["delta_t_ev"] = (a["delta_t"] - x["delta_t"]) * H2EV
    out["delta_zpe_ev"] = a["zpe_ev"] - x["zpe_ev"]
    out["t0_ev"] = out["te_cbs_ev"] + out["delta_t_ev"] + out["delta_zpe_ev"]
    out["error_ev"] = out["t0_ev"] - t0_exp
    print(f"{name}: Te TZ {out['te_tz_ev']:.3f} QZ {out['te_qz_ev']:.3f} CBS {out['te_cbs_ev']:.3f} "
          f"dT {out['delta_t_ev']:+.3f} dZPE {out['delta_zpe_ev']:+.3f} -> T0 {out['t0_ev']:.3f} "
          f"(measured {t0_exp:.3f}, error {out['error_ev']:+.3f})", flush=True)
    return out


if __name__ == "__main__":
    outdir = Path(sys.argv[1])
    cache = outdir / "points"
    cache.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2:] or list(CASES)
    results = [run_case(n, cache) for n in names]
    (outdir / "validation.json").write_text(json.dumps(results, indent=1))
    print("case  T0 calc  T0 exp  error")
    for r in results:
        print(f"{r['case']:4s}  {r['t0_ev']:.3f}   {r['measured_t0_ev']:.3f}  {r['error_ev']:+.3f}")
