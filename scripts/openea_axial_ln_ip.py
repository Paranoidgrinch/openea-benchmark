"""First ionization energies of lanthanide atoms with an open 4f spectator shell.

4f^n 6s^2 -> 4f^n 6s (Pr, Nd, Sm, Eu, Tb, Dy, Ho, Er, Tm, Yb): the Hund component of
maximum M_L and M_S of the f shell is one axial determinant (complex m orbitals); the
atom is solved with (l, m)-blocked UHF (f cannot mix with d), then CCSD(T) (or CCSDT)
on top, sfX2C one-electron Hamiltonian, e-aug-cc-pVXZ-X2C, Cholesky integrals.

Spin-orbit: in LS coupling the first-order shift of the lowest level is -zeta (L+1)/2
(n < 7) or -zeta L/2 (n > 7) for the 4f^n Hund term and is the same for the 4f^n 6s
high-spin term of the cation, so the spin-free IP compares with the measured IP up to
second-order and intermediate-coupling effects.  NIST values are validation only.

Usage: openea_axial_ln_ip.py OUTDIR ELEMENT X [methods] [threads]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

H2EV = 27.211386245988
F_COUNT = {"Pr": 3, "Nd": 4, "Sm": 6, "Eu": 7, "Tb": 9, "Dy": 10, "Ho": 11, "Er": 12, "Tm": 13, "Yb": 14}
NIST_IP_EV = {"Pr": 5.4702, "Nd": 5.5250, "Sm": 5.6437, "Eu": 5.6704, "Tb": 5.8638, "Dy": 5.9391,
              "Ho": 6.0215, "Er": 6.1077, "Tm": 6.1843, "Yb": 6.2542}   # NIST ASD, validation only
CHOLESKY_TOL = 1.0e-8   # Pr aDZ SCF: 1e-6 -> 2.7e-5 Eh, 1e-8 -> 3e-7 Eh absolute error


def f_shell(n: int) -> tuple[dict, dict, dict]:
    """(closed f shells, open alpha m, open beta m) of the Hund max-M_L component."""
    if n == 14:
        return {3: 1}, {}, {}
    if n <= 7:
        return {}, {3: list(range(3, 3 - n, -1))}, {}
    return {}, {3: list(range(-3, 4))}, {3: list(range(3, 3 - (n - 7), -1))}


def occupation(n: int, cation: bool):
    from openea_benchmark.axial.scf import atomic_occupation
    f_closed, f_alpha, f_beta = f_shell(n)
    closed = {0: 5 if cation else 6, 1: 4, 2: 2, **f_closed}
    alpha = {**f_alpha}
    if cation:
        alpha[0] = [0]
    spin = len(sum(alpha.values(), [])) - len(sum(f_beta.values(), []))
    return atomic_occupation(closed, alpha, f_beta), spin


def point(el: str, x: str, cation: bool, methods: tuple, cache: Path) -> dict:
    tag = f"{el}_{x}Z_{'cation' if cation else 'neutral'}_{'+'.join(methods)}"
    path = cache / f"{tag}.json"
    if path.is_file():
        return json.loads(path.read_text())
    from pyscf import gto, scf
    from openea_benchmark.adaptive.stage3_execution import valence_frozen_core_orbitals
    from openea_benchmark.axial.basis import axial_integrals
    from openea_benchmark.axial.cc import axial_cc
    from openea_benchmark.axial.scf import atomic_blocks, axial_uhf, minao_density
    from openea_benchmark.heavy_elements import basis_name_for, resolve_named_basis
    occ, spin = occupation(F_COUNT[el], cation)
    basis = resolve_named_basis(basis_name_for(el, f"aug-cc-pV{x}Z"), el)
    mol = gto.M(atom=f"{el} 0 0 0", basis={el: basis.shells}, charge=int(cation), spin=spin, verbose=0)
    t0 = time.time()
    ints = axial_integrals(mol, hcore_ao=scf.ROHF(mol).sfx2c1e().get_hcore(), cholesky_tol=CHOLESKY_TOL)
    res = axial_uhf(ints, occ, minao_density(mol, ints), block_of_ao=atomic_blocks(mol, ints))
    nfrozen = valence_frozen_core_orbitals(mol)
    corr = axial_cc(ints, res, mol.nelectron, nfrozen, methods)
    out = {"element": el, "basis": basis_name_for(el, f"aug-cc-pV{x}Z"), "nao": mol.nao_nr(),
           "charge": int(cation), "spin": spin, "scf": res.e_tot, "scf_converged": res.converged,
           "s2": res.s2, "nfrozen": nfrozen, "cholesky_tol": CHOLESKY_TOL, "cholesky_vectors": ints.eri.nvec,
           "correlation": corr, "seconds": time.time() - t0}
    if res.converged:
        path.write_text(json.dumps(out, indent=1))
    return out


def main(argv):
    outdir, el, x = Path(argv[1]), argv[2], argv[3]
    methods = tuple(argv[4].split(",")) if len(argv) > 4 else ("ccsd(t)",)
    if len(argv) > 5:
        from pyscf import lib
        lib.num_threads(int(argv[5]))
    cache = outdir / "points"
    cache.mkdir(parents=True, exist_ok=True)
    e = {}
    for cation in (False, True):
        p = point(el, x, cation, methods, cache)
        e[cation] = {"scf": p["scf"], **{k: p["scf"] + v for k, v in p["correlation"].items()}}
        print(f"{el} {x}Z {'cation ' if cation else 'neutral'}: nao {p['nao']} conv {p['scf_converged']} "
              f"<S2> {p['s2']:.3f} nvec {p['cholesky_vectors']} " +
              " ".join(f"{k} {v:.8f}" for k, v in e[cation].items()) + f" ({p['seconds']:.0f} s)", flush=True)
    ip = {k: (e[True][k] - e[False][k]) * H2EV for k in e[False]}
    print(f"{el} {x}Z IP (eV): " + " ".join(f"{k} {v:.4f}" for k, v in ip.items()) +
          f"   NIST {NIST_IP_EV[el]:.4f} (validation only)", flush=True)
    (outdir / f"{el}_{x}Z_{'+'.join(methods)}.json").write_text(json.dumps({"ip_ev": ip, "nist_ev": NIST_IP_EV[el]},
                                                                           indent=1))


if __name__ == "__main__":
    main(sys.argv)
