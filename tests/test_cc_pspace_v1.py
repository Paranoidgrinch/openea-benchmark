"""Symmetry-restricted CC(P) with every allowed triple = CCSDT (cc_pspace.py)."""
import contextlib
import io

import numpy as np
import pytest
from pyscf import gto, scf

from openea_benchmark.cc_pspace import run_symmetric_ccsdt, symmetric_triples


def _driver(mf, nfrozen):
    from ccpy.drivers.driver import Driver
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
    system, h = load_pyscf_integrals(mf, nfrozen=nfrozen)
    d = Driver(system, h)
    d.options["energy_convergence"] = d.options["amp_convergence"] = 1e-9
    return system, d


def _brute(oa, va, ob, vb, combine):
    from itertools import combinations, product
    f = (lambda *x: np.bitwise_xor.reduce(np.array(x))) if combine == "xor" else (lambda *x: sum(x))
    out = {"aaa": set(), "aab": set(), "abb": set(), "bbb": set()}
    for (i, j, k), (a, b, c) in product(combinations(range(len(oa)), 3), combinations(range(len(va)), 3)):
        if f(oa[i], oa[j], oa[k]) == f(va[a], va[b], va[c]):
            out["aaa"].add((a + 1, b + 1, c + 1, i + 1, j + 1, k + 1))
    for (i, j, k), (a, b, c) in product(combinations(range(len(ob)), 3), combinations(range(len(vb)), 3)):
        if f(ob[i], ob[j], ob[k]) == f(vb[a], vb[b], vb[c]):
            out["bbb"].add((a + 1, b + 1, c + 1, i + 1, j + 1, k + 1))
    for (i, j), k, (a, b), c in product(combinations(range(len(oa)), 2), range(len(ob)),
                                        combinations(range(len(va)), 2), range(len(vb))):
        if f(oa[i], oa[j], ob[k]) == f(va[a], va[b], vb[c]):
            out["aab"].add((a + 1, b + 1, c + 1, i + 1, j + 1, k + 1))
    for i, (j, k), a, (b, c) in product(range(len(oa)), combinations(range(len(ob)), 2),
                                        range(len(va)), combinations(range(len(vb)), 2)):
        if f(oa[i], ob[j], ob[k]) == f(va[a], vb[b], vb[c]):
            out["abb"].add((a + 1, b + 1, c + 1, i + 1, j + 1, k + 1))
    return out


@pytest.mark.parametrize("combine", ["xor", "sum"])
def test_triples_list_equals_brute_force(combine):
    rng = np.random.default_rng(3)
    hi = 4 if combine == "xor" else 3
    lo = 0 if combine == "xor" else -3
    oa, va, ob, vb = (rng.integers(lo, hi, size=n) for n in (4, 7, 3, 8))
    ex = symmetric_triples(oa, va, ob, vb, combine)
    ref = _brute(oa, va, ob, vb, combine)
    for case in ref:
        got = {tuple(int(x) for x in row) for row in ex[case]} if ref[case] else set()
        assert got == ref[case], case
        assert len(ex[case]) == max(1, len(ref[case]))


def test_symmetric_ccp_equals_dense_ccsdt():
    mol = gto.M(atom="C 0 0 0; N 0 0 1.1718", basis="cc-pvdz", spin=1, symmetry="C2v", verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-11)
    system, dense = _driver(mf, 2)
    with contextlib.redirect_stdout(io.StringIO()):
        dense.run_cc(method="ccsdt")
    _, packed = _driver(mf, 2)
    irr = system.point_group_irrep_to_number
    lab = [irr[s] for s in system.orbital_symmetries]
    with contextlib.redirect_stdout(io.StringIO()):
        counts = run_symmetric_ccsdt(packed, lab, lab, system.noccupied_alpha, system.noccupied_beta)
    assert sum(counts.values()) > 0
    assert packed.correlation_energy == pytest.approx(dense.correlation_energy, abs=1e-8)
