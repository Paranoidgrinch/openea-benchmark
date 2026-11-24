"""Coupled cluster (CCpy) on an axial determinant, in real arithmetic.

``semicanonical=True`` (the CBS/Stage-3 definition of ROHF-CCSD(T), see
``post_ccsd_t_stage``): the ROHF determinant in UHF form with the alpha and beta
occupied and virtual spaces canonicalized separately (within each m block, so the
orbitals stay axial) and the lowest ``nfrozen`` orbitals of each spin frozen; CCpy's
spin-integrated (T) is then the standard semicanonical one.  ``semicanonical=False``:
CCpy on the ROHF orbitals (ROHF core frozen).  CCSD, CCSDT and CCSDTQ do not depend on
the choice beyond the frozen-core definition.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np

from .basis import AxialIntegrals
from .scf import AxialSCFResult, AxialUHFResult


def _mo_eri(eri, ca, cb=None):
    """(pq|rs) over real axial MO coefficients (alpha pair, beta pair)."""
    cb = ca if cb is None else cb
    return np.einsum("ijkl,ip,jq,kr,ls->pqrs", eri, ca, ca, cb, cb, optimize=True)


def semicanonical_orbitals(scf: AxialSCFResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(C_alpha, C_beta, eps_alpha, eps_beta): occupied and virtual blocks of each spin's Fock
    diagonalized separately within every m block."""
    out = []
    for fock, occupied in ((scf.fock_alpha, scf.mo_occ > 0), (scf.fock_beta, scf.mo_occ > 1)):
        c = scf.mo_coeff.copy()
        eps = np.zeros(c.shape[1])
        for space in (occupied, ~occupied):
            for k in np.unique(scf.mo_m[space]):
                cols = np.flatnonzero(space & (scf.mo_m == k))
                f = c[:, cols].T @ fock @ c[:, cols]
                e, v = np.linalg.eigh(f)
                c[:, cols], eps[cols] = c[:, cols] @ v, e
        # energy order within occupied and virtual (CCpy takes the first orbitals as occupied)
        order = np.concatenate([np.flatnonzero(occupied)[np.argsort(eps[occupied])],
                                np.flatnonzero(~occupied)[np.argsort(eps[~occupied])]])
        out.append((c[:, order], eps[order]))
    (ca, ea), (cb, eb) = out
    return ca, cb, ea, eb


def axial_cc(ints: AxialIntegrals, scf: AxialSCFResult, nelectron: int, nfrozen: int,
             methods=("ccsd(t)",), *, semicanonical: bool = True, convergence: float = 1e-9,
             max_iterations: int = 200) -> dict:
    """Correlation energies {method: E_corr}; methods: ccsd, ccsd(t), ccsdt, ccsdtq."""
    from ccpy.drivers.driver import Driver
    from ccpy.models.integrals import Integral, build_f, build_v
    from ccpy.models.system import System, UHFSystem

    uhf = isinstance(scf, AxialUHFResult)
    if uhf:
        n = scf.mo_coeff[0].shape[1]
        na, nb = scf.nocc
    else:
        n = scf.mo_coeff.shape[1]
        na, nb = int(np.sum(scf.mo_occ > 0)), int(np.sum(scf.mo_occ > 1))
    mult = na - nb + 1
    if uhf or semicanonical:
        # UHF: canonical per spin (occupied first); ROHF: semicanonical UHF form
        ca, cb, ea, eb = (*scf.mo_coeff, *scf.mo_energy) if uhf else semicanonical_orbitals(scf)
        h = {"a": ca.T @ ints.hcore @ ca, "b": cb.T @ ints.hcore @ cb}
        aa = _mo_eri(ints.eri, ca).transpose(0, 2, 1, 3)
        bb = _mo_eri(ints.eri, cb).transpose(0, 2, 1, 3)
        ab = _mo_eri(ints.eri, ca, cb).transpose(0, 2, 1, 3)   # <p q~|r s~> = (pr|qs~)
        v = {"aa": aa - aa.transpose(0, 1, 3, 2), "ab": ab, "bb": bb - bb.transpose(0, 1, 3, 2)}
        system = UHFSystem(nelectron, n, mult, nfrozen, point_group="C1",
                           orbital_symmetries={"a": ["A"] * n, "b": ["A"] * n},
                           mo_energies={"a": ea, "b": eb},
                           mo_occupation={"a": (np.arange(n) < na).astype(float),
                                          "b": (np.arange(n) < nb).astype(float)},
                           nuclear_repulsion=ints.energy_nuc, reference_energy=scf.e_tot)
        fock = build_f(h, v, system)
        corr = slice(nfrozen, n)
        hamiltonian = Integral(system, 2, {"a": fock["a"][corr, corr], "b": fock["b"][corr, corr],
                                           **{k: x[corr, corr, corr, corr] for k, x in v.items()}}, sorted=True)
    else:
        from ccpy.energy.hf_energy import calc_hf_frozen_core_energy
        from ccpy.models.integrals import getHamiltonian
        c = scf.mo_coeff
        e1 = np.asfortranarray(c.T @ ints.hcore @ c)
        e2 = np.asfortranarray(_mo_eri(ints.eri, c).transpose(0, 2, 1, 3))
        system = System(nelectron, n, mult, nfrozen, point_group="C1", orbital_symmetries=["A"] * n,
                        nuclear_repulsion=ints.energy_nuc, mo_energies=scf.mo_energy, mo_occupation=scf.mo_occ)
        system.reference_energy = scf.e_tot
        system.frozen_energy = calc_hf_frozen_core_energy(e1, e2, system)
        hamiltonian = getHamiltonian(e1, e2, system, True, True)
    out = {}
    for method in methods:
        driver = Driver(system, hamiltonian)
        driver.options["energy_convergence"] = driver.options["amp_convergence"] = convergence
        driver.options["maximum_iterations"] = max_iterations
        with contextlib.redirect_stdout(io.StringIO()):
            if method == "ccsd(t)":
                driver.run_cc(method="ccsd")
                out["ccsd"] = float(driver.correlation_energy)
                driver.run_ccp3(method="ccsd(t)")
                out["ccsd(t)"] = out["ccsd"] + float(driver.deltap3[0]["A"])
            else:
                driver.run_cc(method=method)
                out[method] = float(driver.correlation_energy)
    return out
