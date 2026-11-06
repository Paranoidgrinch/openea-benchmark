"""Beyond-CCSDT increment by DMRG in frozen-natural-orbital spaces (Route A).

When exact CCSDTQ does not fit the machine (transition-metal oxides: 3.3 TB even in
cc-pVDZ), the higher-excitation increment is computed as

    delta(state, n) = E_DMRG(n) - E_CCSDT(n)

in the same truncated orbital space: frozen valence core, all occupied orbitals, and
the ``n`` virtual natural orbitals of largest occupation from the spin-summed UMP2
one-particle density (semicanonical ROHF orbitals) projected onto the ROHF virtual
space.  The reference determinant is unchanged; CCSDT (CCpy) and DMRG (block2, SU(2),
Fiedler ordering) see the same Hamiltonian, so their difference is the correlation
missing in CCSDT within that space.  DMRG energies are extrapolated linearly in the
discarded weight over the last bond dimensions; the distance of the largest bond
dimension from the extrapolated value is reported.

Two truncation levels are computed.  The EA increment is the one of the larger
space; its uncertainty is the change between the two levels plus the DMRG
extrapolation distances.  Validation on PO (verify_v1, exact CCSDTQ - CCSDT in
cc-pVDZ, 40 h and >100 GB): see the post-CC stage docstring.

No empirical parameter enters: the truncation sizes and bond dimensions are cost
settings, and their effect is measured (level change, extrapolation distance).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .electronic_ea_stage import _save_json


@dataclass(frozen=True)
class FNODMRGPolicy:
    """Cost settings (not fitted): active-space sizes and the DMRG schedule."""
    # largest DMRG active space (correlated occupied + kept virtual orbitals).  PO cc-pVDZ
    # (2026-11-06): with M <= 1000, 18-19 active orbitals extrapolate to 0.04/0.25 mEh,
    # 22-23 orbitals only to 0.4/1.0 mEh; the level change 8 -> 12 virtuals was 0.0008 eV.
    max_active_orbitals: int = 20
    # the smaller level keeps this fraction of the larger level's virtuals
    second_level_fraction: float = 0.6
    min_virtuals: int = 4
    bond_dimensions: tuple[int, ...] = (250, 500, 1000, 2000)
    sweeps_per_bond_dimension: int = 10
    dmrg_stack_memory_mb: int = 6000


def truncation_levels(n_occ_active: int, n_virtual: int, policy: FNODMRGPolicy) -> tuple[int, int]:
    """(larger, smaller) number of kept virtual natural orbitals.  ``n_occ_active`` is the
    larger of the two states, so neutral and anion keep the same number of virtuals."""
    large = max(policy.min_virtuals, min(n_virtual, policy.max_active_orbitals - n_occ_active))
    small = max(policy.min_virtuals, int(round(policy.second_level_fraction * large)))
    return large, min(small, large - 1)


def natural_orbital_reference(mf, nfrozen: int):
    """``mf`` (C1, occupation-ordered ROHF/RHF) with its virtual block rotated to UMP2
    natural orbitals, largest occupation first.  Returns (mf, occupations, n_docc, n_socc)."""
    from pyscf import mp, scf
    occ = np.asarray(mf.mo_occ)
    C = np.asarray(mf.mo_coeff)
    ndocc, nsocc = int((occ > 1.5).sum()), int(((occ > 0.5) & (occ < 1.5)).sum())
    nocc, nmo = ndocc + nsocc, C.shape[1]
    dm = mf.make_rdm1()
    fock = mf.get_fock(dm=dm)
    fa, fb = (fock.focka, fock.fockb) if hasattr(fock, "focka") else (fock, fock)

    def semicanonical(f, n_occ_spin):
        Cs, e = C.copy(), np.zeros(nmo)
        for sl in (slice(nfrozen, n_occ_spin), slice(n_occ_spin, nmo)):
            w, v = np.linalg.eigh(C[:, sl].T @ f @ C[:, sl])
            Cs[:, sl], e[sl] = C[:, sl] @ v, w
        e[:nfrozen] = -1.0e3
        return Cs, e

    Ca, ea = semicanonical(fa, nocc)
    Cb, eb = semicanonical(fb, ndocc)
    uhf = scf.UHF(mf.mol)
    uhf.mo_coeff, uhf.mo_energy, uhf.e_tot = (Ca, Cb), (ea, eb), mf.e_tot
    uhf.mo_occ = (np.r_[np.ones(nocc), np.zeros(nmo - nocc)], np.r_[np.ones(ndocc), np.zeros(nmo - ndocc)])
    ump2 = mp.UMP2(uhf, frozen=nfrozen)
    ump2.kernel()
    da, db = ump2.make_rdm1(ao_repr=True)
    S = mf.mol.intor("int1e_ovlp")
    Cv = C[:, nocc:]
    w, v = np.linalg.eigh(Cv.T @ S @ (da + db) @ S @ Cv)
    order = np.argsort(-w)
    mf.mo_coeff = np.hstack([C[:, :nocc], Cv @ v[:, order]])
    return mf, w[order], ndocc, nsocc


def ccsdt_energy(mf, nfrozen: int, ndelete: int, convergence: float, max_iter: int) -> float:
    from ccpy.drivers.driver import Driver
    driver = Driver.from_pyscf(mf, nfrozen=nfrozen, ndelete=ndelete)
    driver.options["energy_convergence"] = convergence
    driver.options["amp_convergence"] = convergence
    driver.options["maximum_iterations"] = max_iter
    driver.run_cc(method="ccsdt")
    return float(mf.e_tot) + float(driver.correlation_energy)


def dmrg_energy(mf, nfrozen: int, n_active: int, policy: FNODMRGPolicy, scratch: str, threads: int) -> dict:
    from pyblock2._pyscf.ao2mo import integrals as itg
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    _, n_elec, spin, ecore, h1e, g2e, _ = itg.get_rhf_integrals(mf, ncore=nfrozen, ncas=n_active, g2e_symm=1)
    driver = DMRGDriver(scratch=scratch, symm_type=SymmetryTypes.SU2, n_threads=threads,
                        stack_mem=int(policy.dmrg_stack_memory_mb * 1e6))
    driver.initialize_system(n_sites=n_active, n_elec=n_elec, spin=spin)
    idx = driver.orbital_reordering(h1e, g2e)
    h1e, g2e = h1e[idx][:, idx], g2e[idx][:, idx][:, :, idx][:, :, :, idx]
    mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
    ket = driver.get_random_mps(tag="KET", bond_dim=policy.bond_dimensions[0], nroots=1)
    n = policy.sweeps_per_bond_dimension
    third = max(1, n // 3)
    noises = [1e-5] * third + [1e-6] * third + [0.0] * (n - 2 * third)
    rows = []
    for m in policy.bond_dimensions:
        e = driver.dmrg(mpo, ket, n_sweeps=n, bond_dims=[m] * n, noises=noises, thrds=[1e-8] * n, iprint=0)
        rows.append({"bond_dimension": m, "energy_hartree": float(e),
                     "discarded_weight": float(driver.get_dmrg_results()[1][-1])})
    tail = rows[-3:]
    slope, e0 = np.polyfit([r["discarded_weight"] for r in tail], [r["energy_hartree"] for r in tail], 1)
    last = rows[-1]["energy_hartree"]
    # variational: the extrapolated value cannot lie above the largest-M energy
    e0 = min(float(e0), last)
    return {"energy_hartree": e0, "largest_bond_energy_hartree": last,
            "extrapolation_distance_hartree": last - e0, "sweeps": rows}


def point_signature(spec: Any, basis: str, level: int, policy: FNODMRGPolicy, ref_policy: Any) -> str:
    return sha256(json.dumps({"state": asdict(spec), "basis": basis, "level": level, "policy": asdict(policy),
                              "reference_policy": asdict(ref_policy)}, sort_keys=True, default=str)
                  .encode()).hexdigest()


def run_fno_dmrg_point(spec: Any, basis: str, level: int, n_occ_active_max: int, policy: FNODMRGPolicy,
                       ref_policy: Any, path: str, scratch: str) -> dict:
    """One state, one truncation level (0 = larger space, 1 = smaller); ``n_occ_active_max``
    (the larger of both states) fixes the kept virtuals for neutral and anion alike."""
    out = Path(path)
    signature = point_signature(spec, basis, level, policy, ref_policy)
    if out.is_file():
        cached = json.loads(out.read_text())
        if cached.get("status") == "COMPLETED" and cached.get("signature") == signature:
            return cached
    from pyscf import lib
    from .adaptive.stage3_execution import valence_frozen_core_orbitals
    from .post_ccsd_t_stage import reference_c1
    record: dict[str, Any] = {"signature": signature, "role": spec.role, "basis": basis, "level": level}
    try:
        t0 = time.perf_counter()
        mf = reference_c1(spec, basis, ref_policy)
        nfz = valence_frozen_core_orbitals(mf.mol)
        mf, no_occ, ndocc, nsocc = natural_orbital_reference(mf, nfz)
        n_occ_active = ndocc + nsocc - nfz
        n_virtual = mf.mo_coeff.shape[1] - ndocc - nsocc
        large, small = truncation_levels(n_occ_active_max, n_virtual, policy)
        keep = (large, small)[level]
        e_ccsdt = ccsdt_energy(mf, nfz, n_virtual - keep, ref_policy.cc_convergence, ref_policy.cc_max_iterations)
        Path(scratch).mkdir(parents=True, exist_ok=True)
        dm = dmrg_energy(mf, nfz, n_occ_active + keep, policy, scratch, lib.num_threads())
        record.update(status="COMPLETED", nfrozen=nfz, n_occ_active=n_occ_active, n_virtual=n_virtual,
                      n_keep=keep, kept_no_occupation_sum=float(no_occ[:keep].sum()),
                      dropped_no_occupation_sum=float(no_occ[keep:].sum()),
                      reference_hartree=float(mf.e_tot), ccsdt_hartree=e_ccsdt, dmrg=dm,
                      increment_hartree=dm["energy_hartree"] - e_ccsdt,
                      wall_seconds=time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 - recorded, the layer fails closed
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)  # MPS/renormalized operators, per point
    _save_json(out, record)
    return record


def ea_increment(points: dict[tuple[str, int], dict], hartree_to_ev: float) -> dict:
    """EA increment and its half-width from the four points {(role, level): record}."""
    inc = {lv: (points[("neutral", lv)]["increment_hartree"] - points[("anion", lv)]["increment_hartree"])
           * hartree_to_ev for lv in (0, 1)}
    extrap = sum(points[(role, 0)]["dmrg"]["extrapolation_distance_hartree"]
                 for role in ("neutral", "anion")) * hartree_to_ev
    return {"delta_ev": inc[0], "delta_by_level_ev": inc, "level_change_ev": abs(inc[0] - inc[1]),
            "dmrg_extrapolation_ev": extrap, "half_width_ev": abs(inc[0] - inc[1]) + extrap}
