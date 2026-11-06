"""Beyond-CCSDT increment by CCSDTQ in frozen-natural-orbital spaces (Route A).

When exact CCSDTQ does not fit the machine (transition-metal oxides: 3.3 TB even in
cc-pVDZ), the quadruples increment is computed as

    delta(state, n) = E_CCSDTQ(n) - E_CCSDT(n)

in the same truncated orbital space: frozen valence core, all occupied orbitals, and
the ``n`` virtual natural orbitals of largest occupation from the spin-summed UMP2
one-particle density (semicanonical ROHF orbitals) projected onto the ROHF virtual
space.  The reference determinant is unchanged; both methods run in CCpy on the same
Hamiltonian, so their difference is the quadruples correlation within that space.

Two truncation levels are computed.  The larger keeps as many virtuals as the
CCSDTQ memory limit of the stage allows (the same estimate that decides on exact
CCSDTQ), the smaller a fixed fraction of it.  The EA increment is the one of the
larger space; its uncertainty is the change between the two levels.

FNO-DMRG was the first version (2026-11-06) and was replaced the same day: in these
spaces of weakly occupied natural orbitals (dynamic correlation) DMRG converged badly
(PO anion, 18 orbitals, M <= 1000: 0.38 mEh extrapolation distance, M = 2000 too slow).

No empirical parameter enters: the truncation sizes are cost settings, and their
effect is measured (level change).
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
class FNOPolicy:
    """Cost settings (not fitted)."""
    # the smaller level keeps this fraction of the larger level's virtuals
    second_level_fraction: float = 0.6
    min_virtuals: int = 4
    # in-core CCpy copies of the T4 array (amplitudes, residual, DIIS vectors, intermediates)
    amplitude_copies: int = 16


def t4_amplitude_mb(n_occ_alpha: int, n_occ_beta: int, n_virtual_alpha: int, n_virtual_beta: int) -> float:
    """One copy of the T4 array: 8 bytes x sum over spin blocks of (o v)^4."""
    x, y = n_occ_alpha * n_virtual_alpha, n_occ_beta * n_virtual_beta
    return 8 * sum(x ** (4 - k) * y ** k for k in range(5)) / 1.0e6


def truncation_levels(n_occ_alpha: int, n_occ_beta: int, n_virtual: int, memory_limit_mb: float,
                      policy: FNOPolicy) -> tuple[int, int]:
    """(larger, smaller) number of kept virtual natural orbitals (beta virtuals = kept +
    open shells).  Occupations are the larger of both states, so neutral and anion keep the
    same number of virtuals."""
    open_shells = n_occ_alpha - n_occ_beta
    large = policy.min_virtuals
    for keep in range(policy.min_virtuals, n_virtual + 1):
        if policy.amplitude_copies * t4_amplitude_mb(n_occ_alpha, n_occ_beta, keep, keep + open_shells) \
                > memory_limit_mb:
            break
        large = keep
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


def cc_energy(mf, nfrozen: int, ndelete: int, method: str, convergence: float, max_iter: int) -> float:
    """Total CCpy energy (``method`` "ccsdt" or "ccsdtq"); a new driver per method."""
    from ccpy.drivers.driver import Driver
    driver = Driver.from_pyscf(mf, nfrozen=nfrozen, ndelete=ndelete)
    driver.options["energy_convergence"] = convergence
    driver.options["amp_convergence"] = convergence
    driver.options["maximum_iterations"] = max_iter
    driver.run_cc(method=method)
    return float(mf.e_tot) + float(driver.correlation_energy)


def point_signature(spec: Any, basis: str, n_keep: int, policy: FNOPolicy, ref_policy: Any) -> str:
    return sha256(json.dumps({"method": "FNO_CCSDTQ", "state": asdict(spec), "basis": basis, "n_keep": n_keep,
                              "policy": asdict(policy), "reference_policy": asdict(ref_policy)},
                             sort_keys=True, default=str).encode()).hexdigest()


def run_fno_point(spec: Any, basis: str, n_keep: int, policy: FNOPolicy, ref_policy: Any, path: str) -> dict:
    """One state in the space of ``n_keep`` virtual natural orbitals: CCSDT and CCSDTQ."""
    out = Path(path)
    signature = point_signature(spec, basis, n_keep, policy, ref_policy)
    if out.is_file():
        cached = json.loads(out.read_text())
        if cached.get("status") == "COMPLETED" and cached.get("signature") == signature:
            return cached
    from .adaptive.stage3_execution import valence_frozen_core_orbitals
    from .post_ccsd_t_stage import reference_c1
    record: dict[str, Any] = {"signature": signature, "role": spec.role, "basis": basis, "n_keep": n_keep}
    try:
        t0 = time.perf_counter()
        mf = reference_c1(spec, basis, ref_policy)
        nfz = valence_frozen_core_orbitals(mf.mol)
        mf, no_occ, ndocc, nsocc = natural_orbital_reference(mf, nfz)
        n_virtual = mf.mo_coeff.shape[1] - ndocc - nsocc
        keep = min(n_keep, n_virtual)
        conv, it = ref_policy.cc_convergence, ref_policy.cc_max_iterations
        e = {m: cc_energy(mf, nfz, n_virtual - keep, m, conv, it) for m in ("ccsdt", "ccsdtq")}
        record.update(status="COMPLETED", nfrozen=nfz, n_virtual=n_virtual, n_keep=keep,
                      kept_no_occupation_sum=float(no_occ[:keep].sum()),
                      dropped_no_occupation_sum=float(no_occ[keep:].sum()),
                      reference_hartree=float(mf.e_tot), ccsdt_hartree=e["ccsdt"], ccsdtq_hartree=e["ccsdtq"],
                      increment_hartree=e["ccsdtq"] - e["ccsdt"], wall_seconds=time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 - recorded, the layer fails closed
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    _save_json(out, record)
    return record


def ea_increment(points: dict[tuple[str, int], dict], hartree_to_ev: float) -> dict:
    """EA increment and its half-width from the four points {(role, level): record}."""
    inc = {lv: (points[("neutral", lv)]["increment_hartree"] - points[("anion", lv)]["increment_hartree"])
           * hartree_to_ev for lv in (0, 1)}
    return {"delta_ev": inc[0], "delta_by_level_ev": inc, "level_change_ev": abs(inc[0] - inc[1]),
            "half_width_ev": abs(inc[0] - inc[1])}
