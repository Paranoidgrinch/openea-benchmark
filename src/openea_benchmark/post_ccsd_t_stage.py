"""Automated post-CCSD(T) layer: iterative triples and connected quadruples (Patch 49).

    Delta_T = EA[CCSDT]  - EA[CCSD(T)]   in aug-cc-pVDZ and aug-cc-pVTZ
    Delta_Q = EA[CCSDTQ] - EA[CCSDT]     in aug-cc-pVDZ, else cc-pVDZ (exact quadruples)

frozen (noble-gas) core, nonrelativistic, at the TZ equilibrium of the selected
states.  All three methods run in CCpy on one reference per state and basis:
the ROHF/RHF reference is first converged in the state's symmetry block (point
group, irrep occupation, seeded from its high-level checkpoint), then carried
into C1 (CCpy needs C1 metadata) and must keep its energy there.

Central values: Delta_T(aug-cc-pVTZ), bound |Delta_T(TZ) - Delta_T(DZ)| (only the
aDZ value when aTZ CCSDT exceeds the memory limit: bound 50 % of it);
Delta_Q(first affordable of aug-cc-pVDZ, cc-pVDZ), bound 50 % of |Delta_Q| (pilot; DZ quadruples are basis-
incomplete).  CR-CC(2,4) is not used: the OH study (runs/oh_crcc24_transfer_*)
showed its DZ->TZ transfer disagreeing with exact CCSDTQ by 0.1 eV.

Triples and quadruples partly cancel, so the layer is included in EA_0 only
when *both* are available.  Exact CCSDTQ amplitudes grow as (o v)^4; when the
estimated storage exceeds ``q_memory_limit_mb`` the layer reports
``QUADRUPLES_NOT_AFFORDABLE`` with Delta_T as evidence only (fail closed).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core_valence_stage import selected_state_specs
from .electronic_ea_stage import HARTREE_TO_EV, _save_json
from .point_pool import run_parallel

METHODS = ("CCSD(T)", "CCSDT", "CCSDTQ")
# File tokens; "CCSD(T)" without parentheses would collide with "CCSDT".
METHOD_TOKEN = {"CCSD(T)": "CCSD_pT", "CCSDT": "CCSDT", "CCSDTQ": "CCSDTQ"}


@dataclass(frozen=True)
class PostCCSDTStagePolicy:
    """Explicit pilot policy, not a validated production threshold set."""
    triples_bases: tuple[str, ...] = ("aug-cc-pvdz", "aug-cc-pvtz")
    # First affordable basis is used; cc-pVDZ (W4 practice for the quadruples
    # increment) when aug-cc-pVDZ CCSDTQ exceeds the storage limit.
    quadruples_bases: tuple[str, ...] = ("aug-cc-pvdz", "cc-pvdz")
    quadruples_relative_bound: float = 0.5
    # Per-point memory limit; every point reserves its estimate in the point-pool
    # memory gate, and points above point_pool.BIG_TASK_MB (100 GB) run one at a
    # time machine-wide (user decision 2026-10-13: NO, O2, ClO CCSDTQ/cc-pVDZ need
    # 130-200 GB; run them one after the other).
    q_memory_limit_mb: int = 210000
    max_memory_mb: int = 48000
    scf_conv_tol: float = 1.0e-10
    cc_convergence: float = 1.0e-8
    cc_max_iterations: int = 200
    c1_energy_tolerance_hartree: float = 1.0e-7


def _mol(spec: Any, basis: str, symmetry: Any, max_memory_mb: int):
    from pyscf import gto
    return gto.M(atom=[(spec.atoms[0], (0.0, 0.0, 0.0)), (spec.atoms[1], (0.0, 0.0, float(spec.r_angstrom)))],
                 basis=basis, charge=spec.charge, spin=spec.spin_2s, unit="Angstrom", symmetry=symmetry,
                 verbose=0, max_memory=max_memory_mb, dump_input=False)


def reference_c1(spec: Any, basis: str, policy: PostCCSDTStagePolicy):
    """Reference converged in the symmetry block, then carried into C1 unchanged."""
    from pyscf import scf
    open_shell = spec.spin_2s > 0
    mol = _mol(spec, basis, spec.point_group or False, policy.max_memory_mb)
    mf = scf.ROHF(mol) if open_shell else scf.RHF(mol)
    if spec.irrep_nelec is not None:
        mf.irrep_nelec = ({k: (int(v[0]), int(v[1])) for k, v in spec.irrep_nelec.items()} if open_shell
                          else {k: int(v[0]) + int(v[1]) for k, v in spec.irrep_nelec.items()})
    guess = scf.rohf.init_guess_by_chkfile if open_shell else scf.hf.init_guess_by_chkfile
    mf.conv_tol, mf.max_cycle, mf.max_memory = policy.scf_conv_tol, 200, policy.max_memory_mb
    mf.kernel(dm0=guess(mol, spec.source_checkpoint_path, project=True))
    if not mf.converged:
        raise RuntimeError(f"symmetry-block SCF not converged ({basis})")
    mol1 = _mol(spec, basis, "C1", policy.max_memory_mb)
    mf1 = scf.ROHF(mol1) if open_shell else scf.RHF(mol1)
    mf1.conv_tol, mf1.max_cycle, mf1.max_memory = policy.scf_conv_tol, 50, policy.max_memory_mb
    mf1.kernel(dm0=mf.make_rdm1())
    if not mf1.converged or abs(mf1.e_tot - mf.e_tot) > policy.c1_energy_tolerance_hartree:
        raise RuntimeError(f"C1 reference left the symmetry-block solution ({mf1.e_tot} vs {mf.e_tot})")
    return mf1


# CCpy keeps diis_size (6) amplitude and residual vectors in core, plus the
# amplitudes, the residual and intermediates: ~16 copies of the highest-rank
# amplitude array.  2026-10-13: CCSDTQ/cc-pVDZ of NO, ClO and N2 together with
# CCSDT/aug-cc-pVTZ points took ~450 GB; the earlier 3-copy estimate was 5x low.
AMPLITUDE_COPIES = 2 * 6 + 4


def cc_storage_mb(spec: Any, basis: str, method: str) -> float:
    """Memory estimate of a CCpy CCSDT or CCSDTQ point: copies x 8 bytes x sum over spin blocks of (o v)^n."""
    from .adaptive.stage3_execution import valence_frozen_core_orbitals
    rank = {"CCSD(T)": 2, "CCSDT": 3, "CCSDTQ": 4}[method]
    mol = _mol(spec, basis, False, 1000)
    nfz = valence_frozen_core_orbitals(mol)
    na, nb = mol.nelec
    nmo = mol.nao_nr()
    x, y = (na - nfz) * (nmo - na), (nb - nfz) * (nmo - nb)
    return AMPLITUDE_COPIES * 8 * sum(x ** (rank - k) * y ** k for k in range(rank + 1)) / 1.0e6


def quadruples_storage_mb(spec: Any, basis: str) -> float:
    return cc_storage_mb(spec, basis, "CCSDTQ")


def _run_point(spec: Any, basis: str, method: str, policy: PostCCSDTStagePolicy, path: str) -> dict:
    out = Path(path)
    signature = sha256(json.dumps({"state": {k: v for k, v in asdict(spec).items()}, "basis": basis,
                                   "method": method, "policy": asdict(policy)}, sort_keys=True, default=str)
                       .encode()).hexdigest()
    if out.is_file():
        cached = json.loads(out.read_text())
        if cached.get("signature") == signature and cached.get("status") == "COMPLETED":
            return cached
    from ccpy.drivers.driver import Driver
    from .adaptive.stage3_execution import valence_frozen_core_orbitals
    record: dict[str, Any] = {"signature": signature, "role": spec.role, "basis": basis, "method": method}
    try:
        mf = reference_c1(spec, basis, policy)
        nfz = valence_frozen_core_orbitals(mf.mol)
        driver = Driver.from_pyscf(mf, nfrozen=nfz)
        driver.options["energy_convergence"] = policy.cc_convergence
        driver.options["amp_convergence"] = policy.cc_convergence
        driver.options["maximum_iterations"] = policy.cc_max_iterations
        if method == "CCSD(T)":
            driver.run_cc(method="ccsd")
            corr = float(driver.correlation_energy)
            driver.run_ccp3(method="ccsd(t)")
            corr += float(driver.deltap3[0]["A"])
        else:
            driver.run_cc(method=method.lower())
            corr = float(driver.correlation_energy)
        record.update(status="COMPLETED", nfrozen=nfz, reference_hartree=float(mf.e_tot),
                      total_hartree=float(mf.e_tot) + corr)
    except Exception as exc:  # noqa: BLE001 - recorded, layer fails closed
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    _save_json(out, record)
    return record


def run_post_ccsd_t_stage(*, electronic_ea: Mapping[str, Any], requests: Sequence[Any],
                          point_records: Mapping[str, Mapping[str, Any]], refinement_dir: Path, output_dir: Path,
                          policy: PostCCSDTStagePolicy = PostCCSDTStagePolicy(), point_fn=None) -> dict:
    output_dir = Path(output_dir)
    record: dict[str, Any] = {"stage": "POST_CCSD_T", "policy": asdict(policy), "is_production_ea": False,
                              "validation_metadata_used_in_computation": False}

    def done(**fields) -> dict:
        record.update(fields)
        _save_json(output_dir / "post_ccsd_t.json", record)
        return record

    if electronic_ea.get("status") != "ELECTRONIC_EA_EVALUATED":
        return done(status="NOT_ATTEMPTED", reason="single-basis electronic EA not evaluated")
    states = selected_state_specs(electronic_ea, requests, point_records, refinement_dir)
    if isinstance(states, str):
        return done(status="NOT_ATTEMPTED", reason=states)
    def need(basis, method):
        return max(cc_storage_mb(s, basis, method) for s in states.values())

    estimates = {b: need(b, "CCSDTQ") for b in policy.quadruples_bases}
    record["quadruples_storage_estimate_mb"] = estimates
    q_basis = next((b for b in policy.quadruples_bases if estimates[b] <= policy.q_memory_limit_mb), None)
    run_q = q_basis is not None
    record["quadruples_basis"] = q_basis
    t_bases = tuple(b for b in policy.triples_bases if need(b, "CCSDT") <= policy.q_memory_limit_mb)
    record["triples_bases_affordable"] = list(t_bases)
    if not t_bases:
        return done(status="TRIPLES_NOT_AFFORDABLE", delta_post_ev=None, bound_ev=None,
                    reason=f"CCSDT exceeds {policy.q_memory_limit_mb} MB in every triples basis")
    tasks = [(basis, method) for basis in t_bases for method in ("CCSD(T)", "CCSDT")]
    if run_q:
        if q_basis not in t_bases:
            tasks.append((q_basis, "CCSD(T)"))
            tasks.append((q_basis, "CCSDT"))
        tasks.append((q_basis, "CCSDTQ"))
    fn = point_fn or _run_point
    jobs, reserves = [], []
    for role, spec in states.items():
        for basis, method in tasks:
            reserves.append(cc_storage_mb(spec, basis, method))
            print(f"[POST-CC] {spec.system} {role} {basis} {method}", flush=True)
            jobs.append((spec, basis, method, policy,
                         str(output_dir / "points" / f"{role}__{basis}__{METHOD_TOKEN[method]}.json")))
    results = run_parallel(fn, jobs, reserve_mb=reserves)
    energies = {(r["role"], r["basis"], r["method"]): r for r in results}
    record["points"] = [{k: v for k, v in r.items() if k != "signature"} for r in results]
    failed = [r for r in results if r.get("status") != "COMPLETED"]
    if failed:
        return done(status="EXECUTION_BLOCKED", reason="; ".join(f"{r['role']} {r['basis']} {r['method']}: "
                                                                   f"{r.get('error')}" for r in failed))

    def ea(basis: str, method: str) -> float:
        return (energies[("neutral", basis, method)]["total_hartree"]
                - energies[("anion", basis, method)]["total_hartree"]) * HARTREE_TO_EV

    delta_t = {b: ea(b, "CCSDT") - ea(b, "CCSD(T)") for b in t_bases}
    t_central = delta_t[t_bases[-1]]
    # Two bases: the last basis change; one basis only: 50 % of the increment.
    t_bound = (abs(delta_t[t_bases[-1]] - delta_t[t_bases[0]]) if len(t_bases) > 1
               else 0.5 * abs(t_central))
    record.update(delta_t_by_basis_ev=delta_t, delta_t_ev=t_central, delta_t_bound_ev=t_bound)
    if not run_q:
        return done(status="QUADRUPLES_NOT_AFFORDABLE", delta_post_ev=None, bound_ev=None,
                    reason=f"CCSDTQ storage {estimates} MB > {policy.q_memory_limit_mb} MB; "
                           "triples alone are not applied (partial cancellation with quadruples)")
    delta_q = ea(q_basis, "CCSDTQ") - ea(q_basis, "CCSDT")
    q_bound = policy.quadruples_relative_bound * abs(delta_q)
    return done(status="CLEARED", delta_q_ev=delta_q, delta_q_bound_ev=q_bound,
                delta_post_ev=t_central + delta_q, bound_ev=t_bound + q_bound)
