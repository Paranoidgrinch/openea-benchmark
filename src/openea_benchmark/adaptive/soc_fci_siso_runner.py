"""Generic, explicitly authorized MR-CASCI / FCI-SISO point diagnostic.

The *whole* spin-free manifold uses one CASSCF-optimized orbital basis.
A successful run is NOT a production SOC correction or a bound on SOC error.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from math import comb, isfinite
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np

from .mr_casscf_nevpt2_runner import MRPointRequest, MRPointSettings

HARTREE_TO_EV = 27.211386245988
# Commit used by the existing FeH research pilot; no floating/master installation.
FCI_SISO_PIN = "e0f103104f45d10881d8a3526e1f654559142141"


class SOCPointStatus(str, Enum):
    COMPLETE_REVIEW_REQUIRED = "COMPLETE_REVIEW_REQUIRED"
    BLOCKED_NOT_AUTHORIZED = "BLOCKED_NOT_AUTHORIZED"
    BLOCKED_INVALID_INPUT = "BLOCKED_INVALID_INPUT"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    SCF_NOT_CONVERGED = "SCF_NOT_CONVERGED"
    CASSCF_NOT_CONVERGED = "CASSCF_NOT_CONVERGED"
    STATE_MANIFOLD_UNRESOLVED = "STATE_MANIFOLD_UNRESOLVED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SOCSpinManifold:
    spin_2s: int
    nroots: int

    def __post_init__(self) -> None:
        if self.spin_2s < 0 or not 1 <= self.nroots <= 10:
            raise ValueError("Invalid explicitly requested spin manifold")


@dataclass(frozen=True)
class SOCPointRequest:
    point: MRPointRequest
    spin_manifolds: tuple[SOCSpinManifold, ...]
    ground_spin_2s: int
    ground_state_review_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.spin_manifolds or not self.ground_state_review_ids:
            raise ValueError("SOC requires explicit manifolds and ground-state review provenance")
        spins = [m.spin_2s for m in self.spin_manifolds]
        if len(set(spins)) != len(spins) or self.ground_spin_2s not in spins:
            raise ValueError("SOC ground spin must be present exactly once")
        if self.ground_spin_2s != self.point.spin_2s:
            raise ValueError("SOC ground spin must match the MR reference")
        nelec = sum(self.point.nelecas)
        if any(s > nelec or (nelec - s) % 2 for s in spins):
            raise ValueError("SOC spin manifold incompatible with active electron number")
        if sum(x.nroots for x in self.spin_manifolds) > 20:
            raise ValueError("Too many spin-free SOC roots")
        if sum(x.nroots * (x.spin_2s + 1) for x in self.spin_manifolds) > 80:
            raise ValueError("SOC spin-projection space exceeds diagnostic cap")


@dataclass(frozen=True)
class SOCPointAuthorization:
    authorized: bool
    rationale: str
    evidence_ids: tuple[str, ...] = ()
    max_fci_determinants: int = 200_000
    max_spin_orbit_states: int = 80

    def __post_init__(self) -> None:
        if not self.rationale.strip() or self.max_fci_determinants < 1 or self.max_spin_orbit_states < 1:
            raise ValueError("SOC authorization requires valid reason and budgets")
        if self.authorized and not self.evidence_ids:
            raise ValueError("SOC execution needs authorization evidence")


@dataclass(frozen=True)
class SOCPointSettings:
    mr: MRPointSettings = MRPointSettings()
    fci_siso_checkout: str = ""
    fci_siso_revision: str = FCI_SISO_PIN
    use_amfi: bool = True

    def __post_init__(self) -> None:
        if self.fci_siso_revision != FCI_SISO_PIN:
            raise ValueError("Unreviewed FCI-SISO version is not supported")
        if not self.use_amfi:
            raise ValueError("Non-AMFI SOC must be validated separately")


@dataclass(frozen=True)
class SOCSpinFreeRoot:
    spin_2s: int
    state_index: int
    energy_hartree: float

    def __post_init__(self) -> None:
        if self.spin_2s < 0 or self.state_index < 0 or not isfinite(self.energy_hartree):
            raise ValueError("Invalid spin-free SOC state")


@dataclass(frozen=True)
class SOCPointResult:
    request_id: str
    status: SOCPointStatus
    reason: str
    spin_free_roots: tuple[SOCSpinFreeRoot, ...] = ()
    soc_energies_hartree: tuple[float, ...] = ()
    spin_free_ground_hartree: float | None = None
    soc_ground_hartree: float | None = None
    delta_soc_hartree: float | None = None
    pyscf_version: str | None = None
    scalar_hamiltonian: str | None = None
    fci_siso_source_sha256: str | None = None
    source_checkpoint_sha256: str | None = None
    result_signature: str | None = None
    method_role: str = "DIAGNOSTIC_METHOD_DEVELOPMENT"
    ground_state_identity_validated: bool = False
    state_manifold_complete: bool = False
    soc_uncertainty_bounded: bool = False
    production_soc_validated: bool = False

    def __post_init__(self) -> None:
        if (self.ground_state_identity_validated or self.state_manifold_complete or
                self.soc_uncertainty_bounded or self.production_soc_validated):
            raise ValueError("SOC pilot cannot certify state identity or production")
        if self.method_role != "DIAGNOSTIC_METHOD_DEVELOPMENT":
            raise ValueError("SOC method role cannot be promoted")
        if self.status is not SOCPointStatus.COMPLETE_REVIEW_REQUIRED:
            if (self.spin_free_roots or self.soc_energies_hartree or
                    any(x is not None for x in (self.delta_soc_hartree,
                                                self.spin_free_ground_hartree,
                                                self.soc_ground_hartree))):
                raise ValueError("Unresolved SOC cannot advertise a usable correction")
        else:
            if self.scalar_hamiltonian not in {"NONE", "SFX2C1E"}:
                raise ValueError("Completed SOC must report scalar Hamiltonian")
            if not self.spin_free_roots or not self.soc_energies_hartree:
                raise ValueError("Completed SOC needs physical spin-free and SOC levels")
            if any(not isfinite(x) for x in self.soc_energies_hartree):
                raise ValueError("Non-finite SOC energy")
            if tuple(sorted(self.soc_energies_hartree)) != self.soc_energies_hartree:
                raise ValueError("SOC levels must be sorted")
            if any(x is None or not isfinite(x) for x in (
                    self.spin_free_ground_hartree, self.soc_ground_hartree,
                    self.delta_soc_hartree)):
                raise ValueError("SOC needs finite comparable energies")
            if abs(self.soc_ground_hartree - self.spin_free_ground_hartree - self.delta_soc_hartree) > 1e-8:
                raise ValueError("Inconsistent SOC energy shift")
            if abs(self.soc_energies_hartree[0] - self.soc_ground_hartree) > 1e-8:
                raise ValueError("SOC ground does not equal lowest coupled level")
            if abs(min(x.energy_hartree for x in self.spin_free_roots) - self.spin_free_ground_hartree) > 1e-8:
                raise ValueError("Spin-free ground not minimum of supplied manifold")


def _load_pinned_siso(path: str):
    """Load an explicitly supplied checkout without global sys.path modifications."""
    checkout = Path(path).expanduser().resolve()
    source = checkout / "fcisiso.py"
    if not source.is_file() or not (checkout / ".git").exists():
        raise FileNotFoundError("Pinned FCI-SISO checkout with fcisiso.py is required")
    resolved = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if resolved != FCI_SISO_PIN:
        raise ValueError(f"Wrong FCI-SISO revision: {resolved}")
    # Ignore untracked bytecode/cache files created by import; enforce tracked-code integrity.
    if subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
                      capture_output=True, text=True, check=True).stdout.strip():
        raise ValueError("Dirty FCI-SISO checkout cannot establish code provenance")
    spec = spec_from_file_location("openea_pinned_fcisiso", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load pinned FCI-SISO module")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "FCISISO", None)) or not callable(getattr(module, "extract_ci_list", None)):
        raise RuntimeError("Pinned FCI-SISO lacks expected API")
    return module, sha256(source.read_bytes()).hexdigest()


def _run_pyscf(request: SOCPointRequest, settings: SOCPointSettings) -> SOCPointResult:
    import copy
    import pyscf
    from pyscf import fci, gto, mcscf, scf

    p = request.point
    siso_module, digest = _load_pinned_siso(settings.fci_siso_checkout)
    mol = gto.M(
        atom=[(p.atoms[0], (0, 0, 0)), (p.atoms[1], (0, 0, p.r_angstrom))],
        basis=dict(p.basis_by_element), charge=p.charge, spin=p.spin_2s,
        unit="Angstrom", symmetry=False, verbose=settings.mr.verbose,
        max_memory=settings.mr.max_memory_mb,
    )
    if getattr(mol, "_ecp", None):
        raise ValueError("ECP with SOC requires separately validated SOC integrals")
    nactive = sum(p.nelecas)
    if (mol.nelectron - nactive) < 0 or (mol.nelectron - nactive) % 2:
        raise ValueError("Total electron count incompatible with selected CAS")
    ncore = (mol.nelectron - nactive) // 2
    if ncore + p.ncas > mol.nao_nr() or max(p.active_orbital_indices) >= mol.nao_nr():
        raise ValueError("Requested CAS exceeds orbital space")
    mf = scf.RHF(mol) if p.spin_2s == 0 else scf.ROHF(mol)
    if settings.mr.scalar_relativistic == "SFX2C1E":
        mf = mf.sfx2c1e()
    if p.spin_2s == 0:
        dm0 = scf.hf.init_guess_by_chkfile(mol, p.source_checkpoint_path,
                                           project=settings.mr.checkpoint_project)
    else:
        dm0 = scf.rohf.init_guess_by_chkfile(mol, p.source_checkpoint_path,
                                             project=settings.mr.checkpoint_project)
    mf.conv_tol = settings.mr.scf_conv_tol
    mf.max_cycle = settings.mr.scf_max_cycle
    mf.kernel(dm0=dm0)
    if not mf.converged:
        return SOCPointResult(p.request_id, SOCPointStatus.SCF_NOT_CONVERGED,
                              "Reoptimized HF reference did not converge")
    mc = mcscf.CASSCF(mf, p.ncas, p.nelecas)
    mo = mc.sort_mo(list(p.active_orbital_indices), mo_coeff=mf.mo_coeff, base=0)
    ground_roots = next(x.nroots for x in request.spin_manifolds if x.spin_2s == p.spin_2s)
    if ground_roots > 1:
        mc.state_average_(weights=[1.0 / ground_roots] * ground_roots)
    target_s2 = p.spin_2s * (p.spin_2s + 2) / 4.0
    mc.fcisolver.spin = p.spin_2s
    mc.fix_spin_(ss=target_s2, shift=0.5)
    mc.conv_tol = settings.mr.casscf_conv_tol
    mc.max_cycle_macro = settings.mr.casscf_max_macro
    mc.kernel(mo)
    if not bool(mc.converged):
        return SOCPointResult(p.request_id, SOCPointStatus.CASSCF_NOT_CONVERGED,
                              "Ground-spin CASSCF orbital optimization did not converge")

    # Common optimized orbitals for all spin manifolds; never separately relax.
    solvers = []
    for manifold in request.spin_manifolds:
        solver = fci.direct_spin1.FCI(mol)
        solver.spin = manifold.spin_2s
        solver.nroots = manifold.nroots
        ss = manifold.spin_2s * (manifold.spin_2s + 2) / 4.0
        solver = fci.addons.fix_spin(solver, shift=0.5, ss=ss)
        solver.spin = manifold.spin_2s
        solver.nroots = manifold.nroots
        solvers.append(solver)
    nroots = sum(m.nroots for m in request.spin_manifolds)
    casci = mcscf.CASCI(mf, p.ncas, p.nelecas)
    mcscf.state_average_mix_(casci, solvers, [1.0 / nroots] * nroots)
    casci.kernel(mc.mo_coeff)
    ci = siso_module.extract_ci_list(casci)
    if len(ci) != nroots:
        raise RuntimeError("Mixed-spin CASCI returned incomplete roots")
    spin_free = tuple(SOCSpinFreeRoot(int(x[2]), i, float(x[4])) for i, x in enumerate(ci))
    if sorted(x.spin_2s for x in spin_free) != sorted(
            m.spin_2s for m in request.spin_manifolds for _ in range(m.nroots)):
        raise RuntimeError("Mixed-spin CASCI returned wrong spin labels")
    spin_free_min = min(x.energy_hartree for x in spin_free)
    # An unexpected spin-free ground manifold is a *scientific* decision conflict.
    if min(x.spin_2s for x in spin_free if abs(x.energy_hartree - spin_free_min) <= 1e-8) != request.ground_spin_2s:
        return SOCPointResult(p.request_id, SOCPointStatus.STATE_MANIFOLD_UNRESOLVED,
                              "Competing spin manifold below assumed ground state")

    ground_ci = min((x for x in ci if x[2] == request.ground_spin_2s), key=lambda x: x[4])
    gamma = np.asarray(fci.direct_spin1.make_rdm1(ground_ci[5], p.ncas,
                                                   (ground_ci[0], ground_ci[1])), dtype=float)
    c = np.asarray(casci.mo_coeff)
    if np.iscomplexobj(c) or gamma.shape != (p.ncas, p.ncas):
        raise RuntimeError("Unexpected complex MOs or active 1RDM dimension")
    dm_mo = np.zeros((c.shape[1], c.shape[1]))
    dm_mo[:ncore, :ncore] = 2 * np.eye(ncore)
    dm_mo[ncore:ncore + p.ncas, ncore:ncore + p.ncas] = gamma
    dm_ao = c @ dm_mo @ c.T
    tr = float(np.trace(dm_ao @ mf.get_ovlp()))
    if abs(tr - mol.nelectron) > 1e-5:
        raise RuntimeError("SOMF density has wrong electron trace")
    mf_soc = copy.copy(mf)
    mf_soc.mo_coeff = casci.mo_coeff
    siso = siso_module.FCISISO(mol, mf_soc, cas=(p.ncas, p.nelecas))
    siso.ci = ci
    coupled = tuple(float(x) for x in np.asarray(
        siso.kernel_we(dmao=dm_ao, amfi=True), dtype=float).reshape(-1))
    expected = sum(m.nroots * (m.spin_2s + 1) for m in request.spin_manifolds)
    if len(coupled) != expected or not np.isfinite(coupled).all():
        raise RuntimeError("SISO returned wrong number of spin-orbit levels")
    coupled = tuple(sorted(coupled))
    return SOCPointResult(
        p.request_id, SOCPointStatus.COMPLETE_REVIEW_REQUIRED,
        "Common-CASCI-spin manifold and AMFI SISO complete; manifold and SOC method require review",
        spin_free_roots=spin_free, soc_energies_hartree=coupled,
        spin_free_ground_hartree=spin_free_min,
        soc_ground_hartree=coupled[0], delta_soc_hartree=coupled[0] - spin_free_min,
        pyscf_version=pyscf.__version__, scalar_hamiltonian=settings.mr.scalar_relativistic,
        fci_siso_source_sha256=digest,
    )


def run_soc_fci_siso_point(
    request: SOCPointRequest,
    authorization: SOCPointAuthorization,
    *,
    settings: SOCPointSettings | None = None,
    backend: Callable[[SOCPointRequest, SOCPointSettings], SOCPointResult] | None = None,
) -> SOCPointResult:
    """Execute explicit SOC diagnostic, never certify a production correction."""
    p = request.point
    if not authorization.authorized:
        return SOCPointResult(p.request_id, SOCPointStatus.BLOCKED_NOT_AUTHORIZED,
                              "SOC execution not explicitly authorized")
    # Require an upper bound that covers each requested spin-projected CI sector.
    max_det = max(comb(p.ncas, (sum(p.nelecas) + m.spin_2s)//2) *
                  comb(p.ncas, (sum(p.nelecas) - m.spin_2s)//2)
                  for m in request.spin_manifolds)
    projected = sum(m.nroots * (m.spin_2s + 1) for m in request.spin_manifolds)
    if max_det > authorization.max_fci_determinants or projected > authorization.max_spin_orbit_states:
        return SOCPointResult(p.request_id, SOCPointStatus.BLOCKED_INVALID_INPUT,
                              "Explicit SOC cost authorization exceeded")
    checkpoint = Path(p.source_checkpoint_path)
    if not checkpoint.is_file():
        return SOCPointResult(p.request_id, SOCPointStatus.BLOCKED_INVALID_INPUT,
                              "Source HF checkpoint does not exist")
    if settings is None:
        settings = SOCPointSettings()
    if not backend and not settings.fci_siso_checkout.strip():
        return SOCPointResult(p.request_id, SOCPointStatus.BACKEND_UNAVAILABLE,
                              "Explicit pinned FCI-SISO checkout path required")
    source_digest = sha256(checkpoint.read_bytes()).hexdigest()
    sig = sha256(json.dumps({"request": asdict(request), "settings": asdict(settings),
                             "source_digest": source_digest,
                             "backend": "FCI_SISO_AMFI_CASCI_v1"}, sort_keys=True).encode()).hexdigest()
    try:
        res = (backend or _run_pyscf)(request, settings)
    except (FileNotFoundError, ImportError) as exc:
        return SOCPointResult(p.request_id, SOCPointStatus.BACKEND_UNAVAILABLE,
                              f"{type(exc).__name__}: {exc}",
                              source_checkpoint_sha256=source_digest, result_signature=sig)
    except Exception as exc:
        return SOCPointResult(p.request_id, SOCPointStatus.ERROR,
                              f"{type(exc).__name__}: {exc}",
                              source_checkpoint_sha256=source_digest, result_signature=sig)
    if not isinstance(res, SOCPointResult) or res.request_id != p.request_id:
        raise ValueError("SOC backend returned mismatched request")
    if res.status is SOCPointStatus.COMPLETE_REVIEW_REQUIRED:
        if res.scalar_hamiltonian != settings.mr.scalar_relativistic:
            raise ValueError("SOC backend returned inconsistent scalar Hamiltonian")
        if len(res.spin_free_roots) != sum(m.nroots for m in request.spin_manifolds):
            raise ValueError("SOC backend omitted spin-free roots")
        spins = sorted(x.spin_2s for x in res.spin_free_roots)
        expected_spins = sorted(m.spin_2s for m in request.spin_manifolds for _ in range(m.nroots))
        if spins != expected_spins:
            raise ValueError("SOC backend returned wrong spin sectors")
        if len(res.soc_energies_hartree) != projected:
            raise ValueError("SOC backend omitted spin projections")
        if res.fci_siso_source_sha256 is None and backend is None:
            raise ValueError("Real SOC backend omitted source integrity digest")
    return replace(res, source_checkpoint_sha256=source_digest, result_signature=sig)


@dataclass(frozen=True)
class SOCEACandidate:
    status: str
    reason: str
    delta_ea_soc_ev: float | None = None
    production_validated: bool = False
    uncertainty_bounded: bool = False

    def __post_init__(self) -> None:
        if self.production_validated or self.uncertainty_bounded:
            raise ValueError("SOC candidate cannot certify production corrections")
        if self.status != "CANDIDATE_REVIEW_REQUIRED" and self.delta_ea_soc_ev is not None:
            raise ValueError("Unresolved SOC cannot advertise an EA shift")


def assess_soc_ea_pair(neutral_request: SOCPointRequest, neutral: SOCPointResult,
                       anion_request: SOCPointRequest, anion: SOCPointResult) -> SOCEACandidate:
    """Diagnostic delta(EA) = delta E(neutral) - delta E(anion) at declared geometries."""
    pn, pa = neutral_request.point, anion_request.point
    if (pn.role, pa.role) != ("neutral", "anion") or pn.atoms != pa.atoms or pn.charge - pa.charge != 1:
        return SOCEACandidate("UNRESOLVED", "Neutral/anion roles, composition, or charges mismatch")
    if (pn.basis_label, dict(pn.basis_by_element)) != (pa.basis_label, dict(pa.basis_by_element)):
        return SOCEACandidate("UNRESOLVED", "SOC neutral/anion bases mismatch")
    if neutral.status is not SOCPointStatus.COMPLETE_REVIEW_REQUIRED or anion.status is not SOCPointStatus.COMPLETE_REVIEW_REQUIRED:
        return SOCEACandidate("UNRESOLVED", "Both SOC point calculations must complete")
    if neutral.request_id != pn.request_id or anion.request_id != pa.request_id:
        return SOCEACandidate("UNRESOLVED", "SOC request identity mismatch")
    if neutral.scalar_hamiltonian != anion.scalar_hamiltonian:
        return SOCEACandidate("UNRESOLVED", "Neutral/anion SOC Hamiltonian modes mismatch")
    if (not neutral.result_signature or not anion.result_signature or
            not neutral.fci_siso_source_sha256 or not anion.fci_siso_source_sha256 or
            neutral.fci_siso_source_sha256 != anion.fci_siso_source_sha256):
        return SOCEACandidate("UNRESOLVED", "SOC point implementation provenance incomplete or mismatched")
    return SOCEACandidate("CANDIDATE_REVIEW_REQUIRED",
                           "CASCI/AMFI SISO EA shift only; spin-manifold, basis and SOC uncertainties unbounded",
                           (neutral.delta_soc_hartree - anion.delta_soc_hartree) * HARTREE_TO_EV)
