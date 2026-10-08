"""Explicitly authorized, generic CASSCF -> multi-root CASCI -> SC-NEVPT2.

This is a *method-development* runner, not a validated OpenEA production
energy/correction. There is no automated active-space or state selection.
Missing root continuity and model errors are not silently set to zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from math import comb, isfinite
from pathlib import Path
from typing import Callable, Mapping, Any

import numpy as np


class MRPointStatus(str, Enum):
    COMPLETE_REVIEW_REQUIRED = "COMPLETE_REVIEW_REQUIRED"
    BLOCKED_NOT_AUTHORIZED = "BLOCKED_NOT_AUTHORIZED"
    BLOCKED_INVALID_INPUT = "BLOCKED_INVALID_INPUT"
    SCF_NOT_CONVERGED = "SCF_NOT_CONVERGED"
    CASSCF_NOT_CONVERGED = "CASSCF_NOT_CONVERGED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class MRPointAuthorization:
    authorized: bool
    rationale: str
    evidence_ids: tuple[str, ...] = ()
    max_fci_determinants: int = 200_000

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("MR authorization needs a rationale")
        if self.authorized and not self.evidence_ids:
            raise ValueError("Authorized MR jobs need provenance evidence")
        if self.max_fci_determinants < 1:
            raise ValueError("FCI determinant budget must be positive")


@dataclass(frozen=True)
class MRPointRequest:
    request_id: str
    system: str
    role: str
    atoms: tuple[str, str]
    charge: int
    spin_2s: int
    state_manifold_id: str
    r_angstrom: float
    basis_label: str
    basis_by_element: Mapping[str, str]
    source_root_id: str
    source_checkpoint_path: str
    # Explicit 0-based indices in the reconverged RHF/ROHF MO order.
    active_orbital_indices: tuple[int, ...]
    active_electrons_alpha: int
    active_electrons_beta: int
    nroots: int
    active_space_review_ids: tuple[str, ...]
    state_manifold_review_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.system.strip():
            raise ValueError("MR request requires stable IDs")
        if self.role not in {"neutral", "anion"}:
            raise ValueError("MR role must be neutral or anion")
        if len(self.atoms) != 2 or any(not x.strip() for x in self.atoms):
            raise ValueError("MR runner requires a diatomic")
        if self.spin_2s < 0 or not isfinite(self.r_angstrom) or self.r_angstrom <= 0:
            raise ValueError("Invalid geometry or spin")
        if not self.state_manifold_id.strip() or not self.source_root_id.strip() or not self.source_checkpoint_path.strip():
            raise ValueError("MR provenance IDs and source checkpoint are required")
        if not self.basis_label.strip() or set(self.basis_by_element) != set(self.atoms):
            raise ValueError("Explicit basis_by_element for both atoms required")
        if not all(str(v).strip() for v in self.basis_by_element.values()):
            raise ValueError("Basis entry must be non-empty")
        active = self.active_orbital_indices
        if not active or any(x < 0 for x in active) or len(set(active)) != len(active):
            raise ValueError("Active orbitals must be explicit unique zero-based indices")
        if self.active_electrons_alpha < 0 or self.active_electrons_beta < 0:
            raise ValueError("Invalid active electron counts")
        if max(self.active_electrons_alpha, self.active_electrons_beta) > len(active):
            raise ValueError("Active electron count exceeds active orbitals")
        if self.active_electrons_alpha - self.active_electrons_beta != self.spin_2s:
            raise ValueError("Active spin imbalance must match molecular spin")
        if not 1 <= self.nroots <= 10:
            raise ValueError("nroots must be explicitly between 1 and 10")
        if not self.active_space_review_ids or not self.state_manifold_review_ids:
            raise ValueError("Active-space and state-manifold provenance are required")

    @property
    def ncas(self) -> int:
        return len(self.active_orbital_indices)

    @property
    def nelecas(self) -> tuple[int, int]:
        return self.active_electrons_alpha, self.active_electrons_beta

    @property
    def fci_determinants(self) -> int:
        return comb(self.ncas, self.active_electrons_alpha) * comb(self.ncas, self.active_electrons_beta)


@dataclass(frozen=True)
class MRPointSettings:
    scf_conv_tol: float = 1e-10
    scf_max_cycle: int = 120
    casscf_conv_tol: float = 1e-8
    casscf_max_macro: int = 70
    max_memory_mb: int = 12000
    checkpoint_project: bool = True
    scalar_relativistic: str = "NONE"
    verbose: int = 0

    def __post_init__(self) -> None:
        if any(not isfinite(x) or x <= 0 for x in (self.scf_conv_tol, self.casscf_conv_tol)):
            raise ValueError("Positive finite convergence tolerances required")
        if min(self.scf_max_cycle, self.casscf_max_macro, self.max_memory_mb) <= 0:
            raise ValueError("Invalid cycle or memory limit")
        if self.scalar_relativistic not in {"NONE", "SFX2C1E"}:
            raise ValueError("Unsupported scalar relativistic Hamiltonian")


@dataclass(frozen=True)
class MRRootEnergy:
    root_index: int
    casci_hartree: float
    sc_nevpt2_correction_hartree: float
    sc_nevpt2_total_hartree: float
    spin_square: float
    root_identity_review_required: bool = True
    # Spin-summed active-space one-particle RDM from *this* CASCI root.
    # A fingerprint, not a many-electron wavefunction overlap or a Dyson orbital.
    active_rdm1: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if any(not isfinite(x) for x in (
            self.casci_hartree, self.sc_nevpt2_correction_hartree,
            self.sc_nevpt2_total_hartree, self.spin_square,
        )):
            raise ValueError("MR root energies and spin expectation must be finite")
        if abs(self.casci_hartree + self.sc_nevpt2_correction_hartree - self.sc_nevpt2_total_hartree) > 1e-8:
            raise ValueError("Inconsistent NEVPT2 total")
        if not self.root_identity_review_required:
            raise ValueError("Automatic MR root identity validation is forbidden")
        if self.active_rdm1 is not None:
            if np.iscomplexobj(self.active_rdm1):
                raise ValueError("Complex 1RDM requires a separately validated MR method")
            g = np.asarray(self.active_rdm1, dtype=float)
            if g.ndim != 2 or g.shape[0] != g.shape[1] or not np.isfinite(g).all():
                raise ValueError("Invalid active-space 1RDM")
            if not np.allclose(g, g.T, atol=1e-7):
                raise ValueError("Active-space 1RDM must be Hermitian/real")


@dataclass(frozen=True)
class MRPointResult:
    request_id: str
    status: MRPointStatus
    reason: str
    method_role: str = "DIAGNOSTIC_METHOD_DEVELOPMENT"
    pyscf_version: str | None = None
    scf_energy_hartree: float | None = None
    cas_orbital_optimization_converged: bool | None = None
    roots: tuple[MRRootEnergy, ...] = ()
    active_space_review_ids: tuple[str, ...] = ()
    state_manifold_review_ids: tuple[str, ...] = ()
    source_checkpoint_sha256: str | None = None
    result_signature: str | None = None
    is_production_ea: bool = False
    mr_production_validated: bool = False
    # Columns of the optimized CASSCF/CASCI MOs corresponding to the active
    # orbitals, in the AO ordering of the declared geometry/basis.
    active_mo_coeff_ao: tuple[tuple[float, ...], ...] | None = None
    # AO coefficients of all doubly occupied *inactive* CASSCF/CASCI orbitals.
    # These are needed to compare the entire 1RDM when two CAS partitions
    # have different ncore; active-space 1RDMs alone are not comparable.
    inactive_mo_coeff_ao: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.is_production_ea or self.mr_production_validated:
            raise ValueError("MR point execution cannot certify production")
        if self.status is MRPointStatus.COMPLETE_REVIEW_REQUIRED and not self.roots:
            raise ValueError("Completed MR point requires roots")
        if self.status is not MRPointStatus.COMPLETE_REVIEW_REQUIRED and self.roots:
            raise ValueError("Incomplete MR result cannot advertise usable roots")
        if self.active_mo_coeff_ao is not None:
            if np.iscomplexobj(self.active_mo_coeff_ao):
                raise ValueError("Complex active orbitals require a separately validated MR method")
            c = np.asarray(self.active_mo_coeff_ao, dtype=float)
            if c.ndim != 2 or not c.size or not np.isfinite(c).all():
                raise ValueError("Invalid AO active-orbital coefficients")
            if self.status is not MRPointStatus.COMPLETE_REVIEW_REQUIRED:
                raise ValueError("Incomplete MR result cannot advertise orbital fingerprints")

        if self.inactive_mo_coeff_ao is not None:
            if np.iscomplexobj(self.inactive_mo_coeff_ao):
                raise ValueError("Complex inactive orbitals require a separately validated MR method")
            ccore = np.asarray(self.inactive_mo_coeff_ao, dtype=float)
            if ccore.ndim != 2 or not np.isfinite(ccore).all():
                raise ValueError("Invalid inactive AO orbital coefficients")
            if self.status is not MRPointStatus.COMPLETE_REVIEW_REQUIRED:
                raise ValueError("Incomplete MR result cannot advertise inactive orbitals")
            if self.active_mo_coeff_ao is not None and ccore.shape[0] != len(self.active_mo_coeff_ao):
                raise ValueError("Inactive/active AO coefficient dimensions must agree")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_pyscf(request: MRPointRequest, settings: MRPointSettings) -> MRPointResult:
    # The backend is optional for pure architecture tests.
    import pyscf
    from pyscf import fci, gto, mcscf, mrpt, scf

    atom = [(request.atoms[0], (0., 0., 0.)),
            (request.atoms[1], (0., 0., request.r_angstrom))]
    mol = gto.M(atom=atom, basis=dict(request.basis_by_element),
                charge=request.charge, spin=request.spin_2s,
                unit="Angstrom", symmetry=False, verbose=settings.verbose,
                max_memory=settings.max_memory_mb)
    if getattr(mol, "_ecp", None):
        raise ValueError("MR pilot only supports all-electron orbital Hamiltonians")
    # Core must be doubly occupied; otherwise this CAS partition is not valid.
    total_electrons = mol.nelectron
    nactive = sum(request.nelecas)
    if total_electrons < nactive or (total_electrons - nactive) % 2:
        raise ValueError("Active electrons incompatible with total molecular electrons")
    ncore = (total_electrons - nactive) // 2
    if request.ncas + ncore > mol.nao_nr():
        raise ValueError("Active space plus doubly occupied core exceeds orbital dimension")
    if max(request.active_orbital_indices) >= mol.nao_nr():
        raise ValueError("Active orbital index beyond HF orbital dimension")

    mf = scf.RHF(mol) if request.spin_2s == 0 else scf.ROHF(mol)
    if settings.scalar_relativistic == "SFX2C1E":
        mf = mf.sfx2c1e()
    if request.spin_2s == 0:
        dm0 = scf.hf.init_guess_by_chkfile(mol, request.source_checkpoint_path,
                                           project=settings.checkpoint_project)
    else:
        dm0 = scf.rohf.init_guess_by_chkfile(mol, request.source_checkpoint_path,
                                             project=settings.checkpoint_project)
    mf.conv_tol = settings.scf_conv_tol
    mf.max_cycle = settings.scf_max_cycle
    mf.kernel(dm0=dm0)
    if not mf.converged:
        return MRPointResult(request.request_id, MRPointStatus.SCF_NOT_CONVERGED,
                             "Reoptimized HF reference did not converge", pyscf_version=pyscf.__version__)
    if max(request.active_orbital_indices) >= mf.mo_coeff.shape[1]:
        raise ValueError("Active orbital selection inconsistent with converged HF")
    if request.fci_determinants < request.nroots:
        raise ValueError("Requested roots exceed active-space determinant dimension")

    mc = mcscf.CASSCF(mf, request.ncas, request.nelecas)
    # PySCF sort_mo uses 1-based orbital indices unless base=0 is explicit.
    # PySCF's 1-D mask[caslst] treats an index *tuple* as multi-axis
    # indexing.  MRPointRequest stores an immutable tuple, so materialize a
    # list explicitly; pass the freshly reconverged MO coefficients too.
    orbitals = mc.sort_mo(list(request.active_orbital_indices),
                          mo_coeff=mf.mo_coeff, base=0)
    target_ss = request.spin_2s * (request.spin_2s + 2) / 4.0
    if request.nroots > 1:
        mc = mc.state_average_(weights=[1 / request.nroots] * request.nroots)
    mc.fcisolver.spin = request.spin_2s
    # Native MCSCF spin constraint; check each root S^2 again after CASCI.
    mc.fix_spin_(ss=target_ss, shift=0.5)
    mc.conv_tol = settings.casscf_conv_tol
    mc.max_cycle_macro = settings.casscf_max_macro
    mc.kernel(orbitals)
    if not bool(mc.converged):
        return MRPointResult(request.request_id, MRPointStatus.CASSCF_NOT_CONVERGED,
                             "State-average CASSCF orbital optimization did not converge",
                             pyscf_version=pyscf.__version__, scf_energy_hartree=float(mf.e_tot),
                             cas_orbital_optimization_converged=False)

    # Per PySCF's documented state-average NEVPT2 recipe: a new multi-root
    # CASCI at the optimized orbitals, then NEVPT2(root=k) on that CASCI.
    casci = mcscf.CASCI(mf, request.ncas, request.nelecas)
    casci.fcisolver.nroots = request.nroots
    casci.fcisolver.spin = request.spin_2s
    casci.fix_spin_(ss=target_ss, shift=0.5)
    casci.kernel(mc.mo_coeff)
    casci_energies = [float(casci.e_tot)] if request.nroots == 1 else [float(x) for x in casci.e_tot]
    ci_vectors = [casci.ci] if request.nroots == 1 else list(casci.ci)
    if len(casci_energies) != request.nroots or len(ci_vectors) != request.nroots:
        raise RuntimeError("Multi-root CASCI did not return all requested states")
    roots = []
    for k, (energy, ci) in enumerate(zip(casci_energies, ci_vectors)):
        ss, _ = fci.spin_op.spin_square(ci, request.ncas, request.nelecas)
        ss = float(ss)
        if abs(ss - target_ss) > 0.1:
            raise RuntimeError(f"CASCI root {k} spin contamination: S2={ss}, target={target_ss}")
        corr = float(mrpt.NEVPT(casci, root=k).kernel())
        gamma_raw = np.asarray(casci.fcisolver.make_rdm1(ci, request.ncas, request.nelecas))
        if np.iscomplexobj(gamma_raw):
            raise RuntimeError("Spin-free MR pilot received complex active-root density")
        gamma = np.asarray(gamma_raw, dtype=float)
        if gamma.shape != (request.ncas, request.ncas):
            raise RuntimeError(f"CASCI root {k} returned unexpected active 1RDM dimensions")
        roots.append(MRRootEnergy(
            k, energy, corr, energy + corr, ss,
            active_rdm1=tuple(tuple(float(x) for x in row) for row in gamma),
        ))
    coeff_raw = np.asarray(casci.mo_coeff[:, ncore:ncore + request.ncas])
    if np.iscomplexobj(coeff_raw):
        raise RuntimeError("Spin-free MR pilot received complex optimized orbitals")
    active_coeff = np.asarray(coeff_raw, dtype=float)
    if active_coeff.shape != (mol.nao_nr(), request.ncas):
        raise RuntimeError("CASCI optimized active orbitals have unexpected AO dimensions")
    core_raw = np.asarray(casci.mo_coeff[:, :ncore])
    if np.iscomplexobj(core_raw):
        raise RuntimeError("Spin-free MR pilot received complex inactive orbitals")
    if core_raw.shape != (mol.nao_nr(), ncore):
        raise RuntimeError("CASCI inactive orbitals have unexpected AO dimensions")
    return MRPointResult(
        request_id=request.request_id,
        status=MRPointStatus.COMPLETE_REVIEW_REQUIRED,
        reason="CASSCF/CASCI/SC-NEVPT2 completed; root and active-space validity require review",
        pyscf_version=pyscf.__version__,
        scf_energy_hartree=float(mf.e_tot),
        cas_orbital_optimization_converged=True,
        roots=tuple(roots),
        active_space_review_ids=request.active_space_review_ids,
        state_manifold_review_ids=request.state_manifold_review_ids,
        active_mo_coeff_ao=tuple(tuple(float(x) for x in row) for row in active_coeff),
        inactive_mo_coeff_ao=tuple(
            tuple(float(x) for x in row)
            for row in np.asarray(core_raw, dtype=float)
        ),
    )


def run_mr_casscf_nevpt2_point(
    request: MRPointRequest,
    authorization: MRPointAuthorization,
    *,
    settings: MRPointSettings | None = None,
    backend: Callable[[MRPointRequest, MRPointSettings], MRPointResult] | None = None,
) -> MRPointResult:
    """Execute one explicitly scoped MR geometry/manifold; never certify MR production."""
    if not authorization.authorized:
        return MRPointResult(request.request_id, MRPointStatus.BLOCKED_NOT_AUTHORIZED,
                             "MR high-cost calculation has not been authorized")
    if request.fci_determinants > authorization.max_fci_determinants:
        return MRPointResult(request.request_id, MRPointStatus.BLOCKED_INVALID_INPUT,
                             "Estimated active-space determinant count exceeds authorized budget")
    if settings is None:
        settings = MRPointSettings()
    checkpoint = Path(request.source_checkpoint_path)
    if not checkpoint.is_file():
        return MRPointResult(request.request_id, MRPointStatus.BLOCKED_INVALID_INPUT,
                             "Source HF checkpoint does not exist")
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    signature = sha256(json.dumps({"request": asdict(request), "settings": asdict(settings),
                                   "source_digest": digest, "backend": "PySCF_SA_CASSCF_CASCI_SCNEVPT2_v1"},
                                  sort_keys=True).encode()).hexdigest()
    try:
        result = (backend or _run_pyscf)(request, settings)
    except Exception as exc:
        return MRPointResult(request.request_id, MRPointStatus.ERROR,
                             f"{type(exc).__name__}: {exc}",
                             source_checkpoint_sha256=digest, result_signature=signature)
    if not isinstance(result, MRPointResult) or result.request_id != request.request_id:
        raise ValueError("Backend returned mismatched MR point")
    # The local runner records provenance but does not persist/reuse a result;
    # no accidental checkpoint cross-reuse before the MR state-selection policy.
    if result.status is MRPointStatus.COMPLETE_REVIEW_REQUIRED:
        if len(result.roots) != request.nroots or tuple(x.root_index for x in result.roots) != tuple(range(request.nroots)):
            raise ValueError("Backend returned incomplete, duplicated, or reordered MR roots")
        target_ss = request.spin_2s * (request.spin_2s + 2) / 4.0
        if any(abs(x.spin_square - target_ss) > 0.1 for x in result.roots):
            raise ValueError("Backend returned MR root with incorrect spin")
        if result.active_mo_coeff_ao is not None:
            c = np.asarray(result.active_mo_coeff_ao)
            if c.ndim != 2 or c.shape[1] != request.ncas:
                raise ValueError("MR active-orbital fingerprint has incorrect active dimension")
        if result.inactive_mo_coeff_ao is not None:
            ccore = np.asarray(result.inactive_mo_coeff_ao, dtype=float)
            if ccore.ndim != 2 or (result.active_mo_coeff_ao is not None
                                      and ccore.shape[0] != len(result.active_mo_coeff_ao)):
                raise ValueError("MR inactive-orbital fingerprint has incorrect AO dimension")
        for root in result.roots:
            if root.active_rdm1 is not None:
                g = np.asarray(root.active_rdm1)
                if g.shape != (request.ncas, request.ncas):
                    raise ValueError("MR root 1RDM has incorrect active dimension")
                if abs(float(np.trace(g)) - sum(request.nelecas)) > 1e-4:
                    raise ValueError("MR root 1RDM has incorrect electron count")
                if np.linalg.eigvalsh(g).min() < -1e-5 or np.linalg.eigvalsh(g).max() > 2 + 1e-5:
                    raise ValueError("MR root 1RDM occupations out of physical bounds")
    return replace(result, source_checkpoint_sha256=digest, result_signature=signature)
