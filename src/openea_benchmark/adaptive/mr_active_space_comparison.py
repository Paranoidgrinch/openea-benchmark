"""Same-geometry CASSCF/CASCI/SC-NEVPT2 active-space sensitivity study.

A matched energy difference is *observed model sensitivity*, NOT a verified
error bound, a proof of wavefunction/root identity, or an automatic CAS choice.
The comparison never performs electronic-structure calculations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .mr_casscf_nevpt2_runner import (
    MRPointRequest, MRPointResult, MRPointSettings, MRPointStatus,
)


class MRActiveSpaceStatus(str, Enum):
    CANDIDATE_REVIEW_REQUIRED = "CANDIDATE_REVIEW_REQUIRED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class MRActiveSpaceThresholds:
    min_density_similarity: float
    min_root_assignment_margin: float
    max_orbital_metric_error: float
    min_ao_metric_eigenvalue: float

    def __post_init__(self) -> None:
        if (not isfinite(self.min_density_similarity)
                or not 0 < self.min_density_similarity <= 1):
            raise ValueError("Explicit density similarity threshold required")
        if (not isfinite(self.min_root_assignment_margin)
                or not 0 <= self.min_root_assignment_margin < 1):
            raise ValueError("Invalid root assignment margin")
        if (not isfinite(self.max_orbital_metric_error)
                or not 0 < self.max_orbital_metric_error < 0.1):
            raise ValueError("Invalid orbital metric tolerance")
        if (not isfinite(self.min_ao_metric_eigenvalue)
                or not 0 < self.min_ao_metric_eigenvalue < 1):
            raise ValueError("Invalid AO conditioning threshold")


@dataclass(frozen=True)
class MRActiveSpaceRootShift:
    left_root: int
    right_root: int
    density_similarity: float
    casci_shift_hartree: float
    sc_nevpt2_total_shift_hartree: float

    def __post_init__(self) -> None:
        if any(not isfinite(x) for x in (
            self.density_similarity, self.casci_shift_hartree,
            self.sc_nevpt2_total_shift_hartree,
        )):
            raise ValueError("Nonfinite active-space model sensitivity")


@dataclass(frozen=True)
class MRActiveSpaceComparison:
    status: MRActiveSpaceStatus
    reason: str
    left_request_id: str
    right_request_id: str
    density_similarities: tuple[tuple[float, ...], ...] = ()
    candidate_root_shifts: tuple[MRActiveSpaceRootShift, ...] = ()
    root_identity_validated: bool = False
    active_space_converged: bool = False
    method_uncertainty_bounded: bool = False
    mr_production_validated: bool = False

    def __post_init__(self) -> None:
        if any((self.root_identity_validated, self.active_space_converged,
                self.method_uncertainty_bounded, self.mr_production_validated)):
            raise ValueError("A two-CAS diagnostic cannot certify scientific closure")
        if self.status is MRActiveSpaceStatus.UNRESOLVED and self.candidate_root_shifts:
            raise ValueError("Unresolved root matching cannot advertise matched energy shifts")


def _unresolved(a: MRPointRequest, b: MRPointRequest, reason: str,
                scores: np.ndarray | None = None) -> MRActiveSpaceComparison:
    return MRActiveSpaceComparison(
        MRActiveSpaceStatus.UNRESOLVED, reason, a.request_id, b.request_id,
        density_similarities=() if scores is None else tuple(
            tuple(float(x) for x in row) for row in scores
        ),
    )


def pyscf_mr_same_geometry_ao_metric(request: MRPointRequest) -> np.ndarray:
    """AO metric for the *declared* geometry and orbital basis; no SCF job."""
    from pyscf import gto

    mol = gto.M(
        atom=[(request.atoms[0], (0., 0., 0.)),
              (request.atoms[1], (0., 0., request.r_angstrom))],
        basis=dict(request.basis_by_element), charge=request.charge,
        spin=request.spin_2s, symmetry=False, unit="Angstrom", verbose=0,
    )
    return np.asarray(mol.intor_symmetric("int1e_ovlp"))


def _expected_signature(request: MRPointRequest, result: MRPointResult,
                        settings: MRPointSettings) -> str:
    return sha256(json.dumps({
        "request": asdict(request), "settings": asdict(settings),
        "source_digest": result.source_checkpoint_sha256,
        "backend": "PySCF_SA_CASSCF_CASCI_SCNEVPT2_v1",
    }, sort_keys=True).encode()).hexdigest()


def assess_mr_active_space_pair(
    left: MRPointRequest,
    left_result: MRPointResult,
    right: MRPointRequest,
    right_result: MRPointResult,
    *,
    thresholds: MRActiveSpaceThresholds,
    left_settings: MRPointSettings | None = None,
    right_settings: MRPointSettings | None = None,
    overlap_provider: Callable[[MRPointRequest], Any] | None = None,
) -> MRActiveSpaceComparison:
    """Review-only, no calculations or automated CAS/state/EA acceptance.

    Both calculations must refer to the same Hamiltonian, geometry, role,
    source checkpoint and *root manifold*. The source digest and full
    request/settings signatures must match their separately executed jobs.
    Matched total 1RDMs include the inactive doubly occupied orbitals, so
    CAS(n,m) can be compared against a differently sized CAS(n',m').
    """
    if left.request_id == right.request_id:
        return _unresolved(left, right, "Distinct MR request IDs required")
    if any((left.system != right.system, left.atoms != right.atoms,
            left.charge != right.charge, left.spin_2s != right.spin_2s,
            left.role != right.role, left.state_manifold_id != right.state_manifold_id,
            left.r_angstrom != right.r_angstrom,
            left.basis_label != right.basis_label,
            dict(left.basis_by_element) != dict(right.basis_by_element),
            left.source_root_id != right.source_root_id,
            left.state_manifold_review_ids != right.state_manifold_review_ids)):
        return _unresolved(left, right, "Hamiltonian/geometry/spin/manifold provenance differs")
    if (left.active_orbital_indices == right.active_orbital_indices
            and left.nelecas == right.nelecas):
        return _unresolved(left, right, "No different active-space model supplied")
    if left.nroots != right.nroots or left.nroots < 2:
        return _unresolved(left, right, "Comparable multi-root coverage is required")
    if any(x.status is not MRPointStatus.COMPLETE_REVIEW_REQUIRED for x in
           (left_result, right_result)):
        return _unresolved(left, right, "Incomplete MR electronic calculations")
    settings_a, settings_b = (left_settings or MRPointSettings(),
                              right_settings or MRPointSettings())
    if settings_a != settings_b:
        return _unresolved(left, right, "Different electronic-structure settings")
    if not left_result.source_checkpoint_sha256 or (
        left_result.source_checkpoint_sha256 != right_result.source_checkpoint_sha256
    ):
        return _unresolved(left, right, "Different or unknown source HF checkpoints")
    if (left_result.pyscf_version != right_result.pyscf_version):
        return _unresolved(left, right, "Different MR backend versions")
    if any((res.request_id != req.request_id or
            res.state_manifold_review_ids != req.state_manifold_review_ids or
            res.active_space_review_ids != req.active_space_review_ids or
            res.result_signature != _expected_signature(req, res, settings_a) or
            len(res.roots) != req.nroots or
            tuple(r.root_index for r in res.roots) != tuple(range(req.nroots)) or
            not res.cas_orbital_optimization_converged or
            res.active_mo_coeff_ao is None or res.inactive_mo_coeff_ao is None or
            any(root.active_rdm1 is None or
                abs(root.spin_square - req.spin_2s * (req.spin_2s + 2) / 4.0) > 0.1
                for root in res.roots)
           ) for req, res in ((left, left_result), (right, right_result))):
        return _unresolved(left, right, "MR root/active-space data or signatures incomplete")
    # Do not reuse results if their source files changed since execution.
    try:
        for req in (left, right):
            if sha256(Path(req.source_checkpoint_path).read_bytes()).hexdigest() != left_result.source_checkpoint_sha256:
                return _unresolved(left, right, "Source checkpoint changed since MR calculation")
    except OSError:
        return _unresolved(left, right, "Source checkpoint is unavailable")
    try:
        s = np.asarray((overlap_provider or pyscf_mr_same_geometry_ao_metric)(left), dtype=float)
    except Exception as exc:
        return _unresolved(left, right, f"AO metric unavailable: {type(exc).__name__}: {exc}")
    if (s.ndim != 2 or s.shape[0] != s.shape[1] or
            not np.isfinite(s).all() or not np.allclose(s, s.T, atol=1e-9)):
        return _unresolved(left, right, "Invalid AO overlap metric")
    if s.shape[0] == 0:
        return _unresolved(left, right, "Empty AO overlap metric")
    try:
        eigen, u = np.linalg.eigh(s)
    except np.linalg.LinAlgError:
        return _unresolved(left, right, "Could not diagonalize AO overlap metric")
    if eigen[0] < thresholds.min_ao_metric_eigenvalue:
        return _unresolved(left, right, "Near-linearly-dependent AO metric")
    shalf = (u * np.sqrt(eigen)) @ u.T

    def densities(req: MRPointRequest, res: MRPointResult) -> list[np.ndarray]:
        ca = np.asarray(res.active_mo_coeff_ao, dtype=float)
        cc = np.asarray(res.inactive_mo_coeff_ao, dtype=float)
        if (ca.shape != (s.shape[0], req.ncas) or cc.ndim != 2 or
                cc.shape[0] != s.shape[0]):
            raise ValueError("CAS/inactive AO orbital dimensions mismatch")
        if 2 * cc.shape[1] + sum(req.nelecas) != 2 * np.asarray(left_result.inactive_mo_coeff_ao).shape[1] + sum(left.nelecas):
            raise ValueError("Different total electron counts across CAS partitions")
        c = np.concatenate((cc, ca), axis=1)
        if not np.isfinite(c).all() or np.max(np.abs(c.T @ s @ c - np.eye(c.shape[1]))) > thresholds.max_orbital_metric_error:
            raise ValueError("Incorrectly normalized CAS/inactive orbitals")
        result = []
        for root in res.roots:
            gamma = np.asarray(root.active_rdm1, dtype=float)
            if (gamma.shape != (req.ncas, req.ncas) or not np.isfinite(gamma).all()
                    or not np.allclose(gamma, gamma.T, atol=1e-7)
                    or abs(np.trace(gamma) - sum(req.nelecas)) > 1e-4):
                raise ValueError("Invalid active-space 1RDM")
            occ = np.linalg.eigvalsh(gamma)
            if occ[0] < -1e-5 or occ[-1] > 2 + 1e-5:
                raise ValueError("Unphysical natural occupation numbers")
            d = 2 * cc @ cc.T + ca @ gamma @ ca.T
            physical = shalf @ d @ shalf
            result.append((physical + physical.T) / 2)
        return result

    try:
        da, db = densities(left, left_result), densities(right, right_result)
        scores = np.empty((left.nroots, right.nroots), dtype=float)
        for i, a in enumerate(da):
            for j, b in enumerate(db):
                denom = sqrt(float(np.sum(a * a) * np.sum(b * b)))
                if denom < 1e-12:
                    raise ValueError("Zero total 1RDM norm")
                scores[i, j] = float(np.sum(a * b) / denom)
    except (ValueError, TypeError, FloatingPointError) as exc:
        return _unresolved(left, right, f"Invalid MR densities: {exc}")
    if not np.isfinite(scores).all() or np.max(scores) > 1 + 1e-7 or np.min(scores) < -1e-7:
        return _unresolved(left, right, "Unphysical density overlap", scores=scores)
    # Conservative mutual-best matching; no silent greedy/global near-tie.
    # The total 1RDM includes inactive closed shells: they may dominate these
    # similarities.  Scores are merely candidates, never scientific closure.
    pairs: list[tuple[int, int]] = []
    for i in range(left.nroots):
        order = np.argsort(-scores[i])
        j, second = int(order[0]), int(order[1])
        col_order = np.argsort(-scores[:, j])
        if (int(col_order[0]) != i or
                scores[i, j] < thresholds.min_density_similarity or
                scores[i, j] - scores[i, second] <= thresholds.min_root_assignment_margin or
                scores[i, j] - scores[int(col_order[1]), j] <= thresholds.min_root_assignment_margin):
            return _unresolved(left, right, "Ambiguous/nonunique root-density matching", scores=scores)
        pairs.append((i, j))
    if len({j for _, j in pairs}) != left.nroots:
        return _unresolved(left, right, "Incomplete unique root assignment", scores=scores)
    shifts = tuple(MRActiveSpaceRootShift(
        left_root=i, right_root=j, density_similarity=float(scores[i, j]),
        casci_shift_hartree=(right_result.roots[j].casci_hartree
                            - left_result.roots[i].casci_hartree),
        sc_nevpt2_total_shift_hartree=(
            right_result.roots[j].sc_nevpt2_total_hartree
            - left_result.roots[i].sc_nevpt2_total_hartree),
    ) for i, j in pairs)
    return MRActiveSpaceComparison(
        MRActiveSpaceStatus.CANDIDATE_REVIEW_REQUIRED,
        "Distinct CAS models have candidate root matches; energy shifts are not accuracy bounds",
        left.request_id, right.request_id,
        density_similarities=tuple(tuple(float(x) for x in row) for row in scores),
        candidate_root_shifts=shifts,
    )
