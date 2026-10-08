"""Fail-closed MR active-space and root-continuity *candidates* along a PEC.

The score is the AO-cross-metric overlap of spin-summed active one-particle
CASCI densities, not a many-electron CI overlap, not a Dyson overlap, and not
a proof of state identity. Degenerate root densities are explicitly ambiguous.
No energy or EA is accepted for production by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from typing import Any, Callable

import numpy as np

from .mr_casscf_nevpt2_runner import MRPointRequest, MRPointResult, MRPointStatus


class MRPECContinuityStatus(str, Enum):
    CANDIDATE_REVIEW_REQUIRED = "CANDIDATE_REVIEW_REQUIRED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class MRPECContinuityThresholds:
    # Explicit physical/numerical policy; no generic universal defaults.
    max_delta_r_angstrom: float
    min_active_subspace_singular_value: float
    min_root_density_similarity: float
    min_root_assignment_margin: float
    max_active_orthonormality_error: float

    def __post_init__(self) -> None:
        if not (isfinite(self.max_delta_r_angstrom) and self.max_delta_r_angstrom > 0):
            raise ValueError("A positive geometry step limit is required")
        if not (isfinite(self.min_active_subspace_singular_value)
                and 0 < self.min_active_subspace_singular_value <= 1):
            raise ValueError("Invalid active-subspace threshold")
        if not (isfinite(self.min_root_density_similarity)
                and 0 < self.min_root_density_similarity <= 1):
            raise ValueError("Invalid root-density threshold")
        if not (isfinite(self.min_root_assignment_margin)
                and 0 <= self.min_root_assignment_margin <= 1):
            raise ValueError("Invalid root-assignment margin")
        if not (isfinite(self.max_active_orthonormality_error)
                and 0 < self.max_active_orthonormality_error < 0.1):
            raise ValueError("Invalid AO-metric orthonormality tolerance")


@dataclass(frozen=True)
class MRPECContinuityResult:
    status: MRPECContinuityStatus
    reason: str
    left_request_id: str
    right_request_id: str
    root_candidates: tuple[tuple[int, int], ...] = ()
    root_density_similarities: tuple[tuple[float, ...], ...] = ()
    active_subspace_singular_values: tuple[float, ...] = ()
    scientific_state_identity_cleared: bool = False
    mr_production_validated: bool = False

    def __post_init__(self) -> None:
        if self.scientific_state_identity_cleared or self.mr_production_validated:
            raise ValueError("MR PEC continuity candidates cannot certify state identity or production")
        if self.status is MRPECContinuityStatus.UNRESOLVED and self.root_candidates:
            raise ValueError("An unresolved MR comparison cannot advertise a root mapping")


def _pending(left: MRPointRequest, right: MRPointRequest, reason: str,
             *, scores: np.ndarray | None = None,
             singular: np.ndarray | None = None) -> MRPECContinuityResult:
    return MRPECContinuityResult(
        MRPECContinuityStatus.UNRESOLVED, reason, left.request_id, right.request_id,
        root_density_similarities=() if scores is None else tuple(tuple(float(v) for v in row) for row in scores),
        active_subspace_singular_values=() if singular is None else tuple(float(v) for v in singular),
    )


def pyscf_mr_ao_overlap_matrices(
    left: MRPointRequest, right: MRPointRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute S_AA, S_AB, S_BB for the *declared* geometries and bases.

    Pure integral construction, no HF/CASSCF execution. Do not silently
    reuse a single-basis overlap across distinct PEC geometries.
    """
    from pyscf import gto

    def molecule(req: MRPointRequest) -> Any:
        return gto.M(
            atom=[(req.atoms[0], (0.0, 0.0, 0.0)),
                  (req.atoms[1], (0.0, 0.0, req.r_angstrom))],
            basis=dict(req.basis_by_element), charge=req.charge,
            spin=req.spin_2s, unit="Angstrom", symmetry=False, verbose=0,
        )

    a, b = molecule(left), molecule(right)
    return (np.asarray(a.intor_symmetric("int1e_ovlp")),
            np.asarray(gto.intor_cross("int1e_ovlp", a, b)),
            np.asarray(b.intor_symmetric("int1e_ovlp")))


def assess_mr_pec_continuity(
    left: MRPointRequest,
    left_result: MRPointResult,
    right: MRPointRequest,
    right_result: MRPointResult,
    *,
    thresholds: MRPECContinuityThresholds,
    overlap_provider: Callable[
        [MRPointRequest, MRPointRequest], tuple[Any, Any, Any]
    ] | None = None,
) -> MRPECContinuityResult:
    """Return *only* proposed root pairs and diagnostic scores.

    Root matches are an aid to a later human/independent scientific review;
    they cannot close G1, G2, or enable MR production. Fails closed when
    geometries, provenance, active spaces, or root densities cannot be compared.
    """
    if left.request_id == right.request_id:
        return _pending(left, right, "Distinct MR point IDs are required")
    if any((left.system != right.system, left.atoms != right.atoms,
            left.charge != right.charge, left.spin_2s != right.spin_2s,
            left.role != right.role, left.state_manifold_id != right.state_manifold_id,
            left.nelecas != right.nelecas, left.ncas != right.ncas)):
        return _pending(left, right, "Incompatible molecule/spin/manifold/active electron-space")
    if abs(left.r_angstrom - right.r_angstrom) > thresholds.max_delta_r_angstrom:
        return _pending(left, right, "Geometry step exceeds authorized comparison range")
    if (left_result.status is not MRPointStatus.COMPLETE_REVIEW_REQUIRED
            or right_result.status is not MRPointStatus.COMPLETE_REVIEW_REQUIRED):
        return _pending(left, right, "Incomplete MR electronic-point calculation")
    for request, result in ((left, left_result), (right, right_result)):
        if result.request_id != request.request_id:
            return _pending(left, right, "MR request/result ID mismatch")
        if (result.active_space_review_ids != request.active_space_review_ids
                or result.state_manifold_review_ids != request.state_manifold_review_ids):
            return _pending(left, right, "Active-space or state-manifold provenance mismatch")
        if (not result.result_signature or len(result.result_signature) != 64
                or not result.source_checkpoint_sha256 or len(result.source_checkpoint_sha256) != 64):
            return _pending(left, right, "Source-checkpoint/result signatures missing")
        if result.active_mo_coeff_ao is None or any(r.active_rdm1 is None for r in result.roots):
            return _pending(left, right, "MR AO active orbitals or CASCI root 1RDMs missing")
        if (len(result.roots) != request.nroots
                or tuple(r.root_index for r in result.roots) != tuple(range(request.nroots))):
            return _pending(left, right, "MR roots missing or not in expected indexed order")
    if left.nroots != right.nroots:
        return _pending(left, right, "Different computed root-manifold coverage")

    ca = np.asarray(left_result.active_mo_coeff_ao, dtype=float)
    cb = np.asarray(right_result.active_mo_coeff_ao, dtype=float)
    if (ca.ndim != 2 or cb.ndim != 2 or ca.shape[1] != left.ncas or cb.shape[1] != right.ncas):
        return _pending(left, right, "Invalid active-MO dimensions")
    try:
        aa, ab, bb = (np.asarray(s, dtype=float) for s in
                      (overlap_provider or pyscf_mr_ao_overlap_matrices)(left, right))
    except Exception as exc:
        return _pending(left, right, f"Cross-geometry AO overlap unavailable: {type(exc).__name__}: {exc}")
    if (aa.shape != (ca.shape[0], ca.shape[0])
            or ab.shape != (ca.shape[0], cb.shape[0])
            or bb.shape != (cb.shape[0], cb.shape[0])
            or not all(np.isfinite(mat).all() for mat in (ca, cb, aa, ab, bb))):
        return _pending(left, right, "Invalid AO overlap/active coefficient dimensions or values")
    metric_limit = thresholds.max_active_orthonormality_error
    if (np.max(np.abs(ca.T @ aa @ ca - np.eye(left.ncas))) > metric_limit
            or np.max(np.abs(cb.T @ bb @ cb - np.eye(right.ncas))) > metric_limit):
        return _pending(left, right, "Active CASSCF orbitals not normalized in the AO metric")
    x = ca.T @ ab @ cb
    singular = np.linalg.svd(x, compute_uv=False)
    if np.any(singular > 1 + 10 * metric_limit):
        return _pending(left, right, "Invalid cross-subspace overlap exceeds unity", singular=singular)
    if singular.min() < thresholds.min_active_subspace_singular_value:
        return _pending(left, right, "Active spaces changed/rotated beyond subspace threshold", singular=singular)

    def density_matrix(request: MRPointRequest, result: MRPointResult, i: int) -> np.ndarray:
        gamma = np.asarray(result.roots[i].active_rdm1, dtype=float)
        if gamma.shape != (request.ncas, request.ncas) or not np.isfinite(gamma).all():
            raise ValueError("Malformed active 1RDM")
        if np.max(np.abs(gamma - gamma.T)) > 1e-6:
            raise ValueError("Non-symmetric active 1RDM")
        eigen = np.linalg.eigvalsh(gamma)
        if (eigen.min() < -1e-5 or eigen.max() > 2 + 1e-5
                or abs(np.trace(gamma) - sum(request.nelecas)) > 1e-4):
            raise ValueError("Unphysical active electron occupations")
        return gamma

    try:
        ga = [density_matrix(left, left_result, i) for i in range(left.nroots)]
        gb = [density_matrix(right, right_result, i) for i in range(right.nroots)]
        scores = np.empty((left.nroots, right.nroots), dtype=float)
        for i, a in enumerate(ga):
            for j, b in enumerate(gb):
                norm = sqrt(float(np.trace(a @ a) * np.trace(b @ b)))
                if norm < 1e-12:
                    raise ValueError("Zero-norm active one-particle density")
                scores[i, j] = float(np.trace(a @ x @ b @ x.T)) / norm
        if not np.isfinite(scores).all() or np.any(scores < -1e-6) or np.any(scores > 1 + 1e-5):
            raise ValueError("Invalid cross-geometry density similarities")
        scores = np.clip(scores, 0, 1)
    except (ValueError, FloatingPointError) as exc:
        return _pending(left, right, f"MR root density comparison invalid: {exc}", singular=singular)

    pairs: list[tuple[int, int]] = []
    margin = thresholds.min_root_assignment_margin
    for i in range(left.nroots):
        j = int(np.argmax(scores[i]))
        best = float(scores[i, j])
        # Reciprocal unique best (not a forced global assignment). Root
        # swaps are allowed; competing roots must be resolved explicitly.
        if int(np.argmax(scores[:, j])) != i:
            return _pending(left, right, "Competing roots share the same best match",
                            scores=scores, singular=singular)
        row_other = max((float(scores[i, k]) for k in range(right.nroots) if k != j), default=-1.0)
        col_other = max((float(scores[k, j]) for k in range(left.nroots) if k != i), default=-1.0)
        # Even when the *user* chooses zero separation margin, an exact
        # / numerical tie must never be treated as a unique state mapping.
        if (best < thresholds.min_root_density_similarity
                or best - max(row_other, col_other) <= max(margin, 1e-8)):
            return _pending(left, right, "Root similarity below threshold or assignment ambiguous",
                            scores=scores, singular=singular)
        pairs.append((i, j))
    if len({j for _, j in pairs}) != len(pairs):
        return _pending(left, right, "Root assignment is not bijective",
                        scores=scores, singular=singular)
    return MRPECContinuityResult(
        MRPECContinuityStatus.CANDIDATE_REVIEW_REQUIRED,
        "AO-metric active-space and 1RDM diagnostics propose root correspondence; "
        "CI-level/many-body, spin/symmetry and state-manifold scientific review remains required",
        left.request_id, right.request_id, root_candidates=tuple(pairs),
        root_density_similarities=tuple(tuple(float(v) for v in row) for row in scores),
        active_subspace_singular_values=tuple(float(v) for v in singular),
    )
