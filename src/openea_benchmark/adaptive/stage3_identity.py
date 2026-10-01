"""High-level HF state-identity and geometry-continuity review for Stage 3.

The Stage-2 identity machinery in :mod:`openea_benchmark.state_identity` and
:mod:`openea_benchmark.branch_continuity` was designed around RKS/UKS scout
checkpoints.  Stage 3 deliberately uses RHF/ROHF references.  This module is a
thin evidence adapter between those worlds: it extracts spin-resolved HF
fingerprints/occupied subspaces from RHF/ROHF checkpoints, but delegates the
actual state and continuity decisions to the existing OpenEA threshold policy.

Scientific invariants
---------------------
* Similar energies never establish state identity.
* Identity thresholds and branch thresholds are explicit inputs; this module
  provides no production defaults.
* Multiple Stage-2 initializations are compared at every sampled geometry.
* Geometry continuity is checked only after same-geometry initialization
  identity has been cleared.
* DISTINCT or AMBIGUOUS evidence fails closed as an UNRESOLVED review.
* A cleared review does not assign a ground state, compute an EA, or authorize
  pruning.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from openea_benchmark.branch_continuity import (
    BranchComparison,
    BranchRelation,
    BranchThresholds,
    classify_branch_metrics,
)
from openea_benchmark.checkpoint_fingerprint import CheckpointAuditSettings
from openea_benchmark.root_record import SCFRootRecord, SCFRunStatus
from openea_benchmark.state_identity import (
    IdentityThresholds,
    StateComparison,
    StateFingerprint,
    StateRelation,
    compare_states,
    fingerprint_from_orthonormal_density,
)

from .stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3PointResult,
)
from .stage3_pec import (
    HighLevelPEC,
    IdentityReviewStatus,
    StateIdentityReview,
    assemble_high_level_pec,
)


@dataclass(frozen=True)
class HighLevelCheckpointAudit:
    request_id: str
    checkpoint_path: str
    reference: str
    checkpoint_energy_hartree: float
    result_energy_hartree: float
    energy_delta_hartree: float
    checkpoint_r_angstrom: float
    request_r_angstrom: float
    geometry_delta_angstrom: float
    electron_count: float
    expected_electron_count: int
    spin_population: float
    expected_spin_2s: int
    overlap_min_eigenvalue: float
    overlap_max_eigenvalue: float


@dataclass(frozen=True)
class HighLevelIdentityResolution:
    initialization_review: StateIdentityReview
    geometry_continuity_review: StateIdentityReview
    initialization_comparisons: tuple[StateComparison, ...]
    continuity_comparisons: tuple[BranchComparison, ...]
    checkpoint_audits: tuple[HighLevelCheckpointAudit, ...]
    pec: HighLevelPEC


@dataclass(frozen=True)
class _CheckpointEvidence:
    request: Stage3ExecutionRequest
    result: Stage3PointResult
    root: SCFRootRecord
    fingerprint: StateFingerprint
    mol: Any
    alpha_occ_coeff: np.ndarray
    beta_occ_coeff: np.ndarray
    audit: HighLevelCheckpointAudit


def _result_map(
    requests: Sequence[Stage3ExecutionRequest],
    results: Sequence[Stage3PointResult],
) -> dict[str, Stage3PointResult]:
    request_ids = [item.request_id for item in requests]
    result_ids = [item.request_id for item in results]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Duplicate Stage-3 request IDs")
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Duplicate Stage-3 result IDs")
    missing = sorted(set(request_ids) - set(result_ids))
    extra = sorted(set(result_ids) - set(request_ids))
    if missing or extra:
        raise ValueError(
            f"Stage-3 identity request/result mismatch; missing={missing}, extra={extra}"
        )
    return {item.request_id: item for item in results}


def _pending_review(reason: str) -> StateIdentityReview:
    return StateIdentityReview(
        status=IdentityReviewStatus.UNRESOLVED,
        covered_request_ids=(),
        evidence_ids=(),
        rationale=reason,
    )


def _spin_resolved_hf_density(coeff: np.ndarray, occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return alpha/beta AO densities from RHF or ROHF orbitals.

    PySCF RHF/ROHF uses one spatial-orbital coefficient matrix and occupations
    in {0, 1, 2}.  Singly occupied ROHF orbitals are alpha by convention for
    the non-negative ``mol.spin`` sectors used by OpenEA.
    """
    coeff = np.asarray(coeff)
    occ = np.asarray(occ)
    if coeff.ndim != 2 or occ.ndim != 1 or coeff.shape[1] != occ.shape[0]:
        raise ValueError("RHF/ROHF checkpoint has unexpected MO dimensions")
    if np.any(occ < -1.0e-8) or np.any(occ > 2.0 + 1.0e-8):
        raise ValueError("RHF/ROHF checkpoint has invalid occupations")

    occ_alpha = (occ > 0.5).astype(float)
    occ_beta = (occ > 1.5).astype(float)
    dm_alpha = np.einsum("ui,i,vi->uv", coeff, occ_alpha, coeff.conj())
    dm_beta = np.einsum("ui,i,vi->uv", coeff, occ_beta, coeff.conj())
    return dm_alpha, dm_beta


def _lowdin_spin_density(
    mol: Any,
    dm_alpha: np.ndarray,
    dm_beta: np.ndarray,
    *,
    eigenvalue_floor: float,
) -> tuple[np.ndarray, float, float]:
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"))
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("AO overlap must be square")
    eigvals, eigvecs = np.linalg.eigh(overlap)
    minimum = float(np.min(eigvals))
    maximum = float(np.max(eigvals))
    if minimum <= float(eigenvalue_floor):
        raise ValueError(
            "AO overlap is singular or below configured eigenvalue floor: "
            f"{minimum:.6e}"
        )
    sqrt_s = (eigvecs * np.sqrt(eigvals)) @ eigvecs.conj().T
    dm_orth = np.stack(
        (
            sqrt_s @ dm_alpha @ sqrt_s,
            sqrt_s @ dm_beta @ sqrt_s,
        )
    )
    return dm_orth, minimum, maximum


def _root_for_high_level_result(
    request: Stage3ExecutionRequest,
    result: Stage3PointResult,
) -> SCFRootRecord:
    if result.scf_energy_hartree is None or result.s2 is None:
        raise ValueError(f"{request.request_id}: high-level HF energy/<S^2> missing")
    return SCFRootRecord(
        root_id=request.request_id,
        molecule=request.system,
        atom_a=request.atoms[0],
        atom_b=request.atoms[1],
        charge=request.charge,
        spin_2s=request.spin_2s,
        r_angstrom=request.r_angstrom,
        functional="HF_STAGE3",
        basis=request.basis,
        origin_guess=request.source_root_id,
        scf_path="stage3_high_level_hf",
        reference=request.scf_reference,
        status=SCFRunStatus.CANONICALIZED,
        energy_hartree=float(result.scf_energy_hartree),
        internal_stable=result.internal_stable,
        external_stable=result.external_stable,
        s2=float(result.s2),
        observed_multiplicity=result.multiplicity,
        checkpoint_path=result.high_level_checkpoint_path,
        diagnostic_message="Stage-3 high-level HF identity surrogate root",
    )


def _load_high_level_checkpoint(
    request: Stage3ExecutionRequest,
    result: Stage3PointResult,
    *,
    audit_settings: CheckpointAuditSettings,
) -> _CheckpointEvidence:
    if result.status is not PointExecutionStatus.COMPLETED:
        raise ValueError(f"{request.request_id}: point execution is not COMPLETED")
    if not result.high_level_checkpoint_path:
        raise ValueError(f"{request.request_id}: high-level HF checkpoint missing")
    path = Path(result.high_level_checkpoint_path)
    if not path.is_file():
        raise ValueError(f"{request.request_id}: high-level HF checkpoint not found: {path}")

    from pyscf.scf import chkfile

    mol, data = chkfile.load_scf(str(path))
    if not data or "mo_coeff" not in data or "mo_occ" not in data or "e_tot" not in data:
        raise ValueError(f"{request.request_id}: incomplete high-level HF checkpoint")

    if int(mol.charge) != int(request.charge) or int(mol.spin) != int(request.spin_2s):
        raise ValueError(f"{request.request_id}: checkpoint charge/spin mismatch")
    if int(mol.natm) != 2:
        raise ValueError(f"{request.request_id}: checkpoint is not diatomic")
    symbols = tuple(str(mol.atom_pure_symbol(i)) for i in range(2))
    if symbols != tuple(request.atoms):
        raise ValueError(
            f"{request.request_id}: checkpoint atoms mismatch: {symbols!r} != {request.atoms!r}"
        )

    coords = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=float)
    checkpoint_r = float(np.linalg.norm(coords[1] - coords[0]))
    geometry_delta = checkpoint_r - float(request.r_angstrom)
    if abs(geometry_delta) > float(audit_settings.geometry_tol_angstrom):
        raise ValueError(
            f"{request.request_id}: checkpoint geometry mismatch: {geometry_delta:+.6e} Angstrom"
        )

    checkpoint_energy = float(data["e_tot"])
    if result.scf_energy_hartree is None:
        raise ValueError(f"{request.request_id}: result has no HF energy")
    result_energy = float(result.scf_energy_hartree)
    energy_delta = checkpoint_energy - result_energy
    if not isfinite(checkpoint_energy) or abs(energy_delta) > float(audit_settings.energy_tol_hartree):
        raise ValueError(
            f"{request.request_id}: checkpoint/result HF energy mismatch: {energy_delta:+.6e} Eh"
        )

    coeff = np.asarray(data["mo_coeff"])
    occ = np.asarray(data["mo_occ"])
    dm_alpha, dm_beta = _spin_resolved_hf_density(coeff, occ)
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"))
    nalpha = float(np.einsum("ij,ji->", dm_alpha, overlap).real)
    nbeta = float(np.einsum("ij,ji->", dm_beta, overlap).real)
    electron_count = nalpha + nbeta
    spin_population = nalpha - nbeta
    if abs(electron_count - float(mol.nelectron)) > float(audit_settings.electron_trace_tol):
        raise ValueError(f"{request.request_id}: HF checkpoint electron-count mismatch")
    if abs(spin_population - float(request.spin_2s)) > float(audit_settings.spin_trace_tol):
        raise ValueError(f"{request.request_id}: HF checkpoint spin-population mismatch")

    dm_orth, overlap_min, overlap_max = _lowdin_spin_density(
        mol,
        dm_alpha,
        dm_beta,
        eigenvalue_floor=float(audit_settings.overlap_eigenvalue_floor),
    )
    fingerprint = fingerprint_from_orthonormal_density(dm_orth)

    alpha_occ = coeff[:, occ > 0.5]
    beta_occ = coeff[:, occ > 1.5]
    root = _root_for_high_level_result(request, result)
    audit = HighLevelCheckpointAudit(
        request_id=request.request_id,
        checkpoint_path=str(path),
        reference=request.scf_reference,
        checkpoint_energy_hartree=checkpoint_energy,
        result_energy_hartree=result_energy,
        energy_delta_hartree=energy_delta,
        checkpoint_r_angstrom=checkpoint_r,
        request_r_angstrom=float(request.r_angstrom),
        geometry_delta_angstrom=geometry_delta,
        electron_count=electron_count,
        expected_electron_count=int(mol.nelectron),
        spin_population=spin_population,
        expected_spin_2s=int(request.spin_2s),
        overlap_min_eigenvalue=overlap_min,
        overlap_max_eigenvalue=overlap_max,
    )
    return _CheckpointEvidence(
        request=request,
        result=result,
        root=root,
        fingerprint=fingerprint,
        mol=mol,
        alpha_occ_coeff=alpha_occ,
        beta_occ_coeff=beta_occ,
        audit=audit,
    )


def _same_geometry_groups(
    requests: Sequence[Stage3ExecutionRequest],
) -> tuple[tuple[Stage3ExecutionRequest, ...], ...]:
    grouped: dict[float, list[Stage3ExecutionRequest]] = {}
    for request in requests:
        grouped.setdefault(float(request.r_angstrom), []).append(request)
    return tuple(
        tuple(sorted(items, key=lambda item: item.request_id))
        for _, items in sorted(grouped.items())
    )


def review_high_level_initializations(
    *,
    requests: Sequence[Stage3ExecutionRequest],
    results: Sequence[Stage3PointResult],
    identity_thresholds: IdentityThresholds,
    audit_settings: CheckpointAuditSettings | None = None,
) -> tuple[StateIdentityReview, tuple[StateComparison, ...], tuple[HighLevelCheckpointAudit, ...], Mapping[str, _CheckpointEvidence]]:
    """Review multiple initializations at the same geometry using HF fingerprints."""
    if not requests:
        return _pending_review("No Stage-3 requests supplied"), (), (), {}
    by_result = _result_map(requests, results)
    settings = audit_settings or CheckpointAuditSettings()

    evidence: dict[str, _CheckpointEvidence] = {}
    audits: list[HighLevelCheckpointAudit] = []
    try:
        for request in requests:
            item = _load_high_level_checkpoint(
                request,
                by_result[request.request_id],
                audit_settings=settings,
            )
            evidence[request.request_id] = item
            audits.append(item.audit)
    except Exception as exc:
        return (
            _pending_review(f"High-level checkpoint audit failed: {type(exc).__name__}: {exc}"),
            (),
            tuple(audits),
            evidence,
        )

    comparisons: list[StateComparison] = []
    covered: list[str] = []
    evidence_ids: list[str] = []
    unresolved_reasons: list[str] = []

    for group in _same_geometry_groups(requests):
        covered.extend(item.request_id for item in group)
        if len(group) == 1:
            evidence_ids.append(f"hl:init:single:{group[0].request_id}")
            continue
        for i, left_req in enumerate(group):
            left = evidence[left_req.request_id]
            for right_req in group[i + 1 :]:
                right = evidence[right_req.request_id]
                comparison = compare_states(
                    left.root,
                    left.fingerprint,
                    right.root,
                    right.fingerprint,
                    thresholds=identity_thresholds,
                )
                comparisons.append(comparison)
                evidence_ids.append(
                    f"hl:init:{left_req.request_id}:{right_req.request_id}:{comparison.relation.value}"
                )
                if comparison.relation is not StateRelation.SAME_STATE:
                    unresolved_reasons.append(
                        f"{left_req.request_id} vs {right_req.request_id}: {comparison.relation.value}"
                    )

    if unresolved_reasons:
        review = StateIdentityReview(
            status=IdentityReviewStatus.UNRESOLVED,
            covered_request_ids=tuple(sorted(set(covered))),
            evidence_ids=tuple(evidence_ids),
            rationale="High-level initializations are not proven equivalent: " + "; ".join(unresolved_reasons),
        )
    else:
        review = StateIdentityReview(
            status=IdentityReviewStatus.CLEARED,
            covered_request_ids=tuple(sorted(set(covered))),
            evidence_ids=tuple(evidence_ids),
            rationale="Every same-geometry Stage-3 initialization is either unique or classified SAME_STATE from HF density fingerprints",
        )
    return review, tuple(comparisons), tuple(audits), evidence


def _principal_values(left: np.ndarray, right: np.ndarray, cross_overlap: np.ndarray) -> tuple[float, ...]:
    left = np.asarray(left)
    right = np.asarray(right)
    sab = np.asarray(cross_overlap)
    if left.ndim != 2 or right.ndim != 2 or sab.ndim != 2:
        raise ValueError("Occupied-subspace comparison requires matrices")
    if left.shape[0] != sab.shape[0] or right.shape[0] != sab.shape[1]:
        raise ValueError("Occupied-subspace AO dimensions disagree")
    if left.shape[1] != right.shape[1]:
        raise ValueError("Occupied-subspace dimensions differ")
    if left.shape[1] == 0:
        return ()
    singular = np.linalg.svd(left.conj().T @ sab @ right, compute_uv=False)
    values = []
    for raw in singular:
        value = float(np.real_if_close(raw))
        if value < -1.0e-10 or value > 1.0 + 1.0e-7:
            raise ValueError(f"Invalid occupied-subspace singular value: {value}")
        values.append(min(1.0, max(0.0, value)))
    return tuple(sorted(values, reverse=True))


def _overlap_summary(values: tuple[float, ...]) -> tuple[float, float]:
    if not values:
        return 1.0, 1.0
    array = np.asarray(values, dtype=float)
    return float(array.min()), float(array.mean())


def _compare_high_level_geometry(
    left: _CheckpointEvidence,
    right: _CheckpointEvidence,
    *,
    thresholds: BranchThresholds,
) -> BranchComparison:
    from pyscf import gto

    if left.root.molecule != right.root.molecule or left.root.charge != right.root.charge:
        raise ValueError("High-level continuity comparison context mismatch")
    if left.root.spin_2s != right.root.spin_2s or left.root.basis != right.root.basis:
        raise ValueError("High-level continuity comparison spin/basis mismatch")
    if left.root.reference != right.root.reference:
        raise ValueError("High-level continuity comparison reference mismatch")

    cross_overlap = gto.intor_cross("int1e_ovlp", left.mol, right.mol)
    alpha_sv = _principal_values(left.alpha_occ_coeff, right.alpha_occ_coeff, cross_overlap)
    beta_sv = _principal_values(left.beta_occ_coeff, right.beta_occ_coeff, cross_overlap)
    alpha_min, alpha_mean = _overlap_summary(alpha_sv)
    beta_min, beta_mean = _overlap_summary(beta_sv)
    dr = abs(float(right.root.r_angstrom) - float(left.root.r_angstrom))
    ds2 = abs(float(right.root.s2) - float(left.root.s2))
    delta_energy_mev = (
        float(right.root.energy_hartree) - float(left.root.energy_hartree)
    ) * 27.211386245988 * 1000.0
    relation = classify_branch_metrics(
        delta_r_angstrom=dr,
        delta_s2=ds2,
        alpha_occ_overlap_min=alpha_min,
        beta_occ_overlap_min=beta_min,
        thresholds=thresholds,
    )
    return BranchComparison(
        root_a=left.root.root_id,
        root_b=right.root.root_id,
        r_a_angstrom=float(left.root.r_angstrom),
        r_b_angstrom=float(right.root.r_angstrom),
        delta_r_angstrom=dr,
        delta_energy_mev=delta_energy_mev,
        delta_s2=ds2,
        alpha_singular_values=alpha_sv,
        beta_singular_values=beta_sv,
        alpha_occ_overlap_min=alpha_min,
        alpha_occ_overlap_mean=alpha_mean,
        beta_occ_overlap_min=beta_min,
        beta_occ_overlap_mean=beta_mean,
        relation=relation,
    )


def review_high_level_geometry_continuity(
    *,
    requests: Sequence[Stage3ExecutionRequest],
    evidence: Mapping[str, _CheckpointEvidence],
    initialization_review: StateIdentityReview,
    branch_thresholds: BranchThresholds,
) -> tuple[StateIdentityReview, tuple[BranchComparison, ...]]:
    """Review the canonical same-state representative across adjacent geometries."""
    if initialization_review.status is not IdentityReviewStatus.CLEARED:
        return _pending_review("Geometry continuity deferred until initialization identity is CLEARED"), ()

    groups = _same_geometry_groups(requests)
    canonical = [group[0] for group in groups]
    if any(item.request_id not in evidence for item in canonical):
        return _pending_review("Canonical high-level checkpoint evidence is incomplete"), ()

    if len(canonical) <= 1:
        ids = tuple(item.request_id for item in canonical)
        return (
            StateIdentityReview(
                status=IdentityReviewStatus.CLEARED,
                covered_request_ids=ids,
                evidence_ids=tuple(f"hl:continuity:single:{item}" for item in ids),
                rationale="Only one sampled geometry is present; no inter-geometry continuity edge is required",
            ),
            (),
        )

    comparisons: list[BranchComparison] = []
    evidence_ids: list[str] = []
    unresolved: list[str] = []
    for left_req, right_req in zip(canonical[:-1], canonical[1:]):
        comparison = _compare_high_level_geometry(
            evidence[left_req.request_id],
            evidence[right_req.request_id],
            thresholds=branch_thresholds,
        )
        comparisons.append(comparison)
        evidence_ids.append(
            f"hl:continuity:{left_req.request_id}:{right_req.request_id}:{comparison.relation.value}"
        )
        if comparison.relation is not BranchRelation.CONTINUOUS:
            unresolved.append(
                f"{left_req.request_id} -> {right_req.request_id}: {comparison.relation.value}"
            )

    status = IdentityReviewStatus.CLEARED if not unresolved else IdentityReviewStatus.UNRESOLVED
    rationale = (
        "Every adjacent canonical Stage-3 HF point is classified CONTINUOUS from occupied-subspace overlaps"
        if not unresolved
        else "High-level geometry continuity is not proven: " + "; ".join(unresolved)
    )
    return (
        StateIdentityReview(
            status=status,
            covered_request_ids=tuple(item.request_id for item in canonical),
            evidence_ids=tuple(evidence_ids),
            rationale=rationale,
        ),
        tuple(comparisons),
    )


def resolve_high_level_identity_and_pec(
    *,
    requests: Sequence[Stage3ExecutionRequest],
    results: Sequence[Stage3PointResult],
    identity_thresholds: IdentityThresholds,
    branch_thresholds: BranchThresholds,
    audit_settings: CheckpointAuditSettings | None = None,
    duplicate_energy_tolerance_hartree: float = 1.0e-7,
) -> HighLevelIdentityResolution:
    """Generate independent HF identity reviews and assemble the guarded PEC."""
    init_review, init_comparisons, audits, evidence = review_high_level_initializations(
        requests=requests,
        results=results,
        identity_thresholds=identity_thresholds,
        audit_settings=audit_settings,
    )
    continuity_review, continuity_comparisons = review_high_level_geometry_continuity(
        requests=requests,
        evidence=evidence,
        initialization_review=init_review,
        branch_thresholds=branch_thresholds,
    )
    pec = assemble_high_level_pec(
        requests=requests,
        results=results,
        initialization_identity_review=init_review,
        geometry_continuity_review=continuity_review,
        duplicate_energy_tolerance_hartree=duplicate_energy_tolerance_hartree,
    )
    return HighLevelIdentityResolution(
        initialization_review=init_review,
        geometry_continuity_review=continuity_review,
        initialization_comparisons=init_comparisons,
        continuity_comparisons=continuity_comparisons,
        checkpoint_audits=audits,
        pec=pec,
    )
