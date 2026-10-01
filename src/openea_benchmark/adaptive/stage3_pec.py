"""Conservative assembly of Stage-3 point results into a high-level local PEC.

This layer deliberately stops short of an EA, a ground-state assignment, or a
continuous PEC fit.  It checks execution completeness, keeps independent DFT
initializations explicit, requires separately supplied electronic-state
identity/continuity evidence, and only then permits a *discrete* local-minimum
scout on the sampled high-level grid.

Scientific invariants
---------------------
* Equal/similar energies are not proof of electronic-state identity.
* Multiple initializations are never averaged into a new physical energy.
  Once independently reviewed as equivalent, one deterministic canonical
  duplicate is retained and the numerical spread is reported.
* Geometry-to-geometry continuity is an independent requirement.
* A strict sampled local minimum is a bracketed candidate only; it is not an
  equilibrium geometry and no interpolation is performed here.
* No output from this module is an EA, a ground-state assignment, or authority
  to prune another electronic state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

from .stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3PointResult,
)


class IdentityReviewStatus(str, Enum):
    PENDING = "PENDING"
    CLEARED = "CLEARED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class StateIdentityReview:
    """Independent state-identity evidence covering explicit execution requests."""

    status: IdentityReviewStatus
    covered_request_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("State-identity review requires rationale")
        if len(self.covered_request_ids) != len(set(self.covered_request_ids)):
            raise ValueError("covered_request_ids must be unique")
        if self.status is IdentityReviewStatus.CLEARED and not self.evidence_ids:
            raise ValueError("Cleared identity review requires evidence")


class HighLevelPointStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    EXECUTION_INCOMPLETE = "EXECUTION_INCOMPLETE"
    ENERGY_MISSING = "ENERGY_MISSING"
    INITIALIZATION_IDENTITY_UNRESOLVED = "INITIALIZATION_IDENTITY_UNRESOLVED"
    INITIALIZATION_ENERGY_DISAGREEMENT = "INITIALIZATION_ENERGY_DISAGREEMENT"


@dataclass(frozen=True)
class HighLevelPECPoint:
    r_angstrom: float
    status: HighLevelPointStatus
    energy_method: str | None
    energy_hartree: float | None
    canonical_request_id: str | None
    request_ids: tuple[str, ...]
    source_root_ids: tuple[str, ...]
    initialization_energy_spread_hartree: float | None
    high_level_checkpoint_paths: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isfinite(float(self.r_angstrom)) or self.r_angstrom <= 0.0:
            raise ValueError("PEC point geometry must be positive and finite")
        if self.status is HighLevelPointStatus.ACCEPTED:
            if self.energy_hartree is None or self.canonical_request_id is None:
                raise ValueError("Accepted point requires canonical energy/request")


class HighLevelPECStatus(str, Enum):
    EXECUTION_INCOMPLETE = "EXECUTION_INCOMPLETE"
    INITIALIZATION_REVIEW_REQUIRED = "INITIALIZATION_REVIEW_REQUIRED"
    GEOMETRY_CONTINUITY_REQUIRED = "GEOMETRY_CONTINUITY_REQUIRED"
    READY_FOR_DISCRETE_MINIMUM_SCOUT = "READY_FOR_DISCRETE_MINIMUM_SCOUT"


class HighLevelMinimumStatus(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
    NO_BRACKETED_MINIMUM = "NO_BRACKETED_MINIMUM"
    BRACKETED_SINGLE_MINIMUM = "BRACKETED_SINGLE_MINIMUM"
    MULTIPLE_MINIMUM_CANDIDATES = "MULTIPLE_MINIMUM_CANDIDATES"


@dataclass(frozen=True)
class HighLevelMinimumCandidate:
    r_angstrom: float
    energy_hartree: float
    point_index: int
    left_r_angstrom: float
    right_r_angstrom: float
    canonical_request_id: str


@dataclass(frozen=True)
class HighLevelMinimumScout:
    status: HighLevelMinimumStatus
    candidates: tuple[HighLevelMinimumCandidate, ...]
    rationale: str


@dataclass(frozen=True)
class HighLevelPEC:
    job_id: str
    system: str
    charge: int
    spin_2s: int
    component_id: str
    basis: str
    points: tuple[HighLevelPECPoint, ...]
    status: HighLevelPECStatus
    initialization_identity_status: IdentityReviewStatus
    geometry_continuity_status: IdentityReviewStatus
    minimum_scout: HighLevelMinimumScout
    is_production_ea: bool = False
    ground_state_assigned: bool = False
    authorizes_pruning: bool = False

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple):
                return [convert(x) for x in value]
            if hasattr(value, "__dataclass_fields__"):
                return {k: convert(v) for k, v in asdict(value).items()}
            return value
        return convert(self)


def _validate_request_result_contract(
    requests: Sequence[Stage3ExecutionRequest],
    results: Sequence[Stage3PointResult],
) -> dict[str, Stage3PointResult]:
    request_ids = [r.request_id for r in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Duplicate Stage-3 execution request IDs")
    result_ids = [r.request_id for r in results]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Duplicate Stage-3 point result IDs")
    missing = sorted(set(request_ids) - set(result_ids))
    extra = sorted(set(result_ids) - set(request_ids))
    if missing or extra:
        raise ValueError(f"Request/result contract mismatch; missing={missing}, extra={extra}")
    by_result = {r.request_id: r for r in results}
    for req in requests:
        result = by_result[req.request_id]
        if result.job_id != req.job_id:
            raise ValueError(f"{req.request_id}: result belongs to another job")
        for name in ("system", "charge", "spin_2s", "component_id", "basis"):
            if getattr(result, name) != getattr(req, name):
                raise ValueError(f"{req.request_id}: result/request {name} mismatch")
        if abs(float(result.r_angstrom) - float(req.r_angstrom)) > 1.0e-10:
            raise ValueError(f"{req.request_id}: result/request geometry mismatch")
    return by_result


def _requested_energy(
    request: Stage3ExecutionRequest,
    result: Stage3PointResult,
) -> tuple[str, float | None]:
    if "CCSD(T)" in request.methods:
        return "CCSD(T)", result.ccsd_t_total_hartree
    return "CCSD", result.ccsd_total_hartree


def _review_covers(review: StateIdentityReview | None, request_ids: Iterable[str]) -> bool:
    if review is None or review.status is not IdentityReviewStatus.CLEARED:
        return False
    return set(request_ids).issubset(set(review.covered_request_ids))


def _pending_review() -> StateIdentityReview:
    return StateIdentityReview(
        status=IdentityReviewStatus.PENDING,
        covered_request_ids=(),
        evidence_ids=(),
        rationale="Independent electronic-state identity review has not been supplied",
    )


def assemble_high_level_pec(
    *,
    requests: Sequence[Stage3ExecutionRequest],
    results: Sequence[Stage3PointResult],
    initialization_identity_review: StateIdentityReview | None = None,
    geometry_continuity_review: StateIdentityReview | None = None,
    duplicate_energy_tolerance_hartree: float = 1.0e-7,
) -> HighLevelPEC:
    """Assemble one job's point results without inferring state identity from energy."""
    if not requests:
        raise ValueError("High-level PEC assembly requires requests")
    tol = float(duplicate_energy_tolerance_hartree)
    if not isfinite(tol) or tol < 0.0:
        raise ValueError("duplicate_energy_tolerance_hartree must be finite and non-negative")

    job_ids = {r.job_id for r in requests}
    if len(job_ids) != 1:
        raise ValueError("High-level PEC assembly accepts exactly one Stage-3 job")
    signatures = {
        (r.system, r.charge, r.spin_2s, r.component_id, r.basis)
        for r in requests
    }
    if len(signatures) != 1:
        raise ValueError("Stage-3 requests disagree on PEC identity")
    system, charge, spin_2s, component_id, basis = next(iter(signatures))
    job_id = next(iter(job_ids))

    by_result = _validate_request_result_contract(requests, results)
    init_review = initialization_identity_review or _pending_review()
    continuity_review = geometry_continuity_review or _pending_review()

    by_geometry: dict[float, list[Stage3ExecutionRequest]] = {}
    for req in requests:
        by_geometry.setdefault(float(req.r_angstrom), []).append(req)

    points: list[HighLevelPECPoint] = []
    for r_angstrom in sorted(by_geometry):
        group = sorted(by_geometry[r_angstrom], key=lambda item: item.request_id)
        group_results = [by_result[req.request_id] for req in group]
        request_ids = tuple(req.request_id for req in group)
        root_ids = tuple(req.source_root_id for req in group)
        checkpoints = tuple(
            result.high_level_checkpoint_path
            for result in group_results
            if result.high_level_checkpoint_path
        )

        if any(result.status is not PointExecutionStatus.COMPLETED for result in group_results):
            points.append(HighLevelPECPoint(
                r_angstrom=r_angstrom,
                status=HighLevelPointStatus.EXECUTION_INCOMPLETE,
                energy_method=None,
                energy_hartree=None,
                canonical_request_id=None,
                request_ids=request_ids,
                source_root_ids=root_ids,
                initialization_energy_spread_hartree=None,
                high_level_checkpoint_paths=checkpoints,
                evidence_ids=(),
            ))
            continue

        energies: list[float] = []
        methods: set[str] = set()
        energy_missing = False
        for req, result in zip(group, group_results):
            method, energy = _requested_energy(req, result)
            methods.add(method)
            if energy is None or not isfinite(float(energy)):
                energy_missing = True
            else:
                energies.append(float(energy))
        if energy_missing or len(energies) != len(group):
            points.append(HighLevelPECPoint(
                r_angstrom=r_angstrom,
                status=HighLevelPointStatus.ENERGY_MISSING,
                energy_method=next(iter(methods)) if len(methods) == 1 else None,
                energy_hartree=None,
                canonical_request_id=None,
                request_ids=request_ids,
                source_root_ids=root_ids,
                initialization_energy_spread_hartree=None,
                high_level_checkpoint_paths=checkpoints,
                evidence_ids=(),
            ))
            continue
        if len(methods) != 1:
            raise ValueError("Mixed high-level energy methods within one geometry")

        spread = max(energies) - min(energies) if len(energies) > 1 else 0.0
        if len(group) > 1 and not _review_covers(init_review, request_ids):
            points.append(HighLevelPECPoint(
                r_angstrom=r_angstrom,
                status=HighLevelPointStatus.INITIALIZATION_IDENTITY_UNRESOLVED,
                energy_method=next(iter(methods)),
                energy_hartree=None,
                canonical_request_id=None,
                request_ids=request_ids,
                source_root_ids=root_ids,
                initialization_energy_spread_hartree=spread,
                high_level_checkpoint_paths=checkpoints,
                evidence_ids=init_review.evidence_ids,
            ))
            continue
        if spread > tol:
            points.append(HighLevelPECPoint(
                r_angstrom=r_angstrom,
                status=HighLevelPointStatus.INITIALIZATION_ENERGY_DISAGREEMENT,
                energy_method=next(iter(methods)),
                energy_hartree=None,
                canonical_request_id=None,
                request_ids=request_ids,
                source_root_ids=root_ids,
                initialization_energy_spread_hartree=spread,
                high_level_checkpoint_paths=checkpoints,
                evidence_ids=init_review.evidence_ids,
            ))
            continue

        # Same-state duplicate initializations are not averaged.  The stable,
        # lexicographically first request is the canonical numerical duplicate;
        # the full spread and all provenance are retained.
        canonical_req = group[0]
        canonical_result = by_result[canonical_req.request_id]
        _, canonical_energy = _requested_energy(canonical_req, canonical_result)
        points.append(HighLevelPECPoint(
            r_angstrom=r_angstrom,
            status=HighLevelPointStatus.ACCEPTED,
            energy_method=next(iter(methods)),
            energy_hartree=float(canonical_energy),
            canonical_request_id=canonical_req.request_id,
            request_ids=request_ids,
            source_root_ids=root_ids,
            initialization_energy_spread_hartree=spread,
            high_level_checkpoint_paths=checkpoints,
            evidence_ids=init_review.evidence_ids if len(group) > 1 else (),
        ))

    if any(point.status in (
        HighLevelPointStatus.EXECUTION_INCOMPLETE,
        HighLevelPointStatus.ENERGY_MISSING,
        HighLevelPointStatus.INITIALIZATION_ENERGY_DISAGREEMENT,
    ) for point in points):
        status = HighLevelPECStatus.EXECUTION_INCOMPLETE
    elif any(point.status is HighLevelPointStatus.INITIALIZATION_IDENTITY_UNRESOLVED for point in points):
        status = HighLevelPECStatus.INITIALIZATION_REVIEW_REQUIRED
    else:
        accepted_ids = tuple(
            point.canonical_request_id
            for point in points
            if point.canonical_request_id is not None
        )
        if not _review_covers(continuity_review, accepted_ids):
            status = HighLevelPECStatus.GEOMETRY_CONTINUITY_REQUIRED
        else:
            status = HighLevelPECStatus.READY_FOR_DISCRETE_MINIMUM_SCOUT

    minimum = scout_high_level_local_minimum(points, enabled=(status is HighLevelPECStatus.READY_FOR_DISCRETE_MINIMUM_SCOUT))
    return HighLevelPEC(
        job_id=job_id,
        system=system,
        charge=charge,
        spin_2s=spin_2s,
        component_id=component_id,
        basis=basis,
        points=tuple(points),
        status=status,
        initialization_identity_status=init_review.status,
        geometry_continuity_status=continuity_review.status,
        minimum_scout=minimum,
    )


def scout_high_level_local_minimum(
    points: Sequence[HighLevelPECPoint],
    *,
    enabled: bool = True,
) -> HighLevelMinimumScout:
    if not enabled:
        return HighLevelMinimumScout(
            HighLevelMinimumStatus.NOT_EVALUATED,
            (),
            "Identity/continuity prerequisites are not cleared",
        )
    accepted = [p for p in points if p.status is HighLevelPointStatus.ACCEPTED]
    if len(accepted) != len(points):
        return HighLevelMinimumScout(
            HighLevelMinimumStatus.NOT_EVALUATED,
            (),
            "Not every sampled geometry has an accepted high-level point",
        )
    if len(points) < 3:
        return HighLevelMinimumScout(
            HighLevelMinimumStatus.INSUFFICIENT_POINTS,
            (),
            "At least three ordered sampled points are required for strict bracketing",
        )

    candidates: list[HighLevelMinimumCandidate] = []
    for index in range(1, len(points) - 1):
        left, center, right = points[index - 1], points[index], points[index + 1]
        assert left.energy_hartree is not None and center.energy_hartree is not None and right.energy_hartree is not None
        if center.energy_hartree < left.energy_hartree and center.energy_hartree < right.energy_hartree:
            assert center.canonical_request_id is not None
            candidates.append(HighLevelMinimumCandidate(
                r_angstrom=center.r_angstrom,
                energy_hartree=center.energy_hartree,
                point_index=index,
                left_r_angstrom=left.r_angstrom,
                right_r_angstrom=right.r_angstrom,
                canonical_request_id=center.canonical_request_id,
            ))

    if not candidates:
        status = HighLevelMinimumStatus.NO_BRACKETED_MINIMUM
        rationale = "No sampled interior point is strictly below both immediate neighbours"
    elif len(candidates) == 1:
        status = HighLevelMinimumStatus.BRACKETED_SINGLE_MINIMUM
        rationale = "Exactly one strict discrete local-minimum candidate is bracketed"
    else:
        status = HighLevelMinimumStatus.MULTIPLE_MINIMUM_CANDIDATES
        rationale = "Multiple strict discrete local-minimum candidates remain"
    return HighLevelMinimumScout(status, tuple(candidates), rationale)
