"""Audit reusable HF/CCSD/(T) evidence before component-resolved CBS work.

The audit is deliberately non-destructive and calculation-free.  It inspects
trusted local OpenEA stage checkpoints and reports which electronic energy
components are already available at which basis/geometry.

Scientific invariants
---------------------
* No finite-basis energies are averaged.
* d-aug evidence is not silently substituted for the aug cardinal series.
* Missing component evidence remains MISSING; it is never inferred from total
  CCSD(T) energies.
* Only COMPLETED point results count as reusable component evidence.
* The audit does not assign a CBS energy or production EA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
import pickle
from typing import Any, Iterable


class ComponentAvailability(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"


@dataclass(frozen=True)
class PointComponents:
    request_id: str
    basis: str
    r_angstrom: float
    status: str
    scf_energy_hartree: float | None
    ccsd_correlation_hartree: float | None
    ccsd_total_hartree: float | None
    triples_correction_hartree: float | None
    ccsd_t_total_hartree: float | None
    component_availability: ComponentAvailability

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["component_availability"] = self.component_availability.value
        return data


@dataclass(frozen=True)
class LoopComponentEvidence:
    source_path: str
    basis: str
    role: str
    loop_status: str
    point_count: int
    complete_component_point_count: int
    accepted_request_ids: tuple[str, ...]
    minimum_candidate_request_ids: tuple[str, ...]
    points: tuple[PointComponents, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "basis": self.basis,
            "role": self.role,
            "loop_status": self.loop_status,
            "point_count": self.point_count,
            "complete_component_point_count": self.complete_component_point_count,
            "accepted_request_ids": list(self.accepted_request_ids),
            "minimum_candidate_request_ids": list(self.minimum_candidate_request_ids),
            "points": [p.to_dict() for p in self.points],
        }


@dataclass(frozen=True)
class BasisComponentSummary:
    basis: str
    neutral_loop_found: bool
    anion_loop_found: bool
    neutral_complete_component_points: int
    anion_complete_component_points: int
    neutral_minimum_component_available: bool
    anion_minimum_component_available: bool
    ready_for_component_extraction: bool
    evidence_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CBSComponentAudit:
    scanned_run_dirs: tuple[str, ...]
    loops: tuple[LoopComponentEvidence, ...]
    basis_summaries: tuple[BasisComponentSummary, ...]
    required_cardinal_bases: tuple[str, ...]
    missing_required_bases: tuple[str, ...]
    ready_required_bases: tuple[str, ...]
    diffuse_reference_basis: str
    diffuse_reference_available: bool
    next_actions: tuple[str, ...]
    is_production_ea: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_run_dirs": list(self.scanned_run_dirs),
            "loops": [x.to_dict() for x in self.loops],
            "basis_summaries": [x.to_dict() for x in self.basis_summaries],
            "required_cardinal_bases": list(self.required_cardinal_bases),
            "missing_required_bases": list(self.missing_required_bases),
            "ready_required_bases": list(self.ready_required_bases),
            "diffuse_reference_basis": self.diffuse_reference_basis,
            "diffuse_reference_available": self.diffuse_reference_available,
            "next_actions": list(self.next_actions),
            "is_production_ea": self.is_production_ea,
        }


def _status_value(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    return str(getattr(value, "value", value))


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _point_components(result: Any) -> PointComponents:
    status = _status_value(getattr(result, "status", None))
    values = {
        "scf_energy_hartree": _finite_or_none(
            getattr(result, "scf_energy_hartree", None)
        ),
        "ccsd_correlation_hartree": _finite_or_none(
            getattr(result, "ccsd_correlation_hartree", None)
        ),
        "ccsd_total_hartree": _finite_or_none(
            getattr(result, "ccsd_total_hartree", None)
        ),
        "triples_correction_hartree": _finite_or_none(
            getattr(result, "triples_correction_hartree", None)
        ),
        "ccsd_t_total_hartree": _finite_or_none(
            getattr(result, "ccsd_t_total_hartree", None)
        ),
    }
    required = (
        values["scf_energy_hartree"],
        values["ccsd_correlation_hartree"],
        values["triples_correction_hartree"],
        values["ccsd_t_total_hartree"],
    )
    present = sum(x is not None for x in required)
    if status == "COMPLETED" and present == len(required):
        availability = ComponentAvailability.COMPLETE
    elif present:
        availability = ComponentAvailability.PARTIAL
    else:
        availability = ComponentAvailability.MISSING

    return PointComponents(
        request_id=str(getattr(result, "request_id", "")),
        basis=str(getattr(result, "basis", "")),
        r_angstrom=float(getattr(result, "r_angstrom")),
        status=status,
        component_availability=availability,
        **values,
    )


def _accepted_request_ids(loop: Any) -> tuple[str, ...]:
    pec = getattr(loop, "final_pec", None)
    points = getattr(pec, "points", ()) or ()
    result = []
    for point in points:
        request_id = getattr(point, "canonical_request_id", None)
        status = _status_value(getattr(point, "status", None))
        if request_id and status == "ACCEPTED":
            result.append(str(request_id))
    return tuple(result)


def _minimum_candidate_request_ids(loop: Any) -> tuple[str, ...]:
    pec = getattr(loop, "final_pec", None)
    scout = getattr(pec, "minimum_scout", None)
    candidates = getattr(scout, "candidates", ()) or ()
    result = []
    for candidate in candidates:
        request_id = getattr(candidate, "canonical_request_id", None)
        if request_id:
            result.append(str(request_id))
    return tuple(result)


def extract_loop_component_evidence(
    loop: Any,
    *,
    source_path: Path,
    role: str,
) -> LoopComponentEvidence:
    results = tuple(getattr(loop, "results", ()) or ())
    points = tuple(_point_components(item) for item in results)
    bases = {p.basis for p in points if p.basis}
    if len(bases) != 1:
        raise ValueError(
            f"{source_path}: expected one basis in loop, found {sorted(bases)}"
        )
    basis = next(iter(bases))
    return LoopComponentEvidence(
        source_path=str(source_path),
        basis=basis,
        role=role,
        loop_status=_status_value(getattr(loop, "status", None)),
        point_count=len(points),
        complete_component_point_count=sum(
            p.component_availability is ComponentAvailability.COMPLETE
            for p in points
        ),
        accepted_request_ids=_accepted_request_ids(loop),
        minimum_candidate_request_ids=_minimum_candidate_request_ids(loop),
        points=points,
    )


def _minimum_component_available(loop: LoopComponentEvidence) -> bool:
    by_id = {p.request_id: p for p in loop.points}
    ids = loop.minimum_candidate_request_ids or loop.accepted_request_ids
    return bool(ids) and all(
        request_id in by_id
        and by_id[request_id].component_availability
        is ComponentAvailability.COMPLETE
        for request_id in ids
    )


def summarize_basis_components(
    loops: Iterable[LoopComponentEvidence],
) -> tuple[BasisComponentSummary, ...]:
    grouped: dict[str, list[LoopComponentEvidence]] = {}
    for loop in loops:
        grouped.setdefault(loop.basis, []).append(loop)

    output = []
    for basis in sorted(grouped):
        items = grouped[basis]
        neutral = [x for x in items if x.role == "neutral"]
        anion = [x for x in items if x.role == "anion"]
        neutral_ready = any(_minimum_component_available(x) for x in neutral)
        anion_ready = any(_minimum_component_available(x) for x in anion)
        output.append(
            BasisComponentSummary(
                basis=basis,
                neutral_loop_found=bool(neutral),
                anion_loop_found=bool(anion),
                neutral_complete_component_points=sum(
                    x.complete_component_point_count for x in neutral
                ),
                anion_complete_component_points=sum(
                    x.complete_component_point_count for x in anion
                ),
                neutral_minimum_component_available=neutral_ready,
                anion_minimum_component_available=anion_ready,
                ready_for_component_extraction=neutral_ready and anion_ready,
                evidence_paths=tuple(x.source_path for x in items),
            )
        )
    return tuple(output)


def discover_stage_loop_pickles(run_dirs: Iterable[Path]) -> tuple[tuple[Path, str], ...]:
    found = []
    for run_dir in run_dirs:
        root = Path(run_dir)
        for path in root.glob("basis_stage_checkpoints/*/neutral_loop.pkl"):
            found.append((path, "neutral"))
        for path in root.glob("basis_stage_checkpoints/*/anion_loop.pkl"):
            found.append((path, "anion"))
    return tuple(sorted(found, key=lambda x: str(x[0])))


def audit_cbs_components(
    *,
    run_dirs: Iterable[Path],
    required_cardinal_bases: tuple[str, ...] = (
        "aug-cc-pvqz",
        "aug-cc-pv5z",
    ),
    diffuse_reference_basis: str = "d-aug-cc-pv5z",
) -> CBSComponentAudit:
    run_dirs = tuple(Path(x) for x in run_dirs)
    loops = []
    for path, role in discover_stage_loop_pickles(run_dirs):
        # TRUST BOUNDARY: only locally generated OpenEA run checkpoints under
        # the explicitly supplied run directories are loaded.
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        loops.append(
            extract_loop_component_evidence(
                obj,
                source_path=path,
                role=role,
            )
        )

    summaries = summarize_basis_components(loops)
    ready = {
        item.basis
        for item in summaries
        if item.ready_for_component_extraction
    }
    missing = tuple(
        basis for basis in required_cardinal_bases if basis not in ready
    )
    ready_required = tuple(
        basis for basis in required_cardinal_bases if basis in ready
    )
    diffuse_available = diffuse_reference_basis in ready

    actions = []
    if missing:
        actions.append(
            "COMPUTE_OR_RECOVER_COMPONENT_EVIDENCE_FOR:"
            + ",".join(missing)
        )
    if not diffuse_available:
        actions.append(
            "RECOVER_DIFFUSE_REFERENCE_COMPONENT_EVIDENCE:"
            + diffuse_reference_basis
        )
    if not missing:
        actions.append("BUILD_COMPONENT_RESOLVED_CBS_EXTRAPOLATION")
    else:
        actions.append("DO_NOT_EXTRAPOLATE_CBS_YET")
    if diffuse_available:
        actions.append("KEEP_DAUG_MINUS_AUG_AS_SEPARATE_DIFFUSE_CORRECTION")
    actions.append("NO_NEW_CALCULATION_WAS_RUN_BY_THIS_AUDIT")

    return CBSComponentAudit(
        scanned_run_dirs=tuple(str(x) for x in run_dirs),
        loops=tuple(loops),
        basis_summaries=summaries,
        required_cardinal_bases=required_cardinal_bases,
        missing_required_bases=missing,
        ready_required_bases=ready_required,
        diffuse_reference_basis=diffuse_reference_basis,
        diffuse_reference_available=diffuse_available,
        next_actions=tuple(actions),
    )
