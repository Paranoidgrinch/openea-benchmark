"""Targeted component evidence for component-resolved CBS extrapolation."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
import pickle
from typing import Any, Iterable

from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus, Stage3ExecutionRequest, Stage3PointResult,
)


class ComponentEvidenceStatus(str, Enum):
    READY = "READY"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ReferenceGeometry:
    role: str
    basis: str
    r_angstrom: float
    request_id: str
    source_loop_path: str
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class CBSComponentPoint:
    role: str
    basis: str
    cardinal_number: int
    r_angstrom: float
    status: str
    scf_energy_hartree: float | None
    ccsd_correlation_hartree: float | None
    ccsd_total_hartree: float | None
    triples_correction_hartree: float | None
    ccsd_t_total_hartree: float | None
    request_id: str
    reusable: bool
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class CBSComponentEvidence:
    status: ComponentEvidenceStatus
    reference_basis: str
    reference_geometries: tuple[ReferenceGeometry, ...]
    points: tuple[CBSComponentPoint, ...]
    required_bases: tuple[str, ...]
    missing_points: tuple[str, ...]
    next_actions: tuple[str, ...]
    is_production_ea: bool = False
    def to_dict(self):
        return {
            "status": self.status.value,
            "reference_basis": self.reference_basis,
            "reference_geometries": [x.to_dict() for x in self.reference_geometries],
            "points": [x.to_dict() for x in self.points],
            "required_bases": list(self.required_bases),
            "missing_points": list(self.missing_points),
            "next_actions": list(self.next_actions),
            "is_production_ea": self.is_production_ea,
        }


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _minimum_candidate_ids(loop: Any) -> tuple[str, ...]:
    pec = getattr(loop, "final_pec", None)
    scout = getattr(pec, "minimum_scout", None)
    candidates = getattr(scout, "candidates", ()) or ()
    return tuple(
        str(getattr(x, "canonical_request_id"))
        for x in candidates
        if getattr(x, "canonical_request_id", None)
    )


def extract_reference_geometry(loop: Any, *, role: str, source_loop_path: Path) -> ReferenceGeometry:
    ids = _minimum_candidate_ids(loop)
    if not ids:
        raise ValueError(f"{role}: no minimum candidate IDs")
    by_id = {str(getattr(x, "request_id", "")): x for x in (getattr(loop, "results", ()) or ())}
    if any(rid not in by_id for rid in ids):
        raise ValueError(f"{role}: minimum candidate result missing")
    matched = [by_id[rid] for rid in ids]
    if any(_status_value(getattr(x, "status", None)) != "COMPLETED" for x in matched):
        raise ValueError(f"{role}: minimum candidate is not COMPLETED")
    geometries = {round(float(getattr(x, "r_angstrom")), 12) for x in matched}
    if len(geometries) != 1:
        raise ValueError(f"{role}: minimum candidate geometry is ambiguous: {sorted(geometries)}")
    r = float(next(iter(geometries)))
    if not isfinite(r) or r <= 0:
        raise ValueError(f"{role}: invalid reference geometry")
    bases = {str(getattr(x, "basis", "")) for x in matched}
    if len(bases) != 1:
        raise ValueError(f"{role}: minimum candidates mix basis sets")
    return ReferenceGeometry(
        role=role,
        basis=next(iter(bases)),
        r_angstrom=r,
        request_id=str(getattr(matched[0], "request_id")),
        source_loop_path=str(source_loop_path),
    )


def load_reference_geometries(*, run_dir: Path, reference_basis: str = "d-aug-cc-pv5z"):
    stage = Path(run_dir) / "basis_stage_checkpoints" / reference_basis
    out = []
    for role in ("neutral", "anion"):
        path = stage / f"{role}_loop.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            loop = pickle.load(handle)
        geom = extract_reference_geometry(loop, role=role, source_loop_path=path)
        if geom.basis != reference_basis:
            raise ValueError(f"{role}: checkpoint basis mismatch")
        out.append(geom)
    return tuple(out)


def make_cbs_single_point_request(*, role: str, basis: str, cardinal_number: int,
                                  r_angstrom: float, source_checkpoint_path: Path):
    if role not in {"neutral", "anion"}:
        raise ValueError("role must be neutral or anion")
    charge = 0 if role == "neutral" else -1
    spin_2s = 1 if role == "neutral" else 0
    job_id = f"oh_cbs_component__{role}__X{cardinal_number}"
    return Stage3ExecutionRequest(
        request_id=f"{job_id}__fixed_reference_geometry",
        job_id=job_id, system="OH", atoms=("O", "H"),
        charge=charge, spin_2s=spin_2s,
        component_id=f"{job_id}__component",
        r_angstrom=float(r_angstrom), basis=basis,
        methods=("CCSD", "CCSD(T)"),
        requested_reference="ROHF",
        scf_reference="ROHF" if spin_2s else "RHF",
        source_link_status="SINGLE_DFT_INITIALIZATION",
        source_root_id=f"{role}_fixed_reference_geometry",
        source_checkpoint_path=str(source_checkpoint_path),
        source_origin_guess="CBS_COMPONENT_FIXED_REFERENCE_GEOMETRY",
        grid_index=0, initialization_index=0,
        dft_center_r_angstrom=float(r_angstrom),
        dft_center_energy_hartree=0.0,
        requires_independent_state_identity_validation=False,
        authorizes_pruning=False,
    )


def point_from_stage3_result(result: Stage3PointResult, *, role: str, cardinal_number: int):
    vals = (
        result.scf_energy_hartree, result.ccsd_correlation_hartree,
        result.ccsd_total_hartree, result.triples_correction_hartree,
        result.ccsd_t_total_hartree,
    )
    reusable = result.status is PointExecutionStatus.COMPLETED and all(x is not None for x in vals)
    return CBSComponentPoint(
        role=role, basis=result.basis, cardinal_number=cardinal_number,
        r_angstrom=result.r_angstrom, status=result.status.value,
        scf_energy_hartree=result.scf_energy_hartree,
        ccsd_correlation_hartree=result.ccsd_correlation_hartree,
        ccsd_total_hartree=result.ccsd_total_hartree,
        triples_correction_hartree=result.triples_correction_hartree,
        ccsd_t_total_hartree=result.ccsd_t_total_hartree,
        request_id=result.request_id, reusable=reusable,
    )


def summarize_component_evidence(*, reference_basis: str,
                                 reference_geometries: Iterable[ReferenceGeometry],
                                 points: Iterable[CBSComponentPoint],
                                 required_bases: tuple[str, ...]):
    points = tuple(points)
    by_key = {(p.role, p.basis): p for p in points}
    missing = []
    for basis in required_bases:
        for role in ("neutral", "anion"):
            p = by_key.get((role, basis))
            if p is None or not p.reusable:
                missing.append(f"{role}:{basis}")
    if not missing:
        status = ComponentEvidenceStatus.READY
        actions = (
            "BUILD_COMPONENT_RESOLVED_CBS_EXTRAPOLATION",
            "KEEP_DAUG_MINUS_AUG_DIFFUSE_CORRECTION_SEPARATE",
            "PROPAGATE_COMPONENT_AND_GEOMETRY_EVIDENCE",
        )
    elif len(missing) < 2 * len(required_bases):
        status = ComponentEvidenceStatus.PARTIAL
        actions = ("RESUME_ONLY_MISSING_COMPONENT_SINGLE_POINTS", "DO_NOT_EXTRAPOLATE_CBS_YET")
    else:
        status = ComponentEvidenceStatus.BLOCKED
        actions = ("COMPUTE_REQUIRED_COMPONENT_SINGLE_POINTS", "DO_NOT_EXTRAPOLATE_CBS_YET")
    return CBSComponentEvidence(
        status=status, reference_basis=reference_basis,
        reference_geometries=tuple(reference_geometries),
        points=points, required_bases=required_bases,
        missing_points=tuple(missing), next_actions=actions,
    )
