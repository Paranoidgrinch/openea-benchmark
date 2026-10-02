"""Evidence-based equilibrium geometry/energy resolution for Stage-3 PECs.

The Stage-3 refinement loop guarantees a narrow *sampled* minimum bracket.
This module adds a separate model-stability check before converting that
discrete evidence into a two-sided equilibrium-energy interval.

No single interpolation is treated as truth.  An ensemble of exact local
three-point quadratic models is built from the retained minimum and multiple
left/right high-level points.  Resolution is allowed only when:
- the mandatory nearest-neighbour model is physically convex,
- enough independent local models are admissible,
- every admitted vertex remains inside the Stage-3 minimum bracket,
- model-to-model geometry and energy spreads satisfy explicit thresholds.

The resulting interval is an evidence/convergence interval, not a statistical
confidence interval and not a complete final EA uncertainty budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from statistics import median

from .decision import EnergyInterval
from .stage3_bridge import (
    Stage3AttachmentBridgeStatus,
    Stage3AttachmentRecord,
)


class EquilibriumStatus(str, Enum):
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class EquilibriumResolverSettings:
    max_side_points: int
    minimum_admissible_models: int
    minimum_curvature_hartree_per_angstrom2: float
    max_model_geometry_spread_angstrom: float
    max_model_energy_spread_hartree: float
    point_energy_tolerance_hartree: float

    def __post_init__(self) -> None:
        if int(self.max_side_points) < 1:
            raise ValueError("max_side_points must be >= 1")
        if int(self.minimum_admissible_models) < 2:
            raise ValueError("minimum_admissible_models must be >= 2")

        for name in (
            "minimum_curvature_hartree_per_angstrom2",
            "max_model_geometry_spread_angstrom",
            "max_model_energy_spread_hartree",
            "point_energy_tolerance_hartree",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        if self.max_model_geometry_spread_angstrom <= 0.0:
            raise ValueError("max_model_geometry_spread_angstrom must be positive")
        if self.max_model_energy_spread_hartree <= 0.0:
            raise ValueError("max_model_energy_spread_hartree must be positive")


@dataclass(frozen=True)
class QuadraticMinimumModel:
    left_r_angstrom: float
    right_r_angstrom: float
    curvature_hartree_per_angstrom2: float
    vertex_r_angstrom: float
    vertex_energy_hartree: float
    is_nearest_neighbor_model: bool


@dataclass(frozen=True)
class EquilibriumEstimate:
    status: EquilibriumStatus
    job_id: str
    geometry_interval_angstrom: tuple[float, float] | None
    geometry_central_angstrom: float | None
    energy_interval_hartree: EnergyInterval | None
    energy_central_hartree: float | None
    models: tuple[QuadraticMinimumModel, ...]
    generated_model_count: int
    rejected_model_count: int
    evidence_quality: str
    evidence: tuple[str, ...]
    rationale: str
    is_production_ea: bool = False
    ground_state_assigned: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("equilibrium estimate requires rationale")
        if self.is_production_ea or self.ground_state_assigned or self.authorizes_pruning:
            raise ValueError(
                "equilibrium resolver cannot assign production EA/ground state or authorize pruning"
            )
        if self.status is EquilibriumStatus.RESOLVED:
            if (
                self.geometry_interval_angstrom is None
                or self.geometry_central_angstrom is None
                or self.energy_interval_hartree is None
                or self.energy_central_hartree is None
            ):
                raise ValueError("resolved equilibrium estimate is incomplete")


def _unresolved(
    job_id: str,
    reason: str,
    evidence: tuple[str, ...],
    *,
    models: tuple[QuadraticMinimumModel, ...] = (),
    generated_model_count: int = 0,
    rejected_model_count: int = 0,
) -> EquilibriumEstimate:
    return EquilibriumEstimate(
        status=EquilibriumStatus.UNRESOLVED,
        job_id=job_id,
        geometry_interval_angstrom=None,
        geometry_central_angstrom=None,
        energy_interval_hartree=None,
        energy_central_hartree=None,
        models=models,
        generated_model_count=generated_model_count,
        rejected_model_count=rejected_model_count,
        evidence_quality="UNKNOWN",
        evidence=evidence,
        rationale=reason,
    )


def _fit_three_point_quadratic(
    *,
    left_r: float,
    left_e: float,
    center_r: float,
    center_e: float,
    right_r: float,
    right_e: float,
    nearest: bool,
) -> QuadraticMinimumModel | None:
    """Fit E = E0 + b*x + a*x^2 with x = R - R_center."""
    xl = left_r - center_r
    xr = right_r - center_r

    if not (xl < 0.0 < xr):
        return None

    yl = left_e - center_e
    yr = right_e - center_e
    determinant = xl * xr * (xr - xl)
    if determinant == 0.0:
        return None

    a = (xl * yr - xr * yl) / determinant
    b = (yl * xr * xr - yr * xl * xl) / determinant

    if not isfinite(a) or not isfinite(b):
        return None

    if a <= 0.0:
        return None

    x_vertex = -b / (2.0 * a)
    r_vertex = center_r + x_vertex
    e_vertex = center_e + b * x_vertex + a * x_vertex * x_vertex

    if not isfinite(r_vertex) or not isfinite(e_vertex):
        return None

    return QuadraticMinimumModel(
        left_r_angstrom=left_r,
        right_r_angstrom=right_r,
        curvature_hartree_per_angstrom2=a,
        vertex_r_angstrom=r_vertex,
        vertex_energy_hartree=e_vertex,
        is_nearest_neighbor_model=nearest,
    )


def resolve_stage3_equilibrium(
    record: Stage3AttachmentRecord,
    *,
    settings: EquilibriumResolverSettings,
) -> EquilibriumEstimate:
    """Resolve a Stage-3 sampled minimum into an evidence-based energy interval."""
    job_id = record.job_id

    if record.status is not Stage3AttachmentBridgeStatus.READY or record.branch is None:
        return _unresolved(
            job_id,
            "Stage-3 attachment bridge is not READY",
            ("STAGE3_ATTACHMENT_NOT_READY",),
        )

    if (
        record.discrete_minimum_r_angstrom is None
        or record.discrete_minimum_energy_hartree is None
        or record.minimum_bracket_angstrom is None
    ):
        return _unresolved(
            job_id,
            "Stage-3 attachment record lacks discrete minimum evidence",
            ("DISCRETE_MINIMUM_EVIDENCE_INCOMPLETE",),
        )

    r_values = tuple(float(x) for x in record.branch.r_points)
    energies = tuple(float(x) for x in record.branch.energies)

    if len(r_values) != len(energies) or len(r_values) < 4:
        return _unresolved(
            job_id,
            "At least four aligned PEC points are required for model-stability evidence",
            ("INSUFFICIENT_PEC_POINTS_FOR_MODEL_ENSEMBLE",),
        )

    if len(set(r_values)) != len(r_values):
        raise ValueError("PEC geometries must be unique")

    pairs = sorted(zip(r_values, energies), key=lambda item: item[0])
    if any(
        not isfinite(r) or r <= 0.0 or not isfinite(e)
        for r, e in pairs
    ):
        raise ValueError("PEC points must be finite and geometries positive")

    center_r = float(record.discrete_minimum_r_angstrom)
    center_e = float(record.discrete_minimum_energy_hartree)
    left_bracket, right_bracket = map(float, record.minimum_bracket_angstrom)

    if not (left_bracket < center_r < right_bracket):
        return _unresolved(
            job_id,
            "Discrete minimum is not strictly inside its Stage-3 bracket",
            ("INVALID_MINIMUM_BRACKET",),
        )

    center_matches = [
        (r, e) for r, e in pairs
        if abs(r - center_r) <= 1.0e-10
    ]
    if len(center_matches) != 1:
        return _unresolved(
            job_id,
            "Discrete minimum geometry is not uniquely represented in PEC points",
            ("DISCRETE_MINIMUM_POINT_NOT_UNIQUE",),
        )

    sampled_center_e = center_matches[0][1]
    if abs(sampled_center_e - center_e) > max(
        1.0e-12, settings.point_energy_tolerance_hartree
    ):
        return _unresolved(
            job_id,
            "Bridge minimum energy disagrees with the PEC point at the same geometry",
            ("DISCRETE_MINIMUM_ENERGY_MISMATCH",),
        )

    left_pool = [(r, e) for r, e in pairs if r < center_r]
    right_pool = [(r, e) for r, e in pairs if r > center_r]
    if not left_pool or not right_pool:
        return _unresolved(
            job_id,
            "PEC does not contain points on both sides of the retained minimum",
            ("MINIMUM_NOT_TWO_SIDED",),
        )

    left_pool = sorted(left_pool, key=lambda item: center_r - item[0])[
        : settings.max_side_points
    ]
    right_pool = sorted(right_pool, key=lambda item: item[0] - center_r)[
        : settings.max_side_points
    ]

    nearest_left = left_pool[0][0]
    nearest_right = right_pool[0][0]

    generated = 0
    rejected = 0
    admitted: list[QuadraticMinimumModel] = []
    nearest_model_admitted = False

    for left_r, left_e in left_pool:
        for right_r, right_e in right_pool:
            generated += 1
            nearest = (
                abs(left_r - nearest_left) <= 1.0e-12
                and abs(right_r - nearest_right) <= 1.0e-12
            )
            model = _fit_three_point_quadratic(
                left_r=left_r,
                left_e=left_e,
                center_r=center_r,
                center_e=center_e,
                right_r=right_r,
                right_e=right_e,
                nearest=nearest,
            )

            if model is None:
                rejected += 1
                continue

            if (
                model.curvature_hartree_per_angstrom2
                < settings.minimum_curvature_hartree_per_angstrom2
            ):
                rejected += 1
                continue

            if not (
                left_bracket - 1.0e-12
                <= model.vertex_r_angstrom
                <= right_bracket + 1.0e-12
            ):
                rejected += 1
                continue

            if (
                model.vertex_energy_hartree
                > center_e + settings.point_energy_tolerance_hartree
            ):
                rejected += 1
                continue

            admitted.append(model)
            nearest_model_admitted = nearest_model_admitted or nearest

    admitted_tuple = tuple(admitted)

    if not nearest_model_admitted:
        return _unresolved(
            job_id,
            "Nearest-neighbour quadratic model is not physically admissible",
            ("NEAREST_NEIGHBOR_MODEL_REJECTED",),
            models=admitted_tuple,
            generated_model_count=generated,
            rejected_model_count=rejected,
        )

    if len(admitted) < settings.minimum_admissible_models:
        return _unresolved(
            job_id,
            "Too few independent local models survive the equilibrium checks",
            ("INSUFFICIENT_ADMISSIBLE_EQUILIBRIUM_MODELS",),
            models=admitted_tuple,
            generated_model_count=generated,
            rejected_model_count=rejected,
        )

    model_r = [m.vertex_r_angstrom for m in admitted]
    model_e = [m.vertex_energy_hartree for m in admitted]
    r_spread = max(model_r) - min(model_r)
    e_spread = max(model_e) - min(model_e)

    if r_spread > settings.max_model_geometry_spread_angstrom:
        return _unresolved(
            job_id,
            "Local equilibrium models disagree beyond the geometry-spread threshold",
            ("EQUILIBRIUM_GEOMETRY_MODEL_SPREAD_TOO_LARGE",),
            models=admitted_tuple,
            generated_model_count=generated,
            rejected_model_count=rejected,
        )

    if e_spread > settings.max_model_energy_spread_hartree:
        return _unresolved(
            job_id,
            "Local equilibrium models disagree beyond the energy-spread threshold",
            ("EQUILIBRIUM_ENERGY_MODEL_SPREAD_TOO_LARGE",),
            models=admitted_tuple,
            generated_model_count=generated,
            rejected_model_count=rejected,
        )

    tol = settings.point_energy_tolerance_hartree
    energy_lower = min(model_e) - tol
    # The sampled minimum is a conservative upper bound on the continuous
    # minimum, widened only by the explicit point-energy tolerance.
    energy_upper = center_e + tol

    if energy_lower > energy_upper:
        return _unresolved(
            job_id,
            "Constructed equilibrium-energy interval is inconsistent",
            ("EQUILIBRIUM_ENERGY_INTERVAL_INVALID",),
            models=admitted_tuple,
            generated_model_count=generated,
            rejected_model_count=rejected,
        )

    central_e = median(model_e)
    central_r = median(model_r)

    return EquilibriumEstimate(
        status=EquilibriumStatus.RESOLVED,
        job_id=job_id,
        geometry_interval_angstrom=(left_bracket, right_bracket),
        geometry_central_angstrom=central_r,
        energy_interval_hartree=EnergyInterval(
            lower_hartree=energy_lower,
            upper_hartree=energy_upper,
        ),
        energy_central_hartree=central_e,
        models=admitted_tuple,
        generated_model_count=generated,
        rejected_model_count=rejected,
        evidence_quality="CONVERGENCE_ESTIMATED",
        evidence=(
            "STAGE3_BRACKET_TARGET_MET",
            "NEAREST_NEIGHBOR_MODEL_ADMISSIBLE",
            "LOCAL_QUADRATIC_MODEL_ENSEMBLE_STABLE",
            "ENERGY_UPPER_BOUND_FROM_SAMPLED_MINIMUM",
            "EQUILIBRIUM_INTERVAL_IS_MODEL_EVIDENCE_NOT_STATISTICAL_CI",
        ),
        rationale=(
            "Multiple independent local quadratic models place the equilibrium "
            "inside the Stage-3 bracket and agree within explicit geometry/energy thresholds"
        ),
    )
