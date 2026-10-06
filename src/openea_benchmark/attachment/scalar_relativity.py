"""Scalar-relativistic correction assessment.

Definition at cardinal X:
    Delta_SR(X) = EA_SFX2C1E(X) - EA_NR(X)

The NR and SFX2C1E calculations must use the same geometry, orbital basis,
electron correlation space, and CCSD(T) settings.  The central correction is
the highest-cardinal directly calculated value; no SR extrapolation is
assumed in v1.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class ScalarRelativityPoint:
    cardinal: int
    basis: str
    ea_nr_ev: float
    ea_sfx2c1e_ev: float
    delta_sr_ev: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ScalarRelativityAssessment:
    status: str
    action: str
    central_correction_ev: float
    convergence_bound_ev: float | None
    highest_cardinal: int
    points: tuple[ScalarRelativityPoint, ...]
    evidence: tuple[str, ...]
    is_production_ea: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["points"] = [p.to_dict() for p in self.points]
        return data

def assess_scalar_relativity(
    points: list[ScalarRelativityPoint],
    *,
    target_change_ev: float = 0.0005,
    max_cardinal: int = 5,
) -> ScalarRelativityAssessment:
    if not points:
        raise ValueError("At least one scalar-relativity point is required")
    pts = tuple(sorted(points, key=lambda p: p.cardinal))
    latest = pts[-1]
    if len(pts) == 1:
        return ScalarRelativityAssessment(
            status="NEED_MORE_EVIDENCE",
            action=f"COMPUTE_X{latest.cardinal + 1}",
            central_correction_ev=latest.delta_sr_ev,
            convergence_bound_ev=None,
            highest_cardinal=latest.cardinal,
            points=pts,
            evidence=(
                "SAME_BASIS_NR_MINUS_SFX2C_PAIR",
                "ONE_CARDINAL_POINT_ONLY",
                "SFX2C1E_ONE_BODY_ONLY",
                "SOC_NOT_INCLUDED",
            ),
        )

    change = abs(pts[-1].delta_sr_ev - pts[-2].delta_sr_ev)
    if change <= target_change_ev:
        return ScalarRelativityAssessment(
            status="CLEARED",
            action="NONE",
            central_correction_ev=latest.delta_sr_ev,
            convergence_bound_ev=change,
            highest_cardinal=latest.cardinal,
            points=pts,
            evidence=(
                "SAME_BASIS_NR_SFX2C_PAIR",
                "LATEST_CARDINAL_CHANGE_WITHIN_TARGET",
                "NO_SR_EXTRAPOLATION_ASSUMED",
                "SFX2C1E_ONE_BODY_ONLY",
                "SOC_NOT_INCLUDED",
            ),
        )

    if latest.cardinal < max_cardinal:
        return ScalarRelativityAssessment(
            status="NEED_MORE_EVIDENCE",
            action=f"COMPUTE_X{latest.cardinal + 1}",
            central_correction_ev=latest.delta_sr_ev,
            convergence_bound_ev=change,
            highest_cardinal=latest.cardinal,
            points=pts,
            evidence=(
                "SAME_BASIS_NR_SFX2C_PAIR",
                "LATEST_CARDINAL_CHANGE_EXCEEDS_TARGET",
                "SFX2C1E_ONE_BODY_ONLY",
                "SOC_NOT_INCLUDED",
            ),
        )

    return ScalarRelativityAssessment(
        status="UNRESOLVED",
        action="REVIEW_SCALAR_RELATIVITY_MODEL",
        central_correction_ev=latest.delta_sr_ev,
        convergence_bound_ev=change,
        highest_cardinal=latest.cardinal,
        points=pts,
        evidence=(
            "SAME_BASIS_NR_SFX2C_PAIR",
            "MAX_CARDINAL_REACHED_WITHOUT_CONVERGENCE",
            "SFX2C1E_ONE_BODY_ONLY",
                "SOC_NOT_INCLUDED",
        ),
    )
