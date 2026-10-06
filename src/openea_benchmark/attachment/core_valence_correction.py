"""Adaptive convergence policy for the additive core-valence correction.

At a fixed reference geometry and fixed core-valence basis X:
    Delta_CV(X) = EA_all-electron(X) - EA_frozen-core(X)

The correction is evaluated with aug-cc-pwCVXZ.  No extrapolation formula is
assumed in v1.  Instead the highest-cardinal correction is used as central
value and the latest cardinal change is a convergence uncertainty.

TZ/QZ are evaluated first.  If their change exceeds the target, 5Z is
requested.  With three points, contraction is also required.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class CoreValenceStatus(str, Enum):
    CLEARED = "CLEARED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class CVCardinalPoint:
    cardinal: int
    basis: str
    ea_all_electron_ev: float
    ea_frozen_core_ev: float
    delta_cv_ev: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoreValenceAssessment:
    status: CoreValenceStatus
    action: str
    central_correction_ev: float | None
    convergence_bound_ev: float | None
    contraction_ratio: float | None
    highest_cardinal: int | None
    points: tuple[CVCardinalPoint, ...]
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["points"] = [p.to_dict() for p in self.points]
        return d


def assess_core_valence(
    points: tuple[CVCardinalPoint, ...],
    *,
    target_change_ev: float = 0.002,
    max_contraction_ratio: float = 0.8,
    max_cardinal: int = 5,
) -> CoreValenceAssessment:
    ordered = tuple(sorted(points, key=lambda x: x.cardinal))
    if not ordered:
        return CoreValenceAssessment(
            status=CoreValenceStatus.NEED_MORE_EVIDENCE,
            action="COMPUTE_TZ",
            central_correction_ev=None,
            convergence_bound_ev=None,
            contraction_ratio=None,
            highest_cardinal=None,
            points=(),
            evidence=("NO_CORE_VALENCE_POINTS",),
        )

    cardinals = [p.cardinal for p in ordered]
    if len(set(cardinals)) != len(cardinals):
        return CoreValenceAssessment(
            status=CoreValenceStatus.BLOCKED,
            action="REPAIR_DUPLICATE_CARDINAL_EVIDENCE",
            central_correction_ev=None,
            convergence_bound_ev=None,
            contraction_ratio=None,
            highest_cardinal=max(cardinals),
            points=ordered,
            evidence=("DUPLICATE_CARDINAL_POINTS",),
        )

    latest = ordered[-1]
    if len(ordered) == 1:
        nxt = latest.cardinal + 1
        return CoreValenceAssessment(
            status=CoreValenceStatus.NEED_MORE_EVIDENCE,
            action=f"COMPUTE_X{nxt}",
            central_correction_ev=latest.delta_cv_ev,
            convergence_bound_ev=None,
            contraction_ratio=None,
            highest_cardinal=latest.cardinal,
            points=ordered,
            evidence=("ONE_CARDINAL_POINT_ONLY",),
        )

    change = abs(ordered[-1].delta_cv_ev - ordered[-2].delta_cv_ev)
    contraction = None

    if len(ordered) >= 3:
        prev = abs(ordered[-2].delta_cv_ev - ordered[-3].delta_cv_ev)
        if prev > 0:
            contraction = change / prev
        elif change == 0:
            contraction = 0.0

    enough = change <= target_change_ev
    if len(ordered) >= 3:
        enough = (
            enough
            and contraction is not None
            and contraction <= max_contraction_ratio
        )

    if enough:
        evidence = [
            "ALL_ELECTRON_MINUS_FROZEN_CORE_SAME_BASIS",
            "LATEST_CARDINAL_CHANGE_WITHIN_TARGET",
            "NO_CV_EXTRAPOLATION_ASSUMED",
        ]
        if contraction is not None:
            evidence.append("CORE_VALENCE_INCREMENT_CONTRACTING")
        return CoreValenceAssessment(
            status=CoreValenceStatus.CLEARED,
            action="NONE",
            central_correction_ev=latest.delta_cv_ev,
            convergence_bound_ev=change,
            contraction_ratio=contraction,
            highest_cardinal=latest.cardinal,
            points=ordered,
            evidence=tuple(evidence),
        )

    if latest.cardinal < max_cardinal:
        return CoreValenceAssessment(
            status=CoreValenceStatus.NEED_MORE_EVIDENCE,
            action=f"COMPUTE_X{latest.cardinal+1}",
            central_correction_ev=latest.delta_cv_ev,
            convergence_bound_ev=change,
            contraction_ratio=contraction,
            highest_cardinal=latest.cardinal,
            points=ordered,
            evidence=(
                "CORE_VALENCE_NOT_YET_CONVERGED",
                "REQUEST_HIGHER_CARDINAL",
            ),
        )

    return CoreValenceAssessment(
        status=CoreValenceStatus.NEED_MORE_EVIDENCE,
        action="MAX_CARDINAL_REACHED_UNRESOLVED",
        central_correction_ev=latest.delta_cv_ev,
        convergence_bound_ev=change,
        contraction_ratio=contraction,
        highest_cardinal=latest.cardinal,
        points=ordered,
        evidence=(
            "CORE_VALENCE_NOT_CONVERGED_AT_MAX_CARDINAL",
            "FAIL_CLOSED",
        ),
    )


def cv_point_from_results(
    *,
    cardinal: int,
    basis: str,
    ae_neutral_h: float,
    ae_anion_h: float,
    fc_neutral_h: float,
    fc_anion_h: float,
    hartree_to_ev: float,
) -> CVCardinalPoint:
    ea_ae = (float(ae_neutral_h)-float(ae_anion_h))*hartree_to_ev
    ea_fc = (float(fc_neutral_h)-float(fc_anion_h))*hartree_to_ev
    return CVCardinalPoint(
        cardinal=cardinal,
        basis=basis,
        ea_all_electron_ev=ea_ae,
        ea_frozen_core_ev=ea_fc,
        delta_cv_ev=ea_ae-ea_fc,
    )
