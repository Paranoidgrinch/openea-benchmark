"""Adaptive convergence policy for the additive core-valence correction.

At a fixed molecular-state geometry and a fixed, explicitly selected
core-valence orbital basis for cardinal X::

    Delta_CV(X) = EA_all-electron(X) - EA_frozen-core(X)

The neutral and anion may have different equilibrium geometries for an
adiabatic EA, but each all-electron/frozen-core pair must be matched in basis,
geometry, state sector, Hamiltonian, and correlation treatment.  Basis-family
selection is deliberately outside this assessment module.

No extrapolation formula is assumed in v1.  The highest-cardinal correction is
used as the central value and the latest cardinal change as a convergence
bound.  Two cardinal points may clear if the latest change is within target;
with three or more points, contraction of the increment is additionally
required.

At the largest available cardinal a series that is still above target but
converges geometrically (two increments of the same sign, contraction ratio
r <= max_contraction_ratio) clears with the remaining geometric tail as its
bound, max(change, change * r / (1 - r)), provided that bound is at most
max_tail_bound_ev.  verify_v1 PbO/ScO/VO: 5Z changes of 3-7 meV with r ~ 0.4
(tail 2-5 meV) failed the fixed 2 meV target only because no 6Z basis exists.
A series that does not contract clears with the spread of all cardinals
(max - min) as its bound, plus the last step when it drifts in one direction,
if that is at most max_tail_bound_ev (NaNa oscillating -12.6/-8.3/-13.7 meV:
5.4 meV; ScO SR drifting 29.0/28.2/25.0 meV: 7.3 meV).  The same rules apply to the scalar-relativity series.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


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


def max_cardinal_bound(values: list[float], *, max_contraction_ratio: float,
                       max_bound_ev: float) -> tuple[float, str] | None:
    """Bound of a series (ordered by cardinal) that stopped above target at the largest
    cardinal: the geometric tail of a monotonic contracting series, else the spread of
    all cardinals; None when neither is at most ``max_bound_ev``."""
    if len(values) < 3:
        return None
    step, prev_step = values[-1] - values[-2], values[-2] - values[-3]
    change = abs(step)
    if prev_step != 0.0 and step * prev_step > 0.0:
        ratio = change / abs(prev_step)
        if ratio <= max_contraction_ratio:
            bound = max(change, change * ratio / (1.0 - ratio))
            if bound <= max_bound_ev:
                return bound, "GEOMETRIC_TAIL_BOUND_AT_MAX_CARDINAL"
            return None
    # Oscillating: the limit lies within the spread.  Drifting (same sign, not
    # contracting): one more step of the last size is added.
    bound = max(values) - min(values) + (change if step * prev_step > 0.0 else 0.0)
    if bound <= max_bound_ev:
        return bound, "CARDINAL_SPREAD_BOUND_AT_MAX_CARDINAL"
    return None


def assess_core_valence(
    points: tuple[CVCardinalPoint, ...],
    *,
    target_change_ev: float = 0.002,
    max_contraction_ratio: float = 0.8,
    max_cardinal: int = 5,
    max_tail_bound_ev: float = 0.01,
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

    tail = max_cardinal_bound([p.delta_cv_ev for p in ordered], max_contraction_ratio=max_contraction_ratio,
                              max_bound_ev=max_tail_bound_ev)
    if tail is not None:
        return CoreValenceAssessment(
            status=CoreValenceStatus.CLEARED,
            action="NONE",
            central_correction_ev=latest.delta_cv_ev,
            convergence_bound_ev=tail[0],
            contraction_ratio=contraction,
            highest_cardinal=latest.cardinal,
            points=ordered,
            evidence=(
                "ALL_ELECTRON_MINUS_FROZEN_CORE_SAME_BASIS",
                tail[1],
                "NO_CV_EXTRAPOLATION_ASSUMED",
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
