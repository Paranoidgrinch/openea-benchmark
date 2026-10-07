"""Post-CCSD(T) correction assessment for OpenEA.

The valence post-CCSD(T) correction is decomposed into

    Delta_T3 = EA[CCSDT]  - EA[CCSD(T)]
    Delta_T4 = EA[CCSDTQ] - EA[CCSDT]

at the same fixed geometries, basis, frozen-core definition, and
nonrelativistic Hamiltonian.

v1 deliberately does not assign an uncomputed higher-order term a value of
zero.  The triples correction is checked over an aug-cc-pV{D,T}Z sequence.
Connected quadruples are first evaluated at aug-cc-pVDZ.  If their magnitude
is already below the configured threshold, the full DZ contribution is also
used as a conservative uncertainty bound.  Otherwise aug-cc-pVTZ CCSDTQ is
requested and the DZ->TZ change becomes the convergence bound.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PostCCPoint:
    cardinal: int
    basis: str
    ea_ccsd_t_ev: float
    ea_ccsdt_ev: float
    delta_t3_ev: float
    ea_ccsdtq_ev: float | None = None
    delta_t4_ev: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PostCCAssessment:
    status: str
    action: str
    central_correction_ev: float
    triples_correction_ev: float
    quadruples_correction_ev: float
    triples_convergence_bound_ev: float | None
    quadruples_convergence_bound_ev: float | None
    combined_bound_ev: float | None
    highest_triples_cardinal: int
    highest_quadruples_cardinal: int | None
    points: tuple[PostCCPoint, ...]
    evidence: tuple[str, ...]
    is_production_ea: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["points"] = [p.to_dict() for p in self.points]
        return data


def assess_post_ccsd_t(
    points: list[PostCCPoint],
    *,
    triples_target_ev: float = 0.001,
    quadruples_target_ev: float = 0.001,
    max_quadruples_cardinal: int = 3,
) -> PostCCAssessment:
    if not points:
        raise ValueError("At least one post-CCSD(T) point is required")

    pts = tuple(sorted(points, key=lambda p: p.cardinal))
    t3_pts = [p for p in pts if p.delta_t3_ev is not None]

    if len(t3_pts) < 2:
        latest = t3_pts[-1]
        return PostCCAssessment(
            status="NEED_MORE_EVIDENCE",
            action=f"COMPUTE_T3_X{latest.cardinal + 1}",
            central_correction_ev=latest.delta_t3_ev,
            triples_correction_ev=latest.delta_t3_ev,
            quadruples_correction_ev=0.0,
            triples_convergence_bound_ev=None,
            quadruples_convergence_bound_ev=None,
            combined_bound_ev=None,
            highest_triples_cardinal=latest.cardinal,
            highest_quadruples_cardinal=None,
            points=pts,
            evidence=(
                "CCSDT_MINUS_CCSD(T)_SAME_BASIS",
                "TRIPLES_ONE_CARDINAL_ONLY",
                "NO_UNCOMPUTED_TERM_ASSIGNED_ZERO",
            ),
        )

    t3_hi = t3_pts[-1]
    t3_prev = t3_pts[-2]
    t3_bound = abs(t3_hi.delta_t3_ev - t3_prev.delta_t3_ev)
    t3_cleared = t3_bound <= triples_target_ev

    t4_pts = [p for p in pts if p.delta_t4_ev is not None]
    if not t4_pts:
        return PostCCAssessment(
            status="NEED_MORE_EVIDENCE",
            action="COMPUTE_T4_X2",
            central_correction_ev=t3_hi.delta_t3_ev,
            triples_correction_ev=t3_hi.delta_t3_ev,
            quadruples_correction_ev=0.0,
            triples_convergence_bound_ev=t3_bound,
            quadruples_convergence_bound_ev=None,
            combined_bound_ev=None,
            highest_triples_cardinal=t3_hi.cardinal,
            highest_quadruples_cardinal=None,
            points=pts,
            evidence=(
                "CCSDT_MINUS_CCSD(T)_SAME_BASIS",
                "CONNECTED_QUADRUPLES_NOT_YET_COMPUTED",
                "NO_UNCOMPUTED_TERM_ASSIGNED_ZERO",
            ),
        )

    t4_pts = sorted(t4_pts, key=lambda p: p.cardinal)
    t4_hi = t4_pts[-1]

    if len(t4_pts) == 1:
        # A directly computed small DZ quadruples correction can be accepted,
        # but its entire magnitude is retained as a conservative basis bound.
        t4_bound = abs(t4_hi.delta_t4_ev)
        if t4_bound <= quadruples_target_ev:
            t4_cleared = True
            t4_action = "NONE"
            t4_evidence = "SMALL_DIRECT_DZ_T4_BOUNDED_BY_FULL_MAGNITUDE"
        elif t4_hi.cardinal < max_quadruples_cardinal:
            t4_cleared = False
            t4_action = f"COMPUTE_T4_X{t4_hi.cardinal + 1}"
            t4_evidence = "DZ_T4_EXCEEDS_SMALL_TERM_THRESHOLD"
        else:
            t4_cleared = False
            t4_action = "REVIEW_T4_BASIS_CONVERGENCE"
            t4_evidence = "T4_MAX_CARDINAL_REACHED_WITHOUT_BOUND"
    else:
        t4_prev = t4_pts[-2]
        t4_bound = abs(t4_hi.delta_t4_ev - t4_prev.delta_t4_ev)
        t4_cleared = t4_bound <= quadruples_target_ev
        t4_action = "NONE" if t4_cleared else "REVIEW_T4_BASIS_CONVERGENCE"
        t4_evidence = (
            "T4_LATEST_CARDINAL_CHANGE_WITHIN_TARGET"
            if t4_cleared
            else "T4_LATEST_CARDINAL_CHANGE_EXCEEDS_TARGET"
        )

    total = t3_hi.delta_t3_ev + t4_hi.delta_t4_ev
    combined = t3_bound + t4_bound

    if not t3_cleared:
        status = "UNRESOLVED"
        action = "REVIEW_T3_BASIS_CONVERGENCE"
    elif not t4_cleared:
        status = "NEED_MORE_EVIDENCE" if t4_action.startswith("COMPUTE_") else "UNRESOLVED"
        action = t4_action
    else:
        status = "CLEARED"
        action = "NONE"

    return PostCCAssessment(
        status=status,
        action=action,
        central_correction_ev=total,
        triples_correction_ev=t3_hi.delta_t3_ev,
        quadruples_correction_ev=t4_hi.delta_t4_ev,
        triples_convergence_bound_ev=t3_bound,
        quadruples_convergence_bound_ev=t4_bound,
        combined_bound_ev=combined,
        highest_triples_cardinal=t3_hi.cardinal,
        highest_quadruples_cardinal=t4_hi.cardinal,
        points=pts,
        evidence=(
            "CCSDT_MINUS_CCSD(T)_SAME_BASIS",
            "CCSDTQ_MINUS_CCSDT_SAME_BASIS",
            "FROZEN_CORE_VALENCE_POST_CC",
            "NONRELATIVISTIC_POST_CC_CORRECTION",
            "TRIPLES_LATEST_CARDINAL_CHANGE_WITHIN_TARGET"
            if t3_cleared else
            "TRIPLES_LATEST_CARDINAL_CHANGE_EXCEEDS_TARGET",
            t4_evidence,
            "NO_POST_CC_EXTRAPOLATION_ASSUMED",
        ),
        is_production_ea=False,
    )
