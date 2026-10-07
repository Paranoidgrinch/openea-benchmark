"""CCSDT triples-reliability diagnostic for OpenEA v1.

Canonical role
--------------
CCSDT is a DIAGNOSTIC of the perturbative-triples approximation in CCSD(T):

    Delta_T3(X) = EA[CCSDT](X) - EA[CCSD(T)](X)

It is not a mandatory production rung.  A small, basis-stable Delta_T3 may be
used as an optional correction with an evidence-based residual bound.  A large
or unstable Delta_T3 triggers POST_CC_WARNING and reference-character
reassessment.  This module NEVER requests CCSDTQ automatically.

Legacy CCSDTQ fields remain readable in PostCCPoint so old validation/checkpoint
records can be inspected without data loss.  They are not added to the v1
production correction by this diagnostic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from openea_benchmark.adaptive.model import MethodRole


@dataclass(frozen=True)
class PostCCPoint:
    cardinal: int
    basis: str
    ea_ccsd_t_ev: float
    ea_ccsdt_ev: float
    delta_t3_ev: float
    # Legacy validation-only fields.  Never required by the production graph.
    ea_ccsdtq_ev: float | None = None
    delta_t4_ev: float | None = None

    def __post_init__(self) -> None:
        if self.cardinal < 1:
            raise ValueError('Cardinal number must be positive')
        if not self.basis.strip():
            raise ValueError('Basis must be specified')
        for value in (self.ea_ccsd_t_ev, self.ea_ccsdt_ev, self.delta_t3_ev):
            if not isfinite(value):
                raise ValueError('Post-CCSD(T) energies/differences must be finite')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PostCCAssessment:
    status: str
    action: str
    central_correction_ev: float | None
    triples_correction_ev: float
    quadruples_correction_ev: float | None
    triples_convergence_bound_ev: float | None
    quadruples_convergence_bound_ev: float | None
    combined_bound_ev: float | None
    highest_triples_cardinal: int
    highest_quadruples_cardinal: int | None
    points: tuple[PostCCPoint, ...]
    evidence: tuple[str, ...]
    method_role: MethodRole = MethodRole.DIAGNOSTIC
    is_production_ea: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['points'] = [p.to_dict() for p in self.points]
        data['method_role'] = self.method_role.value
        return data


def assess_post_ccsd_t(
    points: list[PostCCPoint],
    *,
    triples_target_ev: float = 0.001,
    quadruples_target_ev: float | None = None,
    max_quadruples_cardinal: int | None = None,
) -> PostCCAssessment:
    """Assess whether CCSD(T)'s perturbative triples are reliable enough.

    `triples_target_ev` is the active tolerance against which both the latest
    |Delta_T3| magnitude and its latest cardinal change are judged.  The two
    quadruples arguments are accepted only for source compatibility with the
    pre-refactor OH validation script; they never authorize or request T4 work.
    """

    if not points:
        raise ValueError('At least one post-CCSD(T) point is required')
    if not isfinite(triples_target_ev) or triples_target_ev <= 0:
        raise ValueError('triples_target_ev must be finite and positive')

    pts = tuple(sorted(points, key=lambda p: p.cardinal))
    t3_pts = list(pts)
    latest = t3_pts[-1]
    legacy_t4 = [p for p in pts if p.delta_t4_ev is not None]
    highest_t4 = max((p.cardinal for p in legacy_t4), default=None)

    common_evidence = [
        'METHOD_ROLE_DIAGNOSTIC',
        'CCSDT_MINUS_CCSD(T)_SAME_BASIS',
        'FROZEN_CORE_VALENCE_POST_CC',
        'NONRELATIVISTIC_POST_CC_DIAGNOSTIC',
        'NO_AUTOMATIC_CCSDTQ_ESCALATION',
        'NO_UNCOMPUTED_TERM_ASSIGNED_ZERO',
    ]
    if legacy_t4:
        common_evidence.append('LEGACY_CCSDTQ_DATA_PRESENT_BUT_NOT_USED_IN_PRODUCTION_DIAGNOSTIC')
    if quadruples_target_ev is not None or max_quadruples_cardinal is not None:
        common_evidence.append('LEGACY_QUADRUPLES_CONTROL_ARGUMENTS_IGNORED')

    if len(t3_pts) < 2:
        return PostCCAssessment(
            status='NEED_MORE_EVIDENCE',
            action=f'COMPUTE_T3_X{latest.cardinal + 1}',
            central_correction_ev=None,
            triples_correction_ev=latest.delta_t3_ev,
            quadruples_correction_ev=None,
            triples_convergence_bound_ev=None,
            quadruples_convergence_bound_ev=None,
            combined_bound_ev=None,
            highest_triples_cardinal=latest.cardinal,
            highest_quadruples_cardinal=highest_t4,
            points=pts,
            evidence=tuple(common_evidence + [
                'TRIPLES_ONE_CARDINAL_ONLY',
                'OPTIONAL_DELTA_T3_CORRECTION_NOT_AUTHORIZED',
            ]),
        )

    previous = t3_pts[-2]
    t3_bound = abs(latest.delta_t3_ev - previous.delta_t3_ev)
    magnitude_small = abs(latest.delta_t3_ev) <= triples_target_ev
    basis_stable = t3_bound <= triples_target_ev

    if magnitude_small and basis_stable:
        return PostCCAssessment(
            status='CLEARED',
            action='NONE',
            central_correction_ev=latest.delta_t3_ev,
            triples_correction_ev=latest.delta_t3_ev,
            quadruples_correction_ev=None,
            triples_convergence_bound_ev=t3_bound,
            quadruples_convergence_bound_ev=None,
            combined_bound_ev=t3_bound,
            highest_triples_cardinal=latest.cardinal,
            highest_quadruples_cardinal=highest_t4,
            points=pts,
            evidence=tuple(common_evidence + [
                'TRIPLES_LATEST_MAGNITUDE_WITHIN_ACTIVE_TARGET',
                'TRIPLES_LATEST_CARDINAL_CHANGE_WITHIN_ACTIVE_TARGET',
                'OPTIONAL_DELTA_T3_CORRECTION_AUTHORIZED',
            ]),
        )

    reasons = []
    if not magnitude_small:
        reasons.append('TRIPLES_LATEST_MAGNITUDE_EXCEEDS_ACTIVE_TARGET')
    if not basis_stable:
        reasons.append('TRIPLES_LATEST_CARDINAL_CHANGE_EXCEEDS_ACTIVE_TARGET')

    return PostCCAssessment(
        status='POST_CC_WARNING',
        action='REASSESS_REFERENCE_CHARACTER',
        central_correction_ev=None,
        triples_correction_ev=latest.delta_t3_ev,
        quadruples_correction_ev=None,
        triples_convergence_bound_ev=t3_bound,
        quadruples_convergence_bound_ev=None,
        combined_bound_ev=None,
        highest_triples_cardinal=latest.cardinal,
        highest_quadruples_cardinal=highest_t4,
        points=pts,
        evidence=tuple(common_evidence + reasons + [
            'OPTIONAL_DELTA_T3_CORRECTION_NOT_AUTHORIZED',
            'RETURN_TO_REFERENCE_CHARACTER_GATE',
        ]),
    )
