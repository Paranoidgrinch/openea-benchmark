"""V0.1 routing catalogue: recommendations, never automatic method approval.

The scheduler will later translate these diagnostic requests into backend jobs
only after explicit scientific validation of capability and numerical policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from .model import DiagnosticID, Review, ReviewStatus


@dataclass(frozen=True)
class DiagnosticSpec:
    identifier: DiagnosticID
    gate: str
    question: str
    required_evidence: str
    escalation: str
    priority: int  # 0 = prerequisite, 1 = dominant uncertainty, 2 = refinement


DIAGNOSTIC_CATALOG: dict[DiagnosticID, DiagnosticSpec] = {
    DiagnosticID.D01_SPIN: DiagnosticSpec(DiagnosticID.D01_SPIN, 'G1', 'Is the spin representation appropriate?',
        'Target S, computed <S^2>, alternative-reference context', 'Spin-/reference-adapted control', 0),
    DiagnosticID.D02_SCF_STABILITY: DiagnosticSpec(DiagnosticID.D02_SCF_STABILITY, 'G1', 'Is the SCF root suitably stable?',
        'Available internal/external tests with explicit test scope', 'Restart from unstable mode or use alternative reference', 0),
    DiagnosticID.D03_MULTIREFERENCE: DiagnosticSpec(DiagnosticID.D03_MULTIREFERENCE, 'G3a', 'Does static correlation invalidate SR accuracy?',
        'Several indicators; natural occupations / active-space stability when indicated', 'AVAS/APC -> CASSCF -> validated CAS expansion', 1),
    DiagnosticID.D04_STATE_COMPETITION: DiagnosticSpec(DiagnosticID.D04_STATE_COMPETITION, 'G1', 'Can another state be ground?',
        'Tracked candidate energies with justified intervals', 'Targeted alternative-state PEC and correlation', 0),
    DiagnosticID.D05_REFERENCE_SENSITIVITY: DiagnosticSpec(DiagnosticID.D05_REFERENCE_SENSITIVITY, 'G3a', 'Are correlated results reference sensitive?',
        'Distinct physical roots and compatible correlated controls', 'Additional reference or MR route', 1),
    DiagnosticID.D06_TRIPLES_RELIABILITY: DiagnosticSpec(DiagnosticID.D06_TRIPLES_RELIABILITY, 'G3c', 'Is post-(T) residual acceptable?',
        'CCSD/(T), amplitudes, CCSDT control where justified', 'Reference-character reassessment; do not automatically escalate coupled-cluster rank', 1),
    DiagnosticID.D07_DIFFUSE_BASIS: DiagnosticSpec(DiagnosticID.D07_DIFFUSE_BASIS, 'G2', 'Is extra-electron diffusity resolved?',
        'Augmentation convergence of energy and attachment density', 'd-aug/exponent variation + conditioning checks', 0),
    DiagnosticID.D08_ATTACHMENT: DiagnosticSpec(DiagnosticID.D08_ATTACHMENT, 'G2', 'Is attachment physically bound/resolved?',
        'Vertical threshold, attachment character, continuum sensitivity', 'EOM-EA and/or stabilization/continuum treatment', 0),
    DiagnosticID.D09_PEC_ASYMPTOTES: DiagnosticSpec(DiagnosticID.D09_PEC_ASYMPTOTES, 'G1', 'Are PEC minima and channels covered?',
        'Branch continuity, nuclear minima, large-R limits and fragments', 'Adaptive PEC and fragment-channel discovery', 0),
    DiagnosticID.D10_SCALAR_RELATIVITY: DiagnosticSpec(DiagnosticID.D10_SCALAR_RELATIVITY, 'G3d', 'Is scalar relativity accounted for?',
        'Hamiltonian provenance or bounded relative correction', 'Compatible X2C/DKH comparison', 2),
    DiagnosticID.D11_SOC: DiagnosticSpec(DiagnosticID.D11_SOC, 'G3d', 'Is spin-orbit effect adequately resolved?',
        'Electronic term spectrum + bounded SOC impact on EA', 'Validated OpenMolcas/RASSI-SOC route', 2),
    DiagnosticID.D12_NUCLEAR_MOTION: DiagnosticSpec(DiagnosticID.D12_NUCLEAR_MOTION, 'G3d', 'Are zero-point energies resolved?',
        'Vibrational eigenvalues + PEC convergence', 'Refined PEC and radial nuclear solve', 2),
}


@dataclass(frozen=True)
class PlannedDiagnostic:
    identifier: DiagnosticID
    priority: int
    action: str
    reason: str


def next_diagnostics(
    active_ids: set[DiagnosticID],
    reviews: Mapping[DiagnosticID, Review],
) -> tuple[PlannedDiagnostic, ...]:
    """Return a stable, transparent diagnostic request queue.

    Caller enables IDs from chemical screening or new observed evidence.
    No ad-hoc chemistry thresholds; neither a job executor nor a claim that
    the selected method is necessarily applicable to every system.
    """
    queued: list[PlannedDiagnostic] = []
    for identifier in active_ids:
        spec = DIAGNOSTIC_CATALOG[identifier]
        status = reviews.get(identifier, Review(ReviewStatus.PENDING)).status
        if status in (ReviewStatus.CLEARED, ReviewStatus.NOT_APPLICABLE):
            continue
        if status is ReviewStatus.PENDING:
            action = f'Perform diagnostic: {spec.required_evidence}'
        else:
            action = f'Investigate/escalate: {spec.escalation}'
        queued.append(PlannedDiagnostic(identifier, spec.priority, action,
                                         f'{spec.gate}: {spec.question} [{status.value}]'))
    return tuple(sorted(queued, key=lambda item: (item.priority, item.identifier.value)))
