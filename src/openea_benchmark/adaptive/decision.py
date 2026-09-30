"""Fail-closed EA classification. No numerical chemistry, no external refs."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .model import EAEstimate, GateSet, Interval


class DecisionCategory(str, Enum):
    BOUND = 'BOUND'
    UNBOUND = 'UNBOUND'
    UNRESOLVED = 'UNRESOLVED'


class PrecisionStatus(str, Enum):
    TARGET_MET = 'TARGET_MET'
    TARGET_NOT_MET = 'TARGET_NOT_MET'
    NOT_APPLICABLE = 'NOT_APPLICABLE'
    UNDETERMINED = 'UNDETERMINED'


@dataclass(frozen=True)
class DecisionInput:
    molecule: str
    ea0_ev: Interval | None
    gates: GateSet
    target_half_width_ev: float = 0.020
    critical_open_questions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.molecule.strip():
            raise ValueError('Molecule must be specified')
        if not isfinite(self.target_half_width_ev) or self.target_half_width_ev <= 0:
            raise ValueError('Precision target must be finite and positive')


@dataclass(frozen=True)
class DecisionOutput:
    molecule: str
    category: DecisionCategory
    precision: PrecisionStatus
    ea0_ev: Interval | None
    reasons: tuple[str, ...]
    # A transparent scientific verdict; NOT a probabilistic confidence claim.
    estimated_not_certified: bool = True


def evaluate_decision(case: DecisionInput) -> DecisionOutput:
    """Classify only with independently supplied, reviewed G1/G2/G3 evidence.

    No `UNBOUND` on basis-set failure alone. No `BOUND` on a positive
    single-point value alone. Numerical precision is separate from sign.
    """
    reasons = tuple(case.gates.open_gates()) + case.critical_open_questions
    if reasons:
        return DecisionOutput(
            case.molecule, DecisionCategory.UNRESOLVED,
            PrecisionStatus.UNDETERMINED, case.ea0_ev, reasons,
        )
    if case.ea0_ev is None:
        return DecisionOutput(
            case.molecule, DecisionCategory.UNRESOLVED,
            PrecisionStatus.UNDETERMINED, None, ('MISSING_DEFENSIBLE_EA_INTERVAL',),
        )
    iv = case.ea0_ev
    if iv.lower > 0.0:
        precision = (PrecisionStatus.TARGET_MET
                     if iv.half_width <= case.target_half_width_ev
                     else PrecisionStatus.TARGET_NOT_MET)
        return DecisionOutput(case.molecule, DecisionCategory.BOUND, precision, iv,
                              ('POSITIVE_EA_INTERVAL_WITH_G1_G2_G3_CLEARED',))
    if iv.upper <= 0.0:
        return DecisionOutput(case.molecule, DecisionCategory.UNBOUND,
                              PrecisionStatus.NOT_APPLICABLE, iv,
                              ('NONPOSITIVE_EA_INTERVAL_WITH_G1_G2_G3_CLEARED',))
    return DecisionOutput(case.molecule, DecisionCategory.UNRESOLVED,
                          PrecisionStatus.UNDETERMINED, iv,
                          ('EA_INTERVAL_OVERLAPS_ZERO',))


def evaluate_estimate(
    molecule: str,
    estimate: EAEstimate,
    gates: GateSet,
    target_half_width_ev: float = 0.020,
    critical_open_questions: tuple[str, ...] = (),
) -> DecisionOutput:
    missing = tuple('UNKNOWN_CORRECTION:' + item for item in estimate.missing_components)
    return evaluate_decision(DecisionInput(
        molecule=molecule,
        ea0_ev=estimate.estimated_interval(),
        gates=gates,
        target_half_width_ev=target_half_width_ev,
        critical_open_questions=critical_open_questions + missing,
    ))
