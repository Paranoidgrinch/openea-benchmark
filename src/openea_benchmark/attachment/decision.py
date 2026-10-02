"""Conservative electron-affinity decision logic.

This module separates neutral/anion pairing, molecular-anion stability against
fragmentation, electron binding from the sign of the EA interval, and precision
target attainment. Intervals are conservative model/evidence intervals, not
statistical confidence intervals.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .asymptote import BindingAssessment, BindingStatus
from .model import AttachmentCandidate

HARTREE_TO_EV = 27.211386245988


class EADecisionStatus(str, Enum):
    BOUND = "BOUND"
    UNBOUND = "UNBOUND"
    UNRESOLVED = "UNRESOLVED"


class PrecisionStatus(str, Enum):
    TARGET_MET = "TARGET_MET"
    TARGET_NOT_MET = "TARGET_NOT_MET"
    NOT_ASSESSED = "NOT_ASSESSED"


@dataclass(frozen=True)
class EnergyInterval:
    lower_hartree: float
    upper_hartree: float

    def __post_init__(self) -> None:
        lo = float(self.lower_hartree)
        hi = float(self.upper_hartree)
        if not isfinite(lo) or not isfinite(hi):
            raise ValueError("energy interval bounds must be finite")
        if lo > hi:
            raise ValueError("energy interval lower bound exceeds upper bound")


@dataclass(frozen=True)
class EADecision:
    status: EADecisionStatus
    precision_status: PrecisionStatus
    rationale: tuple[str, ...]
    ea_lower_ev: float | None = None
    ea_upper_ev: float | None = None
    ea_central_ev: float | None = None
    half_width_ev: float | None = None
    target_half_width_ev: float | None = None

    @property
    def has_numeric_bound_ea(self) -> bool:
        return (
            self.status is EADecisionStatus.BOUND
            and self.ea_lower_ev is not None
            and self.ea_upper_ev is not None
        )


def _precision_status(
    half_width_ev: float,
    target_half_width_ev: float | None,
) -> PrecisionStatus:
    if target_half_width_ev is None:
        return PrecisionStatus.NOT_ASSESSED

    target = float(target_half_width_ev)
    if not isfinite(target) or target <= 0.0:
        raise ValueError("target_half_width_ev must be finite and > 0")

    return (
        PrecisionStatus.TARGET_MET
        if half_width_ev <= target
        else PrecisionStatus.TARGET_NOT_MET
    )


def _validated_nominal(
    name: str,
    value: float | None,
    interval: EnergyInterval,
) -> float | None:
    if value is None:
        return None

    x = float(value)
    if not isfinite(x):
        raise ValueError(f"{name} nominal energy must be finite")

    tol = 1.0e-12
    if x < interval.lower_hartree - tol or x > interval.upper_hartree + tol:
        raise ValueError(f"{name} nominal energy lies outside its interval")

    return x


def decide_ea(
    *,
    candidate: AttachmentCandidate,
    anion_binding: BindingAssessment,
    neutral_energy: EnergyInterval | None,
    anion_energy: EnergyInterval | None,
    neutral_nominal_hartree: float | None = None,
    anion_nominal_hartree: float | None = None,
    target_half_width_ev: float | None = None,
) -> EADecision:
    """Make a fail-closed adiabatic electronic-EA decision.

    EA_e = E_neutral(min) - E_anion(min)
    and interval propagation is
        [L_N - U_A, U_N - L_A].
    """
    if candidate.pairing_status != "VALID":
        return EADecision(
            EADecisionStatus.UNRESOLVED,
            PrecisionStatus.NOT_ASSESSED,
            ("ATTACHMENT_PAIRING_NOT_CLEARED",),
        )

    if anion_binding.status is BindingStatus.UNBOUND:
        return EADecision(
            EADecisionStatus.UNBOUND,
            PrecisionStatus.NOT_ASSESSED,
            ("ANION_NOT_PHYSICALLY_BOUND",),
        )

    if anion_binding.status is not BindingStatus.BOUND:
        return EADecision(
            EADecisionStatus.UNRESOLVED,
            PrecisionStatus.NOT_ASSESSED,
            ("ANION_BINDING_UNRESOLVED",),
        )

    if neutral_energy is None or anion_energy is None:
        return EADecision(
            EADecisionStatus.UNRESOLVED,
            PrecisionStatus.NOT_ASSESSED,
            ("ENERGY_INTERVAL_INCOMPLETE",),
        )

    neutral_nominal = _validated_nominal(
        "neutral", neutral_nominal_hartree, neutral_energy
    )
    anion_nominal = _validated_nominal(
        "anion", anion_nominal_hartree, anion_energy
    )

    if (neutral_nominal is None) != (anion_nominal is None):
        raise ValueError(
            "neutral and anion nominal energies must be supplied together"
        )

    ea_lower_ev = (
        neutral_energy.lower_hartree - anion_energy.upper_hartree
    ) * HARTREE_TO_EV
    ea_upper_ev = (
        neutral_energy.upper_hartree - anion_energy.lower_hartree
    ) * HARTREE_TO_EV
    half_width_ev = 0.5 * (ea_upper_ev - ea_lower_ev)

    central_ev = None
    if neutral_nominal is not None:
        central_ev = (neutral_nominal - anion_nominal) * HARTREE_TO_EV

    if ea_upper_ev <= 0.0:
        return EADecision(
            EADecisionStatus.UNBOUND,
            PrecisionStatus.NOT_ASSESSED,
            ("EA_INTERVAL_NONPOSITIVE",),
        )

    if ea_lower_ev <= 0.0:
        return EADecision(
            EADecisionStatus.UNRESOLVED,
            PrecisionStatus.NOT_ASSESSED,
            ("EA_INTERVAL_OVERLAPS_ZERO",),
            ea_lower_ev=ea_lower_ev,
            ea_upper_ev=ea_upper_ev,
            ea_central_ev=central_ev,
            half_width_ev=half_width_ev,
        )

    precision = _precision_status(half_width_ev, target_half_width_ev)
    return EADecision(
        EADecisionStatus.BOUND,
        precision,
        (
            "ATTACHMENT_PAIRING_CLEARED",
            "ANION_PHYSICALLY_BOUND",
            "EA_INTERVAL_STRICTLY_POSITIVE",
        ),
        ea_lower_ev=ea_lower_ev,
        ea_upper_ev=ea_upper_ev,
        ea_central_ev=central_ev,
        half_width_ev=half_width_ev,
        target_half_width_ev=target_half_width_ev,
    )
