from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .decision import EnergyInterval


class BindingStatus(str, Enum):
    BOUND = "BOUND"
    UNBOUND = "UNBOUND"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class DissociationChannel:
    channel_id: str
    fragment_a: str
    fragment_b: str
    asymptotic_energy_hartree: float | None
    source: str = "UNKNOWN"


@dataclass(frozen=True)
class BindingAssessment:
    status: BindingStatus
    evidence: tuple[str, ...]
    energy_margin_hartree: float | None = None


def evaluate_binding_status(
    minimum_energy_hartree: float | None,
    minimum_status: str,
    channels: tuple[DissociationChannel, ...],
    safety_margin_hartree: float = 0.0,
) -> BindingAssessment:
    if minimum_energy_hartree is None or minimum_status != "BRACKETED_SINGLE_MINIMUM":
        return BindingAssessment(BindingStatus.UNRESOLVED, ("NO_CONFIRMED_MINIMUM",))

    energies = [
        c.asymptotic_energy_hartree
        for c in channels
        if c.asymptotic_energy_hartree is not None
    ]
    if not energies:
        return BindingAssessment(
            BindingStatus.UNRESOLVED,
            ("NO_ASYMPTOTIC_CHANNEL_AVAILABLE",),
        )

    lowest_asymptote = min(energies)
    margin = lowest_asymptote - minimum_energy_hartree
    if margin > safety_margin_hartree:
        return BindingAssessment(
            BindingStatus.BOUND,
            ("MINIMUM_BELOW_LOWEST_DISSOCIATION_CHANNEL",),
            margin,
        )
    return BindingAssessment(
        BindingStatus.UNBOUND,
        ("MINIMUM_NOT_BELOW_DISSOCIATION_LIMIT",),
        margin,
    )


def evaluate_binding_interval(
    minimum_energy: "EnergyInterval | None",
    channels: tuple[DissociationChannel, ...],
    *,
    safety_margin_hartree: float = 0.0,
) -> BindingAssessment:
    if not isfinite(float(safety_margin_hartree)) or safety_margin_hartree < 0.0:
        raise ValueError("safety_margin_hartree must be finite and non-negative")

    if minimum_energy is None:
        return BindingAssessment(
            BindingStatus.UNRESOLVED,
            ("MINIMUM_ENERGY_INTERVAL_MISSING",),
        )

    if not channels:
        return BindingAssessment(
            BindingStatus.UNRESOLVED,
            ("NO_DISSOCIATION_CHANNELS_SUPPLIED",),
        )

    if any(c.asymptotic_energy_hartree is None for c in channels):
        return BindingAssessment(
            BindingStatus.UNRESOLVED,
            ("DISSOCIATION_CHANNEL_ENERGY_INCOMPLETE",),
        )

    energies = tuple(float(c.asymptotic_energy_hartree) for c in channels)
    if any(not isfinite(e) for e in energies):
        raise ValueError("dissociation-channel energies must be finite")

    threshold = min(energies)
    lower_margin = threshold - minimum_energy.upper_hartree
    upper_margin = threshold - minimum_energy.lower_hartree

    if lower_margin > safety_margin_hartree:
        return BindingAssessment(
            BindingStatus.BOUND,
            (
                "ANION_ENERGY_INTERVAL_BELOW_ALL_SUPPLIED_DISSOCIATION_CHANNELS",
                "DISSOCIATION_MARGIN_INTERVAL_CLEARED",
            ),
            lower_margin,
        )

    if upper_margin <= 0.0:
        return BindingAssessment(
            BindingStatus.UNBOUND,
            ("ANION_ENERGY_INTERVAL_NOT_BELOW_LOWEST_DISSOCIATION_CHANNEL",),
            upper_margin,
        )

    return BindingAssessment(
        BindingStatus.UNRESOLVED,
        ("ANION_DISSOCIATION_MARGIN_INTERVAL_OVERLAPS_ZERO_OR_SAFETY_MARGIN",),
        lower_margin,
    )
