from dataclasses import dataclass
from enum import Enum


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
        return BindingAssessment(
            BindingStatus.UNRESOLVED,
            ("NO_CONFIRMED_MINIMUM",),
        )

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
