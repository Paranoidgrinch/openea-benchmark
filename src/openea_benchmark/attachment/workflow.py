from dataclasses import dataclass
from enum import Enum
from typing import Any

from .asymptote import (
    BindingAssessment,
    DissociationChannel,
    evaluate_binding_interval,
)
from .decision import (
    EADecision,
    EADecisionStatus,
    PrecisionStatus,
    decide_ea,
)
from .equilibrium import (
    EquilibriumEstimate,
    EquilibriumResolverSettings,
    EquilibriumStatus,
    resolve_stage3_equilibrium,
)
from .model import AttachmentCandidate
from .stage3_bridge import (
    Stage3AttachmentRecord,
    bridge_stage3_loop_to_attachment,
    pair_stage3_attachment_records,
)


class ElectronicEAWorkflowStatus(str, Enum):
    BOUND = "BOUND"
    UNBOUND = "UNBOUND"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class ElectronicEAWorkflowResult:
    status: ElectronicEAWorkflowStatus
    neutral_stage3: Stage3AttachmentRecord
    anion_stage3: Stage3AttachmentRecord
    neutral_equilibrium: EquilibriumEstimate | None
    anion_equilibrium: EquilibriumEstimate | None
    attachment_candidate: AttachmentCandidate | None
    anion_binding: BindingAssessment | None
    ea_decision: EADecision
    evidence: tuple[str, ...]
    is_electronic_ea_only: bool = True
    includes_zpe: bool = False
    is_production_ea: bool = False
    ground_state_assigned: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if not self.is_electronic_ea_only:
            raise ValueError("this workflow is explicitly electronic-EA only")
        if self.includes_zpe:
            raise ValueError("Stage-3 electronic-EA workflow does not include ZPE")
        if self.is_production_ea or self.ground_state_assigned or self.authorizes_pruning:
            raise ValueError(
                "electronic-EA workflow cannot claim production EA/ground state or authorize pruning"
            )


def _unresolved_decision(reason: str) -> EADecision:
    return EADecision(
        status=EADecisionStatus.UNRESOLVED,
        precision_status=PrecisionStatus.NOT_ASSESSED,
        rationale=(reason,),
    )


def _status_from_decision(decision: EADecision) -> ElectronicEAWorkflowStatus:
    if decision.status is EADecisionStatus.BOUND:
        return ElectronicEAWorkflowStatus.BOUND
    if decision.status is EADecisionStatus.UNBOUND:
        return ElectronicEAWorkflowStatus.UNBOUND
    return ElectronicEAWorkflowStatus.UNRESOLVED


def evaluate_stage3_electronic_ea(
    *,
    neutral_loop_result: Any,
    anion_loop_result: Any,
    equilibrium_settings: EquilibriumResolverSettings,
    anion_dissociation_channels: tuple[DissociationChannel, ...],
    dissociation_safety_margin_hartree: float = 0.0,
    target_half_width_ev: float | None = None,
    neutral_state_id: str | None = None,
    anion_state_id: str | None = None,
    neutral_state_label: str | None = None,
    anion_state_label: str | None = None,
) -> ElectronicEAWorkflowResult:
    neutral = bridge_stage3_loop_to_attachment(
        neutral_loop_result,
        state_id=neutral_state_id,
        state_label=neutral_state_label,
    )
    anion = bridge_stage3_loop_to_attachment(
        anion_loop_result,
        state_id=anion_state_id,
        state_label=anion_state_label,
    )

    pairs = pair_stage3_attachment_records(neutral, anion)
    if len(pairs) != 1:
        decision = _unresolved_decision("STAGE3_ATTACHMENT_PAIR_NOT_UNIQUELY_AVAILABLE")
        return ElectronicEAWorkflowResult(
            status=ElectronicEAWorkflowStatus.UNRESOLVED,
            neutral_stage3=neutral,
            anion_stage3=anion,
            neutral_equilibrium=None,
            anion_equilibrium=None,
            attachment_candidate=None,
            anion_binding=None,
            ea_decision=decision,
            evidence=("ATTACHMENT_PAIR_NOT_READY",),
        )

    candidate = pairs[0]
    if candidate.pairing_status != "VALID":
        decision = _unresolved_decision("ATTACHMENT_PAIRING_NOT_CLEARED")
        return ElectronicEAWorkflowResult(
            status=ElectronicEAWorkflowStatus.UNRESOLVED,
            neutral_stage3=neutral,
            anion_stage3=anion,
            neutral_equilibrium=None,
            anion_equilibrium=None,
            attachment_candidate=candidate,
            anion_binding=None,
            ea_decision=decision,
            evidence=("ATTACHMENT_PAIRING_NOT_CLEARED",),
        )

    neutral_eq = resolve_stage3_equilibrium(neutral, settings=equilibrium_settings)
    anion_eq = resolve_stage3_equilibrium(anion, settings=equilibrium_settings)

    if (
        neutral_eq.status is not EquilibriumStatus.RESOLVED
        or anion_eq.status is not EquilibriumStatus.RESOLVED
    ):
        decision = _unresolved_decision("EQUILIBRIUM_ENERGY_NOT_RESOLVED")
        return ElectronicEAWorkflowResult(
            status=ElectronicEAWorkflowStatus.UNRESOLVED,
            neutral_stage3=neutral,
            anion_stage3=anion,
            neutral_equilibrium=neutral_eq,
            anion_equilibrium=anion_eq,
            attachment_candidate=candidate,
            anion_binding=None,
            ea_decision=decision,
            evidence=("EQUILIBRIUM_ENERGY_NOT_RESOLVED",),
        )

    binding = evaluate_binding_interval(
        anion_eq.energy_interval_hartree,
        anion_dissociation_channels,
        safety_margin_hartree=dissociation_safety_margin_hartree,
    )

    decision = decide_ea(
        candidate=candidate,
        anion_binding=binding,
        neutral_energy=neutral_eq.energy_interval_hartree,
        anion_energy=anion_eq.energy_interval_hartree,
        neutral_nominal_hartree=neutral_eq.energy_central_hartree,
        anion_nominal_hartree=anion_eq.energy_central_hartree,
        target_half_width_ev=target_half_width_ev,
    )

    return ElectronicEAWorkflowResult(
        status=_status_from_decision(decision),
        neutral_stage3=neutral,
        anion_stage3=anion,
        neutral_equilibrium=neutral_eq,
        anion_equilibrium=anion_eq,
        attachment_candidate=candidate,
        anion_binding=binding,
        ea_decision=decision,
        evidence=(
            "STAGE3_NEUTRAL_BRIDGE_EVALUATED",
            "STAGE3_ANION_BRIDGE_EVALUATED",
            "EQUILIBRIUM_INTERVALS_EVALUATED",
            "ANION_DISSOCIATION_STABILITY_EVALUATED",
            "ELECTRONIC_EA_INTERVAL_EVALUATED",
            "ZPE_NOT_INCLUDED",
            "NOT_PRODUCTION_EA",
        ),
    )
