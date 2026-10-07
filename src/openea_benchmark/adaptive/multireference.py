"""Fail-closed multireference branch contract for OpenEA v1."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .model import ScientificResolutionStatus


class MRCapabilityStatus(str, Enum):
    VALIDATED_AVAILABLE = 'VALIDATED_AVAILABLE'
    UNAVAILABLE = 'UNAVAILABLE'
    INADEQUATE = 'INADEQUATE'


@dataclass(frozen=True)
class MRProductionCapability:
    """Evidence that a production-quality MR route is or is not available.

    Merely having CASSCF/NEVPT2 code installed is not enough to set
    VALIDATED_AVAILABLE.  The method family must be validated for the intended
    state/PEC/EA role and carry provenance.
    """

    status: MRCapabilityStatus
    method_family: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    rationale: str = ''

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError('MR capability requires an explicit rationale')
        if self.status is MRCapabilityStatus.VALIDATED_AVAILABLE:
            if not self.method_family:
                raise ValueError('Validated MR capability requires a method family')
            if not self.evidence_ids:
                raise ValueError('Validated MR capability requires provenance')
        elif self.method_family:
            raise ValueError('Unavailable/inadequate MR capability must not advertise a production method')


class MRBranchStatus(str, Enum):
    PRODUCTION_AUTHORIZED = 'PRODUCTION_AUTHORIZED'
    TERMINAL_UNRESOLVED = 'TERMINAL_UNRESOLVED'


@dataclass(frozen=True)
class MRBranchResolution:
    status: MRBranchStatus
    method_family: tuple[str, ...]
    scientific_status: ScientificResolutionStatus | None
    reason_code: str
    evidence_ids: tuple[str, ...]
    rationale: str


def resolve_multireference_branch(
    capability: MRProductionCapability | None,
) -> MRBranchResolution:
    """Authorize MR production only with an explicitly validated capability.

    The current OpenEA-v1 default is fail-closed: absence of such a capability
    yields the scientific terminal result UNRESOLVED with the contract reason
    MULTIREFERENCE_METHOD_REQUIRED.
    """

    if capability is not None and capability.status is MRCapabilityStatus.VALIDATED_AVAILABLE:
        return MRBranchResolution(
            MRBranchStatus.PRODUCTION_AUTHORIZED,
            capability.method_family,
            None,
            'VALIDATED_MULTIREFERENCE_METHOD_AVAILABLE',
            capability.evidence_ids,
            capability.rationale,
        )

    evidence = () if capability is None else capability.evidence_ids
    detail = (
        'No validated multireference production capability is registered.'
        if capability is None
        else capability.rationale
    )
    return MRBranchResolution(
        MRBranchStatus.TERMINAL_UNRESOLVED,
        (),
        ScientificResolutionStatus.UNRESOLVED,
        'MULTIREFERENCE_METHOD_REQUIRED',
        evidence,
        detail,
    )
