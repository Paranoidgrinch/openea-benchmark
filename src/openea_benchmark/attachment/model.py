from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class ElectronicState:
    state_id: str
    charge: int
    multiplicity: int
    spin_2s: int
    energy_hartree: float | None = None
    label: str | None = None
    identity_status: str = "UNKNOWN"

@dataclass(frozen=True)
class PECBranch:
    branch_id: str
    state: ElectronicState
    r_points: Tuple[float, ...] = ()
    energies: Tuple[float, ...] = ()
    minimum_r: float | None = None
    minimum_status: str = "UNKNOWN"
    continuity_status: str = "UNKNOWN"

@dataclass(frozen=True)
class AttachmentCandidate:
    candidate_id: str
    neutral_branch: PECBranch
    anion_branch: PECBranch
    pairing_status: str
    evidence: Tuple[str, ...]
    ea_status: str = "UNASSESSED"
