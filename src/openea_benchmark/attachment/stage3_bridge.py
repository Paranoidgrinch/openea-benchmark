"""Fail-closed bridge from converged Stage-3 PEC refinement to attachment records.

Stage 3 resolves a discrete, identity-cleared high-level PEC minimum bracket.
It does NOT yet provide a defensible two-sided energy interval for the true
continuous PEC minimum.  This bridge preserves that distinction explicitly.

Consequently:
- a converged Stage-3 loop can become an attachment PECBranch;
- the sampled minimum energy is retained as discrete evidence;
- no EnergyInterval is fabricated;
- production EA evaluation must wait for a later equilibrium-energy resolver.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

from .model import ElectronicState, PECBranch
from .pairing import generate_attachment_candidates


class Stage3AttachmentBridgeStatus(str, Enum):
    READY = "READY"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class Stage3AttachmentRecord:
    status: Stage3AttachmentBridgeStatus
    job_id: str
    branch: PECBranch | None
    discrete_minimum_r_angstrom: float | None
    discrete_minimum_energy_hartree: float | None
    minimum_bracket_angstrom: tuple[float, float] | None
    canonical_request_id: str | None
    energy_interval_hartree: None
    evidence: tuple[str, ...]
    rationale: str
    is_production_ea: bool = False
    ground_state_assigned: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("Stage-3 attachment record requires rationale")
        if self.is_production_ea or self.ground_state_assigned or self.authorizes_pruning:
            raise ValueError(
                "Stage-3 attachment bridge cannot assign production EA/ground state or authorize pruning"
            )
        if self.energy_interval_hartree is not None:
            raise ValueError(
                "Stage-3 attachment bridge must not fabricate a minimum-energy interval"
            )


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _unresolved(job_id: str, reason: str, evidence: tuple[str, ...]) -> Stage3AttachmentRecord:
    return Stage3AttachmentRecord(
        status=Stage3AttachmentBridgeStatus.UNRESOLVED,
        job_id=job_id,
        branch=None,
        discrete_minimum_r_angstrom=None,
        discrete_minimum_energy_hartree=None,
        minimum_bracket_angstrom=None,
        canonical_request_id=None,
        energy_interval_hartree=None,
        evidence=evidence,
        rationale=reason,
    )


def bridge_stage3_loop_to_attachment(
    loop_result: Any,
    *,
    state_id: str | None = None,
    state_label: str | None = None,
) -> Stage3AttachmentRecord:
    """Convert one converged Stage-3 loop result into an attachment PEC branch.

    The output intentionally has no two-sided minimum-energy interval.
    """
    job_id = str(getattr(loop_result, "job_id", "") or "")
    if not job_id:
        raise ValueError("Stage-3 loop result requires job_id")

    if any(
        bool(getattr(loop_result, name, False))
        for name in ("is_production_ea", "ground_state_assigned", "authorizes_pruning")
    ):
        raise ValueError("Stage-3 loop result violates non-EA/non-pruning contract")

    if _enum_value(getattr(loop_result, "status", None)) != "CONVERGED":
        return _unresolved(
            job_id,
            "Stage-3 refinement loop has not converged",
            ("STAGE3_LOOP_NOT_CONVERGED",),
        )

    plan = getattr(loop_result, "final_refinement_plan", None)
    if plan is None or _enum_value(getattr(plan, "action", None)) != "BRACKET_TARGET_MET":
        return _unresolved(
            job_id,
            "Stage-3 loop lacks an explicit BRACKET_TARGET_MET terminal action",
            ("BRACKET_TARGET_NOT_MET",),
        )

    pec = getattr(loop_result, "final_pec", None)
    if pec is None:
        return _unresolved(job_id, "Final Stage-3 PEC is missing", ("FINAL_PEC_MISSING",))

    if _enum_value(getattr(pec, "status", None)) != "READY_FOR_DISCRETE_MINIMUM_SCOUT":
        return _unresolved(
            job_id,
            "Final Stage-3 PEC validity gates are not cleared",
            ("FINAL_PEC_NOT_READY",),
        )

    if _enum_value(getattr(pec, "initialization_identity_status", None)) != "CLEARED":
        return _unresolved(
            job_id,
            "Same-geometry high-level state identity is unresolved",
            ("INITIALIZATION_IDENTITY_NOT_CLEARED",),
        )

    if _enum_value(getattr(pec, "geometry_continuity_status", None)) != "CLEARED":
        return _unresolved(
            job_id,
            "Geometry-to-geometry electronic continuity is unresolved",
            ("GEOMETRY_CONTINUITY_NOT_CLEARED",),
        )

    scout = getattr(pec, "minimum_scout", None)
    if scout is None or _enum_value(getattr(scout, "status", None)) != "BRACKETED_SINGLE_MINIMUM":
        return _unresolved(
            job_id,
            "Stage-3 PEC does not contain exactly one bracketed discrete minimum",
            ("SINGLE_MINIMUM_NOT_ESTABLISHED",),
        )

    candidates = tuple(getattr(scout, "candidates", ()) or ())
    if len(candidates) != 1:
        return _unresolved(
            job_id,
            "Stage-3 minimum scout does not expose exactly one candidate",
            ("SINGLE_MINIMUM_CANDIDATE_COUNT_INVALID",),
        )
    minimum = candidates[0]

    raw_points = tuple(getattr(pec, "points", ()) or ())
    if len(raw_points) < 3:
        return _unresolved(
            job_id,
            "Fewer than three Stage-3 PEC points are available",
            ("INSUFFICIENT_STAGE3_POINTS",),
        )

    points = sorted(raw_points, key=lambda p: float(getattr(p, "r_angstrom")))
    r_values = []
    energies = []
    for point in points:
        if _enum_value(getattr(point, "status", None)) != "ACCEPTED":
            return _unresolved(
                job_id,
                "Not every final Stage-3 PEC point is accepted",
                ("STAGE3_POINT_NOT_ACCEPTED",),
            )
        r = float(getattr(point, "r_angstrom"))
        e = getattr(point, "energy_hartree", None)
        if e is None:
            return _unresolved(
                job_id,
                "Accepted Stage-3 PEC point lacks an energy",
                ("STAGE3_POINT_ENERGY_MISSING",),
            )
        e = float(e)
        if not isfinite(r) or r <= 0.0 or not isfinite(e):
            raise ValueError("Stage-3 PEC contains non-finite/invalid point data")
        r_values.append(r)
        energies.append(e)

    spin_2s = int(getattr(pec, "spin_2s"))
    if spin_2s < 0:
        raise ValueError("Stage-3 spin_2s must be non-negative")

    component_id = str(getattr(pec, "component_id"))
    basis = str(getattr(pec, "basis"))
    system = str(getattr(pec, "system"))
    charge = int(getattr(pec, "charge"))

    resolved_state_id = state_id or (
        f"{system}|q={charge}|2S={spin_2s}|component={component_id}|basis={basis}"
    )

    minimum_energy = float(getattr(minimum, "energy_hartree"))
    minimum_r = float(getattr(minimum, "r_angstrom"))
    left_r = float(getattr(minimum, "left_r_angstrom"))
    right_r = float(getattr(minimum, "right_r_angstrom"))
    request_id = str(getattr(minimum, "canonical_request_id"))

    state = ElectronicState(
        state_id=resolved_state_id,
        charge=charge,
        multiplicity=spin_2s + 1,
        spin_2s=spin_2s,
        energy_hartree=minimum_energy,
        label=state_label,
        identity_status="CLEARED",
    )

    branch = PECBranch(
        branch_id=job_id,
        state=state,
        r_points=tuple(r_values),
        energies=tuple(energies),
        minimum_r=minimum_r,
        minimum_status="BRACKETED_SINGLE_MINIMUM",
        continuity_status="CLEARED",
    )

    return Stage3AttachmentRecord(
        status=Stage3AttachmentBridgeStatus.READY,
        job_id=job_id,
        branch=branch,
        discrete_minimum_r_angstrom=minimum_r,
        discrete_minimum_energy_hartree=minimum_energy,
        minimum_bracket_angstrom=(left_r, right_r),
        canonical_request_id=request_id,
        energy_interval_hartree=None,
        evidence=(
            "STAGE3_LOOP_CONVERGED",
            "BRACKET_TARGET_MET",
            "INITIALIZATION_IDENTITY_CLEARED",
            "GEOMETRY_CONTINUITY_CLEARED",
            "BRACKETED_SINGLE_MINIMUM",
            "DISCRETE_MINIMUM_ONLY",
            "EQUILIBRIUM_ENERGY_INTERVAL_REQUIRED",
        ),
        rationale=(
            "Stage-3 provides an identity-cleared, converged discrete minimum bracket; "
            "a continuous equilibrium-energy interval is still required before production EA evaluation"
        ),
    )


def pair_stage3_attachment_records(
    neutral: Stage3AttachmentRecord,
    anion: Stage3AttachmentRecord,
):
    """Delegate neutral/anion pairing only after both Stage-3 bridges are READY."""
    if neutral.status is not Stage3AttachmentBridgeStatus.READY:
        return ()
    if anion.status is not Stage3AttachmentBridgeStatus.READY:
        return ()
    if neutral.branch is None or anion.branch is None:
        return ()
    return generate_attachment_candidates((neutral.branch,), (anion.branch,))
