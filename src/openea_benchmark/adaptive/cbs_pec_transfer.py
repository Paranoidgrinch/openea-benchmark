"""Matched CBS/other-model PEC correction interpolation (diagnostic only).

This is a *numerical transfer*, not an electronic structure solver or a
certified error bound.  Delta(R) is formed only from two explicitly matched,
reviewed electronic calculations at each sampled bond length.  A target
outside the measured interval is never extrapolated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Any, Sequence

from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV
from .stage3_execution import PointExecutionStatus, Stage3PointResult


class TransferStatus(str, Enum):
    CANDIDATE_REVIEW_REQUIRED = "CANDIDATE_REVIEW_REQUIRED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class MatchedCorrectionPoint:
    """Two state-matched calculations at the *same* geometry."""

    role: str
    atoms: tuple[str, str]
    charge: int
    spin_2s: int
    state_id: str
    r_angstrom: float
    lower_model_id: str
    upper_model_id: str
    lower_energy_hartree: float
    upper_energy_hartree: float
    lower_source_id: str
    upper_source_id: str
    state_identity_reviewed: bool

    def __post_init__(self):
        if self.role not in ("neutral", "anion") or len(self.atoms) != 2:
            raise ValueError("Matched correction needs neutral/anion diatomic role")
        if self.charge != (0 if self.role == "neutral" else -1):
            raise ValueError("Neutral/anion role must have charge 0/-1")
        if any(not atom.strip() for atom in self.atoms) or self.spin_2s < 0:
            raise ValueError("Invalid atoms or spin")
        for name in ("state_id", "lower_model_id", "upper_model_id", "lower_source_id", "upper_source_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"Missing {name}")
        if self.lower_model_id == self.upper_model_id:
            raise ValueError("Different electronic models are required for transfer")
        if self.lower_source_id == self.upper_source_id:
            raise ValueError("Lower and upper calculations must be independent evidence")
        if not isfinite(self.r_angstrom) or self.r_angstrom <= 0.0:
            raise ValueError("Invalid bond length")
        if not all(isfinite(x) for x in (self.lower_energy_hartree, self.upper_energy_hartree)):
            raise ValueError("Non-finite electronic energy")
        if not self.state_identity_reviewed:
            raise ValueError("Matched electronic states must be reviewed before transfer")

    @property
    def correction_hartree(self) -> float:
        return self.upper_energy_hartree - self.lower_energy_hartree


@dataclass(frozen=True)
class SpeciesTransfer:
    role: str
    r_target_angstrom: float
    correction_hartree: float
    linear_hartree: float
    quadratic_hartree: float
    local_model_disagreement_ev: float
    max_observed_interior_loo_ev: float
    sample_bond_lengths_angstrom: tuple[float, ...]
    source_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PECTransferAssessment:
    status: TransferStatus
    neutral: SpeciesTransfer | None
    anion: SpeciesTransfer | None
    delta_ea_candidate_ev: float | None
    observed_interpolation_sensitivity_ev: float | None
    reason: str
    uncertainty_bounded: bool = False
    is_production_ea: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "neutral": None if self.neutral is None else self.neutral.to_dict(),
            "anion": None if self.anion is None else self.anion.to_dict(),
            "delta_ea_candidate_ev": self.delta_ea_candidate_ev,
            "observed_interpolation_sensitivity_ev": self.observed_interpolation_sensitivity_ev,
            "reason": self.reason,
            "uncertainty_bounded": False,
            "is_production_ea": False,
        }


def matched_stage3_point(
    *, role: str, state_id: str, atoms: tuple[str, str],
    lower_model_id: str, upper_model_id: str,
    lower: Stage3PointResult, upper: Stage3PointResult,
    state_identity_reviewed: bool,
) -> MatchedCorrectionPoint:
    """Adapt *real completed* Stage-3 energy results, never fabricate energies.

    Geometry/charge/spin/system and electronic reference must agree.  Model
    comparability and root identity still require an upstream scientific review.
    """
    if lower.status is not PointExecutionStatus.COMPLETED or upper.status is not PointExecutionStatus.COMPLETED:
        raise ValueError("Both Stage-3 electronic calculations must be completed")
    if lower.ccsd_t_total_hartree is None or upper.ccsd_t_total_hartree is None:
        raise ValueError("Missing correlated CCSD(T) energy")
    for field in ("system", "charge", "spin_2s", "r_angstrom", "scf_reference"):
        if getattr(lower, field) != getattr(upper, field):
            raise ValueError(f"Stage-3 correction points differ in {field}")
    if lower.charge != (0 if role == "neutral" else -1):
        raise ValueError("Role/charge mismatch")
    if not lower.source_checkpoint_path or not upper.source_checkpoint_path:
        raise ValueError("Stage-3 source checkpoint provenance missing")
    return MatchedCorrectionPoint(
        role=role, atoms=atoms, charge=lower.charge, spin_2s=lower.spin_2s,
        state_id=state_id, r_angstrom=float(lower.r_angstrom),
        lower_model_id=lower_model_id, upper_model_id=upper_model_id,
        lower_energy_hartree=float(lower.ccsd_t_total_hartree),
        upper_energy_hartree=float(upper.ccsd_t_total_hartree),
        lower_source_id=lower.request_id, upper_source_id=upper.request_id,
        state_identity_reviewed=state_identity_reviewed,
    )


def _linear(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    return y0 + (target - x0) * (y1 - y0) / (x1 - x0)


def _quadratic(points: Sequence[MatchedCorrectionPoint], target: float) -> float:
    result = 0.0
    for i, pi in enumerate(points):
        weight = 1.0
        for j, pj in enumerate(points):
            if i != j:
                weight *= (target - pj.r_angstrom) / (pi.r_angstrom - pj.r_angstrom)
        result += weight * pi.correction_hartree
    return result


def _species(points: Sequence[MatchedCorrectionPoint], role: str, target: float) -> SpeciesTransfer:
    if not isfinite(target) or target <= 0.0:
        raise ValueError("Target geometry must be finite and positive")
    if len(points) < 3:
        raise ValueError(f"{role}: at least three matched geometry samples required")
    data = sorted(points, key=lambda p: p.r_angstrom)
    signature = {(p.role, p.atoms, p.charge, p.spin_2s, p.state_id, p.lower_model_id, p.upper_model_id) for p in data}
    if len(signature) != 1 or data[0].role != role:
        raise ValueError(f"{role}: mismatched state/model across PEC")
    if any(a.r_angstrom == b.r_angstrom for a, b in zip(data, data[1:])):
        raise ValueError(f"{role}: duplicate geometries")
    source_ids = [s for p in data for s in (p.lower_source_id, p.upper_source_id)]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{role}: reused calculation provenance")
    if not data[0].r_angstrom <= target <= data[-1].r_angstrom:
        raise ValueError(f"{role}: target geometry outside computed correction span; extrapolation forbidden")
    i = next((i for i in range(len(data) - 1) if data[i].r_angstrom <= target <= data[i + 1].r_angstrom), None)
    assert i is not None
    low, high = data[i:i + 2]
    linear = _linear(low.r_angstrom, low.correction_hartree, high.r_angstrom, high.correction_hartree, target)
    # Select nearest three actual points without extrapolating target.
    index = min(max(i - (target - low.r_angstrom > high.r_angstrom - target), 0), len(data) - 3)
    quad = _quadratic(data[index:index + 3], target)
    loo = max(abs(data[j].correction_hartree - _linear(
        data[j - 1].r_angstrom, data[j - 1].correction_hartree,
        data[j + 1].r_angstrom, data[j + 1].correction_hartree,
        data[j].r_angstrom)) for j in range(1, len(data) - 1))
    return SpeciesTransfer(
        role=role, r_target_angstrom=target, correction_hartree=linear,
        linear_hartree=linear, quadratic_hartree=quad,
        local_model_disagreement_ev=abs(quad - linear) * HARTREE_TO_EV,
        max_observed_interior_loo_ev=loo * HARTREE_TO_EV,
        sample_bond_lengths_angstrom=tuple(p.r_angstrom for p in data),
        source_ids=tuple(source_ids),
    )


def evaluate_pec_correction_transfer(
    *, points: Sequence[MatchedCorrectionPoint],
    neutral_target_angstrom: float, anion_target_angstrom: float,
) -> PECTransferAssessment:
    """Interpolate two matched PEC correction grids at the two equilibrium R.

    Raises ValueError for invalid/mismatched evidence rather than returning an
    apparently precise number.  Returns review-only diagnostic otherwise.
    """
    neutral_points = [p for p in points if p.role == "neutral"]
    anion_points = [p for p in points if p.role == "anion"]
    if not neutral_points or not anion_points:
        raise ValueError("Both neutral and anion matched corrections required")
    if neutral_points[0].atoms != anion_points[0].atoms:
        raise ValueError("Neutral/anion atom ordering mismatch")
    if anion_points[0].charge != neutral_points[0].charge - 1:
        raise ValueError("Neutral/anion charge mismatch")
    if (neutral_points[0].lower_model_id, neutral_points[0].upper_model_id) != (
        anion_points[0].lower_model_id, anion_points[0].upper_model_id
    ):
        raise ValueError("Neutral/anion electronic model pair must be identical")
    all_sources = [source for p in points for source in (p.lower_source_id, p.upper_source_id)]
    if len(all_sources) != len(set(all_sources)):
        raise ValueError("Calculation evidence must not be reused across neutral/anion PECs")
    neutral = _species(neutral_points, "neutral", neutral_target_angstrom)
    anion = _species(anion_points, "anion", anion_target_angstrom)
    delta = (neutral.correction_hartree - anion.correction_hartree) * HARTREE_TO_EV
    spread = neutral.local_model_disagreement_ev + anion.local_model_disagreement_ev
    return PECTransferAssessment(
        status=TransferStatus.CANDIDATE_REVIEW_REQUIRED,
        neutral=neutral, anion=anion,
        delta_ea_candidate_ev=delta,
        observed_interpolation_sensitivity_ev=spread,
        reason="Numerical matched-geometry PEC transfer only; observed interpolation/model differences are not certified bounds",
    )
