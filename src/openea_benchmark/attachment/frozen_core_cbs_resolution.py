"""Resolve the corrected frozen-core valence CBS baseline for OpenEA.

Consumes the six-point frozen-core evidence run:
neutral/anion x aug-QZ, aug-5Z, d-aug-5Z.

The cardinal QZ/5Z extrapolation reuses the component-resolved CBS model.
The d-aug-minus-aug correction is measured directly in the same frozen-core
correlation space.

The older all-electron diffuse-axis residual may be carried only as
INDIRECTLY_ESTIMATED uncertainty evidence; it is never treated as a direct
frozen-core convergence test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from openea_benchmark.attachment.component_resolved_cbs import (
    HARTREE_TO_EV,
    resolve_cbs,
)


@dataclass(frozen=True)
class FrozenCoreCBSResolved:
    status: str
    cardinal_status: str
    uncertainty_status: str
    ea_aug_cbs_ev: float
    diffuse_correction_ev: float
    ea_cbs_plus_diffuse_ev: float
    cbs_model_sensitivity_bound_ev: float
    indirect_diffuse_residual_ev: float | None
    conservative_half_width_ev: float
    lower_ev: float
    upper_ev: float
    evidence: tuple[str, ...]
    next_actions: tuple[str, ...]
    is_production_ea: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find(points: list[Mapping[str, Any]], role: str, basis: str) -> Mapping[str, Any]:
    hits = [p for p in points if p["role"] == role and p["basis"] == basis]
    if len(hits) != 1:
        raise ValueError(f"expected one point for {role}:{basis}, found {len(hits)}")
    if not bool(hits[0].get("reusable")):
        raise ValueError(f"point not reusable: {role}:{basis}")
    return hits[0]


def fixed_ea(points: list[Mapping[str, Any]], basis: str) -> float:
    n = float(_find(points, "neutral", basis)["ccsd_t_total_hartree"])
    a = float(_find(points, "anion", basis)["ccsd_t_total_hartree"])
    return (n - a) * HARTREE_TO_EV


def resolve_frozen_core_cbs(
    *,
    points: list[Mapping[str, Any]],
    indirect_diffuse_residual_ev: float | None,
    model_spread_target_ev: float = 0.005,
) -> FrozenCoreCBSResolved:
    aug5 = fixed_ea(points, "aug-cc-pv5z")
    daug5 = fixed_ea(points, "d-aug-cc-pv5z")

    cardinal_points = [
        p for p in points
        if p["basis"] in {"aug-cc-pvqz", "aug-cc-pv5z"}
    ]
    base = resolve_cbs(
        points=cardinal_points,
        daug_fixed_ea_ev=daug5,
        full_pec_aug5_ea_ev=None,
        diffuse_residual_estimate_ev=None,
        model_spread_target_ev=model_spread_target_ev,
    )

    direct_diff = daug5 - aug5
    if abs(direct_diff - base.diffuse_correction_ev) > 1e-10:
        raise RuntimeError("internal diffuse correction mismatch")

    half = float(base.cbs_model_sensitivity_bound_ev)
    evidence = [
        "FROZEN_CORE_CORRELATION_SPACE_EXPLICIT",
        "QZ_5Z_COMPONENT_EVIDENCE_COMPLETE",
        "SCF_CCSD_TRIPLES_EXTRAPOLATED_SEPARATELY",
        "FROZEN_CORE_DAUG_MINUS_AUG_DIRECTLY_TESTED",
        "NO_BASIS_AVERAGING",
    ]

    if indirect_diffuse_residual_ev is not None:
        indirect = abs(float(indirect_diffuse_residual_ev))
        half += indirect
        uncertainty_status = "CONSERVATIVE_WITH_INDIRECT_DIFFUSE_RESIDUAL"
        evidence.append(
            "DIFFUSE_RESIDUAL_INDIRECTLY_ESTIMATED_FROM_PRIOR_ALL_ELECTRON_AXIS"
        )
    else:
        indirect = None
        uncertainty_status = "INCOMPLETE_DIFFUSE_RESIDUAL"
        evidence.append("FROZEN_CORE_DIFFUSE_RESIDUAL_UNKNOWN")

    center = float(base.ea_cbs_plus_diffuse_primary_ev)

    cardinal_status = base.status.value
    status = (
        "READY_FOR_CORE_VALENCE"
        if cardinal_status == "CLEARED"
        else "NEED_MORE_CARDINAL_EVIDENCE"
    )
    if status == "READY_FOR_CORE_VALENCE":
        next_actions = (
            "COMPUTE_CORE_VALENCE_CORRECTION_WITH_CORE_VALENCE_BASIS",
            "KEEP_FROZEN_CORE_CBS_AS_BASELINE",
        )
    else:
        next_actions = (
            "DO_NOT_COMPUTE_ADDITIVE_CORRECTIONS_YET",
            "RESOLVE_CARDINAL_CBS_MODEL",
        )

    return FrozenCoreCBSResolved(
        status=status,
        cardinal_status=cardinal_status,
        uncertainty_status=uncertainty_status,
        ea_aug_cbs_ev=float(base.primary_model.ea_cbs_aug_ev),
        diffuse_correction_ev=direct_diff,
        ea_cbs_plus_diffuse_ev=center,
        cbs_model_sensitivity_bound_ev=float(base.cbs_model_sensitivity_bound_ev),
        indirect_diffuse_residual_ev=indirect,
        conservative_half_width_ev=half,
        lower_ev=center-half,
        upper_ev=center+half,
        evidence=tuple(evidence),
        next_actions=next_actions,
        is_production_ea=False,
    )
