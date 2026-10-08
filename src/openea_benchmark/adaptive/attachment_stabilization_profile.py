"""G2 scale-stabilization *diagnostics*, not continuum-exclusion evidence.

A stationary EA-EOM pseudostate in a finite Gaussian basis is not by itself
proof of an electron-bound state. This module deliberately cannot issue a
CLEARED Review or a terminal BOUND/UNBOUND outcome. It describes the observed
root-tracked spectrum and the adequacy of the exponent-scaling experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite, log
import json

from .attachment_eom_runner import G2EOMSeries, G2EOMStatus, G2EOMSubpoint
from .attachment_root_continuity import RootContinuityReport, RootContinuityStatus
from .model import Review, ReviewStatus


class StabilizationProfileStatus(str, Enum):
    OBSERVED_REVIEW_REQUIRED = "OBSERVED_REVIEW_REQUIRED"
    INSUFFICIENT_SCALE_COVERAGE = "INSUFFICIENT_SCALE_COVERAGE"
    ROOT_PATH_UNRESOLVED = "ROOT_PATH_UNRESOLVED"
    POSSIBLE_ROOT_MIXING = "POSSIBLE_ROOT_MIXING"


@dataclass(frozen=True)
class StabilizationProfileSettings:
    min_scales_below_one: int = 1
    min_scales_above_one: int = 1
    min_log_scale_span: float = 0.20
    root_gap_warning_ev: float = 0.02
    drift_warning_ev: float = 0.05

    def __post_init__(self) -> None:
        if self.min_scales_below_one < 1 or self.min_scales_above_one < 1:
            raise ValueError("Both sides of the unscaled baseline require explicit samples")
        for name in ("min_log_scale_span", "root_gap_warning_ev", "drift_warning_ev"):
            val = getattr(self, name)
            if not isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class StabilizationProfilePoint:
    request_key: str
    scale_factor: float
    root_index: int
    attachment_ea_ev: float
    delta_from_baseline_ev: float
    nearest_other_root_gap_ev: float | None
    algebraic_one_particle_fraction: float | None
    raw_evidence_id: str


@dataclass(frozen=True)
class StabilizationProfileReport:
    status: StabilizationProfileStatus
    baseline_key: str
    baseline_root_index: int | None
    points: tuple[StabilizationProfilePoint, ...]
    energy_span_ev: float | None
    max_abs_slope_ev_per_log_scale: float | None
    min_other_root_gap_ev: float | None
    flags: tuple[str, ...]
    review: Review
    evidence_id: str
    method_role: str = "DIAGNOSTIC"
    boundness_decision: str = "UNRESOLVED"

    def __post_init__(self) -> None:
        if self.review.status is not ReviewStatus.UNRESOLVED:
            raise ValueError("Exponent stabilization profiles cannot clear G2")
        if self.method_role != "DIAGNOSTIC" or self.boundness_decision != "UNRESOLVED":
            raise ValueError("A stabilization profile is not a boundness verdict")
        if not self.evidence_id.startswith("G2_STABILIZATION_PROFILE:"):
            raise ValueError("Invalid stabilization profile evidence namespace")


def _root_gap(sub: G2EOMSubpoint, root_index: int) -> float | None:
    roots = {r.root_index: r for r in sub.result.roots}
    selected = roots.get(root_index)
    if selected is None:
        raise ValueError(f"Root {root_index} missing from {sub.request.key}")
    gaps = [abs(selected.attachment_ea_ev - r.attachment_ea_ev)
            for i, r in roots.items() if i != root_index]
    return min(gaps) if gaps else None


def analyze_g2_stabilization_profile(
    series: G2EOMSeries,
    continuity: RootContinuityReport,
    *,
    settings: StabilizationProfileSettings = StabilizationProfileSettings(),
) -> StabilizationProfileReport:
    """Describe a root-*proposed* scaled-EOM curve with no continuum claim.

    The proposed path may be incomplete. The report then remains unresolved,
    rather than silently selecting an energy-ordered root as a substitute.
    Scaling must concern one reviewed selector policy at one fixed augmentation
    baseline, with identical source geometry and neutral-reference provenance.
    """
    if series.status is not G2EOMStatus.COMPLETE_ROOT_REVIEW_REQUIRED:
        raise ValueError("Incomplete EOM series cannot produce a stabilization profile")
    if continuity.review.status is not ReviewStatus.UNRESOLVED:
        raise ValueError("Root overlap proposals must not be pre-cleared")
    if not continuity.evidence_id.startswith("G2_ROOT_PROPOSAL:"):
        raise ValueError("Stabilization must consume an actual root-continuity proposal")
    by_key = {p.request.key: p for p in series.subpoints}
    if len(by_key) != len(series.subpoints):
        raise ValueError("Duplicate G2 request keys")
    scaled = [p for p in series.subpoints if p.request.scale_factor is not None]
    if not scaled:
        raise ValueError("No exponent scaling points available")
    if continuity.source_evidence_ids != tuple(sorted(p.evidence_id for p in series.subpoints)):
        raise ValueError("Root-continuity proposal does not match raw series evidence IDs")
    levels = {p.request.basis.augmentation_level for p in scaled}
    if len(levels) != 1:
        raise ValueError("Stabilization profile must use one fixed augmentation baseline")
    level = next(iter(levels))
    baseline_key = f"aug:{level}"
    if baseline_key not in by_key or by_key[baseline_key].request.scale_factor is not None:
        raise ValueError("Unscaled parent augmentation point missing")
    baseline = by_key[baseline_key]
    ref = baseline.request
    selectors = scaled[0].request.selectors
    if not selectors:
        raise ValueError("Stabilization requires reviewed explicit diffuse-shell selectors")
    keys = set()
    factors = set()
    for sub in scaled:
        req = sub.request
        if req.key in keys or req.scale_factor in factors:
            raise ValueError("Duplicate scale factor/request in stabilization scan")
        keys.add(req.key)
        factors.add(req.scale_factor)
        if (req.state != ref.state or req.source_checkpoint_sha256 != ref.source_checkpoint_sha256 or
            req.basis != ref.basis or req.selectors != selectors or
            sub.result.reference_kind != baseline.result.reference_kind or
            sub.result.pyscf_version != baseline.result.pyscf_version):
            raise ValueError("Incompatible basis, scaling selectors, neutral source or EOM backend")
        if req.scale_factor == 1.0:
            raise ValueError("Scale=1 duplicates the unscaled baseline; use the baseline record")
    roots = dict(continuity.proposed_roots)
    selection_complete = (continuity.status is RootContinuityStatus.CANDIDATE_REVIEW_REQUIRED and
                          baseline_key in roots and all(p.request.key in roots for p in scaled))
    points: list[StabilizationProfilePoint] = []
    flags: list[str] = []
    span = slope = min_gap = None
    baseline_root = roots.get(baseline_key) if selection_complete else None
    if not selection_complete:
        flags.append("ROOT_PATH_UNRESOLVED")
        status = StabilizationProfileStatus.ROOT_PATH_UNRESOLVED
    else:
        parent_root = next((r for r in baseline.result.roots if r.root_index == baseline_root), None)
        if parent_root is None:
            raise ValueError("Proposed baseline root absent from EOM result")
        for sub in sorted(scaled, key=lambda s: s.request.scale_factor):
            rid = roots[sub.request.key]
            root = next((r for r in sub.result.roots if r.root_index == rid), None)
            if root is None:
                raise ValueError("Proposed scaled root absent from EOM result")
            gap = _root_gap(sub, rid)
            points.append(StabilizationProfilePoint(
                sub.request.key, float(sub.request.scale_factor), rid,
                root.attachment_ea_ev, root.attachment_ea_ev - parent_root.attachment_ea_ev,
                gap, root.one_particle_amplitude_fraction, sub.evidence_id))
        curve = [(0.0, parent_root.attachment_ea_ev)] + [
            (log(p.scale_factor), p.attachment_ea_ev) for p in points]
        curve.sort()
        energies = [y for _, y in curve]
        span = max(energies) - min(energies)
        slope = max(abs((curve[i+1][1] - curve[i][1]) / (curve[i+1][0] - curve[i][0]))
                    for i in range(len(curve)-1))
        gaps = [g for g in [_root_gap(baseline, baseline_root)] +
                [p.nearest_other_root_gap_ev for p in points] if g is not None]
        min_gap = min(gaps) if gaps else None
        if min_gap is not None and min_gap <= settings.root_gap_warning_ev:
            flags.append("NEAR_DEGENERATE_EOM_ROOTS")
        if span >= settings.drift_warning_ev:
            flags.append("LARGE_SCALE_DEPENDENCE")
        left = sum(p.scale_factor < 1 for p in points)
        right = sum(p.scale_factor > 1 for p in points)
        if (left < settings.min_scales_below_one or right < settings.min_scales_above_one or
            max(x for x, _ in curve) - min(x for x, _ in curve) < settings.min_log_scale_span):
            flags.append("INSUFFICIENT_BIDIRECTIONAL_SCALE_COVERAGE")
            status = StabilizationProfileStatus.INSUFFICIENT_SCALE_COVERAGE
        elif "NEAR_DEGENERATE_EOM_ROOTS" in flags:
            status = StabilizationProfileStatus.POSSIBLE_ROOT_MIXING
        else:
            status = StabilizationProfileStatus.OBSERVED_REVIEW_REQUIRED
    signature = sha256(json.dumps({
        "root_proposal": continuity.evidence_id,
        "raw_ids": sorted(p.evidence_id for p in [baseline, *scaled]),
        "scales": sorted(factors),
        "root_indices": [(p.request_key, p.root_index) for p in points],
        "baseline_energy": None if baseline_root is None else (baseline_key, parent_root.attachment_ea_ev),
        "energies": [(p.request_key, p.attachment_ea_ev, p.nearest_other_root_gap_ev,
                      p.algebraic_one_particle_fraction) for p in points],
        "settings": (settings.min_scales_below_one, settings.min_scales_above_one,
                     settings.min_log_scale_span, settings.root_gap_warning_ev, settings.drift_warning_ev),
    }, sort_keys=True, allow_nan=False).encode()).hexdigest()
    evidence_id = f"G2_STABILIZATION_PROFILE:{signature}"
    reason = ("Finite-basis exponent-scaling is a diagnostic only. Flat curves, "
              "negative EA, and AO root-overlap proposals do not alone exclude continuum pseudostates; "
              "independent continuum-discrimination review is required.")
    review = Review(ReviewStatus.UNRESOLVED, (evidence_id, continuity.evidence_id), reason)
    return StabilizationProfileReport(status, baseline_key, baseline_root,
                                      tuple(points), span, slope, min_gap,
                                      tuple(flags), review, evidence_id)
