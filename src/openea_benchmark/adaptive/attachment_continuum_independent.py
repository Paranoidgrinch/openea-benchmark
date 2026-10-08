"""External continuum-method validation and CAP-EA trajectory diagnostics.

Finite Gaussian attachment energies and exponent-stabilization profiles cannot
independently certify exclusion of the electron continuum.  In particular, a
CAP-EOM complex resonance energy needs a *reviewed* threshold, root tracking,
CAP strength trajectory, deperturbation and basis convergence.  This module
never derives a CLEARED continuum review from a trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite
import json

from .model import Review, ReviewStatus


class ContinuumMethod(str, Enum):
    CAP_EOM_EA_CCSD = "CAP_EOM_EA_CCSD"
    ELECTRON_SCATTERING = "ELECTRON_SCATTERING"
    R_MATRIX = "R_MATRIX"
    RIGOROUS_SPECTRAL = "RIGOROUS_SPECTRAL"


class ContinuumScope(str, Enum):
    IDENTIFIED_ROOT = "IDENTIFIED_ROOT"
    ALL_RELEVANT_STATES = "ALL_RELEVANT_STATES"


class ContinuumFinding(str, Enum):
    BOUND_IDENTIFIED_STATE = "BOUND_IDENTIFIED_STATE"
    RESONANT_IDENTIFIED_STATE = "RESONANT_IDENTIFIED_STATE"
    NO_BOUND_STATES_IN_REVIEWED_SECTOR = "NO_BOUND_STATES_IN_REVIEWED_SECTOR"


@dataclass(frozen=True)
class IndependentContinuumDossier:
    """Out-of-band scientific attestation, NEVER generated from Gaussian EOM.

    ``NO_BOUND_STATES_IN_REVIEWED_SECTOR`` requires exhaustive *physical*
    coverage of the electronic sector containing the candidate ground state.
    Finding a resonance for one EOM root is NOT that statement.

    CLEARED records still represent a human/external scientific judgement;
    type validation is provenance hygiene, not proof of its scientific truth.
    """

    method: ContinuumMethod
    scope: ContinuumScope
    finding: ContinuumFinding
    system: str
    neutral_state_id: str
    source_root_id: str
    r_angstrom: float
    detachment_threshold_id: str
    raw_method_evidence_ids: tuple[str, ...]
    method_validity_review: Review
    threshold_review: Review
    root_identity_review: Review
    completeness_review: Review
    scientific_review: Review
    reviewed_sector_id: str = ""
    state_inventory_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(s.strip() for s in (self.system, self.neutral_state_id,
                                      self.source_root_id, self.detachment_threshold_id)):
            raise ValueError("Continuum dossier requires explicit candidate/threshold identity")
        if not isfinite(self.r_angstrom) or self.r_angstrom <= 0:
            raise ValueError("Continuum dossier requires a positive matching geometry")
        if not self.raw_method_evidence_ids or any(not s.strip() for s in self.raw_method_evidence_ids):
            raise ValueError("Continuum dossier requires actual method-specific source data")
        prohibited = ("G2_EOM:", "G2_ROOT_PROPOSAL:", "G2_STABILIZATION_PROFILE:",
                      "G2_CAP_TRAJECTORY:")
        if any(s.startswith(prohibited) for s in self.raw_method_evidence_ids):
            raise ValueError("Derived diagnostic identifiers are not independent physical evidence")
        # CAP-EOM can establish a resonance *candidate* of a selected root,
        # not an exhaustive sector search or L2 boundness proof on its own.
        if (self.method is ContinuumMethod.CAP_EOM_EA_CCSD and
                self.finding is not ContinuumFinding.RESONANT_IDENTIFIED_STATE):
            raise ValueError("CAP-EOM alone cannot certify all-root no-bound or L2 boundness")
        if self.finding is ContinuumFinding.NO_BOUND_STATES_IN_REVIEWED_SECTOR:
            if self.scope is not ContinuumScope.ALL_RELEVANT_STATES:
                raise ValueError("An isolated resonance cannot exclude all bound states")
            if self.completeness_review.status is not ReviewStatus.CLEARED:
                raise ValueError("No-bound assertion requires independently reviewed state completeness")
            if not self.reviewed_sector_id.strip() or not self.state_inventory_evidence_ids or any(
                not x.strip() for x in self.state_inventory_evidence_ids
            ):
                raise ValueError("Global no-bound assertion requires full sector and state inventory provenance")
        elif self.scope is not ContinuumScope.IDENTIFIED_ROOT:
            raise ValueError("Single-state findings must have IDENTIFIED_ROOT scope")
        for name in ("method_validity_review", "threshold_review", "root_identity_review",
                     "scientific_review"):
            review = getattr(self, name)
            if review.status is not ReviewStatus.CLEARED:
                raise ValueError(f"{name} must be explicitly CLEARED")
        # Synthetic status-only assertions and recycling raw EOM evidence are forbidden.
        all_ids = tuple(x for r in (self.method_validity_review, self.threshold_review,
                                    self.root_identity_review, self.scientific_review)
                        for x in r.evidence_ids)
        if not all_ids or any(x.startswith(prohibited) for x in all_ids):
            raise ValueError("Independent review evidence cannot be raw G2 diagnostics")

    def matches(self, system: str, neutral_state_id: str,
                source_root_id: str, r_angstrom: float) -> bool:
        return (self.system == system and self.neutral_state_id == neutral_state_id and
                self.source_root_id == source_root_id and
                abs(self.r_angstrom - r_angstrom) < 1.0e-9)


class CAPTrajectoryStatus(str, Enum):
    RESONANCE_CANDIDATE_REVIEW_REQUIRED = "RESONANCE_CANDIDATE_REVIEW_REQUIRED"
    SUBTHRESHOLD_CANDIDATE_REVIEW_REQUIRED = "SUBTHRESHOLD_CANDIDATE_REVIEW_REQUIRED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INCONSISTENT = "INCONSISTENT"


@dataclass(frozen=True)
class CAPPoint:
    """Deperturbed complex EOM attachment energy above neutral + free e-.

    ``imag_energy_ev`` is Im(E_anion - E_neutral). A physical resonance width
    is Γ = -2 Im(E); units for position, imaginary part and Γ are eV.
    The method *supplying* the CAP energy is external to this module.
    """
    basis_id: str
    eta: float
    position_ev: float
    imag_energy_ev: float
    first_order_deperturbed: bool
    source_evidence_id: str

    def __post_init__(self) -> None:
        if not self.basis_id.strip() or not self.source_evidence_id.strip():
            raise ValueError("CAP point needs basis and source evidence")
        if not all(isfinite(v) for v in (self.eta, self.position_ev, self.imag_energy_ev)):
            raise ValueError("CAP energies and strength must be finite")
        if self.eta <= 0 or self.imag_energy_ev > 1e-10:
            raise ValueError("CAP eta must be positive and absorptive Im(E) <= 0")

    @property
    def width_ev(self) -> float:
        return max(0.0, -2.0 * self.imag_energy_ev)


@dataclass(frozen=True)
class CAPTrajectorySettings:
    minimum_eta_points_per_basis: int = 3
    minimum_independent_bases: int = 2
    max_position_range_ev: float = 0.10
    max_width_range_ev: float = 0.10
    min_resolved_width_ev: float = 0.001

    def __post_init__(self) -> None:
        if self.minimum_eta_points_per_basis < 3 or self.minimum_independent_bases < 2:
            raise ValueError("CAP trajectory requires >=3 eta points per >=2 bases")
        for name in ("max_position_range_ev", "max_width_range_ev", "min_resolved_width_ev"):
            val = getattr(self, name)
            if not isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class CAPTrajectoryReport:
    status: CAPTrajectoryStatus
    energy_range_ev: tuple[float, float] | None
    width_range_ev: tuple[float, float] | None
    evidence_id: str
    review: Review
    reason_codes: tuple[str, ...]
    method_role: str = "DIAGNOSTIC"
    boundness_decision: str = "UNRESOLVED"

    def __post_init__(self) -> None:
        if self.review.status is not ReviewStatus.UNRESOLVED:
            raise ValueError("A finite CAP scan never independently closes G2")
        if self.method_role != "DIAGNOSTIC" or self.boundness_decision != "UNRESOLVED":
            raise ValueError("CAP diagnostics are not terminal boundness assertions")


def assess_cap_trajectory(
    points: tuple[CAPPoint, ...], *,
    threshold_review: Review,
    root_tracking_review: Review,
    cap_method_review: Review,
    settings: CAPTrajectorySettings = CAPTrajectorySettings(),
) -> CAPTrajectoryReport:
    """Evaluate precomputed CAP trajectory, without auto-certifying boundness.

    A CAP-EOM result is a resonance diagnostic only after additional method-
    specific scrutiny (CAP onset, eta optimization/deperturbation, analytic
    continuation, numerical convergence, and physical channel completeness).
    Merely reporting a small imaginary energy does not certify an L2 state.
    """
    raw = tuple(sorted(points, key=lambda p: (p.basis_id, p.eta)))
    digest = sha256(json.dumps({"points": [
        (p.basis_id, p.eta, p.position_ev, p.imag_energy_ev,
         p.first_order_deperturbed, p.source_evidence_id) for p in raw],
        "settings": (settings.minimum_eta_points_per_basis, settings.minimum_independent_bases,
                     settings.max_position_range_ev, settings.max_width_range_ev,
                     settings.min_resolved_width_ev),
        "reviews": [(r.status.value, r.evidence_ids) for r in
                    (threshold_review, root_tracking_review, cap_method_review)],
    }, sort_keys=True).encode()).hexdigest()
    evidence_id = f"G2_CAP_TRAJECTORY:{digest}"
    by_basis: dict[str, list[CAPPoint]] = {}
    for p in raw:
        by_basis.setdefault(p.basis_id, []).append(p)
    reasons: list[str] = []
    if len(by_basis) < settings.minimum_independent_bases:
        reasons.append("INDEPENDENT_BASIS_SERIES_MISSING")
    for basis_id, group in by_basis.items():
        if len(group) < settings.minimum_eta_points_per_basis or len(set(p.eta for p in group)) != len(group):
            reasons.append(f"INSUFFICIENT_OR_DUPLICATE_ETA:{basis_id}")
    if not raw or any(not p.first_order_deperturbed for p in raw):
        reasons.append("FIRST_ORDER_CAP_DEPERTURBATION_MISSING")
    if any(r.status is not ReviewStatus.CLEARED for r in
           (threshold_review, root_tracking_review, cap_method_review)):
        reasons.append("CAP_THRESHOLD_ROOT_OR_METHOD_NOT_REVIEWED")
    if reasons:
        return CAPTrajectoryReport(
            CAPTrajectoryStatus.INSUFFICIENT_EVIDENCE, None, None, evidence_id,
            Review(ReviewStatus.UNRESOLVED, (evidence_id,), "CAP series is incomplete or lacks required scientific reviews."),
            tuple(reasons))
    energies = [p.position_ev for p in raw]
    widths = [p.width_ev for p in raw]
    erange = (min(energies), max(energies))
    wrange = (min(widths), max(widths))
    if (erange[1] - erange[0] > settings.max_position_range_ev or
            wrange[1] - wrange[0] > settings.max_width_range_ev):
        status = CAPTrajectoryStatus.INCONSISTENT
        reasons.append("CAP_ETA_OR_BASIS_SENSITIVITY")
    elif erange[0] > 0 and wrange[0] >= settings.min_resolved_width_ev:
        status = CAPTrajectoryStatus.RESONANCE_CANDIDATE_REVIEW_REQUIRED
        reasons.append("POSITIVE_RESONANCE_POSITION_AND_FINITE_WIDTH")
    elif erange[1] < 0 and wrange[1] < settings.min_resolved_width_ev:
        status = CAPTrajectoryStatus.SUBTHRESHOLD_CANDIDATE_REVIEW_REQUIRED
        reasons.append("SUBTHRESHOLD_NUMERICAL_CANDIDATE_NOT_CERTIFIED_BOUND")
    else:
        status = CAPTrajectoryStatus.INCONSISTENT
        reasons.append("THRESHOLD_CROSSING_OR_UNRESOLVED_WIDTH")
    return CAPTrajectoryReport(
        status, erange, wrange, evidence_id,
        Review(ReviewStatus.UNRESOLVED, (evidence_id,),
               "CAP trajectory is a candidate diagnostic; independent continuum validation and state completeness remain required."),
        tuple(reasons))
