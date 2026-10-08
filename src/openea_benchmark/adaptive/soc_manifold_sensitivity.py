"""Observed state-manifold sensitivity of a real SOC calculation.

The comparison is deliberately narrower than an SOC convergence proof: same
geometry, Hamiltonian, CAS and ground-spin orbital-optimization protocol; only
additional spin-free states are permitted. Even a zero shift has no error bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable

from .soc_fci_siso_runner import (
    HARTREE_TO_EV, SOCPointAuthorization, SOCPointRequest, SOCPointResult,
    SOCPointSettings, SOCPointStatus, run_soc_fci_siso_point,
)


class SOCManifoldSensitivityStatus(str, Enum):
    CANDIDATE_REVIEW_REQUIRED = "CANDIDATE_REVIEW_REQUIRED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class SOCManifoldSensitivitySettings:
    # This checks whether the *retained* spin-free energies agree, not whether
    # uncalculated SOC states or the SOC model are converged.
    max_retained_spin_free_drift_hartree: float = 1.0e-5

    def __post_init__(self) -> None:
        if not isfinite(self.max_retained_spin_free_drift_hartree) or not (
            0 < self.max_retained_spin_free_drift_hartree <= 1.0e-3
        ):
            raise ValueError("Invalid spin-free drift tolerance")


@dataclass(frozen=True)
class SOCManifoldSensitivity:
    status: SOCManifoldSensitivityStatus
    reason: str
    base_request_id: str
    expanded_request_id: str
    retained_spin_free_max_drift_hartree: float | None = None
    base_soc_shift_hartree: float | None = None
    expanded_soc_shift_hartree: float | None = None
    observed_shift_change_hartree: float | None = None
    observed_shift_change_ev: float | None = None
    spin_free_state_identity_cleared: bool = False
    spin_manifold_complete: bool = False
    soc_uncertainty_bounded: bool = False
    production_soc_validated: bool = False

    def __post_init__(self) -> None:
        if (self.spin_free_state_identity_cleared or self.spin_manifold_complete or
                self.soc_uncertainty_bounded or self.production_soc_validated):
            raise ValueError("A manifold-sensitivity test cannot certify SOC")
        numeric = (self.base_soc_shift_hartree, self.expanded_soc_shift_hartree,
                   self.observed_shift_change_hartree, self.observed_shift_change_ev)
        if self.status is SOCManifoldSensitivityStatus.UNRESOLVED and any(
            x is not None for x in numeric
        ):
            raise ValueError("Unresolved SOC sensitivity cannot advertise a shift")
        if self.status is SOCManifoldSensitivityStatus.CANDIDATE_REVIEW_REQUIRED:
            if any(x is None or not isfinite(x) for x in numeric):
                raise ValueError("Completed comparison requires finite observed shifts")
            if abs(self.expanded_soc_shift_hartree - self.base_soc_shift_hartree -
                   self.observed_shift_change_hartree) > 1.e-10:
                raise ValueError("Inconsistent SOC shift change")
            if abs(self.observed_shift_change_ev - self.observed_shift_change_hartree *
                   HARTREE_TO_EV) > 1.e-9:
                raise ValueError("Inconsistent SOC energy conversion")
        if self.retained_spin_free_max_drift_hartree is not None and (
            not isfinite(self.retained_spin_free_max_drift_hartree) or
            self.retained_spin_free_max_drift_hartree < 0
        ):
            raise ValueError("Invalid retained-root drift")


def _unresolved(base: SOCPointRequest, expanded: SOCPointRequest,
                reason: str, drift: float | None = None) -> SOCManifoldSensitivity:
    return SOCManifoldSensitivity(SOCManifoldSensitivityStatus.UNRESOLVED,
                                  reason, base.point.request_id,
                                  expanded.point.request_id,
                                  retained_spin_free_max_drift_hartree=drift)


def compare_soc_manifold_expansion(
    base_request: SOCPointRequest, base: SOCPointResult,
    expanded_request: SOCPointRequest, expanded: SOCPointResult,
    *, settings: SOCManifoldSensitivitySettings | None = None,
) -> SOCManifoldSensitivity:
    """Compare actual spin-manifold extensions, not arbitrary different SOC models.

    Ground-spin CAS optimizations must have the same root count.  Other sectors
    may gain roots or new spin sectors.  Root correspondence within a retained
    sector uses *energy order only*: a candidate, never a proof of identity.
    """
    settings = settings or SOCManifoldSensitivitySettings()
    if base_request.point != expanded_request.point:
        return _unresolved(base_request, expanded_request,
                           "CAS, geometry, source checkpoint, or electronic model differs")
    if (base_request.ground_spin_2s != expanded_request.ground_spin_2s or
            base_request.ground_state_review_ids != expanded_request.ground_state_review_ids):
        return _unresolved(base_request, expanded_request,
                           "Ground-spin reference or review provenance differs")
    b = {m.spin_2s: m.nroots for m in base_request.spin_manifolds}
    e = {m.spin_2s: m.nroots for m in expanded_request.spin_manifolds}
    if (not all(s in e and e[s] >= n for s, n in b.items()) or
            e[base_request.ground_spin_2s] != b[base_request.ground_spin_2s] or
            sum(e.values()) <= sum(b.values())):
        return _unresolved(base_request, expanded_request,
                           "SOC spin-free roots must strictly expand without changing CASSCF ground-spin roots")
    if base.status is not SOCPointStatus.COMPLETE_REVIEW_REQUIRED or (
        expanded.status is not SOCPointStatus.COMPLETE_REVIEW_REQUIRED
    ):
        return _unresolved(base_request, expanded_request,
                           "Both authorized SOC backends must complete")
    if (base.request_id != base_request.point.request_id or
            expanded.request_id != expanded_request.point.request_id):
        return _unresolved(base_request, expanded_request,
                           "SOC request/result identity mismatch")
    if (base.scalar_hamiltonian != expanded.scalar_hamiltonian or
            not base.result_signature or not expanded.result_signature or
            base.result_signature == expanded.result_signature or
            not base.source_checkpoint_sha256 or
            base.source_checkpoint_sha256 != expanded.source_checkpoint_sha256 or
            not base.fci_siso_source_sha256 or
            base.fci_siso_source_sha256 != expanded.fci_siso_source_sha256):
        return _unresolved(base_request, expanded_request,
                           "Missing/mismatched source, SOC Hamiltonian, or backend provenance")
    # Require exact state coverage and spin projection count, not merely two
    # apparently plausible ground energies.
    for req, result in ((base_request, base), (expanded_request, expanded)):
        expected = {m.spin_2s: m.nroots for m in req.spin_manifolds}
        counts = {s: sum(r.spin_2s == s for r in result.spin_free_roots)
                  for s in expected}
        if counts != expected or len(result.spin_free_roots) != sum(expected.values()):
            return _unresolved(base_request, expanded_request,
                               "Missing or mislabeled spin-free roots")
        if len(result.soc_energies_hartree) != sum(
            m.nroots * (m.spin_2s + 1) for m in req.spin_manifolds
        ):
            return _unresolved(base_request, expanded_request,
                               "Incomplete spin-projected SOC spectrum")
    drifts = []
    for spin, n in b.items():
        # Sorting within each spin is only an energy-ordered fingerprint; near
        # crossings/couplings still need independent many-electron review.
        orig = sorted(r.energy_hartree for r in base.spin_free_roots if r.spin_2s == spin)
        later = sorted(r.energy_hartree for r in expanded.spin_free_roots if r.spin_2s == spin)
        drifts.extend(abs(x-y) for x,y in zip(orig[:n], later[:n]))
    max_drift = max(drifts)
    if max_drift > settings.max_retained_spin_free_drift_hartree:
        return _unresolved(base_request, expanded_request,
                           "Retained spin-free roots changed beyond numerical tolerance", max_drift)
    diff = expanded.delta_soc_hartree - base.delta_soc_hartree
    return SOCManifoldSensitivity(
        SOCManifoldSensitivityStatus.CANDIDATE_REVIEW_REQUIRED,
        "Observed state-manifold sensitivity only; not an upper bound on missing SOC or a validated EA correction",
        base_request.point.request_id, expanded_request.point.request_id,
        retained_spin_free_max_drift_hartree=max_drift,
        base_soc_shift_hartree=base.delta_soc_hartree,
        expanded_soc_shift_hartree=expanded.delta_soc_hartree,
        observed_shift_change_hartree=diff,
        observed_shift_change_ev=diff * HARTREE_TO_EV,
    )


def run_soc_manifold_expansion(
    base_request: SOCPointRequest, expanded_request: SOCPointRequest,
    base_auth: SOCPointAuthorization, expanded_auth: SOCPointAuthorization,
    *, point_settings: SOCPointSettings,
    comparison_settings: SOCManifoldSensitivitySettings | None = None,
    backend: Callable | None = None,
) -> SOCManifoldSensitivity:
    """Explicitly execute *both* independently authorized full SOC calculations."""
    # Reject non-nested input before running expensive first job.
    b = {m.spin_2s: m.nroots for m in base_request.spin_manifolds}
    e = {m.spin_2s: m.nroots for m in expanded_request.spin_manifolds}
    if (base_request.point != expanded_request.point or
        base_request.ground_state_review_ids != expanded_request.ground_state_review_ids or
        base_request.ground_spin_2s != expanded_request.ground_spin_2s or
        not all(k in e and e[k]>=v for k,v in b.items()) or
        e.get(base_request.ground_spin_2s) != b.get(base_request.ground_spin_2s) or
        sum(e.values())<=sum(b.values())):
        return _unresolved(base_request, expanded_request,
                           "Invalid or non-nested spin-manifold expansion")
    if not base_auth.authorized or not expanded_auth.authorized:
        return _unresolved(base_request, expanded_request,
                           "Both SOC jobs require independent authorization")
    x = run_soc_fci_siso_point(base_request, base_auth,
                               settings=point_settings, backend=backend)
    if x.status is not SOCPointStatus.COMPLETE_REVIEW_REQUIRED:
        return _unresolved(base_request, expanded_request,
                           f"Base SOC run failed: {x.status.value}")
    y = run_soc_fci_siso_point(expanded_request, expanded_auth,
                               settings=point_settings, backend=backend)
    return compare_soc_manifold_expansion(base_request, x, expanded_request, y,
                                          settings=comparison_settings)
