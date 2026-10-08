"""Policy/trajectory tests; no CAP electronic-structure calculation is run."""
from dataclasses import replace

import pytest

from openea_benchmark.adaptive.model import Review, ReviewStatus
from openea_benchmark.adaptive.attachment_continuum_independent import (
    CAPPoint, CAPTrajectorySettings, CAPTrajectoryStatus,
    ContinuumFinding, ContinuumMethod, ContinuumScope,
    IndependentContinuumDossier, assess_cap_trajectory,
)


def reviewed(tag):
    return Review(ReviewStatus.CLEARED, (f"external:{tag}",), f"Reviewed {tag}")


def unresolved():
    return Review(ReviewStatus.UNRESOLVED, (), "not yet reviewed")


def sample_points(pos=0.30, imag=-0.01):
    return tuple(
        CAPPoint(basis, eta, pos + offset, imag + change, True,
                 f"CAP_RAW:{basis}:{eta}")
        for basis, offset in (("basis_T", -0.01), ("basis_Q", 0.01))
        for eta, change in ((0.005, -0.0005), (0.010, 0), (0.020, 0.0005))
    )


def analysis(points, root=True, method=True, threshold=True):
    return assess_cap_trajectory(
        points,
        threshold_review=reviewed("threshold") if threshold else unresolved(),
        root_tracking_review=reviewed("root") if root else unresolved(),
        cap_method_review=reviewed("method") if method else unresolved(),
    )


def dossier(finding=ContinuumFinding.BOUND_IDENTIFIED_STATE,
            scope=ContinuumScope.IDENTIFIED_ROOT):
    return IndependentContinuumDossier(
        ContinuumMethod.ELECTRON_SCATTERING, scope, finding,
        "AB", "neutral-1", "source-1", 1.8, "threshold-1",
        ("SCATTERING:raw-numerical-data",),
        reviewed("method"), reviewed("threshold"), reviewed("root"),
        reviewed("complete"), reviewed("scientific"),
        reviewed_sector_id="all-relevant-spin-symmetries",
        state_inventory_evidence_ids=("SCATTERING:channel-root-inventory",),
    )


def test_cap_resonance_trajectory_is_diagnostic_only():
    r = analysis(sample_points())
    assert r.status is CAPTrajectoryStatus.RESONANCE_CANDIDATE_REVIEW_REQUIRED
    assert r.energy_range_ev[0] > 0 and r.width_range_ev[0] > 0
    assert r.review.status is ReviewStatus.UNRESOLVED
    assert r.method_role == "DIAGNOSTIC" and r.boundness_decision == "UNRESOLVED"
    assert r.evidence_id.startswith("G2_CAP_TRAJECTORY:")


def test_cap_width_is_minus_twice_imaginary_energy():
    p = CAPPoint("B", 0.01, 0.2, -0.003, True, "CAP_RAW:1")
    assert p.width_ev == pytest.approx(0.006)


def test_cap_subthreshold_zero_width_is_still_not_bound_proof():
    r = analysis(tuple(replace(p, imag_energy_ev=-1e-5) for p in sample_points(pos=-0.2)))
    assert r.status is CAPTrajectoryStatus.SUBTHRESHOLD_CANDIDATE_REVIEW_REQUIRED
    assert r.review.status is ReviewStatus.UNRESOLVED


@pytest.mark.parametrize("kw", [
    {"root": False}, {"method": False}, {"threshold": False},
])
def test_unreviewed_provenance_blocks_trajectory(kw):
    r = analysis(sample_points(), **kw)
    assert r.status is CAPTrajectoryStatus.INSUFFICIENT_EVIDENCE
    assert "CAP_THRESHOLD_ROOT_OR_METHOD_NOT_REVIEWED" in r.reason_codes


def test_only_one_basis_fails_closed():
    r = analysis(tuple(p for p in sample_points() if p.basis_id == "basis_T"))
    assert r.status is CAPTrajectoryStatus.INSUFFICIENT_EVIDENCE


def test_two_eta_points_insufficient():
    r = analysis(tuple(p for p in sample_points() if p.eta != 0.005))
    assert r.status is CAPTrajectoryStatus.INSUFFICIENT_EVIDENCE


def test_uncorrected_cap_energy_is_not_accepted():
    pts = list(sample_points())
    pts[0] = replace(pts[0], first_order_deperturbed=False)
    assert analysis(tuple(pts)).status is CAPTrajectoryStatus.INSUFFICIENT_EVIDENCE


def test_basis_and_eta_instability_is_not_smoothed_away():
    pts = list(sample_points())
    pts[0] = replace(pts[0], position_ev=0.85)
    r = analysis(tuple(pts))
    assert r.status is CAPTrajectoryStatus.INCONSISTENT


def test_threshold_crossing_is_inconclusive():
    assert analysis(sample_points(pos=0.0, imag=-0.01)).status is CAPTrajectoryStatus.INCONSISTENT


def test_point_validation_rejects_negative_strength_or_positive_imaginary():
    with pytest.raises(ValueError):
        CAPPoint("B", -0.1, 1., -0.01, True, "CAP_RAW:1")
    with pytest.raises(ValueError):
        CAPPoint("B", 0.1, 1., 0.01, True, "CAP_RAW:1")


def test_duplicate_eta_cannot_fake_number_of_points():
    pts = list(sample_points())
    pts[1] = replace(pts[1], eta=pts[0].eta)
    assert analysis(tuple(pts)).status is CAPTrajectoryStatus.INSUFFICIENT_EVIDENCE


def test_dossier_validates_matching_target_and_geometry():
    d = dossier()
    assert d.matches("AB", "neutral-1", "source-1", 1.8)
    assert not d.matches("AB", "neutral-1", "other", 1.8)
    assert not d.matches("AB", "neutral-1", "source-1", 1.9)


def test_resonance_root_is_not_a_global_no_bound_assertion():
    with pytest.raises(ValueError, match="isolated resonance"):
        replace(dossier(), finding=ContinuumFinding.NO_BOUND_STATES_IN_REVIEWED_SECTOR)


def test_global_no_bound_needs_complete_inventory():
    good = dossier(ContinuumFinding.NO_BOUND_STATES_IN_REVIEWED_SECTOR,
                   ContinuumScope.ALL_RELEVANT_STATES)
    with pytest.raises(ValueError, match="state inventory"):
        replace(good, state_inventory_evidence_ids=())
    with pytest.raises(ValueError, match="sector"):
        replace(good, reviewed_sector_id="")
    with pytest.raises(ValueError, match="completeness"):
        replace(good, completeness_review=unresolved())


def test_internal_g2_eom_root_cap_profile_is_not_independent_source():
    for tag in ("G2_EOM:123", "G2_ROOT_PROPOSAL:123", "G2_CAP_TRAJECTORY:123",
                "G2_STABILIZATION_PROFILE:123"):
        with pytest.raises(ValueError, match="not independent"):
            replace(dossier(), raw_method_evidence_ids=(tag,))


def test_unreviewed_scattering_claim_is_rejected():
    with pytest.raises(ValueError, match="scientific_review"):
        replace(dossier(), scientific_review=unresolved())


def test_method_contract_requires_real_threshold_source():
    with pytest.raises(ValueError):
        replace(dossier(), detachment_threshold_id="")


def test_cap_dossier_is_resonance_only_even_if_human_review_claims_bound():
    with pytest.raises(ValueError, match="CAP-EOM alone"):
        replace(dossier(), method=ContinuumMethod.CAP_EOM_EA_CCSD)
    with pytest.raises(ValueError, match="CAP-EOM alone"):
        replace(dossier(ContinuumFinding.NO_BOUND_STATES_IN_REVIEWED_SECTOR,
                        ContinuumScope.ALL_RELEVANT_STATES),
                method=ContinuumMethod.CAP_EOM_EA_CCSD)


def test_cap_dossier_can_record_one_reviewed_resonance_without_closing_g2():
    cap = replace(dossier(), method=ContinuumMethod.CAP_EOM_EA_CCSD,
                  finding=ContinuumFinding.RESONANT_IDENTIFIED_STATE)
    assert cap.scope is ContinuumScope.IDENTIFIED_ROOT
