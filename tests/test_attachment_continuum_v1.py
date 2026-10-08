from openea_benchmark.adaptive.attachment_continuum import (
    AttachmentContinuumSettings,
    AttachmentContinuumStatus,
    EOMEAAssessmentStatus,
    EOMEAAttachmentPoint,
    StabilizationPoint,
    StabilizationStatus,
    assess_attachment_continuum,
    assess_eom_ea_diffuse_series,
    assess_stabilization_series,
    pyscf_eom_eigenvalue_to_attachment_ea_ev,
)
from openea_benchmark.adaptive.method_basis_advisor import AttachmentCharacter
from openea_benchmark.adaptive.model import Review, ReviewStatus
from openea_benchmark.adaptive.scientific_resolution import (
    PhysicalValidityStatus,
    physical_validity_from_binding,
)
from openea_benchmark.attachment.asymptote import BindingAssessment, BindingStatus


SETTINGS = AttachmentContinuumSettings(
    eom_increment_target_ev=0.02,
    eom_contraction_ratio_max=0.8,
    stabilization_span_target_ev=0.02,
)


def cleared(tag="x"):
    return Review(ReviewStatus.CLEARED, (tag,), f"{tag} cleared")


def eom_point(level, ea, *, state=True):
    return EOMEAAttachmentPoint(
        augmentation_level=level,
        attachment_ea_ev=ea,
        state_identity=cleared(f"state-{level}") if state else Review(ReviewStatus.UNRESOLVED, (), "root unresolved"),
        evidence_ids=(f"eom-{level}",),
        one_particle_weight=0.8,
        basis_label=f"aug{level}",
    )


def stab(scale, ea, *, state=True):
    return StabilizationPoint(
        scale_factor=scale,
        attachment_ea_ev=ea,
        state_identity=cleared(f"stab-state-{scale}") if state else Review(ReviewStatus.UNRESOLVED, (), "root unresolved"),
        evidence_ids=(f"stab-{scale}",),
    )


def test_eom_diffuse_series_requires_three_points():
    r = assess_eom_ea_diffuse_series(
        (eom_point(0, 1.0), eom_point(1, 1.01)),
        settings=SETTINGS,
    )
    assert r.status is EOMEAAssessmentStatus.NEED_MORE_EVIDENCE


def test_eom_diffuse_series_clears_strictly_bound_attachment():
    r = assess_eom_ea_diffuse_series(
        (eom_point(0, 1.000), eom_point(1, 1.010), eom_point(2, 1.014)),
        settings=SETTINGS,
    )
    assert r.status is EOMEAAssessmentStatus.BOUND_CONVERGED
    assert r.conservative_ea_lower_ev > 0.0
    assert r.residual_estimate_ev is not None


def test_eom_diffuse_series_can_clear_nonbinding_attachment():
    r = assess_eom_ea_diffuse_series(
        (eom_point(0, -0.30), eom_point(1, -0.31), eom_point(2, -0.314)),
        settings=SETTINGS,
    )
    assert r.status is EOMEAAssessmentStatus.UNBOUND_CONVERGED
    assert r.conservative_ea_upper_ev < 0.0


def test_eom_diffuse_series_fails_closed_on_root_identity():
    r = assess_eom_ea_diffuse_series(
        (eom_point(0, 1.0), eom_point(1, 1.01, state=False), eom_point(2, 1.014)),
        settings=SETTINGS,
    )
    assert r.status is EOMEAAssessmentStatus.UNRESOLVED


def test_stabilization_series_clears_stationary_bound_state():
    r = assess_stabilization_series(
        (stab(0.8, 0.50), stab(1.0, 0.505), stab(1.2, 0.502)),
        settings=SETTINGS,
        continuum_discrimination_review=cleared("independent-continuum-bound"),
    )
    assert r.status is StabilizationStatus.STABLE_BOUND
    assert r.energy_span_ev < SETTINGS.stabilization_span_target_ev


def test_valence_bound_can_close_without_stabilization_when_independent_channels_clear():
    eom = assess_eom_ea_diffuse_series(
        (eom_point(0, 1.0), eom_point(1, 1.01), eom_point(2, 1.014)),
        settings=SETTINGS,
    )
    r = assess_attachment_continuum(
        attachment_character=AttachmentCharacter.VALENCE_BOUND,
        attachment_character_review=cleared("valence-character"),
        direct_diffuse_review=cleared("delta-cc-diffuse"),
        eom_ea=eom,
    )
    assert r.status is AttachmentContinuumStatus.BOUND_ATTACHMENT_CLEARED
    assert r.review.status is ReviewStatus.CLEARED


def test_diffuse_bound_requires_explicit_stabilization_review():
    eom = assess_eom_ea_diffuse_series(
        (eom_point(0, 0.20), eom_point(1, 0.21), eom_point(2, 0.214)),
        settings=SETTINGS,
    )
    r = assess_attachment_continuum(
        attachment_character=AttachmentCharacter.DIFFUSE_BOUND,
        attachment_character_review=cleared("diffuse-character"),
        direct_diffuse_review=cleared("delta-cc-diffuse"),
        eom_ea=eom,
        stabilization=None,
    )
    assert r.status is AttachmentContinuumStatus.NEED_MORE_EVIDENCE
    assert "STABILIZATION_REQUIRED" in r.reasons[0]


def test_diffuse_bound_closes_with_stable_continuum_scan():
    eom = assess_eom_ea_diffuse_series(
        (eom_point(0, 0.20), eom_point(1, 0.21), eom_point(2, 0.214)),
        settings=SETTINGS,
    )
    stabilization = assess_stabilization_series(
        (stab(0.8, 0.205), stab(1.0, 0.210), stab(1.2, 0.208)),
        settings=SETTINGS,
        continuum_discrimination_review=cleared("independent-continuum-bound"),
    )
    r = assess_attachment_continuum(
        attachment_character=AttachmentCharacter.DIFFUSE_BOUND,
        attachment_character_review=cleared("diffuse-character"),
        direct_diffuse_review=cleared("delta-cc-diffuse"),
        eom_ea=eom,
        stabilization=stabilization,
    )
    assert r.status is AttachmentContinuumStatus.BOUND_ATTACHMENT_CLEARED


def test_conflicting_eom_and_stabilization_is_unresolved():
    eom = assess_eom_ea_diffuse_series(
        (eom_point(0, 0.20), eom_point(1, 0.21), eom_point(2, 0.214)),
        settings=SETTINGS,
    )
    stabilization = assess_stabilization_series(
        (stab(0.8, -0.205), stab(1.0, -0.210), stab(1.2, -0.208)),
        settings=SETTINGS,
        continuum_discrimination_review=cleared("independent-continuum-unbound"),
    )
    r = assess_attachment_continuum(
        attachment_character=AttachmentCharacter.NEAR_THRESHOLD,
        attachment_character_review=cleared("near-threshold-character"),
        direct_diffuse_review=cleared("delta-cc-diffuse"),
        eom_ea=eom,
        stabilization=stabilization,
    )
    assert r.status is AttachmentContinuumStatus.UNRESOLVED


def test_typed_no_bound_attachment_can_make_g2_physically_unbound():
    eom = assess_eom_ea_diffuse_series(
        (eom_point(0, -0.30), eom_point(1, -0.31), eom_point(2, -0.314)),
        settings=SETTINGS,
    )
    stabilization = assess_stabilization_series(
        (stab(0.8, -0.305), stab(1.0, -0.310), stab(1.2, -0.308)),
        settings=SETTINGS,
        continuum_discrimination_review=cleared("independent-continuum-unbound"),
    )
    d08 = assess_attachment_continuum(
        attachment_character=AttachmentCharacter.CONTINUUM_LIKE,
        attachment_character_review=cleared("continuum-character-reviewed"),
        direct_diffuse_review=cleared("delta-cc-diffuse"),
        eom_ea=eom,
        stabilization=stabilization,
    )
    assert d08.status is AttachmentContinuumStatus.NO_BOUND_ATTACHMENT
    g2 = physical_validity_from_binding(
        BindingAssessment(BindingStatus.BOUND, ("molecular-minimum",), 0.1),
        electron_attachment_review=d08,
    )
    assert g2.status is PhysicalValidityStatus.NO_PHYSICALLY_BOUND_ANION
    assert g2.nuclear_binding_resolved


def test_pyscf_eom_sign_conversion_is_explicit():
    assert pyscf_eom_eigenvalue_to_attachment_ea_ev(-0.1) > 0.0
    assert pyscf_eom_eigenvalue_to_attachment_ea_ev(0.1) < 0.0


def test_flat_pseudostate_is_not_continuum_exclusion():
    """Regression: flat finite-basis pseudo-continuum is not proof of binding."""
    r = assess_stabilization_series(
        (stab(0.8, 0.50), stab(1.0, 0.505), stab(1.2, 0.502)),
        settings=SETTINGS,
    )
    assert r.status is StabilizationStatus.UNRESOLVED
    assert r.review.status is ReviewStatus.UNRESOLVED
    assert r.energy_span_ev is not None


def test_flat_negative_pseudostate_cannot_force_unbound():
    r = assess_stabilization_series(
        (stab(0.8, -0.5), stab(1.0, -0.505), stab(1.2, -0.502)),
        settings=SETTINGS,
    )
    assert r.status is StabilizationStatus.UNRESOLVED


def test_root_overlap_proposal_is_not_independent_continuum_review():
    r = assess_stabilization_series(
        (stab(0.8, 0.50), stab(1.0, 0.505), stab(1.2, 0.502)),
        settings=SETTINGS,
        continuum_discrimination_review=cleared("G2_ROOT_PROPOSAL:123"),
    )
    assert r.status is StabilizationStatus.UNRESOLVED


def test_unscaled_or_unidirectional_series_rejected():
    r = assess_stabilization_series(
        (stab(0.7, 0.50), stab(0.8, 0.505), stab(0.9, 0.502)),
        settings=SETTINGS,
        continuum_discrimination_review=cleared("independent-continuum"),
    )
    assert r.status is StabilizationStatus.NEED_MORE_EVIDENCE


def test_incomplete_continuum_review_not_promoted():
    r = assess_stabilization_series(
        (stab(0.8, 0.50), stab(1.0, 0.505), stab(1.2, 0.502)),
        settings=SETTINGS,
        continuum_discrimination_review=Review(ReviewStatus.UNRESOLVED, ("external-review",), "still open"),
    )
    assert r.status is StabilizationStatus.UNRESOLVED


def test_profile_identifier_not_continuum_independent():
    r = assess_stabilization_series(
        (stab(0.8, 0.50), stab(1.0, 0.505), stab(1.2, 0.502)),
        settings=SETTINGS,
        continuum_discrimination_review=cleared("G2_STABILIZATION_PROFILE:abc"),
    )
    assert r.status is StabilizationStatus.UNRESOLVED



def test_not_applicable_is_not_valid_eom_root_identity():
    points = tuple(EOMEAAttachmentPoint(p.augmentation_level, p.attachment_ea_ev,
        Review(ReviewStatus.NOT_APPLICABLE, (), "root identity declared irrelevant"), p.evidence_ids)
        for p in (eom_point(0,1.0),eom_point(1,1.01),eom_point(2,1.014)))
    assert assess_eom_ea_diffuse_series(points, settings=SETTINGS).status is EOMEAAssessmentStatus.UNRESOLVED


def test_not_applicable_is_not_valid_stabilization_root_identity():
    points = tuple(StabilizationPoint(p.scale_factor,p.attachment_ea_ev,
        Review(ReviewStatus.NOT_APPLICABLE, (), "root identity declared irrelevant"),p.evidence_ids)
        for p in (stab(.8,.5),stab(1.,.505),stab(1.2,.502)))
    assert assess_stabilization_series(points, settings=SETTINGS,
        continuum_discrimination_review=cleared("external-continuum")).status is StabilizationStatus.UNRESOLVED
