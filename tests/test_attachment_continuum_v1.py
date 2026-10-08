from openea_benchmark.adaptive.attachment_continuum import (
    AttachmentContinuumSettings,
    AttachmentContinuumStatus,
    ValenceAttachmentEvidence,
    EOMEAAssessmentStatus,
    EOMEAAttachmentPoint,
    StabilizationPoint,
    StabilizationStatus,
    assess_attachment_continuum,
    assess_eom_ea_diffuse_series,
    assess_stabilization_series,
    pyscf_eom_eigenvalue_to_attachment_ea_ev,
)
from openea_benchmark.adaptive.attachment_continuum_independent import (
    ContinuumMethod, ContinuumScope, ContinuumFinding, IndependentContinuumDossier,
)
from openea_benchmark.adaptive.method_basis_advisor import AttachmentCharacter
from openea_benchmark.adaptive.model import Interval, Review, ReviewStatus
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


def dossier(finding: ContinuumFinding):
    return IndependentContinuumDossier(
        method=ContinuumMethod.ELECTRON_SCATTERING,
        scope=(ContinuumScope.ALL_RELEVANT_STATES
               if finding is ContinuumFinding.NO_BOUND_STATES_IN_REVIEWED_SECTOR
               else ContinuumScope.IDENTIFIED_ROOT),
        finding=finding,
        system="LiH", neutral_state_id="neutral-root", source_root_id="source-1",
        r_angstrom=1.6, detachment_threshold_id="reviewed-neutral-plus-electron",
        raw_method_evidence_ids=("SCATTERING:datafile:1",),
        method_validity_review=cleared("SCATTERING:validated-method"),
        threshold_review=cleared("SCATTERING:threshold"),
        root_identity_review=cleared("SCATTERING:root-identity"),
        completeness_review=cleared("SCATTERING:sector-completeness"),
        scientific_review=cleared("SCATTERING:independent-assessment"),
        reviewed_sector_id="all-electronic-roots-of-reviewed-symmetry",
        state_inventory_evidence_ids=("SCATTERING:complete-state-inventory",),
    )


def valence_evidence(lower=0.5, upper=0.7, *, localized=True, stable=True, identity=True):
    return ValenceAttachmentEvidence(
        system="LiH", neutral_state_id="neutral-root", anion_state_id="anion-root",
        r_angstrom=1.6,
        vertical_detachment_ev=Interval(lower, upper),
        detachment_threshold_id="N0+free-electron:reviewed",
        orbital_localization=cleared("real-orbital-density-review") if localized else Review(ReviewStatus.UNRESOLVED, (), "diffuse orbital"),
        reference_stability=cleared("scf-cc-stability") if stable else Review(ReviewStatus.PENDING, (), "unstable reference"),
        state_continuity=cleared("neutral-anion-state-match") if identity else Review(ReviewStatus.UNRESOLVED, (), "root switch"),
        source_evidence_ids=("CCSD(T):vertical-detachment-interval-with-residual",),
    )


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
        system="LiH", neutral_state_id="neutral-root", source_root_id="source-1",
        r_angstrom=1.6,
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
        continuum_dossier=dossier(ContinuumFinding.BOUND_IDENTIFIED_STATE),
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
        valence_evidence=valence_evidence(),
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
        continuum_dossier=dossier(ContinuumFinding.BOUND_IDENTIFIED_STATE),
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
        continuum_dossier=dossier(ContinuumFinding.NO_BOUND_STATES_IN_REVIEWED_SECTOR),
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
        continuum_dossier=dossier(ContinuumFinding.NO_BOUND_STATES_IN_REVIEWED_SECTOR),
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


def test_plain_cleared_review_must_not_close_continuum_anymore():
    r = assess_stabilization_series(
        (stab(0.8, 0.50), stab(1.0, 0.505), stab(1.2, 0.502)),
        settings=SETTINGS,
        continuum_discrimination_review=cleared("this-string-is-not-a-continuum-method"),
    )
    assert r.status is StabilizationStatus.UNRESOLVED


def test_single_resonance_cannot_claim_no_bound_state_sector():
    from dataclasses import replace
    with __import__('pytest').raises(ValueError, match="isolated resonance"):
        replace(dossier(ContinuumFinding.BOUND_IDENTIFIED_STATE),
                finding=ContinuumFinding.NO_BOUND_STATES_IN_REVIEWED_SECTOR)


def test_mismatched_geometry_rejects_forged_continuum_match():
    from dataclasses import replace
    r = assess_stabilization_series(
        (stab(0.8, 0.50), stab(1.0, 0.505), stab(1.2, 0.502)),
        settings=SETTINGS,
        continuum_dossier=replace(dossier(ContinuumFinding.BOUND_IDENTIFIED_STATE),
                                  r_angstrom=1.61),
    )
    assert r.status is StabilizationStatus.UNRESOLVED


def test_resonance_of_single_root_never_force_global_unbound():
    from dataclasses import replace
    r = assess_stabilization_series(
        (stab(0.8, -0.50), stab(1.0, -0.505), stab(1.2, -0.502)),
        settings=SETTINGS,
        continuum_dossier=replace(dossier(ContinuumFinding.BOUND_IDENTIFIED_STATE),
                                  finding=ContinuumFinding.RESONANT_IDENTIFIED_STATE),
    )
    assert r.status is StabilizationStatus.UNRESOLVED


# G2 simplification: EA-EOM/continuum is a conditional diagnostic, not a
# universal obligatory method for an ordinary compact valence anion.
def test_valence_g2_can_close_without_eom_or_cap():
    outcome = assess_attachment_continuum(
        attachment_character=AttachmentCharacter.VALENCE_BOUND,
        attachment_character_review=cleared("localized-valence-character"),
        direct_diffuse_review=cleared("CCSD(T)-diffuse-converged"),
        valence_evidence=valence_evidence(),
    )
    assert outcome.status is AttachmentContinuumStatus.BOUND_ATTACHMENT_CLEARED
    assert outcome.eom_ea is None and outcome.stabilization is None
    assert "VALENCE_ATTACHMENT_CLEARED_WITHOUT_MANDATORY_EOM" in outcome.reasons
    assert "real-orbital-density-review" in outcome.evidence_ids


def test_valence_is_not_cleared_by_positive_ea_and_diffuse_only():
    outcome = assess_attachment_continuum(
        attachment_character=AttachmentCharacter.VALENCE_BOUND,
        attachment_character_review=cleared("character"),
        direct_diffuse_review=cleared("diffuse"),
        eom_ea=assess_eom_ea_diffuse_series(
            (eom_point(0, 1.0), eom_point(1, 1.01), eom_point(2, 1.014)),
            settings=SETTINGS),
    )
    assert outcome.status is AttachmentContinuumStatus.NEED_MORE_EVIDENCE


def test_valence_evidence_with_missing_reviews_is_not_enough():
    for ev in (valence_evidence(localized=False), valence_evidence(stable=False),
               valence_evidence(identity=False), valence_evidence(lower=-0.01)):
        outcome = assess_attachment_continuum(
            attachment_character=AttachmentCharacter.VALENCE_BOUND,
            attachment_character_review=cleared("character"),
            direct_diffuse_review=cleared("diffuse"),
            valence_evidence=ev,
        )
        assert outcome.status is AttachmentContinuumStatus.NEED_MORE_EVIDENCE


def test_clear_valence_requires_direct_diffuse_convergence():
    outcome = assess_attachment_continuum(
        attachment_character=AttachmentCharacter.VALENCE_BOUND,
        attachment_character_review=cleared("character"),
        direct_diffuse_review=Review(ReviewStatus.PENDING, (), "still computing"),
        valence_evidence=valence_evidence(),
    )
    assert outcome.status is AttachmentContinuumStatus.NEED_MORE_EVIDENCE


def test_already_known_nonbinding_eom_vetoes_valence_route():
    eom = assess_eom_ea_diffuse_series(
        (eom_point(0, -0.30), eom_point(1, -0.31), eom_point(2, -0.314)),
        settings=SETTINGS,
    )
    outcome = assess_attachment_continuum(
        attachment_character=AttachmentCharacter.VALENCE_BOUND,
        attachment_character_review=cleared("character"),
        direct_diffuse_review=cleared("diffuse"),
        valence_evidence=valence_evidence(),
        eom_ea=eom,
    )
    assert outcome.status is AttachmentContinuumStatus.UNRESOLVED
    assert "VALENCE_ATTACHMENT_DIAGNOSTIC_CONFLICT" in outcome.reasons


def test_advanced_regime_has_no_automatic_eom_or_cap_requirement():
    for character in (AttachmentCharacter.DIFFUSE_BOUND, AttachmentCharacter.NEAR_THRESHOLD,
                      AttachmentCharacter.CONTINUUM_LIKE):
        outcome = assess_attachment_continuum(
            attachment_character=character,
            attachment_character_review=cleared("character"),
            direct_diffuse_review=cleared("diffuse"),
        )
        assert outcome.status is AttachmentContinuumStatus.NEED_MORE_EVIDENCE
        assert outcome.reasons == ("SELECT_ATTACHMENT_ESCALATION_BY_PHYSICAL_REGIME",)


def test_typed_valence_g2_bridge_needs_nuclear_binding():
    r = assess_attachment_continuum(
        attachment_character=AttachmentCharacter.VALENCE_BOUND,
        attachment_character_review=cleared("character"),
        direct_diffuse_review=cleared("diffuse"),
        valence_evidence=valence_evidence(),
    )
    g2 = physical_validity_from_binding(
        BindingAssessment(BindingStatus.BOUND, ("molecular-well",), 0.2),
        electron_attachment_review=r,
    )
    assert g2.status is PhysicalValidityStatus.PHYSICALLY_BOUND_ANION
    assert not g2.nuclear_binding_resolved


def test_not_applicable_cannot_replace_essential_valence_reviews():
    for missing in ("attachment_character_review", "direct_diffuse_review"):
        kw = dict(attachment_character=AttachmentCharacter.VALENCE_BOUND,
                  attachment_character_review=cleared("character"),
                  direct_diffuse_review=cleared("diffuse"),
                  valence_evidence=valence_evidence())
        kw[missing] = Review(ReviewStatus.NOT_APPLICABLE, (), "not applicable by declaration")
        result = assess_attachment_continuum(**kw)
        assert result.status is AttachmentContinuumStatus.NEED_MORE_EVIDENCE


def test_valence_evidence_requires_state_and_geometry_provenance():
    from dataclasses import replace
    import pytest
    for kw in ({"r_angstrom": 0.0}, {"system": ""}, {"neutral_state_id": ""},
               {"anion_state_id": ""}, {"detachment_threshold_id": ""}):
        with pytest.raises(ValueError):
            replace(valence_evidence(), **kw)
