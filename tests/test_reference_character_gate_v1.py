from openea_benchmark.adaptive.model import (
    DiagnosticID,
    ReferenceCharacterStatus,
    Review,
    ReviewStatus,
)
from openea_benchmark.adaptive.reference_character import assess_reference_character


def cleared(tag):
    return Review(ReviewStatus.CLEARED, (tag,), 'review cleared')


def confirmed(tag):
    return Review(ReviewStatus.CONFIRMED, (tag,), 'warning confirmed')


def base_reviews():
    return {
        DiagnosticID.D01_SPIN: cleared('d01'),
        DiagnosticID.D02_SCF_STABILITY: cleared('d02'),
        DiagnosticID.D03_MULTIREFERENCE: cleared('d03'),
        DiagnosticID.D05_REFERENCE_SENSITIVITY: cleared('d05'),
    }


def test_safe_requires_reviewed_reference_evidence():
    result = assess_reference_character(base_reviews())
    assert result.status is ReferenceCharacterStatus.SAFE_SINGLE_REFERENCE
    assert result.authorizes_single_reference
    assert not result.requires_expanded_diagnostics


def test_missing_required_review_fails_closed():
    reviews = base_reviews()
    reviews.pop(DiagnosticID.D03_MULTIREFERENCE)
    result = assess_reference_character(reviews)
    assert result.status is ReferenceCharacterStatus.UNRESOLVED
    assert not result.authorizes_single_reference


def test_composite_d03_can_route_multireference():
    reviews = base_reviews()
    reviews[DiagnosticID.D03_MULTIREFERENCE] = confirmed('d03-risk')
    result = assess_reference_character(reviews)
    assert result.status is ReferenceCharacterStatus.MULTIREFERENCE_RISK
    assert result.requires_multireference_branch
    assert not result.authorizes_single_reference


def test_single_warning_is_borderline_not_automatic_mr():
    reviews = base_reviews()
    reviews[DiagnosticID.D05_REFERENCE_SENSITIVITY] = confirmed('d05-warning')
    result = assess_reference_character(reviews)
    assert result.status is ReferenceCharacterStatus.BORDERLINE
    assert result.authorizes_single_reference
    assert result.requires_expanded_diagnostics


def test_confirmed_scf_instability_blocks_method_selection():
    reviews = base_reviews()
    reviews[DiagnosticID.D02_SCF_STABILITY] = confirmed('unstable')
    result = assess_reference_character(reviews)
    assert result.status is ReferenceCharacterStatus.UNRESOLVED
    assert not result.authorizes_single_reference


def test_post_cc_warning_reenters_gate_as_borderline_evidence():
    reviews = base_reviews()
    reviews[DiagnosticID.D06_TRIPLES_RELIABILITY] = confirmed('post-cc-warning')
    result = assess_reference_character(reviews)
    assert result.status is ReferenceCharacterStatus.BORDERLINE
