"""Canonical OpenEA v1 reference-character gate.

The gate consumes *reviewed* diagnostic evidence.  It deliberately does not
apply raw chemistry thresholds and never promotes one scalar diagnostic into a
multireference verdict.  D03 is treated as a composite multireference review:
a caller may mark D03 CONFIRMED only after the several-indicator assessment
required by the scientific workflow.
"""
from __future__ import annotations

from typing import Mapping

from .model import (
    DiagnosticID,
    ReferenceCharacterAssessment,
    ReferenceCharacterStatus,
    Review,
    ReviewStatus,
)


_REQUIRED_PRE_GATE = (
    DiagnosticID.D01_SPIN,
    DiagnosticID.D02_SCF_STABILITY,
    DiagnosticID.D03_MULTIREFERENCE,
    DiagnosticID.D05_REFERENCE_SENSITIVITY,
)


def _evidence_ids(reviews: Mapping[DiagnosticID, Review]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        evidence_id
        for identifier in sorted(reviews, key=lambda item: item.value)
        for evidence_id in reviews[identifier].evidence_ids
    ))


def assess_reference_character(
    reviews: Mapping[DiagnosticID, Review],
) -> ReferenceCharacterAssessment:
    """Aggregate reviewed D01/D02/D03/D05 and optional D06 evidence.

    Policy is intentionally conservative:
    * unresolved required evidence -> UNRESOLVED;
    * confirmed D03 -> MULTIREFERENCE_RISK (D03 itself must be composite);
    * confirmed SCF instability -> UNRESOLVED until the reference is repaired;
    * confirmed spin/reference/triples warnings -> BORDERLINE, not automatic MR;
    * otherwise -> SAFE_SINGLE_REFERENCE.
    """

    relevant_ids = _REQUIRED_PRE_GATE + (DiagnosticID.D06_TRIPLES_RELIABILITY,)
    supplied = {identifier: reviews[identifier] for identifier in relevant_ids if identifier in reviews}
    evidence = _evidence_ids(supplied)

    missing = tuple(identifier for identifier in _REQUIRED_PRE_GATE if identifier not in supplied)
    incomplete = tuple(
        identifier
        for identifier in _REQUIRED_PRE_GATE
        if identifier in supplied
        and supplied[identifier].status in (ReviewStatus.PENDING, ReviewStatus.UNRESOLVED)
    )
    if missing or incomplete:
        labels = tuple(identifier.value for identifier in missing + incomplete)
        return ReferenceCharacterAssessment(
            ReferenceCharacterStatus.UNRESOLVED,
            evidence_ids=evidence,
            reasons=(
                'Required reference-character evidence is incomplete: ' + ','.join(labels),
            ),
            diagnostic_ids=tuple(supplied),
        )

    d03 = supplied[DiagnosticID.D03_MULTIREFERENCE]
    if d03.status is ReviewStatus.CONFIRMED:
        return ReferenceCharacterAssessment(
            ReferenceCharacterStatus.MULTIREFERENCE_RISK,
            evidence_ids=evidence,
            reasons=(
                'Composite D03 review confirms multireference risk; single-reference production is not authorized.',
            ),
            diagnostic_ids=tuple(supplied),
        )

    d02 = supplied[DiagnosticID.D02_SCF_STABILITY]
    if d02.status is ReviewStatus.CONFIRMED:
        return ReferenceCharacterAssessment(
            ReferenceCharacterStatus.UNRESOLVED,
            evidence_ids=evidence,
            reasons=(
                'SCF instability is confirmed; repair or replace the reference before method selection.',
            ),
            diagnostic_ids=tuple(supplied),
        )

    warnings: list[str] = []
    for identifier, label in (
        (DiagnosticID.D01_SPIN, 'spin/reference representation warning'),
        (DiagnosticID.D05_REFERENCE_SENSITIVITY, 'reference sensitivity warning'),
        (DiagnosticID.D06_TRIPLES_RELIABILITY, 'triples-reliability warning'),
    ):
        review = supplied.get(identifier)
        if review is not None and review.status is ReviewStatus.CONFIRMED:
            warnings.append(label)
        elif review is not None and review.status in (ReviewStatus.PENDING, ReviewStatus.UNRESOLVED):
            warnings.append(label + ' remains unresolved')

    if warnings:
        return ReferenceCharacterAssessment(
            ReferenceCharacterStatus.BORDERLINE,
            evidence_ids=evidence,
            reasons=tuple(warnings),
            diagnostic_ids=tuple(supplied),
        )

    return ReferenceCharacterAssessment(
        ReferenceCharacterStatus.SAFE_SINGLE_REFERENCE,
        evidence_ids=evidence,
        reasons=(
            'Required reference-character reviews are cleared or physically not applicable, with no confirmed warning.',
        ),
        diagnostic_ids=tuple(supplied),
    )
