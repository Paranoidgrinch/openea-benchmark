"""Conservative adapter for the EXISTING OpenEA SCFRootRecord interface.

The adapter is deliberately read-only; it never changes checkpoint files,
SCF statuses, state identity or the actual scientific gates G1/G2/G3.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from .model import DiagnosticID, DiagnosticRecord, Review, ReviewStatus

if TYPE_CHECKING:
    from openea_benchmark.root_record import SCFRootRecord


def stability_diagnostic_from_root(root: 'SCFRootRecord') -> DiagnosticRecord:
    """Translate recorded stability EVIDENCE into a D02 review.

    `CANONICALIZED` in the existing repo only asserts INTERNAL stability.
    External `None` means *not tested / not available*, never `True`.
    Thus this adapter intentionally cannot clear D02 solely from the status.
    An unavailable external test can later be resolved through an independent
    validated reference comparison, but that requires a separate evidence ID.
    """
    root_id = str(root.root_id)
    if not root_id.strip():
        raise ValueError('SCF root must carry root_id')
    status = getattr(root.status, 'value', root.status)
    internal = root.internal_stable
    external = root.external_stable
    refs = (root_id,)
    if status == 'FAILED':
        review = Review(ReviewStatus.UNRESOLVED, refs, 'SCF failed; no stability verdict')
    elif internal is False or external is False:
        review = Review(ReviewStatus.CONFIRMED, refs, 'Detected SCF instability')
    elif status == 'CANONICALIZED' and internal is True and external is True:
        review = Review(ReviewStatus.CLEARED, refs, 'Recorded internal and external stability')
    elif internal is True and external is None:
        review = Review(ReviewStatus.UNRESOLVED, refs,
                        'Internal stability only; external scope unavailable/unexamined')
    else:
        review = Review(ReviewStatus.UNRESOLVED, refs,
                        'Stability evidence incomplete or canonicalization pending')
    return DiagnosticRecord(DiagnosticID.D02_SCF_STABILITY, review, state_refs=refs)
