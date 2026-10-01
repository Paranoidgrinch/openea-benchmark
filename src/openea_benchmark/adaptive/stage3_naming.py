"""Bounded stable identifiers for adaptive Stage-3 execution artifacts.

Stage-3 refinement may form long provenance chains.  Human-readable provenance
belongs in structured fields (``source_root_id``, ``source_checkpoint_path``),
not recursively inside new request IDs or filesystem basenames.  This module
provides deterministic compact names while preserving the full provenance in
those separate fields.
"""
from __future__ import annotations

from hashlib import sha256
from math import isfinite


REQUEST_ID_MAX_CHARS = 120
CHECKPOINT_BASENAME_MAX_CHARS = 96
_DIGEST_CHARS = 16


def _safe_prefix(value: str, limit: int) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("identifier source must be non-empty")
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    safe = safe[:limit].rstrip("-_.")
    return safe or "stage3"


def _digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


def refinement_request_id(
    *,
    job_id: str,
    refinement_round: int,
    geometry_index: int,
    seed_index: int,
    seed_request_id: str,
    target_r_angstrom: float,
) -> str:
    """Return a deterministic, bounded refinement request ID.

    The parent request participates in the digest but is never embedded
    recursively in the visible identifier.
    """
    if refinement_round < 1:
        raise ValueError("refinement_round must be >= 1")
    if geometry_index < 0 or seed_index < 0:
        raise ValueError("geometry_index and seed_index must be >= 0")
    if not str(seed_request_id).strip():
        raise ValueError("seed_request_id must be non-empty")
    r = float(target_r_angstrom)
    if not isfinite(r) or r <= 0.0:
        raise ValueError("target_r_angstrom must be finite and positive")

    job_prefix = _safe_prefix(job_id, 36)
    digest = _digest(
        job_id,
        refinement_round,
        geometry_index,
        seed_index,
        seed_request_id,
        f"{r:.12f}",
    )
    value = (
        f"{job_prefix}__refine{refinement_round:02d}"
        f"__g{geometry_index:03d}__seed{seed_index:02d}__{digest}"
    )
    if len(value) > REQUEST_ID_MAX_CHARS:
        raise AssertionError("bounded Stage-3 refinement ID exceeded its contract")
    return value


def checkpoint_artifact_basename(request_id: str) -> str:
    """Return a bounded deterministic HF checkpoint basename."""
    prefix = _safe_prefix(request_id, 48)
    digest = _digest(request_id)
    value = f"{prefix}__{digest}.hf.chk"
    if len(value) > CHECKPOINT_BASENAME_MAX_CHARS:
        raise AssertionError("bounded Stage-3 checkpoint basename exceeded its contract")
    return value
