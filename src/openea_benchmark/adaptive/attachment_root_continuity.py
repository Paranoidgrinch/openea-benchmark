"""G2 diagnostic root continuity via cross-basis AO 1p EOM directions.

The overlap is a useful *root-candidate* diagnostic, not a Dyson overlap,
not a continuum-exclusion proof, and never an automatic G2 identity review.
All paths end at ROOT_REVIEW_REQUIRED even when numerically unambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite, sqrt
from typing import Callable, Any
import json

import numpy as np

from .attachment_eom_runner import (
    G2EOMRoot, G2EOMSeries, G2EOMStatus, G2EOMSubpoint,
    scaled_diffuse_basis, _parse_basis_from_pyscf,
)
from .model import Review, ReviewStatus


class RootContinuityStatus(str, Enum):
    CANDIDATE_REVIEW_REQUIRED = "CANDIDATE_REVIEW_REQUIRED"
    AMBIGUOUS = "AMBIGUOUS"
    MISSING_ONE_PARTICLE_DATA = "MISSING_ONE_PARTICLE_DATA"
    INVALID_SERIES = "INVALID_SERIES"


@dataclass(frozen=True)
class RootContinuitySettings:
    min_overlap: float = 0.80
    min_competitor_separation: float = 0.10
    min_algebraic_one_particle_fraction: float = 0.05

    def __post_init__(self) -> None:
        if not (0 < self.min_overlap <= 1 and
                0 <= self.min_competitor_separation < 1 and
                0 < self.min_algebraic_one_particle_fraction <= 1):
            raise ValueError("Root-continuity policy thresholds must be valid fractions")


@dataclass(frozen=True)
class RootContinuityLink:
    reference_key: str
    target_key: str
    reference_root: int
    candidate_root: int | None
    normalized_overlap: float | None
    competitor_overlap: float | None
    status: RootContinuityStatus
    reason: str


@dataclass(frozen=True)
class RootContinuityReport:
    status: RootContinuityStatus
    proposed_roots: tuple[tuple[str, int], ...]
    links: tuple[RootContinuityLink, ...]
    review: Review
    evidence_id: str
    notes: tuple[str, ...]
    method_role: str = "DIAGNOSTIC"
    boundness_decision: str = "UNRESOLVED"

    def __post_init__(self) -> None:
        if self.review.status is not ReviewStatus.UNRESOLVED:
            raise ValueError("AO 1p root continuity cannot certify root identity")
        if self.boundness_decision != "UNRESOLVED" or self.method_role != "DIAGNOSTIC":
            raise ValueError("G2 root overlap is not a physical boundness decision")


# (reference subpoint, target subpoint) -> cross-AO overlap matrix;
# supplying a provider enables deterministic tests without a real SCF backend.
AOOverlapProvider = Callable[[G2EOMSubpoint, G2EOMSubpoint], np.ndarray]
BasisLoader = Callable[[str, str], Any]


def _basis_material(sub: G2EOMSubpoint, loader: BasisLoader) -> dict[str, Any]:
    bs = sub.request.basis
    if sub.request.scale_factor is not None:
        return scaled_diffuse_basis(bs.basis_by_element, sub.request.selectors,
                                    sub.request.scale_factor, loader=loader)
    return {k: loader(k, v) for k, v in bs.basis_by_element.items()}


def pyscf_cross_ao_overlap(sub1: G2EOMSubpoint, sub2: G2EOMSubpoint,
                           *, loader: BasisLoader = _parse_basis_from_pyscf) -> np.ndarray:
    """Evaluate <chi_A|chi_B> on the *same* atom positions in distinct bases.

    No SCF calculation is run.  Uses the exact per-element basis and selected
    exponent scalings recorded in each request; no guessed basis labels.
    """
    from pyscf import gto

    st = sub1.request.state
    atoms = [(st.atoms[0], (0., 0., 0.)),
             (st.atoms[1], (0., 0., st.r_angstrom))]
    a = gto.M(atom=atoms, charge=st.charge, spin=st.spin_2s,
              unit="Angstrom", symmetry=False, verbose=0,
              basis=_basis_material(sub1, loader))
    b = gto.M(atom=atoms, charge=st.charge, spin=st.spin_2s,
              unit="Angstrom", symmetry=False, verbose=0,
              basis=_basis_material(sub2, loader))
    return np.asarray(gto.intor_cross("int1e_ovlp", a, b))


def _compatible(a: G2EOMSubpoint, b: G2EOMSubpoint) -> bool:
    left, right = a.request, b.request
    if left.state != right.state or left.source_checkpoint_sha256 != right.source_checkpoint_sha256:
        return False
    return (left.basis.family_id == right.basis.family_id and
            left.basis.electron_model == right.basis.electron_model and
            a.result.reference_kind == b.result.reference_kind and
            a.result.pyscf_version == b.result.pyscf_version)


def _available(root: G2EOMRoot, settings: RootContinuitySettings) -> bool:
    return (root.one_particle_ao_alpha is not None and
            root.one_particle_amplitude_fraction is not None and
            root.one_particle_amplitude_fraction >= settings.min_algebraic_one_particle_fraction)


def normalized_ao_one_particle_overlap(
    a: G2EOMRoot, b: G2EOMRoot, *,
    saa: np.ndarray, sab: np.ndarray, sbb: np.ndarray,
) -> float:
    """Phase-invariant normalized overlap in the *AO metric*, spin resolved.

    This is a comparison of truncated right-EOM 1p amplitudes, not a Dyson
    orbital overlap, not a norm/pole strength, and not an electron-attachment
    boundness observable.
    """
    if a.one_particle_ao_alpha is None or b.one_particle_ao_alpha is None:
        raise ValueError("Missing AO one-particle direction")
    if (a.one_particle_ao_beta is None) != (b.one_particle_ao_beta is None):
        raise ValueError("EOM 1p spin representations cannot be mixed")
    saa, sab, sbb = (np.asarray(x, dtype=float) for x in (saa, sab, sbb))
    ca = [np.asarray(a.one_particle_ao_alpha, dtype=float)]
    cb = [np.asarray(b.one_particle_ao_alpha, dtype=float)]
    if a.one_particle_ao_beta is not None:
        ca.append(np.asarray(a.one_particle_ao_beta, dtype=float))
        cb.append(np.asarray(b.one_particle_ao_beta, dtype=float))
    if (saa.shape != (ca[0].size, ca[0].size) or
            sab.shape != (ca[0].size, cb[0].size) or
            sbb.shape != (cb[0].size, cb[0].size)):
        raise ValueError("EOM AO direction does not match cross-basis AO dimensions")
    if any(x.size != ca[0].size for x in ca) or any(x.size != cb[0].size for x in cb):
        raise ValueError("EOM spin-resolved AO dimension mismatch")
    if not all(np.isfinite(x).all() for x in (saa, sab, sbb, *ca, *cb)):
        raise ValueError("Non-finite overlap matrix or root coefficients")
    if not np.allclose(saa, saa.T, atol=1e-9) or not np.allclose(sbb, sbb.T, atol=1e-9):
        raise ValueError("AO self overlap matrices must be symmetric")
    na = sum(float(v @ saa @ v) for v in ca)
    nb = sum(float(v @ sbb @ v) for v in cb)
    if na <= 1e-14 or nb <= 1e-14:
        raise ValueError("EOM one-particle AO norm too small to compare")
    val = abs(sum(float(x @ sab @ y) for x, y in zip(ca, cb))) / sqrt(na * nb)
    if not isfinite(val) or val > 1.0 + 1e-6:
        raise ValueError("Nonphysical normalized AO overlap; check AO basis metrics")
    return min(val, 1.0)


def propose_g2_root_continuity(
    series: G2EOMSeries, *, starting_root_index: int,
    settings: RootContinuitySettings = RootContinuitySettings(),
    overlap_provider: AOOverlapProvider | None = None,
) -> RootContinuityReport:
    """Suggest cross-basis root paths, but NEVER mark root identity as cleared.

    A root is a candidate only when the 1p direction has appreciable content,
    the best target is unique, and its inverse match to the source is unique.
    Stabilization points are compared to their fixed augmentation baseline.
    Energy ranking and root ordinals play no role in continuation.
    """
    if starting_root_index < 0:
        raise ValueError("Explicit starting root index must be nonnegative")
    if series.status is not G2EOMStatus.COMPLETE_ROOT_REVIEW_REQUIRED:
        raise ValueError("Incomplete G2 EOM diagnostics cannot be root tracked")
    subpoints = series.subpoints
    if len({p.request.key for p in subpoints}) != len(subpoints):
        raise ValueError("Duplicate EOM subpoint keys")
    by_key = {p.request.key: p for p in subpoints}
    aug = sorted((p for p in subpoints if p.request.scale_factor is None),
                 key=lambda p: p.request.basis.augmentation_level)
    if not aug or starting_root_index not in {x.root_index for x in aug[0].result.roots}:
        raise ValueError("Starting root absent from first augmentation baseline")
    if len(subpoints) < 2:
        raise ValueError("Root continuity requires at least two independent basis/scale points")
    if any(not _compatible(aug[0], p) for p in subpoints):
        raise ValueError("G2 root continuity requires the same neutral/source/backend/family")
    if len({p.request.basis.augmentation_level for p in aug}) != len(aug):
        raise ValueError("Duplicate augmentation levels in G2 series")
    if overlap_provider is None:
        overlap_provider = pyscf_cross_ao_overlap

    matrix_cache: dict[tuple[str, str], np.ndarray] = {}

    def matrix(a: G2EOMSubpoint, b: G2EOMSubpoint) -> np.ndarray:
        k = (a.request.key, b.request.key)
        if k not in matrix_cache:
            matrix_cache[k] = np.asarray(overlap_provider(a, b), dtype=float)
        return matrix_cache[k]

    def match(source: G2EOMSubpoint, target: G2EOMSubpoint,
              root_index: int) -> RootContinuityLink:
        prefix = (source.request.key, target.request.key, root_index)
        src = {x.root_index: x for x in source.result.roots}
        dest = {x.root_index: x for x in target.result.roots}
        if root_index not in src:
            return RootContinuityLink(*prefix, None, None, None,
                                      RootContinuityStatus.AMBIGUOUS, "Source root missing")
        if not _available(src[root_index], settings):
            return RootContinuityLink(*prefix, None, None, None,
                                      RootContinuityStatus.MISSING_ONE_PARTICLE_DATA,
                                      "Source root lacks usable right-EOM AO 1p information")
        valid_targets = [r for r in dest.values() if _available(r, settings)]
        if not valid_targets:
            return RootContinuityLink(*prefix, None, None, None,
                                      RootContinuityStatus.MISSING_ONE_PARTICLE_DATA,
                                      "Targets lack usable right-EOM AO 1p information")
        s0, s01, s1 = matrix(source, source), matrix(source, target), matrix(target, target)
        scores = {r.root_index: normalized_ao_one_particle_overlap(
            src[root_index], r, saa=s0, sab=s01, sbb=s1) for r in valid_targets}
        ordered = sorted(scores, key=lambda idx: (-scores[idx], idx))
        best = ordered[0]
        maximum = scores[best]
        runner_up = scores[ordered[1]] if len(ordered) > 1 else 0.0
        if maximum < settings.min_overlap or maximum - runner_up < settings.min_competitor_separation:
            return RootContinuityLink(*prefix, best, maximum, runner_up,
                                      RootContinuityStatus.AMBIGUOUS,
                                      "Cross-basis overlap weak or target roots nearly degenerate")
        # Mutual unique match prevents two tracked roots collapsing into one.
        rival = [normalized_ao_one_particle_overlap(
            r, dest[best], saa=s0, sab=s01, sbb=s1)
            for r in src.values() if r.root_index != root_index and _available(r, settings)]
        if rival and maximum - max(rival) < settings.min_competitor_separation:
            return RootContinuityLink(*prefix, best, maximum, max(rival),
                                      RootContinuityStatus.AMBIGUOUS,
                                      "Target root has competing source-root identity")
        return RootContinuityLink(*prefix, best, maximum, runner_up,
                                  RootContinuityStatus.CANDIDATE_REVIEW_REQUIRED,
                                  "AO-projected right-EOM 1p root candidate; independent review still required")

    start = aug[0]
    path = {start.request.key: starting_root_index}
    links: list[RootContinuityLink] = []
    for src, dst in zip(aug, aug[1:]):
        if src.request.key not in path:
            break
        link = match(src, dst, path[src.request.key])
        links.append(link)
        if link.status is not RootContinuityStatus.CANDIDATE_REVIEW_REQUIRED:
            break
        assert link.candidate_root is not None
        path[dst.request.key] = link.candidate_root
    # Each scaled point is compared directly against its physical parent basis.
    for sub in subpoints:
        if sub.request.scale_factor is None:
            continue
        key = f"aug:{sub.request.basis.augmentation_level}"
        if key not in by_key or key not in path:
            links.append(RootContinuityLink(key, sub.request.key, -1, None, None, None,
                RootContinuityStatus.AMBIGUOUS,
                "Stabilization baseline root not established"))
            continue
        link = match(by_key[key], sub, path[key])
        links.append(link)
        if link.status is RootContinuityStatus.CANDIDATE_REVIEW_REQUIRED:
            assert link.candidate_root is not None
            path[sub.request.key] = link.candidate_root

    if any(x.status is RootContinuityStatus.MISSING_ONE_PARTICLE_DATA for x in links):
        status = RootContinuityStatus.MISSING_ONE_PARTICLE_DATA
    elif any(x.status is RootContinuityStatus.AMBIGUOUS for x in links) or len(path) != len(subpoints):
        status = RootContinuityStatus.AMBIGUOUS
    else:
        status = RootContinuityStatus.CANDIDATE_REVIEW_REQUIRED
    signature = sha256(json.dumps({
        "raw_ids": sorted(p.evidence_id for p in subpoints),
        "settings": (settings.min_overlap, settings.min_competitor_separation,
                     settings.min_algebraic_one_particle_fraction),
        "links": [(x.reference_key, x.target_key, x.reference_root, x.candidate_root,
                   x.normalized_overlap, x.competitor_overlap, x.status.value) for x in links],
        "starting_root": starting_root_index,
    }, sort_keys=True, allow_nan=False).encode()).hexdigest()
    evidence_id = f"G2_ROOT_PROPOSAL:{signature}"
    note = ("Cross-basis AO 1p overlaps are diagnostic, not Dyson amplitudes. "
            "Root identity, spin/multiplicity, and continuum exclusion remain subject to independent review.")
    review = Review(ReviewStatus.UNRESOLVED, (evidence_id,), note)
    return RootContinuityReport(status, tuple(sorted(path.items())), tuple(links),
                                review, evidence_id, (note,))
