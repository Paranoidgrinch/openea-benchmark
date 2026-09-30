"""Bridge conservative PEC/state-selection reports into adaptive D04/D09 evidence.

The existing OpenEA state-selection layer deliberately treats a bracketed
sampled minimum as a *high-level calculation seed*, not as proof of a ground
state or as a rigorous bound on the continuous PEC minimum.  This bridge keeps
that epistemic distinction intact.

In particular, scout/DFT energies are preserved as provenance and scheduling
information but are NOT promoted to production energy intervals.  A finite
``production_interval_ev`` must come from a later, separately justified
high-accuracy uncertainty model.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from .model import DiagnosticID, DiagnosticRecord, Interval, Review, ReviewStatus
from .repository_bridge import state_completeness_gate


@dataclass(frozen=True)
class CandidateEnergyEvidence:
    """Energy evidence for one state/PEC candidate.

    ``sampled_energy_hartree`` is the energy of the retained sampled minimum
    candidate when available.  It is *not* an uncertainty interval for the
    continuous PEC minimum.

    ``production_interval_ev`` is intentionally optional.  The state-selection
    bridge never manufactures one from scout energies or DFT state splittings.
    """

    candidate_ref: str
    group_ref: str
    component_id: str
    minimum_status: str
    sampled_r_angstrom: float | None
    sampled_energy_hartree: float | None
    delta_from_lowest_bracketed_ev: float | None
    production_interval_ev: Interval | None
    evidence_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_ref", self.candidate_ref),
            ("group_ref", self.group_ref),
            ("component_id", self.component_id),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be non-empty")
        for value, name in (
            (self.sampled_r_angstrom, "sampled_r_angstrom"),
            (self.sampled_energy_hartree, "sampled_energy_hartree"),
            (self.delta_from_lowest_bracketed_ev, "delta_from_lowest_bracketed_ev"),
        ):
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"{name} must be finite when supplied")
        if not self.evidence_ids:
            raise ValueError("candidate evidence requires provenance")


@dataclass(frozen=True)
class StateSelectionBridgeResult:
    candidate_evidence: tuple[CandidateEnergyEvidence, ...]
    state_competition: DiagnosticRecord
    pec_asymptotes: DiagnosticRecord
    group_count: int
    all_scouted_components_bracketed: bool

    @property
    def diagnostics(self) -> tuple[DiagnosticRecord, DiagnosticRecord]:
        return (self.state_competition, self.pec_asymptotes)

    @property
    def production_intervals(self) -> tuple[Interval, ...]:
        """Return only separately justified high-accuracy intervals.

        Scout energies are deliberately absent until another layer supplies
        defensible uncertainty intervals.
        """
        return tuple(
            item.production_interval_ev
            for item in self.candidate_evidence
            if item.production_interval_ev is not None
        )


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Sequence):
        return tuple(value)
    return ()


def _selection_groups(report: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """Accept the current report contract plus a direct-group test contract."""
    if isinstance(report, Mapping):
        for key in ("groups", "sectors", "state_groups"):
            groups = _as_sequence(report.get(key))
            if groups:
                if not all(isinstance(item, Mapping) for item in groups):
                    raise TypeError(f"{key} must contain mappings")
                return tuple(groups)  # type: ignore[return-value]
        # Empty explicit groups is a valid unresolved report.
        if any(key in report for key in ("groups", "sectors", "state_groups")):
            return ()
        raise KeyError("state-selection report has no groups/sectors/state_groups")

    groups = _as_sequence(report)
    if not all(isinstance(item, Mapping) for item in groups):
        raise TypeError("state-selection groups must be mappings")
    return tuple(groups)  # type: ignore[return-value]


def _first_nonempty(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _group_ref(group: Mapping[str, Any], index: int) -> str:
    explicit = _first_nonempty(
        group,
        "group_id",
        "sector_id",
        "sector_key",
        "state_group_id",
    )
    if explicit is not None:
        return explicit

    # Preserve useful sector labels without assuming a particular schema.
    labels: list[str] = []
    for key in ("charge", "spin_2s", "multiplicity", "method", "functional"):
        if key in group and group[key] is not None:
            labels.append(f"{key}={group[key]}")
    return ";".join(labels) if labels else f"group-{index}"


def _find_nested_minimum_scout(mapping: Mapping[str, Any], *, depth: int = 0) -> Mapping[str, Any] | None:
    if depth > 5:
        return None

    status = mapping.get("status")
    candidates = mapping.get("candidates")
    if status is not None and isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes, bytearray)):
        return mapping

    for key in (
        "minimum_scout",
        "scout",
        "source_scout",
        "pec_scout",
        "minimum",
        "source_minimum_scout",
    ):
        value = mapping.get(key)
        if isinstance(value, Mapping):
            found = _find_nested_minimum_scout(value, depth=depth + 1)
            if found is not None:
                return found
    return None


def _minimum_status(candidate: Mapping[str, Any]) -> str:
    direct = candidate.get("minimum_status")
    if direct is not None:
        return str(getattr(direct, "value", direct))
    scout = _find_nested_minimum_scout(candidate)
    if scout is not None and scout.get("status") is not None:
        return str(getattr(scout["status"], "value", scout["status"]))
    return "UNKNOWN"


def _sampled_minimum(candidate: Mapping[str, Any]) -> tuple[float | None, float | None]:
    # Prefer an explicit retained/source minimum if the report provides one.
    for key in ("source_minimum", "minimum_candidate", "sampled_minimum"):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            r = value.get("r_angstrom")
            e = value.get("energy_hartree")
            return (
                None if r is None else float(r),
                None if e is None else float(e),
            )

    scout = _find_nested_minimum_scout(candidate)
    if scout is None:
        return None, None
    items = _as_sequence(scout.get("candidates"))
    if len(items) != 1 or not isinstance(items[0], Mapping):
        return None, None
    r = items[0].get("r_angstrom")
    e = items[0].get("energy_hartree")
    return (
        None if r is None else float(r),
        None if e is None else float(e),
    )


def _candidate_evidence(
    candidate: Mapping[str, Any],
    *,
    group_ref: str,
    index: int,
    base_evidence_id: str,
) -> CandidateEnergyEvidence:
    component_id = _first_nonempty(
        candidate,
        "component_id",
        "candidate_id",
        "state_id",
        "manifold_id",
        "branch_id",
    ) or f"candidate-{index}"
    candidate_ref = f"{group_ref}:{component_id}"
    evidence_id = f"{base_evidence_id}:{candidate_ref}"
    r_angstrom, energy_hartree = _sampled_minimum(candidate)

    delta = candidate.get("delta_from_lowest_bracketed_ev")
    delta_ev = None if delta is None else float(delta)
    status = _minimum_status(candidate)

    return CandidateEnergyEvidence(
        candidate_ref=candidate_ref,
        group_ref=group_ref,
        component_id=component_id,
        minimum_status=status,
        sampled_r_angstrom=r_angstrom,
        sampled_energy_hartree=energy_hartree,
        delta_from_lowest_bracketed_ev=delta_ev,
        production_interval_ev=None,
        evidence_ids=(evidence_id,),
        rationale=(
            "State-selection scout evidence only. The sampled/DFT minimum and "
            "relative scout splitting are retained for provenance and scheduling; "
            "they are not promoted to a high-accuracy energy interval."
        ),
    )


def _mirror_external_review(
    *,
    identifier: DiagnosticID,
    local_evidence_id: str,
    external: Review,
    cleared_rationale: str,
    confirmed_rationale: str,
    unresolved_rationale: str,
) -> DiagnosticRecord:
    refs = tuple(dict.fromkeys((local_evidence_id,) + external.evidence_ids))
    if external.status is ReviewStatus.CLEARED:
        review = Review(ReviewStatus.CLEARED, refs, cleared_rationale)
    elif external.status is ReviewStatus.CONFIRMED:
        review = Review(ReviewStatus.CONFIRMED, refs, confirmed_rationale)
    else:
        review = Review(ReviewStatus.UNRESOLVED, refs, unresolved_rationale)
    return DiagnosticRecord(identifier, review)


def bridge_state_selection_report(
    report: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    evidence_id: str,
    higher_level_competition_review: Review | None = None,
    asymptote_review: Review | None = None,
) -> StateSelectionBridgeResult:
    """Translate the existing conservative state-selection report.

    Scientific rules:
    - a bracketed scout minimum is a seed, never a production interval;
    - any unbracketed/open component keeps D04 and D09 scientifically active;
    - even when every scout component is bracketed, D04 remains unresolved
      until a higher-level candidate-competition review is supplied;
    - local minimum bracketing does not clear asymptotic coverage. D09 clears
      only with an explicit independent asymptote review.
    """

    evidence_id = str(evidence_id).strip()
    if not evidence_id:
        raise ValueError("evidence_id must be non-empty")

    groups = _selection_groups(report)
    candidates: list[CandidateEnergyEvidence] = []
    all_bracketed = bool(groups)
    inconsistent_bracketed_candidate = False

    for group_index, group in enumerate(groups):
        group_ref = _group_ref(group, group_index)
        group_all_bracketed = bool(group.get("all_scouted_components_bracketed", False))
        all_bracketed = all_bracketed and group_all_bracketed
        bracketed = _as_sequence(group.get("bracketed_candidates"))
        for candidate_index, candidate in enumerate(bracketed):
            if not isinstance(candidate, Mapping):
                raise TypeError("bracketed_candidates must contain mappings")
            item = _candidate_evidence(
                candidate,
                group_ref=group_ref,
                index=candidate_index,
                base_evidence_id=evidence_id,
            )
            candidates.append(item)
            if item.minimum_status not in ("bracketed_single_minimum", "BRACKETED_SINGLE_MINIMUM"):
                inconsistent_bracketed_candidate = True

    local_refs = (evidence_id,)

    if not groups:
        d04 = DiagnosticRecord(
            DiagnosticID.D04_STATE_COMPETITION,
            Review(ReviewStatus.UNRESOLVED, local_refs, "State-selection report contains no groups"),
            recommended_branch="complete_state_selection",
        )
        d09 = DiagnosticRecord(
            DiagnosticID.D09_PEC_ASYMPTOTES,
            Review(ReviewStatus.UNRESOLVED, local_refs, "No PEC/state-selection groups available"),
            recommended_branch="construct_relevant_pecs",
        )
        return StateSelectionBridgeResult((), d04, d09, 0, False)

    if inconsistent_bracketed_candidate:
        d09 = DiagnosticRecord(
            DiagnosticID.D09_PEC_ASYMPTOTES,
            Review(
                ReviewStatus.CONFIRMED,
                local_refs,
                "Report labels a candidate as bracketed while its minimum scout is not a unique bracketed minimum",
            ),
            state_refs=tuple(item.candidate_ref for item in candidates),
            recommended_branch="audit_state_selection_report",
        )
    elif not all_bracketed:
        d09 = DiagnosticRecord(
            DiagnosticID.D09_PEC_ASYMPTOTES,
            Review(
                ReviewStatus.CONFIRMED,
                local_refs,
                "At least one scouted PEC component remains unbracketed/open; local PEC coverage is incomplete",
            ),
            state_refs=tuple(item.candidate_ref for item in candidates),
            recommended_branch="extend_open_pec_components",
        )
    elif asymptote_review is None:
        d09 = DiagnosticRecord(
            DiagnosticID.D09_PEC_ASYMPTOTES,
            Review(
                ReviewStatus.UNRESOLVED,
                local_refs,
                "All scouted components are locally bracketed, but local bracketing does not establish large-R/asymptotic coverage",
            ),
            state_refs=tuple(item.candidate_ref for item in candidates),
            recommended_branch="resolve_pec_asymptotes",
        )
    else:
        d09 = _mirror_external_review(
            identifier=DiagnosticID.D09_PEC_ASYMPTOTES,
            local_evidence_id=evidence_id,
            external=asymptote_review,
            cleared_rationale="All scouted components are locally bracketed and independent PEC/asymptotic coverage is CLEARED",
            confirmed_rationale="Independent PEC/asymptotic review confirms a relevant unresolved problem",
            unresolved_rationale="Independent PEC/asymptotic review is not yet cleared",
        )

    if not all_bracketed:
        d04 = DiagnosticRecord(
            DiagnosticID.D04_STATE_COMPETITION,
            Review(
                ReviewStatus.CONFIRMED,
                local_refs,
                "Open/unbracketed PEC components can still alter candidate ordering; state competition remains active",
            ),
            state_refs=tuple(item.candidate_ref for item in candidates),
            recommended_branch="extend_open_candidate_pecs",
        )
    elif higher_level_competition_review is None:
        d04 = DiagnosticRecord(
            DiagnosticID.D04_STATE_COMPETITION,
            Review(
                ReviewStatus.UNRESOLVED,
                local_refs,
                "DFT/scout candidate splittings and local brackets do not constitute defensible high-accuracy state-energy intervals",
            ),
            state_refs=tuple(item.candidate_ref for item in candidates),
            recommended_branch="high_accuracy_candidate_comparison",
        )
    else:
        d04 = _mirror_external_review(
            identifier=DiagnosticID.D04_STATE_COMPETITION,
            local_evidence_id=evidence_id,
            external=higher_level_competition_review,
            cleared_rationale="Local candidate PECs are bracketed and the independent higher-level competition review is CLEARED",
            confirmed_rationale="Higher-level candidate competition remains scientifically relevant",
            unresolved_rationale="Higher-level candidate competition is not yet cleared",
        )

    return StateSelectionBridgeResult(
        tuple(candidates),
        d04,
        d09,
        len(groups),
        all_bracketed,
    )


def g1_from_state_selection_bridge(
    *,
    state_search_review: Review,
    bridge: StateSelectionBridgeResult,
) -> Review:
    """Compose G1 using the existing fail-closed three-part G1 policy."""
    return state_completeness_gate(
        state_search_review=state_search_review,
        state_competition_review=bridge.state_competition.review,
        pec_asymptote_review=bridge.pec_asymptotes.review,
    )
