"""Read-only bridge from existing OpenEA discovery records to adaptive reviews.

This module deliberately does *not* alter SCF roots, branch graphs, checkpoints,
or existing scientific policy.  It translates already-computed evidence into
fail-closed adaptive diagnostics.

Core rule
---------
A clean numerical/continuity result can remove a local warning, but it cannot
by itself prove global state completeness.  In particular:

* low spin contamination alone does not clear D01;
* internal SCF stability alone does not clear D02;
* an unambiguous BranchGraph alone does not clear D04/G1 because one graph is
  restricted to a fixed charge, spin sector, functional, basis and reference.

Production thresholds are intentionally supplied by the caller.  This bridge
contains no molecule-specific or experimentally tuned defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, TYPE_CHECKING

from .model import (
    DiagnosticID,
    DiagnosticRecord,
    GateSet,
    Review,
    ReviewStatus,
)
from .adapters import stability_diagnostic_from_root

if TYPE_CHECKING:
    from openea_benchmark.branch_graph import BranchGraph
    from openea_benchmark.root_record import SCFRootRecord


@dataclass(frozen=True)
class SpinWarningPolicy:
    """Development policy for flagging |<S^2>-S(S+1)|.

    The value is a *warning trigger*, not a universal validity threshold.
    Falling below it never proves that a single-reference treatment is valid.
    No production default is provided on purpose.
    """

    max_abs_delta_s2_before_warning: float

    def __post_init__(self) -> None:
        value = float(self.max_abs_delta_s2_before_warning)
        if not isfinite(value) or value < 0.0:
            raise ValueError("spin warning threshold must be finite and >= 0")


@dataclass(frozen=True)
class DiscoveryBridgeResult:
    """Adaptive evidence extracted from existing discovery objects."""

    root_diagnostics: tuple[DiagnosticRecord, ...]
    branch_diagnostics: tuple[DiagnosticRecord, ...]
    state_completeness_gate: Review

    @property
    def all_diagnostics(self) -> tuple[DiagnosticRecord, ...]:
        return self.root_diagnostics + self.branch_diagnostics


def _enum_value(value) -> str:
    return str(getattr(value, "value", value))


def spin_diagnostic_from_root(
    root: "SCFRootRecord",
    *,
    policy: SpinWarningPolicy,
) -> DiagnosticRecord:
    """Translate one SCF root into conservative D01 evidence.

    A large deviation is enough to CONFIRM a spin-representation warning.
    A small deviation is only local negative evidence and therefore remains
    UNRESOLVED until reference/state-level controls clear D01.
    """

    root_id = str(root.root_id)
    if not root_id.strip():
        raise ValueError("SCF root must carry root_id")

    status = _enum_value(root.status)
    refs = (root_id,)

    if status == "FAILED":
        review = Review(
            ReviewStatus.UNRESOLVED,
            refs,
            "SCF failed; spin representation cannot be assessed",
        )
        return DiagnosticRecord(DiagnosticID.D01_SPIN, review, state_refs=refs)

    s2 = getattr(root, "s2", None)
    if s2 is None:
        review = Review(
            ReviewStatus.UNRESOLVED,
            refs,
            "<S^2> is unavailable",
        )
        return DiagnosticRecord(DiagnosticID.D01_SPIN, review, state_refs=refs)

    expected = getattr(root, "expected_s2", None)
    if expected is None:
        spin_2s = int(root.spin_2s)
        spin_s = spin_2s / 2.0
        expected = spin_s * (spin_s + 1.0)

    delta = float(s2) - float(expected)
    abs_delta = abs(delta)
    threshold = policy.max_abs_delta_s2_before_warning

    if abs_delta > threshold:
        review = Review(
            ReviewStatus.CONFIRMED,
            refs,
            (
                "Spin-contamination warning: "
                f"|<S^2>-S(S+1)|={abs_delta:.8g} exceeds caller-supplied "
                f"development trigger {threshold:.8g}. This flags D01 but "
                "does not by itself select an SR or MR production method."
            ),
        )
    else:
        review = Review(
            ReviewStatus.UNRESOLVED,
            refs,
            (
                "No spin-contamination warning at the caller-supplied trigger "
                f"({abs_delta:.8g} <= {threshold:.8g}), but low contamination "
                "alone cannot clear D01."
            ),
        )

    return DiagnosticRecord(DiagnosticID.D01_SPIN, review, state_refs=refs)


def branch_diagnostic_from_graph(
    graph: "BranchGraph",
    *,
    evidence_id: str,
    candidate_competition_review: Review | None = None,
) -> DiagnosticRecord:
    """Translate a BranchGraph into D04 without overclaiming completeness.

    BranchGraph continuity is defined inside one fixed electronic-method and
    spin context in the existing repository.  Therefore:

    * any ambiguous edge/topology CONFIRMS an unresolved state-continuity /
      competition problem;
    * a fully unambiguous graph remains UNRESOLVED for D04 unless the caller
      additionally supplies a CLEARED higher-level candidate-competition
      review covering the relevant alternative spin/state sectors.
    """

    evidence_id = str(evidence_id).strip()
    if not evidence_id:
        raise ValueError("branch graph requires a non-empty evidence_id")

    root_ids: list[str] = []
    for _r, ids in graph.root_ids_by_geometry:
        root_ids.extend(str(item) for item in ids)
    state_refs = tuple(dict.fromkeys(root_ids))
    refs = (evidence_id,)

    if not bool(graph.is_fully_unambiguous):
        details = []
        if getattr(graph, "ambiguous_edges", ()):
            details.append(f"ambiguous_edges={len(graph.ambiguous_edges)}")
        if getattr(graph, "topology_ambiguous_root_ids", ()):
            details.append(
                "topology_ambiguous_roots="
                f"{len(graph.topology_ambiguous_root_ids)}"
            )
        if any(not component.is_unambiguous for component in graph.components):
            details.append("ambiguous_component_topology")
        review = Review(
            ReviewStatus.CONFIRMED,
            refs,
            "Branch graph contains unresolved continuity/topology evidence"
            + (": " + ", ".join(details) if details else ""),
        )
        return DiagnosticRecord(
            DiagnosticID.D04_STATE_COMPETITION,
            review,
            state_refs=state_refs,
            recommended_branch="resolve_state_continuity",
        )

    if candidate_competition_review is None:
        review = Review(
            ReviewStatus.UNRESOLVED,
            refs,
            (
                "Branch graph is internally unambiguous, but this graph covers "
                "only one fixed charge/spin/method context. Cross-state and "
                "cross-spin candidate competition has not been supplied."
            ),
        )
    elif candidate_competition_review.status is ReviewStatus.CLEARED:
        combined_ids = tuple(dict.fromkeys(refs + candidate_competition_review.evidence_ids))
        review = Review(
            ReviewStatus.CLEARED,
            combined_ids,
            (
                "Branch continuity is unambiguous and the supplied higher-level "
                "candidate-competition review is CLEARED."
            ),
        )
    elif candidate_competition_review.status is ReviewStatus.CONFIRMED:
        combined_ids = tuple(dict.fromkeys(refs + candidate_competition_review.evidence_ids))
        review = Review(
            ReviewStatus.CONFIRMED,
            combined_ids,
            "Higher-level candidate competition remains scientifically relevant",
        )
    else:
        review = Review(
            ReviewStatus.UNRESOLVED,
            refs + candidate_competition_review.evidence_ids,
            "Higher-level candidate competition is not yet cleared",
        )

    return DiagnosticRecord(
        DiagnosticID.D04_STATE_COMPETITION,
        review,
        state_refs=state_refs,
    )


def state_completeness_gate(
    *,
    state_search_review: Review,
    state_competition_review: Review,
    pec_asymptote_review: Review,
) -> Review:
    """Build G1 from explicit independent reviews.

    G1 clears only if all three requirements are explicitly CLEARED:
      1. the configured state search has adequate coverage;
      2. remaining candidates cannot change the ground-state assignment;
      3. the relevant PEC/asymptotic coverage is adequate.

    A local branch graph can contribute evidence to (2), but cannot replace
    (1) or (3).
    """

    items = (
        ("STATE_SEARCH", state_search_review),
        ("STATE_COMPETITION", state_competition_review),
        ("PEC_ASYMPTOTES", pec_asymptote_review),
    )
    refs = tuple(
        dict.fromkeys(
            evidence_id
            for _name, review in items
            for evidence_id in review.evidence_ids
        )
    )

    if all(review.status is ReviewStatus.CLEARED for _name, review in items):
        if not refs:
            raise ValueError("G1 clearance requires concrete evidence IDs")
        return Review(
            ReviewStatus.CLEARED,
            refs,
            "State-search coverage, candidate competition and PEC/asymptote coverage are cleared",
        )

    unresolved = [
        f"{name}={review.status.value}"
        for name, review in items
        if review.status is not ReviewStatus.CLEARED
    ]
    return Review(
        ReviewStatus.UNRESOLVED,
        refs,
        "G1 remains open: " + ", ".join(unresolved),
    )


def bridge_discovery_records(
    roots: Iterable["SCFRootRecord"],
    branch_graphs: Iterable[tuple[str, "BranchGraph"]],
    *,
    spin_policy: SpinWarningPolicy,
    state_search_review: Review,
    candidate_competition_review: Review,
    pec_asymptote_review: Review,
) -> DiscoveryBridgeResult:
    """Create a deterministic read-only discovery-to-adaptive evidence bundle."""

    roots_tuple = tuple(roots)
    graphs_tuple = tuple(branch_graphs)

    root_records: list[DiagnosticRecord] = []
    for root in roots_tuple:
        root_records.append(spin_diagnostic_from_root(root, policy=spin_policy))
        root_records.append(stability_diagnostic_from_root(root))

    branch_records = tuple(
        branch_diagnostic_from_graph(
            graph,
            evidence_id=evidence_id,
            candidate_competition_review=candidate_competition_review,
        )
        for evidence_id, graph in graphs_tuple
    )

    # Important: G1 consumes the explicit higher-level competition review,
    # not a local graph result alone.
    gate = state_completeness_gate(
        state_search_review=state_search_review,
        state_competition_review=candidate_competition_review,
        pec_asymptote_review=pec_asymptote_review,
    )

    return DiscoveryBridgeResult(tuple(root_records), branch_records, gate)


def gates_with_state_completeness(
    base: GateSet,
    state_completeness: Review,
) -> GateSet:
    """Return a new immutable GateSet with G1 replaced; G2/G3 are preserved."""

    return GateSet(
        state_completeness=state_completeness,
        attachment_resolution=base.attachment_resolution,
        energy_reliability=base.energy_reliability,
        energy_subgates=base.energy_subgates,
    )
