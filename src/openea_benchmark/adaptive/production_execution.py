"""Capability-aware execution routing for OpenEA production closure actions.

This module is the boundary between *scientific planning* and *job execution*.
It deliberately does not infer missing calculation inputs and does not reuse
molecule-specific validation scripts as universal production runners.

The current repository contains generic adaptive runners for cardinal,
diffuse, matched all-electron/frozen-core core-valence convergence, matched
NR/SFX2C1E scalar-relativity convergence, and explicitly authorized CCSDT
triples-reliability diagnostics. Other production components (SOC and nuclear
motion) may have assessment logic or molecule-specific validation scripts, but
they are not therefore universal executable capabilities.

Scientific invariants
---------------------
* Closure priority is preserved: a lower-priority calculation is never run by
  skipping a higher-priority unresolved action.
* Method-validity/review actions are not converted into electronic-structure
  calculations.
* No action is considered executable merely because an OH validation script
  exists for a similar task.
* Unsupported high-order coupled-cluster escalation is policy-blocked.
* Execution requires an explicitly bound adapter carrying the run context;
  the dispatcher never fabricates molecule, geometry, basis or state data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from .adaptive_cardinal_runner import run_adaptive_cardinal_series
from .adaptive_diffuse_runner import run_adaptive_diffuse_series
from .core_valence_runner import run_adaptive_core_valence_series
from .scalar_relativity_runner import run_adaptive_scalar_relativity_series
from .ccsdt_diagnostic_runner import run_adaptive_ccsdt_diagnostic_series

from .production_evidence import (
    CBS_DIFFUSE_RESIDUAL,
    CBS_GEOMETRY_TRANSFER,
    CORE_VALENCE,
    NUCLEAR_MOTION,
    POST_CC,
    REFERENCE_CHARACTER,
    SCALAR_RELATIVITY,
    SCALAR_RELATIVITY_REMAINDER,
    SOC,
    ProductionClosureAction,
    ProductionEvidenceBundle,
)


class ExecutionCapability(str, Enum):
    BASIS_CARDINAL = "BASIS_CARDINAL"
    BASIS_DIFFUSE = "BASIS_DIFFUSE"
    CBS_MODEL = "CBS_MODEL"
    FIXED_GEOMETRY_TRANSFER = "FIXED_GEOMETRY_TRANSFER"
    CORE_VALENCE = "CORE_VALENCE"
    SCALAR_RELATIVITY = "SCALAR_RELATIVITY"
    POST_CC_TRIPLES = "POST_CC_TRIPLES"
    REFERENCE_CHARACTER = "REFERENCE_CHARACTER"
    CORRELATION_RELIABILITY = "CORRELATION_RELIABILITY"
    SCALAR_RELATIVITY_REMAINDER = "SCALAR_RELATIVITY_REMAINDER"
    SOC = "SOC"
    NUCLEAR_MOTION = "NUCLEAR_MOTION"
    UNKNOWN = "UNKNOWN"


class CapabilityImplementation(str, Enum):
    """What the generic OpenEA core can currently execute for a capability."""

    GENERIC_RUNNER_AVAILABLE = "GENERIC_RUNNER_AVAILABLE"
    ASSESSMENT_ONLY = "ASSESSMENT_ONLY"
    REVIEW_ONLY = "REVIEW_ONLY"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    POLICY_FORBIDDEN = "POLICY_FORBIDDEN"


class ExecutionDisposition(str, Enum):
    """How a closure action may proceed in the current architecture."""

    NEEDS_BOUND_CONTEXT = "NEEDS_BOUND_CONTEXT"
    MANUAL_OR_DIAGNOSTIC_REVIEW = "MANUAL_OR_DIAGNOSTIC_REVIEW"
    CAPABILITY_GAP = "CAPABILITY_GAP"
    POLICY_BLOCKED = "POLICY_BLOCKED"


@dataclass(frozen=True)
class ProductionExecutionRequest:
    action: ProductionClosureAction
    capability: ExecutionCapability
    implementation: CapabilityImplementation
    disposition: ExecutionDisposition
    runner_id: str | None
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("Execution request requires a rationale")
        if self.implementation is CapabilityImplementation.GENERIC_RUNNER_AVAILABLE:
            if not self.runner_id:
                raise ValueError("Generic runner capability requires runner_id")
        elif self.runner_id is not None:
            raise ValueError("Non-executable capability must not advertise a runner_id")


@dataclass(frozen=True)
class ProductionExecutionPlan:
    requests: tuple[ProductionExecutionRequest, ...]

    @property
    def next_request(self) -> ProductionExecutionRequest | None:
        return self.requests[0] if self.requests else None

    @property
    def has_capability_gap(self) -> bool:
        return any(x.disposition is ExecutionDisposition.CAPABILITY_GAP for x in self.requests)


@dataclass(frozen=True)
class ExecutionAdapter:
    """Explicit runtime binding between a capability and its run context."""

    capability: ExecutionCapability
    runner_id: str
    execute: Callable[[ProductionClosureAction], Any]

    def __post_init__(self) -> None:
        if not self.runner_id.strip():
            raise ValueError("Execution adapter requires runner_id")
        if not callable(self.execute):
            raise TypeError("Execution adapter execute must be callable")


class ExecutionAttemptStatus(str, Enum):
    EXECUTED = "EXECUTED"
    NO_ACTION_REQUIRED = "NO_ACTION_REQUIRED"
    BLOCKED_NEEDS_CONTEXT = "BLOCKED_NEEDS_CONTEXT"
    BLOCKED_MANUAL_REVIEW = "BLOCKED_MANUAL_REVIEW"
    BLOCKED_CAPABILITY_GAP = "BLOCKED_CAPABILITY_GAP"
    BLOCKED_POLICY = "BLOCKED_POLICY"


@dataclass(frozen=True)
class ExecutionAttempt:
    status: ExecutionAttemptStatus
    request: ProductionExecutionRequest | None
    result: Any = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("Execution attempt requires a rationale")
        if self.status is ExecutionAttemptStatus.EXECUTED and self.request is None:
            raise ValueError("Executed attempt requires a request")


_FORBIDDEN_TOKENS = ("CCSDTQ", "T4", "CCSDTQP", "FCI")
_CARDINAL_RUNNER_ID = f"{run_adaptive_cardinal_series.__module__}.{run_adaptive_cardinal_series.__qualname__}"
_DIFFUSE_RUNNER_ID = f"{run_adaptive_diffuse_series.__module__}.{run_adaptive_diffuse_series.__qualname__}"
_CORE_VALENCE_RUNNER_ID = f"{run_adaptive_core_valence_series.__module__}.{run_adaptive_core_valence_series.__qualname__}"
_SCALAR_RELATIVITY_RUNNER_ID = f"{run_adaptive_scalar_relativity_series.__module__}.{run_adaptive_scalar_relativity_series.__qualname__}"
_CCSDT_DIAGNOSTIC_RUNNER_ID = f"{run_adaptive_ccsdt_diagnostic_series.__module__}.{run_adaptive_ccsdt_diagnostic_series.__qualname__}"


def _is_forbidden_high_order_action(action_id: str) -> bool:
    upper = action_id.upper()
    return any(token in upper for token in _FORBIDDEN_TOKENS)


def _review_request(
    action: ProductionClosureAction,
    capability: ExecutionCapability,
    reason: str,
) -> ProductionExecutionRequest:
    return ProductionExecutionRequest(
        action,
        capability,
        CapabilityImplementation.REVIEW_ONLY,
        ExecutionDisposition.MANUAL_OR_DIAGNOSTIC_REVIEW,
        None,
        reason,
    )


def _gap_request(
    action: ProductionClosureAction,
    capability: ExecutionCapability,
    reason: str,
) -> ProductionExecutionRequest:
    return ProductionExecutionRequest(
        action,
        capability,
        CapabilityImplementation.NOT_IMPLEMENTED,
        ExecutionDisposition.CAPABILITY_GAP,
        None,
        reason,
    )


def classify_closure_action(action: ProductionClosureAction) -> ProductionExecutionRequest:
    """Classify one scientifically ordered closure action for execution.

    The mapping uses both ``addresses_component`` and ``action_id`` because
    strings such as ``COMPUTE_X4`` occur in several physically different
    correction layers.
    """

    action_id = action.action_id.upper()
    component = action.addresses_component.upper()

    if _is_forbidden_high_order_action(action_id):
        return ProductionExecutionRequest(
            action,
            ExecutionCapability.UNKNOWN,
            CapabilityImplementation.POLICY_FORBIDDEN,
            ExecutionDisposition.POLICY_BLOCKED,
            None,
            "Automatic CCSDTQ/T4/higher-rank escalation is outside the OpenEA-v1 production graph.",
        )

    if component == "CARDINAL_CONVERGENCE":
        return ProductionExecutionRequest(
            action,
            ExecutionCapability.BASIS_CARDINAL,
            CapabilityImplementation.GENERIC_RUNNER_AVAILABLE,
            ExecutionDisposition.NEEDS_BOUND_CONTEXT,
            _CARDINAL_RUNNER_ID,
            "A generic cardinal-convergence runner exists; molecule/state/evaluator context must be explicitly bound before execution.",
        )

    if component in {"DIFFUSE_CONVERGENCE", CBS_DIFFUSE_RESIDUAL}:
        return ProductionExecutionRequest(
            action,
            ExecutionCapability.BASIS_DIFFUSE,
            CapabilityImplementation.GENERIC_RUNNER_AVAILABLE,
            ExecutionDisposition.NEEDS_BOUND_CONTEXT,
            _DIFFUSE_RUNNER_ID,
            "A generic diffuse-convergence runner exists; molecule/state/evaluator context must be explicitly bound before execution.",
        )

    if component == "CBS_MODEL":
        return ProductionExecutionRequest(
            action,
            ExecutionCapability.CBS_MODEL,
            CapabilityImplementation.ASSESSMENT_ONLY,
            ExecutionDisposition.CAPABILITY_GAP,
            None,
            "CBS assessment/extrapolation logic exists, but there is no universal closure-job runner for arbitrary missing CBS evidence yet.",
        )

    if component == CBS_GEOMETRY_TRANSFER:
        return _gap_request(
            action,
            ExecutionCapability.FIXED_GEOMETRY_TRANSFER,
            "Fixed-geometry transfer can be evaluated from Stage-3 results, but no universal production execution adapter is implemented yet.",
        )

    if component == CORE_VALENCE:
        return ProductionExecutionRequest(
            action,
            ExecutionCapability.CORE_VALENCE,
            CapabilityImplementation.GENERIC_RUNNER_AVAILABLE,
            ExecutionDisposition.NEEDS_BOUND_CONTEXT,
            _CORE_VALENCE_RUNNER_ID,
            "A generic matched AE/FC core-valence runner exists; validated neutral/anion state provenance and an explicit per-element core-valence basis policy must be bound before execution.",
        )

    if component == SCALAR_RELATIVITY:
        return ProductionExecutionRequest(
            action,
            ExecutionCapability.SCALAR_RELATIVITY,
            CapabilityImplementation.GENERIC_RUNNER_AVAILABLE,
            ExecutionDisposition.NEEDS_BOUND_CONTEXT,
            _SCALAR_RELATIVITY_RUNNER_ID,
            "A generic matched all-electron NR/SFX2C1E scalar-relativity runner exists; validated neutral/anion state provenance and an explicit relativistically suitable per-element basis policy must be bound before execution.",
        )

    if component == POST_CC:
        if action_id == "REASSESS_REFERENCE_CHARACTER":
            return _review_request(
                action,
                ExecutionCapability.REFERENCE_CHARACTER,
                "POST_CC_WARNING must return to the Reference Character Gate rather than launch a higher-rank calculation.",
            )
        if action_id.startswith("COMPUTE_T3_X"):
            return ProductionExecutionRequest(
                action,
                ExecutionCapability.POST_CC_TRIPLES,
                CapabilityImplementation.GENERIC_RUNNER_AVAILABLE,
                ExecutionDisposition.NEEDS_BOUND_CONTEXT,
                _CCSDT_DIAGNOSTIC_RUNNER_ID,
                "A generic, explicitly authorized CCSDT triples-reliability runner exists; validated neutral/anion state provenance, frozen-core definition, basis policy, source checkpoints, and authorization evidence must be bound before execution.",
            )
        return ProductionExecutionRequest(
            action,
            ExecutionCapability.POST_CC_TRIPLES,
            CapabilityImplementation.ASSESSMENT_ONLY,
            ExecutionDisposition.CAPABILITY_GAP,
            None,
            "Post-CC diagnostic assessment exists, but this action is not a recognized executable DeltaT3 calculation request.",
        )

    if component == REFERENCE_CHARACTER:
        return _review_request(
            action,
            ExecutionCapability.REFERENCE_CHARACTER,
            "Reference-character actions require reviewed electronic-structure diagnostics; they are not blind production calculations.",
        )

    if component == "CORRELATION_RELIABILITY":
        return _review_request(
            action,
            ExecutionCapability.CORRELATION_RELIABILITY,
            "Correlation reliability must be assessed from diagnostic evidence before any optional CCSDT calculation is authorized.",
        )

    if component == SCALAR_RELATIVITY_REMAINDER:
        return _gap_request(
            action,
            ExecutionCapability.SCALAR_RELATIVITY_REMAINDER,
            "Two-electron/picture-change relativistic remainder has no universal OpenEA-v1 production implementation yet.",
        )

    if component == SOC:
        return _gap_request(
            action,
            ExecutionCapability.SOC,
            "SOC is an explicit production-layer capability gap in the current repository.",
        )

    if component == NUCLEAR_MOTION:
        return _gap_request(
            action,
            ExecutionCapability.NUCLEAR_MOTION,
            "Final nuclear-motion/ZPE production execution is not yet implemented in the generic workflow.",
        )

    # Repair/review actions should never be guessed into calculations.
    if action_id.startswith(("REPAIR_", "REVIEW_", "RESOLVE_", "ASSESS_")):
        return _review_request(
            action,
            ExecutionCapability.UNKNOWN,
            "This action is a review/repair step and requires an explicit diagnostic implementation.",
        )

    return ProductionExecutionRequest(
        action,
        ExecutionCapability.UNKNOWN,
        CapabilityImplementation.NOT_IMPLEMENTED,
        ExecutionDisposition.CAPABILITY_GAP,
        None,
        "No universal execution capability is registered for this closure action.",
    )


def build_production_execution_plan(bundle: ProductionEvidenceBundle) -> ProductionExecutionPlan:
    """Translate already-prioritized closure actions without reordering them."""

    return ProductionExecutionPlan(tuple(classify_closure_action(x) for x in bundle.closure_actions))


def execute_next_closure_action(
    plan: ProductionExecutionPlan,
    *,
    adapters: Mapping[ExecutionCapability, ExecutionAdapter] | None = None,
) -> ExecutionAttempt:
    """Execute only the highest-priority closure action when safely bound.

    Lower-priority requests are never skipped.  This is important when, for
    example, reference-method validity is open while a basis refinement could
    technically be launched: the method-validity problem wins.
    """

    request = plan.next_request
    if request is None:
        return ExecutionAttempt(
            ExecutionAttemptStatus.NO_ACTION_REQUIRED,
            None,
            None,
            "No production closure action is pending.",
        )

    if request.disposition is ExecutionDisposition.POLICY_BLOCKED:
        return ExecutionAttempt(
            ExecutionAttemptStatus.BLOCKED_POLICY,
            request,
            None,
            request.reason,
        )
    if request.disposition is ExecutionDisposition.MANUAL_OR_DIAGNOSTIC_REVIEW:
        return ExecutionAttempt(
            ExecutionAttemptStatus.BLOCKED_MANUAL_REVIEW,
            request,
            None,
            request.reason,
        )
    if request.disposition is ExecutionDisposition.CAPABILITY_GAP:
        return ExecutionAttempt(
            ExecutionAttemptStatus.BLOCKED_CAPABILITY_GAP,
            request,
            None,
            request.reason,
        )

    bound = {} if adapters is None else dict(adapters)
    adapter = bound.get(request.capability)
    if adapter is None:
        return ExecutionAttempt(
            ExecutionAttemptStatus.BLOCKED_NEEDS_CONTEXT,
            request,
            None,
            "A generic runner exists, but no explicit runtime adapter/context was supplied.",
        )
    if adapter.capability is not request.capability:
        raise ValueError("Execution adapter capability does not match request")
    if adapter.runner_id != request.runner_id:
        raise ValueError("Execution adapter runner_id does not match the authorized generic runner")

    result = adapter.execute(request.action)
    return ExecutionAttempt(
        ExecutionAttemptStatus.EXECUTED,
        request,
        result,
        "Highest-priority closure action executed through an explicitly bound generic runner adapter.",
    )
