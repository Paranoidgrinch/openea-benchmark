"""Experimental, method-independent OpenEA v1 scientific decision layer.

This package consumes externally validated evidence. It NEVER runs quantum
chemistry or reads experimental reference EAs. Not a production policy yet.
"""
from .model import (
    DiagnosticID,
    DiagnosticRecord,
    EvidenceQuality,
    Review,
    ReviewStatus,
    Interval,
    UncertaintyComponent,
    ErrorBudget,
    EAEstimate,
    GateSet,
    MethodRole,
    ReferenceCharacterStatus,
    ReferenceCharacterAssessment,
    ground_state_interval,
    ea_from_state_intervals,
)
from .decision import (
    DecisionInput,
    DecisionOutput,
    DecisionCategory,
    PrecisionStatus,
    evaluate_decision,
    evaluate_estimate,
)
from .diagnostics import DIAGNOSTIC_CATALOG, PlannedDiagnostic, next_diagnostics
from .adapters import stability_diagnostic_from_root
from .reference_character import assess_reference_character
from .precision_controller import (
    PrecisionActionCandidate,
    PrecisionPlan,
    PrecisionPlanningStatus,
    PrecisionTargetAssessment,
    assess_precision_target,
    plan_precision_refinement,
)

__all__ = [
    'DiagnosticID', 'DiagnosticRecord', 'EvidenceQuality', 'Review', 'ReviewStatus',
    'Interval', 'UncertaintyComponent', 'ErrorBudget', 'EAEstimate', 'GateSet',
    'MethodRole', 'ReferenceCharacterStatus', 'ReferenceCharacterAssessment',
    'ground_state_interval', 'ea_from_state_intervals',
    'DecisionInput', 'DecisionOutput', 'DecisionCategory', 'PrecisionStatus',
    'evaluate_decision', 'evaluate_estimate', 'DIAGNOSTIC_CATALOG',
    'PlannedDiagnostic', 'next_diagnostics', 'stability_diagnostic_from_root',
    'assess_reference_character', 'PrecisionActionCandidate', 'PrecisionPlan',
    'PrecisionPlanningStatus', 'PrecisionTargetAssessment',
    'assess_precision_target', 'plan_precision_refinement',
]
