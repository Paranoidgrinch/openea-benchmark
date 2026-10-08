"""OpenEA v1 adaptive scientific decision and execution layer.

Decision objects remain separated from electronic-structure execution.  The
package also exposes explicit, fail-closed generic runners whose expensive
backends are invoked only after scientific planning and runtime context have
been bound.  Experimental reference EAs are never used as hidden production
inputs.
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
    EnergyReliabilityGateSet,
    GateSet,
    MethodRole,
    ScientificResolutionStatus,
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
from .scientific_gates import (
    basis_diffuse_convergence_review,
    cbs_resolution_review,
    build_energy_reliability_gates,
    correlation_reliability_review,
    physical_corrections_review,
    reference_method_validity_review,
    uncertainty_closure_review,
)
from .multireference import (
    MRBranchResolution,
    MRBranchStatus,
    MRCapabilityStatus,
    MRProductionCapability,
    resolve_multireference_branch,
)
from .planner import ProductionRoutePlan, ProductionRouteStatus, plan_production_route
from .production_evidence import (
    ADIABATIC_NUCLEAR_REMAINDER,
    CBS_DIFFUSE_RESIDUAL,
    CBS_GEOMETRY_TRANSFER,
    CORE_VALENCE,
    NUCLEAR_MOTION,
    POST_CC,
    REFERENCE_CHARACTER,
    SCALAR_RELATIVITY,
    SCALAR_RELATIVITY_REMAINDER,
    SOC,
    ClosurePriority,
    CorrectionEvidence,
    ProductionClosureAction,
    ProductionEvidenceBundle,
    ProductionEvidencePlan,
    ProductionEvidencePlanningStatus,
    bounded_external_correction,
    build_single_reference_production_evidence_bundle,
    core_valence_evidence,
    not_applicable_correction,
    nuclear_motion_evidence,
    pending_correction,
    plan_production_evidence,
    post_cc_evidence,
    scalar_relativity_evidence,
)
from .core_valence_runner import (
    AdaptiveCoreValenceResult,
    AdaptiveCoreValenceStatus,
    CoreValenceBasisSpec,
    CoreValenceCardinalEvidence,
    CoreValenceCorrelationSpace,
    CoreValenceStateSpec,
    CoreValenceSubcalculation,
    run_adaptive_core_valence_series,
)
from .scalar_relativity_runner import (
    AdaptiveScalarRelativityResult,
    AdaptiveScalarRelativityStatus,
    ScalarRelativityBasisSpec,
    ScalarRelativityCardinalEvidence,
    ScalarRelativityHamiltonian,
    ScalarRelativityStateSpec,
    ScalarRelativitySubcalculation,
    run_adaptive_scalar_relativity_series,
)
from .ccsdt_diagnostic_runner import (
    AdaptiveCCSDTDiagnosticResult,
    AdaptiveCCSDTDiagnosticStatus,
    CCSDTCardinalEvidence,
    CCSDTDiagnosticAuthorization,
    CCSDTDiagnosticBasisSpec,
    CCSDTDiagnosticExecutionSettings,
    CCSDTDiagnosticMethod,
    CCSDTDiagnosticStateSpec,
    CCSDTMethodExecutionStatus,
    CCSDTMethodRequest,
    CCSDTMethodResult,
    CCSDTSubcalculation,
    run_adaptive_ccsdt_diagnostic_series,
)
from .nuclear_motion import (
    ANGSTROM_TO_BOHR,
    HARTREE_TO_EV,
    U_TO_ELECTRON_MASS,
    DiatomicMassSpecification,
    NuclearMotionAssessment,
    NuclearMotionModelEvidence,
    NuclearMotionPEC,
    NuclearMotionSettings,
    NuclearMotionStatus,
    VibrationalBindingAssessment,
    VibrationalBindingStatus,
    VibrationalGroundStateResult,
    VibrationalSolveStatus,
    assess_vibrational_binding,
    derive_nuclear_motion_model_evidence,
    nuclear_motion_pec_from_high_level,
    run_diatomic_nuclear_motion,
    solve_vibrational_ground_state,
)
from .nuclear_motion_orchestration import (
    NuclearPECModelLevel,
    NuclearPECModelRun,
    NuclearPECModelRunStatus,
    NuclearPECRefinementBatch,
    NuclearPECRefinementRun,
    NuclearPECRefinementRunStatus,
    NuclearPECStage3Context,
    plan_nuclear_pec_refinement,
    run_nuclear_pec_model_convergence,
    run_nuclear_pec_refinement,
)
from .production_execution import (
    CapabilityImplementation,
    ExecutionAdapter,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionCapability,
    ExecutionDisposition,
    ProductionExecutionPlan,
    ProductionExecutionRequest,
    build_production_execution_plan,
    classify_closure_action,
    execute_next_closure_action,
)
from .attachment_continuum import (
    AttachmentContinuumAssessment,
    AttachmentContinuumSettings,
    AttachmentContinuumStatus,
    ValenceAttachmentEvidence,
    EOMEAAssessmentStatus,
    EOMEAAttachmentAssessment,
    EOMEAAttachmentPoint,
    StabilizationAssessment,
    StabilizationPoint,
    StabilizationStatus,
    assess_attachment_continuum,
    assess_eom_ea_diffuse_series,
    assess_stabilization_series,
    direct_diffuse_review_from_assessment,
    pyscf_eom_eigenvalue_to_attachment_ea_ev,
)
from .attachment_eom_runner import (
    G2EOMStatus, G2EOMNeutralState, G2EOMBasis, G2EOMStabilization,
    G2EOMAuthorization, G2EOMSettings, DiffuseShellSelector,
    G2EOMRequest, G2EOMRoot, G2EOMRawResult, G2EOMSubpoint, G2EOMSeries,
    run_g2_eom_diagnostics, scaled_diffuse_basis, evidence_points_from_review,
)
from .attachment_root_continuity import (
    RootContinuityStatus, RootContinuitySettings, RootContinuityLink,
    RootContinuityReport, normalized_ao_one_particle_overlap,
    pyscf_cross_ao_overlap, propose_g2_root_continuity,
)
from .attachment_stabilization_profile import (
    StabilizationProfileStatus, StabilizationProfileSettings,
    StabilizationProfilePoint, StabilizationProfileReport,
    analyze_g2_stabilization_profile,
)
from .scientific_resolution import (
    PhysicalValidityAssessment,
    PhysicalValidityStatus,
    ScientificResolutionPath,
    ScientificResolutionResult,
    physical_validity_from_binding,
    physical_validity_with_nuclear_motion,
    resolve_scientific_outcome,
)
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
    'Interval', 'UncertaintyComponent', 'ErrorBudget', 'EAEstimate',
    'EnergyReliabilityGateSet', 'GateSet', 'MethodRole',
    'ScientificResolutionStatus', 'ReferenceCharacterStatus', 'ReferenceCharacterAssessment',
    'ground_state_interval', 'ea_from_state_intervals',
    'DecisionInput', 'DecisionOutput', 'DecisionCategory', 'PrecisionStatus',
    'evaluate_decision', 'evaluate_estimate', 'DIAGNOSTIC_CATALOG',
    'PlannedDiagnostic', 'next_diagnostics', 'stability_diagnostic_from_root',
    'assess_reference_character', 'reference_method_validity_review',
    'basis_diffuse_convergence_review', 'cbs_resolution_review', 'correlation_reliability_review',
    'physical_corrections_review', 'uncertainty_closure_review',
    'build_energy_reliability_gates', 'MRCapabilityStatus',
    'MRProductionCapability', 'MRBranchStatus', 'MRBranchResolution',
    'resolve_multireference_branch', 'ProductionRouteStatus',
    'ProductionRoutePlan', 'plan_production_route',
    'PrecisionActionCandidate', 'PrecisionPlan', 'PrecisionPlanningStatus',
    'PrecisionTargetAssessment', 'assess_precision_target',
    'plan_precision_refinement', 'CorrectionEvidence', 'ClosurePriority',
    'ProductionClosureAction', 'ProductionEvidenceBundle',
    'ProductionEvidencePlan', 'ProductionEvidencePlanningStatus',
    'build_single_reference_production_evidence_bundle', 'plan_production_evidence',
    'core_valence_evidence', 'scalar_relativity_evidence', 'post_cc_evidence',
    'bounded_external_correction', 'not_applicable_correction', 'nuclear_motion_evidence', 'pending_correction',
    'CORE_VALENCE', 'SCALAR_RELATIVITY', 'SCALAR_RELATIVITY_REMAINDER',
    'SOC', 'NUCLEAR_MOTION', 'ADIABATIC_NUCLEAR_REMAINDER', 'POST_CC', 'CBS_DIFFUSE_RESIDUAL',
    'CBS_GEOMETRY_TRANSFER', 'REFERENCE_CHARACTER',
    'ExecutionCapability', 'CapabilityImplementation', 'ExecutionDisposition',
    'ProductionExecutionRequest', 'ProductionExecutionPlan', 'ExecutionAdapter',
    'ExecutionAttemptStatus', 'ExecutionAttempt', 'classify_closure_action',
    'build_production_execution_plan', 'execute_next_closure_action',
    'CoreValenceCorrelationSpace', 'AdaptiveCoreValenceStatus',
    'CoreValenceStateSpec', 'CoreValenceBasisSpec', 'CoreValenceSubcalculation',
    'CoreValenceCardinalEvidence', 'AdaptiveCoreValenceResult',
    'run_adaptive_core_valence_series',
    'ScalarRelativityHamiltonian', 'AdaptiveScalarRelativityStatus',
    'ScalarRelativityStateSpec', 'ScalarRelativityBasisSpec',
    'ScalarRelativitySubcalculation', 'ScalarRelativityCardinalEvidence',
    'AdaptiveScalarRelativityResult', 'run_adaptive_scalar_relativity_series',
    'CCSDTDiagnosticMethod', 'CCSDTMethodExecutionStatus',
    'AdaptiveCCSDTDiagnosticStatus', 'CCSDTDiagnosticAuthorization',
    'CCSDTDiagnosticStateSpec', 'CCSDTDiagnosticBasisSpec',
    'CCSDTDiagnosticExecutionSettings', 'CCSDTMethodRequest',
    'CCSDTMethodResult', 'CCSDTSubcalculation', 'CCSDTCardinalEvidence',
    'AdaptiveCCSDTDiagnosticResult', 'run_adaptive_ccsdt_diagnostic_series',
    'ANGSTROM_TO_BOHR', 'HARTREE_TO_EV', 'U_TO_ELECTRON_MASS',
    'DiatomicMassSpecification', 'NuclearMotionModelEvidence', 'NuclearMotionPEC', 'NuclearMotionSettings',
    'VibrationalSolveStatus', 'VibrationalGroundStateResult',
    'VibrationalBindingStatus', 'VibrationalBindingAssessment',
    'NuclearMotionStatus', 'NuclearMotionAssessment',
    'nuclear_motion_pec_from_high_level', 'solve_vibrational_ground_state',
    'assess_vibrational_binding', 'derive_nuclear_motion_model_evidence', 'run_diatomic_nuclear_motion',
    'NuclearPECStage3Context', 'NuclearPECRefinementBatch',
    'NuclearPECRefinementRunStatus', 'NuclearPECRefinementRun',
    'NuclearPECModelLevel', 'NuclearPECModelRunStatus', 'NuclearPECModelRun',
    'plan_nuclear_pec_refinement', 'run_nuclear_pec_refinement',
    'run_nuclear_pec_model_convergence',
    'AttachmentContinuumStatus', 'AttachmentContinuumSettings', 'ValenceAttachmentEvidence',
    'EOMEAAssessmentStatus', 'EOMEAAttachmentPoint', 'EOMEAAttachmentAssessment',
    'StabilizationStatus', 'StabilizationPoint', 'StabilizationAssessment',
    'AttachmentContinuumAssessment', 'direct_diffuse_review_from_assessment',
    'assess_eom_ea_diffuse_series', 'assess_stabilization_series', 'assess_attachment_continuum',
    'pyscf_eom_eigenvalue_to_attachment_ea_ev',
    'G2EOMStatus', 'G2EOMNeutralState', 'G2EOMBasis', 'G2EOMStabilization',
    'G2EOMAuthorization', 'G2EOMSettings', 'DiffuseShellSelector',
    'G2EOMRequest', 'G2EOMRoot', 'G2EOMRawResult', 'G2EOMSubpoint', 'G2EOMSeries',
    'run_g2_eom_diagnostics', 'scaled_diffuse_basis', 'evidence_points_from_review',
    'PhysicalValidityStatus', 'PhysicalValidityAssessment',
    'ScientificResolutionPath', 'ScientificResolutionResult',
    'physical_validity_from_binding', 'physical_validity_with_nuclear_motion', 'resolve_scientific_outcome',
    'RootContinuityStatus', 'RootContinuitySettings', 'RootContinuityLink',
    'RootContinuityReport', 'normalized_ao_one_particle_overlap',
    'pyscf_cross_ao_overlap', 'propose_g2_root_continuity',
    'StabilizationProfileStatus', 'StabilizationProfileSettings',
    'StabilizationProfilePoint', 'StabilizationProfileReport',
    'analyze_g2_stabilization_profile',
    'MRActiveSpaceStatus', 'MRActiveSpaceThresholds', 'MRActiveSpaceRootShift',
    'MRActiveSpaceComparison', 'assess_mr_active_space_pair',
    'pyscf_mr_same_geometry_ao_metric',
]

from .attachment_continuum_independent import (
    CAPPoint, CAPTrajectorySettings, CAPTrajectoryStatus, CAPTrajectoryReport,
    ContinuumMethod, ContinuumScope, ContinuumFinding,
    IndependentContinuumDossier, assess_cap_trajectory,
)

# Generic MR execution (method-development status; no validated production EA).
from .mr_casscf_nevpt2_runner import (
    MRPointAuthorization,
    MRPointRequest,
    MRPointSettings,
    MRPointStatus,
    MRPointResult,
    MRRootEnergy,
    run_mr_casscf_nevpt2_point,
)

# MR PEC fingerprint comparison. Candidate only: never clears state identity.
from .mr_pec_continuity import (
    MRPECContinuityStatus, MRPECContinuityThresholds, MRPECContinuityResult,
    assess_mr_pec_continuity, pyscf_mr_ao_overlap_matrices,
)

# Same-geometry MR active-space model-sensitivity diagnostics (never G3 closure).
from .mr_active_space_comparison import (
    MRActiveSpaceStatus, MRActiveSpaceThresholds, MRActiveSpaceRootShift,
    MRActiveSpaceComparison, assess_mr_active_space_pair,
    pyscf_mr_same_geometry_ao_metric,
)

# Conditional MR state-interaction SOC diagnostic (FCI-SISO external pinned code).
from .soc_fci_siso_runner import (
    FCI_SISO_PIN, SOCPointStatus, SOCSpinManifold, SOCPointRequest,
    SOCPointAuthorization, SOCPointSettings, SOCSpinFreeRoot, SOCPointResult,
    SOCEACandidate, run_soc_fci_siso_point, assess_soc_ea_pair,
)

__all__.extend([
    'FCI_SISO_PIN', 'SOCPointStatus', 'SOCSpinManifold', 'SOCPointRequest',
    'SOCPointAuthorization', 'SOCPointSettings', 'SOCSpinFreeRoot',
    'SOCPointResult', 'SOCEACandidate', 'run_soc_fci_siso_point',
    'assess_soc_ea_pair',
])

# Nested SOC spin-manifold sensitivity: observed numerical differences only.
from .soc_manifold_sensitivity import (
    SOCManifoldSensitivityStatus, SOCManifoldSensitivitySettings,
    SOCManifoldSensitivity, compare_soc_manifold_expansion,
    run_soc_manifold_expansion,
)
__all__.extend([
    'SOCManifoldSensitivityStatus', 'SOCManifoldSensitivitySettings',
    'SOCManifoldSensitivity', 'compare_soc_manifold_expansion',
    'run_soc_manifold_expansion',
])

# Finite-grid numerical correction transfer across a PEC: review-only diagnostic.
from .cbs_pec_transfer import (
    TransferStatus, MatchedCorrectionPoint, SpeciesTransfer,
    PECTransferAssessment, matched_stage3_point, evaluate_pec_correction_transfer,
)
__all__.extend([
    'TransferStatus', 'MatchedCorrectionPoint', 'SpeciesTransfer',
    'PECTransferAssessment', 'matched_stage3_point',
    'evaluate_pec_correction_transfer',
])
