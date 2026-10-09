"""
OpenEA-Benchmark scientific workflow infrastructure.

No production electron-affinity method is frozen yet.
"""

from .branch_graph import (
    BranchComponent,
    BranchGraph,
    build_branch_graph,
    build_branch_graph_from_comparisons,
)
from .branch_continuity import (
    BranchComparison,
    BranchRelation,
    BranchThresholds,
    classify_branch_metrics,
    compare_branch_roots,
)
from .checkpoint_fingerprint import (
    CheckpointAuditSettings,
    CheckpointDeduplicationResult,
    CheckpointFingerprintAudit,
    CheckpointFingerprintFailure,
    deduplicate_checkpoint_roots,
    fingerprint_from_checkpoint,
    fingerprints_from_checkpoints,
)
from .local_pec import (
    LocalPEC,
    LocalPECConstructionResult,
    LocalPECPoint,
    LocalPECRejectionReason,
    RejectedLocalPECComponent,
    construct_local_pecs,
)
from .pyscf_backend import (
    DEFAULT_GUESSES,
    DFTMethodSpec,
    DiatomicSpec,
    SCFSettings,
    build_molecule,
    run_guess_panel,
    run_scf_attempt,
)
from .root_record import (
    SCFRootRecord,
    SCFRunStatus,
)
from .state_identity import (
    DeduplicationResult,
    IdentityThresholds,
    RootCluster,
    StateComparison,
    StateFingerprint,
    StateRelation,
    classify_metrics,
    compare_states,
    deduplicate_roots,
    fingerprint_from_orthonormal_density,
)

from .minimum_scout import (
    MinimumCandidate,
    MinimumScoutResult,
    MinimumScoutStatus,
    MinimumScoutThresholds,
    scout_local_pec_minimum,
)

from .vibrational import (
    HarmonicVibration,
    PECPolynomialFit,
    VibrationalAnalysisResult,
    VibrationalAnalysisStatus,
    VibrationalFitSettings,
    analyze_local_pec_vibration,
    diatomic_reduced_mass_amu,
)

__all__ = [
    "HarmonicVibration",
    "PECPolynomialFit",
    "VibrationalAnalysisResult",
    "VibrationalAnalysisStatus",
    "VibrationalFitSettings",
    "analyze_local_pec_vibration",
    "diatomic_reduced_mass_amu",
    "MinimumCandidate",
    "MinimumScoutResult",
    "MinimumScoutStatus",
    "MinimumScoutThresholds",
    "scout_local_pec_minimum",
    "BranchComponent",
    "BranchGraph",
    "BranchComparison",
    "BranchRelation",
    "BranchThresholds",
    "CheckpointAuditSettings",
    "CheckpointDeduplicationResult",
    "CheckpointFingerprintAudit",
    "CheckpointFingerprintFailure",
    "DEFAULT_GUESSES",
    "DFTMethodSpec",
    "DeduplicationResult",
    "DiatomicSpec",
    "IdentityThresholds",
    "LocalPEC",
    "LocalPECConstructionResult",
    "LocalPECPoint",
    "LocalPECRejectionReason",
    "RejectedLocalPECComponent",
    "RootCluster",
    "SCFRootRecord",
    "SCFRunStatus",
    "SCFSettings",
    "StateComparison",
    "StateFingerprint",
    "StateRelation",
    "build_branch_graph",
    "build_branch_graph_from_comparisons",
    "build_molecule",
    "classify_branch_metrics",
    "compare_branch_roots",
    "classify_metrics",
    "compare_states",
    "construct_local_pecs",
    "deduplicate_checkpoint_roots",
    "deduplicate_roots",
    "fingerprint_from_checkpoint",
    "fingerprints_from_checkpoints",
    "fingerprint_from_orthonormal_density",
    "run_guess_panel",
    "run_scf_attempt",
]

from .electronic_manifold import (
    ElectronicManifoldPoint,
    ManifoldConstructionResult,
    ManifoldContinuity,
    ManifoldContinuityRelation,
    ManifoldThresholds,
    RejectedManifoldComponent,
    compare_electronic_manifolds,
    construct_electronic_manifolds,
)
