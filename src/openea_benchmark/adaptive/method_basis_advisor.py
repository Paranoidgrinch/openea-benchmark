from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceAction,
    BasisConvergenceStatus,
    CardinalConvergenceAssessment,
    DiffuseConvergenceAssessment,
)

from .model import (
    ReferenceCharacterAssessment,
    ReferenceCharacterStatus,
    Review,
    ReviewStatus,
)

# Backward-compatible source alias.  New code should use
# ReferenceCharacterStatus / ReferenceCharacterAssessment explicitly.
ReferenceCharacter = ReferenceCharacterStatus


class AttachmentCharacter(str, Enum):
    VALENCE_BOUND = 'VALENCE_BOUND'
    DIFFUSE_BOUND = 'DIFFUSE_BOUND'
    NEAR_THRESHOLD = 'NEAR_THRESHOLD'
    CONTINUUM_LIKE = 'CONTINUUM_LIKE'
    UNRESOLVED = 'UNRESOLVED'


class CorrectionNeed(str, Enum):
    NOT_REQUIRED = 'NOT_REQUIRED'
    TEST = 'TEST'
    REQUIRED = 'REQUIRED'
    UNRESOLVED = 'UNRESOLVED'


class FunctionalSensitivity(str, Enum):
    STABLE = 'STABLE'
    SENSITIVE = 'SENSITIVE'
    UNRESOLVED = 'UNRESOLVED'


class HighAccuracyBranch(str, Enum):
    SINGLE_REFERENCE_CC = 'SINGLE_REFERENCE_CC'
    MULTIREFERENCE = 'MULTIREFERENCE'
    ATTACHMENT_RESOLUTION = 'ATTACHMENT_RESOLUTION'
    UNRESOLVED = 'UNRESOLVED'


class BasisFamilyPolicy(str, Enum):
    CORRELATION_CONSISTENT_VALENCE = 'CORRELATION_CONSISTENT_VALENCE'
    CORRELATION_CONSISTENT_CORE_VALENCE = 'CORRELATION_CONSISTENT_CORE_VALENCE'
    RELATIVISTIC_COMPATIBLE_CORRELATION_CONSISTENT = 'RELATIVISTIC_COMPATIBLE_CORRELATION_CONSISTENT'
    BACKEND_REVIEW_REQUIRED = 'BACKEND_REVIEW_REQUIRED'


class AdvisorAction(str, Enum):
    RESOLVE_REFERENCE_CHARACTER = 'RESOLVE_REFERENCE_CHARACTER'
    EXPAND_REFERENCE_DIAGNOSTICS = 'EXPAND_REFERENCE_DIAGNOSTICS'
    RESOLVE_ATTACHMENT_CHARACTER = 'RESOLVE_ATTACHMENT_CHARACTER'
    RESOLVE_FUNCTIONAL_SENSITIVITY = 'RESOLVE_FUNCTIONAL_SENSITIVITY'
    ENTER_MULTIREFERENCE_BRANCH = 'ENTER_MULTIREFERENCE_BRANCH'
    ENTER_ATTACHMENT_RESOLUTION_BRANCH = 'ENTER_ATTACHMENT_RESOLUTION_BRANCH'
    START_CARDINAL_SERIES = 'START_CARDINAL_SERIES'
    COMPUTE_NEXT_CARDINAL = 'COMPUTE_NEXT_CARDINAL'
    START_DIFFUSE_SERIES = 'START_DIFFUSE_SERIES'
    COMPUTE_DOUBLE_AUGMENTED = 'COMPUTE_DOUBLE_AUGMENTED'
    COMPUTE_MORE_DIFFUSE = 'COMPUTE_MORE_DIFFUSE'
    TEST_CORE_VALENCE = 'TEST_CORE_VALENCE'
    TEST_SCALAR_RELATIVITY = 'TEST_SCALAR_RELATIVITY'
    SELECT_RELATIVISTIC_COMPATIBLE_BASIS = 'SELECT_RELATIVISTIC_COMPATIBLE_BASIS'
    PROCEED_COMPONENT_RESOLVED_CBS = 'PROCEED_COMPONENT_RESOLVED_CBS'


@dataclass(frozen=True)
class ChemicalIntelligenceProfile:
    elements: tuple[str, ...]
    has_transition_metal: bool
    has_lanthanide_or_actinide: bool
    max_atomic_number: int

    def __post_init__(self):
        if not self.elements or any(not e.strip() for e in self.elements):
            raise ValueError('valid elements required')
        if self.max_atomic_number < 1:
            raise ValueError('max_atomic_number must be >=1')


@dataclass(frozen=True)
class MethodBasisEvidence:
    reference_character: ReferenceCharacterAssessment | ReferenceCharacterStatus
    attachment_character: AttachmentCharacter
    functional_sensitivity: FunctionalSensitivity
    core_valence_need: CorrectionNeed
    scalar_relativity_need: CorrectionNeed
    cardinal: CardinalConvergenceAssessment | None = None
    diffuse: DiffuseConvergenceAssessment | None = None
    expanded_reference_diagnostics: Review | None = None

    @property
    def reference_status(self) -> ReferenceCharacterStatus:
        if isinstance(self.reference_character, ReferenceCharacterAssessment):
            return self.reference_character.status
        return self.reference_character


@dataclass(frozen=True)
class MethodBasisPlan:
    reconnaissance_functionals: tuple[str, ...]
    reconnaissance_policy: str
    high_accuracy_branch: HighAccuracyBranch
    high_accuracy_method_family: tuple[str, ...]
    basis_family_policy: BasisFamilyPolicy
    initial_augmentation_level: int | None
    force_double_augmentation: bool
    requested_next_cardinal: int | None
    actions: tuple[AdvisorAction, ...]
    evidence: tuple[str, ...]
    is_production_ea: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self):
        if self.is_production_ea or self.authorizes_pruning:
            raise ValueError('advisor cannot promote production EA/pruning')


_RECON = ('PBE', 'PBE0', 'TPSSh', 'B3LYP')


def _branch(ev: MethodBasisEvidence):
    status = ev.reference_status
    if status is ReferenceCharacterStatus.UNRESOLVED:
        return HighAccuracyBranch.UNRESOLVED, ()
    if status is ReferenceCharacterStatus.MULTIREFERENCE_RISK:
        # The advisor identifies the branch but does not pretend that a
        # production-quality MR method has already been validated.
        return HighAccuracyBranch.MULTIREFERENCE, ()
    if ev.attachment_character in (AttachmentCharacter.NEAR_THRESHOLD, AttachmentCharacter.CONTINUUM_LIKE):
        return HighAccuracyBranch.ATTACHMENT_RESOLUTION, (
            'EOM-EA', 'STABILIZATION/CONTINUUM_DIAGNOSTICS',
        )
    if ev.attachment_character is AttachmentCharacter.UNRESOLVED:
        return HighAccuracyBranch.UNRESOLVED, ()
    return HighAccuracyBranch.SINGLE_REFERENCE_CC, ('CCSD(T)',)


def _basis(profile, ev):
    if profile.has_lanthanide_or_actinide:
        return BasisFamilyPolicy.BACKEND_REVIEW_REQUIRED
    if ev.scalar_relativity_need in (CorrectionNeed.TEST, CorrectionNeed.REQUIRED):
        return BasisFamilyPolicy.RELATIVISTIC_COMPATIBLE_CORRELATION_CONSISTENT
    if ev.core_valence_need is CorrectionNeed.REQUIRED:
        return BasisFamilyPolicy.CORRELATION_CONSISTENT_CORE_VALENCE
    return BasisFamilyPolicy.CORRELATION_CONSISTENT_VALENCE


def advise_method_basis(
    profile: ChemicalIntelligenceProfile,
    ev: MethodBasisEvidence,
) -> MethodBasisPlan:
    actions = []
    audit = [
        'NO_EXPERIMENTAL_EA_USED',
        'DFT_RECONNAISSANCE_NOT_FINAL_EA',
        'NO_FUNCTIONAL_OR_BASIS_AVERAGING',
    ]
    status = ev.reference_status
    branch, methods = _branch(ev)

    if status is ReferenceCharacterStatus.UNRESOLVED:
        actions.append(AdvisorAction.RESOLVE_REFERENCE_CHARACTER)
    elif status is ReferenceCharacterStatus.BORDERLINE:
        expanded = ev.expanded_reference_diagnostics
        if expanded is None or expanded.status is not ReviewStatus.CLEARED:
            actions.append(AdvisorAction.EXPAND_REFERENCE_DIAGNOSTICS)
            audit.append('BORDERLINE_SINGLE_REFERENCE_REQUIRES_EXPANDED_DIAGNOSTICS')
        else:
            audit.append('BORDERLINE_EXPANDED_REFERENCE_DIAGNOSTICS_CLEARED')
        audit.append('BORDERLINE_REQUIRES_ENLARGED_UNCERTAINTY')
    elif status is ReferenceCharacterStatus.SAFE_SINGLE_REFERENCE:
        audit.append('REFERENCE_CHARACTER_SAFE_SINGLE_REFERENCE')
    elif status is ReferenceCharacterStatus.MULTIREFERENCE_RISK:
        audit.append('MULTIREFERENCE_RISK_BLOCKS_SINGLE_REFERENCE_PRODUCTION')

    if ev.attachment_character is AttachmentCharacter.UNRESOLVED:
        actions.append(AdvisorAction.RESOLVE_ATTACHMENT_CHARACTER)
    if ev.functional_sensitivity is FunctionalSensitivity.UNRESOLVED:
        actions.append(AdvisorAction.RESOLVE_FUNCTIONAL_SENSITIVITY)
    elif ev.functional_sensitivity is FunctionalSensitivity.SENSITIVE:
        audit.append('FUNCTIONAL_SENSITIVITY_CONFIRMED')
    else:
        audit.append('FUNCTIONAL_RECONNAISSANCE_STABLE')

    if branch is HighAccuracyBranch.MULTIREFERENCE:
        actions.append(AdvisorAction.ENTER_MULTIREFERENCE_BRANCH)
        audit.append('MR_PRODUCTION_METHOD_NOT_AUTHORIZED_BY_ADVISOR')
    elif branch is HighAccuracyBranch.ATTACHMENT_RESOLUTION:
        actions.append(AdvisorAction.ENTER_ATTACHMENT_RESOLUTION_BRANCH)

    if ev.core_valence_need in (CorrectionNeed.TEST, CorrectionNeed.UNRESOLVED):
        actions.append(AdvisorAction.TEST_CORE_VALENCE)
    if ev.scalar_relativity_need in (CorrectionNeed.TEST, CorrectionNeed.UNRESOLVED):
        actions.append(AdvisorAction.TEST_SCALAR_RELATIVITY)
    elif ev.scalar_relativity_need is CorrectionNeed.REQUIRED:
        actions.append(AdvisorAction.SELECT_RELATIVISTIC_COMPATIBLE_BASIS)
    if profile.has_lanthanide_or_actinide:
        actions.append(AdvisorAction.SELECT_RELATIVISTIC_COMPATIBLE_BASIS)

    if ev.attachment_character is AttachmentCharacter.VALENCE_BOUND:
        initial_aug, force_double = 1, False
    elif ev.attachment_character is AttachmentCharacter.DIFFUSE_BOUND:
        initial_aug, force_double = 1, True
    elif ev.attachment_character in (AttachmentCharacter.NEAR_THRESHOLD, AttachmentCharacter.CONTINUUM_LIKE):
        initial_aug, force_double = 2, True
    else:
        initial_aug, force_double = None, False

    requested = None
    if ev.cardinal is None:
        actions.append(AdvisorAction.START_CARDINAL_SERIES)
    elif ev.cardinal.status is BasisConvergenceStatus.CLEARED:
        audit.append('CARDINAL_CONVERGENCE_CLEARED')
    elif ev.cardinal.action is BasisConvergenceAction.COMPUTE_NEXT_CARDINAL:
        actions.append(AdvisorAction.COMPUTE_NEXT_CARDINAL)
        if ev.cardinal.highest_cardinal is not None:
            requested = ev.cardinal.highest_cardinal + 1
        audit.append('CARDINAL_CONVERGENCE_REQUIRES_NEXT_X')
    else:
        actions.append(AdvisorAction.START_CARDINAL_SERIES)

    if ev.diffuse is None:
        actions.append(AdvisorAction.START_DIFFUSE_SERIES)
    elif ev.diffuse.status is BasisConvergenceStatus.CLEARED:
        audit.append('DIFFUSE_CONVERGENCE_CLEARED')
    elif ev.diffuse.action is BasisConvergenceAction.COMPUTE_DOUBLE_AUGMENTED:
        actions.append(AdvisorAction.COMPUTE_DOUBLE_AUGMENTED)
    else:
        actions.append(AdvisorAction.COMPUTE_MORE_DIFFUSE)

    blockers = {
        AdvisorAction.RESOLVE_REFERENCE_CHARACTER,
        AdvisorAction.EXPAND_REFERENCE_DIAGNOSTICS,
        AdvisorAction.RESOLVE_ATTACHMENT_CHARACTER,
        AdvisorAction.ENTER_MULTIREFERENCE_BRANCH,
        AdvisorAction.ENTER_ATTACHMENT_RESOLUTION_BRANCH,
        AdvisorAction.START_CARDINAL_SERIES,
        AdvisorAction.COMPUTE_NEXT_CARDINAL,
        AdvisorAction.START_DIFFUSE_SERIES,
        AdvisorAction.COMPUTE_DOUBLE_AUGMENTED,
        AdvisorAction.COMPUTE_MORE_DIFFUSE,
        AdvisorAction.TEST_CORE_VALENCE,
        AdvisorAction.TEST_SCALAR_RELATIVITY,
        AdvisorAction.SELECT_RELATIVISTIC_COMPATIBLE_BASIS,
    }
    # BORDERLINE may remain on the SR branch, but production is not authorized
    # until the explicitly requested expanded diagnostics have been reviewed.
    # Even after clearance, its uncertainty budget must remain enlarged.
    if (
        branch is HighAccuracyBranch.SINGLE_REFERENCE_CC
        and ev.cardinal is not None
        and ev.cardinal.status is BasisConvergenceStatus.CLEARED
        and ev.diffuse is not None
        and ev.diffuse.status is BasisConvergenceStatus.CLEARED
        and not any(a in blockers for a in actions)
    ):
        actions.append(AdvisorAction.PROCEED_COMPONENT_RESOLVED_CBS)

    actions = tuple(dict.fromkeys(actions))
    return MethodBasisPlan(
        _RECON,
        'ENSEMBLE_FOR_STATE_DISCOVERY_AND_SENSITIVITY_ONLY;NO_DFT_VOTING_OR_FINAL_EA_AVERAGING',
        branch,
        methods,
        _basis(profile, ev),
        initial_aug,
        force_double,
        requested,
        actions,
        tuple(audit),
    )
