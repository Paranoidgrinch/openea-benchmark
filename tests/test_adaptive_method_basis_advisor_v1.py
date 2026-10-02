from openea_benchmark.adaptive.method_basis_advisor import *
from openea_benchmark.attachment.basis_convergence import *

def profile(tm=False): return ChemicalIntelligenceProfile(('Fe','H') if tm else ('O','H'), tm, False, 26 if tm else 8)
def card_need(): return CardinalConvergenceAssessment(BasisConvergenceStatus.NEED_MORE_EVIDENCE,BasisConvergenceAction.COMPUTE_NEXT_CARDINAL,1,4,0.055,0.48,None,None,'UNKNOWN',('LATEST_CARDINAL_INCREMENT_ABOVE_TARGET',),'test')
def card_ok(): return CardinalConvergenceAssessment(BasisConvergenceStatus.CLEARED,BasisConvergenceAction.NONE,1,5,0.01,0.3,0.014,EAIntervalEV(1.8,1.82,1.84),'CONVERGENCE_ESTIMATED',('TEST',),'test')
def diff_ok(): return DiffuseConvergenceAssessment(BasisConvergenceStatus.CLEARED,BasisConvergenceAction.NONE,4,1,0.003,None,0.003,EAIntervalEV(1.8,1.82,1.84),'CONVERGENCE_ESTIMATED',('TEST',),'test')
def ev(**kw):
    d=dict(reference_character=ReferenceCharacter.SINGLE_REFERENCE,attachment_character=AttachmentCharacter.VALENCE_BOUND,functional_sensitivity=FunctionalSensitivity.STABLE,core_valence_need=CorrectionNeed.NOT_REQUIRED,scalar_relativity_need=CorrectionNeed.NOT_REQUIRED,cardinal=None,diffuse=None); d.update(kw); return MethodBasisEvidence(**d)

def test_oh_qz_requests_5z_not_functional_switch():
    p=advise_method_basis(profile(),ev(cardinal=card_need(),diffuse=diff_ok())); assert AdvisorAction.COMPUTE_NEXT_CARDINAL in p.actions; assert p.requested_next_cardinal==5; assert p.high_accuracy_method_family==('CCSD(T)',); assert p.reconnaissance_functionals==('PBE','PBE0','TPSSh','B3LYP')
def test_functional_sensitivity_no_dft_winner():
    p=advise_method_basis(profile(),ev(functional_sensitivity=FunctionalSensitivity.SENSITIVE)); assert 'NO_DFT_VOTING' in p.reconnaissance_policy; assert p.high_accuracy_method_family==('CCSD(T)',)
def test_multireference_routes_mr():
    p=advise_method_basis(profile(True),ev(reference_character=ReferenceCharacter.MULTIREFERENCE)); assert p.high_accuracy_branch is HighAccuracyBranch.MULTIREFERENCE; assert AdvisorAction.ENTER_MULTIREFERENCE_BRANCH in p.actions
def test_near_threshold_routes_attachment_branch():
    p=advise_method_basis(profile(),ev(attachment_character=AttachmentCharacter.NEAR_THRESHOLD)); assert p.high_accuracy_branch is HighAccuracyBranch.ATTACHMENT_RESOLUTION; assert p.force_double_augmentation; assert p.initial_augmentation_level==2
def test_diffuse_bound_forces_double_policy():
    p=advise_method_basis(profile(),ev(attachment_character=AttachmentCharacter.DIFFUSE_BOUND)); assert p.force_double_augmentation and p.initial_augmentation_level==1
def test_unresolved_reference_fails_closed():
    p=advise_method_basis(profile(),ev(reference_character=ReferenceCharacter.UNRESOLVED)); assert p.high_accuracy_branch is HighAccuracyBranch.UNRESOLVED; assert AdvisorAction.RESOLVE_REFERENCE_CHARACTER in p.actions
def test_core_valence_independent(): assert AdvisorAction.TEST_CORE_VALENCE in advise_method_basis(profile(True),ev(core_valence_need=CorrectionNeed.TEST)).actions
def test_relativity_independent(): assert AdvisorAction.SELECT_RELATIVISTIC_COMPATIBLE_BASIS in advise_method_basis(profile(True),ev(scalar_relativity_need=CorrectionNeed.REQUIRED)).actions
def test_cleared_axes_proceed_cbs(): assert AdvisorAction.PROCEED_COMPONENT_RESOLVED_CBS in advise_method_basis(profile(),ev(cardinal=card_ok(),diffuse=diff_ok())).actions
def test_missing_diffuse_blocks_cbs(): assert AdvisorAction.PROCEED_COMPONENT_RESOLVED_CBS not in advise_method_basis(profile(),ev(cardinal=card_ok())).actions
def test_never_production():
    p=advise_method_basis(profile(),ev()); assert not p.is_production_ea and not p.authorizes_pruning
