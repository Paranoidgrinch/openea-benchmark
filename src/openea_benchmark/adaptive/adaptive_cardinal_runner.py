from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

from openea_benchmark.attachment.basis_convergence import (
    BasisConvergenceSettings,
    CardinalConvergenceAssessment,
    ElectronicEABasisPoint,
    assess_cardinal_convergence,
)
from .method_basis_advisor import (
    AdvisorAction,
    ChemicalIntelligenceProfile,
    HighAccuracyBranch,
    MethodBasisEvidence,
    MethodBasisPlan,
    advise_method_basis,
)

class AdaptiveCardinalStatus(str, Enum):
    CARDINAL_CLEARED = 'CARDINAL_CLEARED'
    CARDINAL_LIMIT_REACHED = 'CARDINAL_LIMIT_REACHED'
    EXECUTION_BLOCKED = 'EXECUTION_BLOCKED'
    ADVISOR_BLOCKED = 'ADVISOR_BLOCKED'

@dataclass(frozen=True)
class AdaptiveCardinalIteration:
    iteration_index: int
    evaluated_cardinals: tuple[int, ...]
    assessment: CardinalConvergenceAssessment
    advisor_plan: MethodBasisPlan
    requested_next_cardinal: int | None

@dataclass(frozen=True)
class AdaptiveCardinalResult:
    status: AdaptiveCardinalStatus
    points: tuple[ElectronicEABasisPoint, ...]
    iterations: tuple[AdaptiveCardinalIteration, ...]
    final_assessment: CardinalConvergenceAssessment | None
    final_advisor_plan: MethodBasisPlan | None
    execution_error_type: str | None = None
    execution_error_message: str | None = None
    is_production_ea: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self):
        if self.is_production_ea or self.authorizes_pruning:
            raise ValueError('adaptive cardinal runner cannot promote production EA or pruning')


def correlation_consistent_basis_name(cardinal_number: int, *, augmentation_level: int) -> str:
    labels = {2:'d',3:'t',4:'q',5:'5',6:'6'}
    if cardinal_number not in labels:
        raise ValueError('validation resolver supports X=2..6')
    if augmentation_level not in (0,1):
        raise ValueError('validation resolver supports augmentation levels 0 or 1')
    return f"{'aug-' if augmentation_level else ''}cc-pv{labels[cardinal_number]}z"


def _ordered(points):
    by_x={}
    for p in points:
        if p.cardinal_number in by_x:
            raise ValueError(f'duplicate cardinal X={p.cardinal_number}')
        by_x[p.cardinal_number]=p
    return tuple(by_x[x] for x in sorted(by_x))


def run_adaptive_cardinal_series(*, initial_cardinals: tuple[int,...], maximum_cardinal: int,
                                 augmentation_level: int, convergence_settings: BasisConvergenceSettings,
                                 chemical_profile: ChemicalIntelligenceProfile,
                                 advisor_evidence_template: MethodBasisEvidence,
                                 evaluate_point: Callable[[int],ElectronicEABasisPoint]) -> AdaptiveCardinalResult:
    if len(initial_cardinals)<3 or tuple(sorted(initial_cardinals))!=initial_cardinals or len(set(initial_cardinals))!=len(initial_cardinals):
        raise ValueError('initial_cardinals must be at least three sorted unique values')
    if maximum_cardinal < max(initial_cardinals):
        raise ValueError('maximum_cardinal below initial series')

    points=[]; iterations=[]
    def do(x):
        try:
            p=evaluate_point(x)
            if p.cardinal_number!=x: raise ValueError(f'evaluator returned X={p.cardinal_number} for X={x}')
            if p.augmentation_level!=augmentation_level: raise ValueError('wrong augmentation level')
            points.append(p); return None
        except Exception as exc:
            return exc

    for x in initial_cardinals:
        err=do(x)
        if err:
            return AdaptiveCardinalResult(AdaptiveCardinalStatus.EXECUTION_BLOCKED,_ordered(points),tuple(iterations),None,None,type(err).__name__,str(err))

    idx=0
    while True:
        ordered=_ordered(points)
        try:
            assessment=assess_cardinal_convergence(ordered,augmentation_level=augmentation_level,settings=convergence_settings)
            plan=advise_method_basis(chemical_profile,replace(advisor_evidence_template,cardinal=assessment))
        except Exception as exc:
            return AdaptiveCardinalResult(AdaptiveCardinalStatus.ADVISOR_BLOCKED,ordered,tuple(iterations),None,None,type(exc).__name__,str(exc))

        requested=plan.requested_next_cardinal
        iterations.append(AdaptiveCardinalIteration(idx,tuple(p.cardinal_number for p in ordered),assessment,plan,requested))

        if assessment.status.value=='CLEARED':
            return AdaptiveCardinalResult(AdaptiveCardinalStatus.CARDINAL_CLEARED,ordered,tuple(iterations),assessment,plan)
        if plan.high_accuracy_branch is not HighAccuracyBranch.SINGLE_REFERENCE_CC:
            return AdaptiveCardinalResult(AdaptiveCardinalStatus.ADVISOR_BLOCKED,ordered,tuple(iterations),assessment,plan,None,'advisor left single-reference CC branch')
        if AdvisorAction.COMPUTE_NEXT_CARDINAL not in plan.actions or requested is None:
            return AdaptiveCardinalResult(AdaptiveCardinalStatus.ADVISOR_BLOCKED,ordered,tuple(iterations),assessment,plan,None,'advisor did not provide next cardinal')
        if requested>maximum_cardinal:
            return AdaptiveCardinalResult(AdaptiveCardinalStatus.CARDINAL_LIMIT_REACHED,ordered,tuple(iterations),assessment,plan)
        if requested in {p.cardinal_number for p in ordered}:
            return AdaptiveCardinalResult(AdaptiveCardinalStatus.ADVISOR_BLOCKED,ordered,tuple(iterations),assessment,plan,None,'advisor requested already evaluated cardinal')
        err=do(requested)
        if err:
            return AdaptiveCardinalResult(AdaptiveCardinalStatus.EXECUTION_BLOCKED,_ordered(points),tuple(iterations),assessment,plan,type(err).__name__,str(err))
        idx+=1
