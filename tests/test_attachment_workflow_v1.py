from types import SimpleNamespace as NS

from openea_benchmark.attachment.asymptote import (
    BindingStatus,
    DissociationChannel,
    evaluate_binding_interval,
)
from openea_benchmark.attachment.decision import EADecisionStatus, EnergyInterval
from openea_benchmark.attachment.equilibrium import EquilibriumResolverSettings
from openea_benchmark.attachment.workflow import (
    ElectronicEAWorkflowStatus,
    evaluate_stage3_electronic_ea,
)

SETTINGS = EquilibriumResolverSettings(
    max_side_points=2,
    minimum_admissible_models=2,
    minimum_curvature_hartree_per_angstrom2=0.01,
    max_model_geometry_spread_angstrom=0.003,
    max_model_energy_spread_hartree=3.0e-5,
    point_energy_tolerance_hartree=1.0e-8,
)

def ev(x): return NS(value=x)

def make_loop(*, job_id, charge, spin_2s, e0, r0=1.003, curvature=0.8, status="CONVERGED"):
    rs = (0.98, 0.99, 1.00, 1.01, 1.02)
    energy = lambda r: e0 + curvature * (r-r0)**2
    center_r = 1.00
    center_e = energy(center_r)
    points = tuple(NS(r_angstrom=r, energy_hartree=energy(r), status=ev("ACCEPTED")) for r in rs)
    minimum = NS(
        r_angstrom=center_r,
        energy_hartree=center_e,
        left_r_angstrom=0.99,
        right_r_angstrom=1.01,
        canonical_request_id=f"{job_id}__center",
    )
    pec = NS(
        status=ev("READY_FOR_DISCRETE_MINIMUM_SCOUT"),
        initialization_identity_status=ev("CLEARED"),
        geometry_continuity_status=ev("CLEARED"),
        minimum_scout=NS(status=ev("BRACKETED_SINGLE_MINIMUM"), candidates=(minimum,)),
        points=points,
        system="OH",
        charge=charge,
        spin_2s=spin_2s,
        component_id="component-1",
        basis="aug-cc-pVTZ",
    )
    return NS(
        status=ev(status),
        job_id=job_id,
        final_refinement_plan=NS(action=ev("BRACKET_TARGET_MET")),
        final_pec=pec,
        is_production_ea=False,
        ground_state_assigned=False,
        authorizes_pruning=False,
    )

def channel(e):
    return DissociationChannel("diss","A","B",e,source="TEST")

def test_interval_binding_bound():
    r = evaluate_binding_interval(EnergyInterval(-75.11,-75.09),(channel(-75.05),))
    assert r.status is BindingStatus.BOUND

def test_interval_binding_unbound():
    r = evaluate_binding_interval(EnergyInterval(-75.01,-74.99),(channel(-75.05),))
    assert r.status is BindingStatus.UNBOUND

def test_interval_binding_overlap_unresolved():
    r = evaluate_binding_interval(EnergyInterval(-75.06,-75.04),(channel(-75.05),))
    assert r.status is BindingStatus.UNRESOLVED

def test_missing_channel_energy_unresolved():
    missing = DissociationChannel("x","A","B",None)
    r = evaluate_binding_interval(EnergyInterval(-75.11,-75.09),(channel(-75.05),missing))
    assert r.status is BindingStatus.UNRESOLVED

def test_end_to_end_bound():
    r = evaluate_stage3_electronic_ea(
        neutral_loop_result=make_loop(job_id="neutral",charge=0,spin_2s=1,e0=-75.0),
        anion_loop_result=make_loop(job_id="anion",charge=-1,spin_2s=0,e0=-75.1),
        equilibrium_settings=SETTINGS,
        anion_dissociation_channels=(channel(-75.05),),
        target_half_width_ev=0.01,
    )
    assert r.status is ElectronicEAWorkflowStatus.BOUND
    assert r.ea_decision.status is EADecisionStatus.BOUND
    assert r.ea_decision.ea_lower_ev > 0

def test_zero_ea_unresolved():
    r = evaluate_stage3_electronic_ea(
        neutral_loop_result=make_loop(job_id="neutral",charge=0,spin_2s=1,e0=-75.1),
        anion_loop_result=make_loop(job_id="anion",charge=-1,spin_2s=0,e0=-75.1),
        equilibrium_settings=SETTINGS,
        anion_dissociation_channels=(channel(-75.05),),
    )
    assert r.status is ElectronicEAWorkflowStatus.UNRESOLVED

def test_dissociatively_unbound():
    r = evaluate_stage3_electronic_ea(
        neutral_loop_result=make_loop(job_id="neutral",charge=0,spin_2s=1,e0=-75.0),
        anion_loop_result=make_loop(job_id="anion",charge=-1,spin_2s=0,e0=-75.1),
        equilibrium_settings=SETTINGS,
        anion_dissociation_channels=(channel(-75.2),),
    )
    assert r.status is ElectronicEAWorkflowStatus.UNBOUND
    assert r.anion_binding.status is BindingStatus.UNBOUND

def test_missing_channels_unresolved():
    r = evaluate_stage3_electronic_ea(
        neutral_loop_result=make_loop(job_id="neutral",charge=0,spin_2s=1,e0=-75.0),
        anion_loop_result=make_loop(job_id="anion",charge=-1,spin_2s=0,e0=-75.1),
        equilibrium_settings=SETTINGS,
        anion_dissociation_channels=(),
    )
    assert r.status is ElectronicEAWorkflowStatus.UNRESOLVED

def test_nonconverged_stage3_fails_closed():
    r = evaluate_stage3_electronic_ea(
        neutral_loop_result=make_loop(job_id="neutral",charge=0,spin_2s=1,e0=-75.0,status="ROUND_LIMIT_REACHED"),
        anion_loop_result=make_loop(job_id="anion",charge=-1,spin_2s=0,e0=-75.1),
        equilibrium_settings=SETTINGS,
        anion_dissociation_channels=(channel(-75.05),),
    )
    assert r.status is ElectronicEAWorkflowStatus.UNRESOLVED
    assert r.attachment_candidate is None

def test_sparse_equilibrium_fails_closed():
    sparse = make_loop(job_id="neutral",charge=0,spin_2s=1,e0=-75.0)
    sparse.final_pec.points = sparse.final_pec.points[1:4]
    r = evaluate_stage3_electronic_ea(
        neutral_loop_result=sparse,
        anion_loop_result=make_loop(job_id="anion",charge=-1,spin_2s=0,e0=-75.1),
        equilibrium_settings=SETTINGS,
        anion_dissociation_channels=(channel(-75.05),),
    )
    assert r.status is ElectronicEAWorkflowStatus.UNRESOLVED

def test_bound_still_not_production_ea():
    r = evaluate_stage3_electronic_ea(
        neutral_loop_result=make_loop(job_id="neutral",charge=0,spin_2s=1,e0=-75.0),
        anion_loop_result=make_loop(job_id="anion",charge=-1,spin_2s=0,e0=-75.1),
        equilibrium_settings=SETTINGS,
        anion_dissociation_channels=(channel(-75.05),),
    )
    assert r.status is ElectronicEAWorkflowStatus.BOUND
    assert r.is_electronic_ea_only is True
    assert r.includes_zpe is False
    assert r.is_production_ea is False
