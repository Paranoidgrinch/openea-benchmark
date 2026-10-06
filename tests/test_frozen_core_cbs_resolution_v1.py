from openea_benchmark.attachment.frozen_core_cbs_resolution import (
    fixed_ea, resolve_frozen_core_cbs,
)

def p(role,basis,scf,corr,t):
    return {"role":role,"basis":basis,"reusable":True,
            "scf_energy_hartree":scf,
            "ccsd_correlation_hartree":corr,
            "triples_correction_hartree":t,
            "ccsd_t_total_hartree":scf+corr+t}

PTS=[
p("neutral","aug-cc-pvqz",-75.42157524157288,-0.2363343364982743,-0.00654860492233577),
p("anion","aug-cc-pvqz",-75.41711820891719,-0.2998624107059611,-0.013330933134440276),
p("neutral","aug-cc-pv5z",-75.42281770364906,-0.24098148407324257,-0.006782522912682475),
p("anion","aug-cc-pv5z",-75.41831334915516,-0.30509618123678034,-0.013733283397307724),
p("neutral","d-aug-cc-pv5z",-75.4228211347261,-0.2410433203782396,-0.006788349955011401),
p("anion","d-aug-cc-pv5z",-75.41833031639177,-0.3052971167658998,-0.01381115497842862),
]

def test_fixed_aug_and_daug_eas_regression():
    assert abs(fixed_ea(PTS,"aug-cc-pv5z")-1.81121988688953)<1e-10
    assert abs(fixed_ea(PTS,"d-aug-cc-pv5z")-1.81733373904191)<1e-10

def test_frozen_core_cbs_regression():
    r=resolve_frozen_core_cbs(points=PTS,indirect_diffuse_residual_ev=0.00653115298148527)
    assert abs(r.ea_aug_cbs_ev-1.82956920812534)<1e-10
    assert abs(r.diffuse_correction_ev-0.00611385215238)<1e-10
    assert abs(r.ea_cbs_plus_diffuse_ev-1.83568306027772)<1e-10
    assert r.status=="READY_FOR_CORE_VALENCE"
    assert r.is_production_ea is False

def test_indirect_diffuse_residual_is_labeled_not_direct():
    r=resolve_frozen_core_cbs(points=PTS,indirect_diffuse_residual_ev=0.006)
    assert "DIFFUSE_RESIDUAL_INDIRECTLY_ESTIMATED_FROM_PRIOR_ALL_ELECTRON_AXIS" in r.evidence
    assert r.uncertainty_status=="CONSERVATIVE_WITH_INDIRECT_DIFFUSE_RESIDUAL"
