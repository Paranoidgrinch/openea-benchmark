import math

from openea_benchmark.attachment.component_resolved_cbs import (
    CBSStatus,
    HARTREE_TO_EV,
    PRIMARY_EXPONENTS,
    build_component_pairs,
    inverse_power_cbs,
    resolve_cbs,
)


def point(role, basis, scf, corr, triples):
    return {
        "role": role,
        "basis": basis,
        "reusable": True,
        "scf_energy_hartree": scf,
        "ccsd_correlation_hartree": corr,
        "triples_correction_hartree": triples,
        "ccsd_t_total_hartree": scf + corr + triples,
    }


POINTS = [
    point("neutral", "aug-cc-pvqz",
          -75.4215752415728, -0.2668430860787891, -0.00678920031278963),
    point("anion", "aug-cc-pvqz",
          -75.41711820891717, -0.3304602085141599, -0.013629906104817194),
    point("neutral", "aug-cc-pv5z",
          -75.42281770364899, -0.2782293399290257, -0.007089724368990185),
    point("anion", "aug-cc-pv5z",
          -75.41831334915528, -0.34242541684901695, -0.014118587309682523),
]


def test_inverse_power_formula_reproduces_limit():
    einf = -10.0
    exponent = 3.0
    a = 2.0
    e4 = einf + a / 4.0**exponent
    e5 = einf + a / 5.0**exponent
    got = inverse_power_cbs(
        e4, e5,
        low_cardinal=4, high_cardinal=5,
        exponent=exponent,
    )
    assert abs(got - einf) < 1e-12


def test_component_pairs_require_neutral_and_anion():
    pairs = build_component_pairs(POINTS)
    assert set(pairs) == {"neutral", "anion"}
    assert pairs["neutral"].scf_5z < pairs["neutral"].scf_qz


def test_primary_oh_cbs_value_is_regression_locked():
    result = resolve_cbs(
        points=POINTS,
        daug_fixed_ea_ev=1.8216532720113219,
        full_pec_aug5_ea_ev=1.8157557003556117,
        diffuse_residual_estimate_ev=0.00653115298148527,
    )
    assert abs(result.primary_model.ea_cbs_aug_ev - 1.834149834980562) < 1e-10


def test_diffuse_correction_uses_fixed_geometry_aug5():
    result = resolve_cbs(
        points=POINTS,
        daug_fixed_ea_ev=1.8216532720113219,
        full_pec_aug5_ea_ev=1.8157557003556117,
        diffuse_residual_estimate_ev=0.00653115298148527,
    )
    assert abs(result.diffuse_correction_ev - 0.0060936530324) < 1e-10


def test_model_sensitivity_is_not_averaged_into_central_value():
    result = resolve_cbs(
        points=POINTS,
        daug_fixed_ea_ev=1.8216532720113219,
        full_pec_aug5_ea_ev=1.8157557003556117,
        diffuse_residual_estimate_ev=0.00653115298148527,
    )
    assert result.ea_cbs_plus_diffuse_primary_ev == (
        result.primary_model.ea_cbs_aug_ev + result.diffuse_correction_ev
    )
    assert result.primary_model.ea_cbs_aug_ev != result.sensitivity_model.ea_cbs_aug_ev


def test_oh_model_sensitivity_clears_5_mev_target():
    result = resolve_cbs(
        points=POINTS,
        daug_fixed_ea_ev=1.8216532720113219,
        full_pec_aug5_ea_ev=1.8157557003556117,
        diffuse_residual_estimate_ev=0.00653115298148527,
        model_spread_target_ev=0.005,
    )
    assert result.status is CBSStatus.CLEARED
    assert result.cbs_model_sensitivity_bound_ev < 0.005


def test_tight_model_target_fails_closed():
    result = resolve_cbs(
        points=POINTS,
        daug_fixed_ea_ev=1.8216532720113219,
        full_pec_aug5_ea_ev=1.8157557003556117,
        diffuse_residual_estimate_ev=0.00653115298148527,
        model_spread_target_ev=0.001,
    )
    assert result.status is CBSStatus.NEED_MORE_EVIDENCE
    assert "DO_NOT_ADVANCE_CBS_AS_CLEARED" in result.next_actions


def test_result_is_never_production_adiabatic_ea():
    result = resolve_cbs(
        points=POINTS,
        daug_fixed_ea_ev=1.8216532720113219,
        full_pec_aug5_ea_ev=1.8157557003556117,
        diffuse_residual_estimate_ev=0.00653115298148527,
    )
    assert result.is_production_ea is False
    assert result.includes_zpe is False
    assert result.includes_core_valence is False
    assert result.includes_scalar_relativity is False
    assert result.includes_post_ccsd_t is False
