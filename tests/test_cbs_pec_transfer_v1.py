from dataclasses import replace
from types import SimpleNamespace

from openea_benchmark.adaptive.stage3_execution import PointExecutionStatus
from openea_benchmark.adaptive.cbs_pec_transfer import matched_stage3_point

import pytest

from openea_benchmark.adaptive.cbs_pec_transfer import (
    MatchedCorrectionPoint, TransferStatus, evaluate_pec_correction_transfer,
)
from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV


def samples(role, offset=0.0):
    return [MatchedCorrectionPoint(
        role=role, atoms=("Li", "H"), charge=0 if role == "neutral" else -1,
        spin_2s=0 if role == "neutral" else 1, state_id=f"{role}_ground",
        r_angstrom=r, lower_model_id="CCSDT_LOW", upper_model_id="CCSDT_HIGH",
        lower_energy_hartree=-7.0 - offset, upper_energy_hartree=-7.0 - offset + 0.01 + 0.002 * (r - 1.5) ** 2,
        lower_source_id=f"{role}_{i}_lo", upper_source_id=f"{role}_{i}_hi",
        state_identity_reviewed=True,
    ) for i, r in enumerate((1.4, 1.5, 1.6, 1.7))]


def evaluate(neutral=None, anion=None, rn=1.55, ra=1.65):
    return evaluate_pec_correction_transfer(
        points=tuple(samples("neutral") if neutral is None else neutral) + tuple(samples("anion", .1) if anion is None else anion),
        neutral_target_angstrom=rn, anion_target_angstrom=ra,
    )


def test_matched_correction_transfer_linear_quadratic_and_ea_sign():
    result = evaluate()
    assert result.status is TransferStatus.CANDIDATE_REVIEW_REQUIRED
    assert result.uncertainty_bounded is False
    assert result.is_production_ea is False
    assert result.delta_ea_candidate_ev == pytest.approx((result.neutral.correction_hartree - result.anion.correction_hartree) * HARTREE_TO_EV)
    assert result.neutral.max_observed_interior_loo_ev > 0
    assert result.neutral.local_model_disagreement_ev > 0
    assert result.to_dict()["status"] == "CANDIDATE_REVIEW_REQUIRED"


def test_constant_correction_interpolates_without_fabricated_error():
    a = [replace(p, upper_energy_hartree=p.lower_energy_hartree + .03) for p in samples("neutral")]
    b = [replace(p, upper_energy_hartree=p.lower_energy_hartree + .02) for p in samples("anion", .1)]
    result = evaluate(a, b)
    assert result.delta_ea_candidate_ev == pytest.approx(.01 * HARTREE_TO_EV)
    assert not result.uncertainty_bounded


@pytest.mark.parametrize("change", [
    {"state_identity_reviewed": False}, {"upper_model_id": "CCSDT_LOW"},
    {"lower_source_id": ""}, {"r_angstrom": float("nan")},
    {"upper_energy_hartree": float("inf")}, {"spin_2s": -1},
])
def test_malformed_matched_point_rejected(change):
    with pytest.raises(ValueError):
        replace(samples("neutral")[0], **change)


@pytest.mark.parametrize("mutate, msg", [
    (lambda p: replace(p, r_angstrom=p[0].r_angstrom) if False else replace(p, r_angstrom=1.4), "duplicate"),
    (lambda p: replace(p, upper_model_id="CHANGED"), "mismatched"),
    (lambda p: replace(p, state_id="OTHER"), "mismatched"),
    (lambda p: replace(p, lower_source_id="neutral_0_lo"), "reused"),
])
def test_mixed_pec_evidence_rejected(mutate, msg):
    ps = samples("neutral")
    ps[2] = mutate(ps[2])
    with pytest.raises(ValueError, match=msg):
        evaluate(neutral=ps)


def test_extrapolation_refused_for_both_species():
    with pytest.raises(ValueError, match="outside computed"):
        evaluate(rn=1.71)
    with pytest.raises(ValueError, match="outside computed"):
        evaluate(ra=1.39)


def test_incomplete_pec_grid_refused():
    with pytest.raises(ValueError, match="at least three"):
        evaluate(neutral=samples("neutral")[:2])


def test_state_role_charges_and_models_must_match():
    b = [replace(p, upper_model_id="OTHER") for p in samples("anion", .1)]
    with pytest.raises(ValueError, match="model pair"):
        evaluate(anion=b)
    b = [replace(p, atoms=("H", "Li")) for p in samples("anion", .1)]
    with pytest.raises(ValueError, match="atom ordering"):
        evaluate(anion=b)


def test_no_manual_review_bypass():
    with pytest.raises(ValueError, match="reviewed"):
        replace(samples("anion")[1], state_identity_reviewed=False)


def stage3_sample(request_id, energy=-7.0, *, r=1.5, status=PointExecutionStatus.COMPLETED):
    return SimpleNamespace(
        status=status, ccsd_t_total_hartree=energy,
        system="LiH", charge=0, spin_2s=0, r_angstrom=r,
        scf_reference="RHF", source_checkpoint_path="/tmp/provenance.chk",
        request_id=request_id,
    )


def test_real_stage3_result_adaptation_preserves_matched_energy_difference():
    point = matched_stage3_point(
        role="neutral", state_id="1Sigma+", atoms=("Li", "H"),
        lower_model_id="avtz", upper_model_id="avqz",
        lower=stage3_sample("point_tz", -7.0),
        upper=stage3_sample("point_qz", -7.01),
        state_identity_reviewed=True,
    )
    assert point.correction_hartree == pytest.approx(-0.01)
    assert point.lower_source_id == "point_tz"


@pytest.mark.parametrize("change, msg", [
    ({"r_angstrom": 1.55}, "r_angstrom"),
    ({"spin_2s": 2}, "spin_2s"),
    ({"scf_reference": "ROHF"}, "scf_reference"),
    ({"ccsd_t_total_hartree": None}, "Missing"),
    ({"status": PointExecutionStatus.ERROR}, "completed"),
])
def test_stage3_adapter_refuses_nonmatched_incomplete_energy(change, msg):
    upper = stage3_sample("qz", -7.01)
    for key, value in change.items():
        setattr(upper, key, value)
    with pytest.raises(ValueError, match=msg):
        matched_stage3_point(
            role="neutral", state_id="1Sigma+", atoms=("Li", "H"),
            lower_model_id="TZ", upper_model_id="QZ",
            lower=stage3_sample("tz"), upper=upper,
            state_identity_reviewed=True,
        )


def test_role_charge_needs_neutral_zero_anion_minus_one():
    with pytest.raises(ValueError, match="charge"):
        replace(samples("neutral")[0], charge=1)


def test_reused_provenance_across_species_is_blocked():
    points = samples("anion", .1)
    points[0] = replace(points[0], lower_source_id="neutral_0_lo")
    with pytest.raises(ValueError, match="across"):
        evaluate(anion=points)
