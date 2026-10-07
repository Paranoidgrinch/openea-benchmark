from __future__ import annotations

from dataclasses import replace

import pytest

from openea_benchmark.adaptive.core_valence_runner import (
    AdaptiveCoreValenceStatus,
    CoreValenceBasisSpec,
    CoreValenceCorrelationSpace,
    CoreValenceStateSpec,
    run_adaptive_core_valence_series,
)
from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionSettings,
    Stage3PointResult,
)
from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV


def states(*, r_neutral=1.00, r_anion=1.05):
    neutral = CoreValenceStateSpec(
        role="neutral",
        system="XY",
        atoms=("X", "Y"),
        charge=0,
        spin_2s=1,
        state_id="neutral_ground",
        r_angstrom=r_neutral,
        source_root_id="neutral_root",
        source_checkpoint_path="/validated/neutral.chk",
        state_identity_validated=True,
    )
    anion = CoreValenceStateSpec(
        role="anion",
        system="XY",
        atoms=("X", "Y"),
        charge=-1,
        spin_2s=0,
        state_id="anion_ground",
        r_angstrom=r_anion,
        source_root_id="anion_root",
        source_checkpoint_path="/validated/anion.chk",
        state_identity_validated=True,
    )
    return neutral, anion


def bases():
    return tuple(
        CoreValenceBasisSpec(
            cardinal=x,
            family_id="EXPLICIT_TEST_CVXZ",
            label=f"X:cv{x}|Y:v{x}",
            basis_by_element={"X": f"cv{x}", "Y": f"v{x}"},
        )
        for x in (3, 4, 5)
    )


def _cardinal(request):
    token = next(x for x in request.job_id.split("__") if len(x) > 1 and x[0] == "X" and x[1:].isdigit())
    return int(token[1:])


def result_for(request, *, total, status=PointExecutionStatus.COMPLETED, error=None):
    return Stage3PointResult(
        request_id=request.request_id,
        job_id=request.job_id,
        status=status,
        system=request.system,
        charge=request.charge,
        spin_2s=request.spin_2s,
        component_id=request.component_id,
        r_angstrom=request.r_angstrom,
        basis=request.basis,
        source_root_id=request.source_root_id,
        source_checkpoint_path=request.source_checkpoint_path,
        scf_reference=request.scf_reference,
        cc_reference="RHF" if request.scf_reference == "RHF" else "SEMICANONICAL_UHF_FROM_ROHF",
        pyscf_version="test",
        scf_converged=status is PointExecutionStatus.COMPLETED,
        scf_energy_hartree=total + 0.2 if status is PointExecutionStatus.COMPLETED else None,
        s2=0.0,
        multiplicity=1.0,
        internal_stable=True,
        external_stable=True if request.scf_reference == "RHF" else None,
        external_stability_available=request.scf_reference == "RHF",
        semicanonicalization="TEST",
        cc_class="TEST_CCSD",
        ccsd_converged=status is PointExecutionStatus.COMPLETED,
        ccsd_correlation_hartree=-0.2 if status is PointExecutionStatus.COMPLETED else None,
        ccsd_total_hartree=total + 0.001 if status is PointExecutionStatus.COMPLETED else None,
        triples_correction_hartree=-0.001 if status is PointExecutionStatus.COMPLETED else None,
        ccsd_t_total_hartree=total if status is PointExecutionStatus.COMPLETED else None,
        t1_diagnostic=None,
        t1_diagnostic_definition=None,
        error_type=error,
        error_message=error,
        is_production_ea=False,
        ground_state_assigned=False,
        authorizes_pruning=False,
        high_level_checkpoint_path=None,
    )


def synthetic_runner(delta_by_x, calls=None):
    def run(request, settings):
        x = _cardinal(request)
        role = "neutral" if request.charge == 0 else "anion"
        ae = not settings.frozen_core
        ea_ev = 1.0 + (delta_by_x[x] if ae else 0.0)
        neutral_energy = -100.0 - 0.001 * x
        total = neutral_energy if role == "neutral" else neutral_energy - ea_ev / HARTREE_TO_EV
        if calls is not None:
            calls.append((request, settings))
        return result_for(request, total=total)
    return run


def test_state_identity_is_a_hard_prerequisite():
    neutral, _ = states()
    with pytest.raises(ValueError, match="state identity"):
        replace(neutral, state_identity_validated=False)


def test_state_pair_must_describe_one_electron_attachment():
    neutral, anion = states()
    with pytest.raises(ValueError, match="minus one electron"):
        run_adaptive_core_valence_series(
            neutral=neutral,
            anion=replace(anion, charge=-2),
            basis_specs=bases(),
            point_runner=synthetic_runner({3: 0.004, 4: 0.0045}),
        )


def test_basis_policy_is_explicit_per_element_and_one_family():
    neutral, anion = states()
    wrong_elements = (
        CoreValenceBasisSpec(3, "F", "bad", {"X": "cv3", "Z": "cv3"}),
        CoreValenceBasisSpec(4, "F", "bad", {"X": "cv4", "Z": "cv4"}),
    )
    with pytest.raises(ValueError, match="map exactly"):
        run_adaptive_core_valence_series(
            neutral=neutral, anion=anion, basis_specs=wrong_elements,
            point_runner=synthetic_runner({3: 0.004, 4: 0.0045}),
        )

    mixed_family = (
        CoreValenceBasisSpec(3, "F1", "x3", {"X": "cv3", "Y": "v3"}),
        CoreValenceBasisSpec(4, "F2", "x4", {"X": "cv4", "Y": "v4"}),
    )
    with pytest.raises(ValueError, match="one declared basis family"):
        run_adaptive_core_valence_series(
            neutral=neutral, anion=anion, basis_specs=mixed_family,
            point_runner=synthetic_runner({3: 0.004, 4: 0.0045}),
        )


def test_ecp_core_replacement_is_not_silently_treated_as_all_electron_cv():
    with pytest.raises(ValueError, match="all-electron"):
        CoreValenceBasisSpec(
            3, "ECP_CV", "ecp", {"X": "ecp-basis", "Y": "ecp-basis"},
            electron_model="ECP",
        )


def test_matched_ae_fc_requests_preserve_state_and_basis_provenance():
    neutral, anion = states()
    calls = []
    result = run_adaptive_core_valence_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        point_runner=synthetic_runner({3: 0.0040, 4: 0.0048}, calls),
    )
    assert result.status is AdaptiveCoreValenceStatus.CORE_VALENCE_CLEARED
    assert len(calls) == 8

    x3_neutral = [
        (req, settings) for req, settings in calls
        if _cardinal(req) == 3 and req.charge == 0
    ]
    assert len(x3_neutral) == 2
    ae = next(x for x in x3_neutral if not x[1].frozen_core)
    fc = next(x for x in x3_neutral if x[1].frozen_core)
    assert ae[0].basis_by_element == fc[0].basis_by_element == {"X": "cv3", "Y": "v3"}
    assert ae[0].r_angstrom == fc[0].r_angstrom == neutral.r_angstrom
    assert ae[0].source_root_id == fc[0].source_root_id == neutral.source_root_id
    assert ae[0].source_checkpoint_path == fc[0].source_checkpoint_path
    assert ae[0].requires_independent_state_identity_validation is True
    assert ae[1].checkpoint_project is True and fc[1].checkpoint_project is True

    spaces = {x.correlation_space for x in result.evidence[0].calculations}
    assert spaces == {
        CoreValenceCorrelationSpace.ALL_ELECTRON,
        CoreValenceCorrelationSpace.FROZEN_CORE,
    }
    policies = {x.frozen_core_policy for x in result.evidence[0].calculations}
    assert policies == {"ALL_ELECTRON_CCSD", "PYSCF_CCSD_SET_FROZEN_CHEMCORE"}


def test_runner_computes_ae_minus_fc_and_adapts_to_x5():
    neutral, anion = states()
    calls = []
    result = run_adaptive_core_valence_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        point_runner=synthetic_runner({3: 0.0020, 4: 0.0050, 5: 0.0065}, calls),
        target_change_ev=0.002,
        maximum_cardinal=5,
    )
    assert result.status is AdaptiveCoreValenceStatus.CORE_VALENCE_CLEARED
    assert [x.cardinal for x in result.points] == [3, 4, 5]
    assert result.assessment.highest_cardinal == 5
    assert result.assessment.contraction_ratio == pytest.approx(0.5)
    assert result.assessment.central_correction_ev == pytest.approx(0.0065)
    assert len(calls) == 12


def test_missing_requested_basis_policy_fails_closed_without_guessing():
    neutral, anion = states()
    result = run_adaptive_core_valence_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases()[:2],
        point_runner=synthetic_runner({3: 0.0020, 4: 0.0050}),
        target_change_ev=0.002,
        maximum_cardinal=5,
    )
    assert result.status is AdaptiveCoreValenceStatus.POLICY_BLOCKED
    assert result.execution_error_type == "MissingCoreValenceBasisPolicy"
    assert "X=5" in result.execution_error_message


def test_scalar_relativity_cannot_be_folded_into_core_valence_runner():
    neutral, anion = states()
    with pytest.raises(ValueError, match="separate OpenEA corrections"):
        run_adaptive_core_valence_series(
            neutral=neutral,
            anion=anion,
            basis_specs=bases(),
            execution_settings=Stage3ExecutionSettings(scalar_relativistic="SFX2C1E"),
            point_runner=synthetic_runner({3: 0.0040, 4: 0.0048}),
        )


def test_execution_failure_blocks_without_fabricating_a_cv_point():
    neutral, anion = states()

    def failing(request, settings):
        if _cardinal(request) == 3 and request.charge == -1 and settings.frozen_core:
            return result_for(
                request, total=-100.0, status=PointExecutionStatus.CCSD_NOT_CONVERGED,
                error="synthetic failure",
            )
        return synthetic_runner({3: 0.0040, 4: 0.0048})(request, settings)

    result = run_adaptive_core_valence_series(
        neutral=neutral, anion=anion, basis_specs=bases(), point_runner=failing
    )
    assert result.status is AdaptiveCoreValenceStatus.EXECUTION_BLOCKED
    assert result.points == ()
    assert "CCSD_NOT_CONVERGED" in result.execution_error_message


def test_subpoint_checkpoints_resume_without_recomputation(tmp_path):
    neutral, anion = states()
    calls = []
    first = run_adaptive_core_valence_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        checkpoint_dir=tmp_path,
        point_runner=synthetic_runner({3: 0.0040, 4: 0.0048}, calls),
    )
    assert first.status is AdaptiveCoreValenceStatus.CORE_VALENCE_CLEARED
    assert len(calls) == 8

    def must_not_run(request, settings):
        raise AssertionError("completed CV subpoint should have been reused")

    second = run_adaptive_core_valence_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        checkpoint_dir=tmp_path,
        point_runner=must_not_run,
    )
    assert second.status is AdaptiveCoreValenceStatus.CORE_VALENCE_CLEARED
    assert all(
        calc.checkpoint_reused
        for evidence in second.evidence
        for calc in evidence.calculations
    )


def test_checkpoint_signature_change_fails_closed(tmp_path):
    neutral, anion = states()
    first = run_adaptive_core_valence_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        checkpoint_dir=tmp_path,
        point_runner=synthetic_runner({3: 0.0040, 4: 0.0048}),
    )
    assert first.status is AdaptiveCoreValenceStatus.CORE_VALENCE_CLEARED

    changed_neutral = replace(neutral, r_angstrom=1.01)
    second = run_adaptive_core_valence_series(
        neutral=changed_neutral,
        anion=anion,
        basis_specs=bases(),
        checkpoint_dir=tmp_path,
        point_runner=synthetic_runner({3: 0.0040, 4: 0.0048}),
    )
    assert second.status is AdaptiveCoreValenceStatus.EXECUTION_BLOCKED
    assert second.execution_error_type == "ValueError"
    assert "signature mismatch" in second.execution_error_message



def test_homonuclear_diatomic_uses_one_element_basis_mapping():
    neutral = CoreValenceStateSpec(
        "neutral", "N2", ("N", "N"), 0, 0, "n2_neutral", 1.10,
        "n2_n_root", "/validated/n2_n.chk", True,
    )
    anion = CoreValenceStateSpec(
        "anion", "N2", ("N", "N"), -1, 1, "n2_anion", 1.12,
        "n2_a_root", "/validated/n2_a.chk", True,
    )
    policy = tuple(
        CoreValenceBasisSpec(x, "N_CVXZ", f"N:cv{x}", {"N": f"cv{x}"})
        for x in (3, 4)
    )
    result = run_adaptive_core_valence_series(
        neutral=neutral, anion=anion, basis_specs=policy,
        point_runner=synthetic_runner({3: 0.0040, 4: 0.0048}),
        maximum_cardinal=4,
    )
    assert result.status is AdaptiveCoreValenceStatus.CORE_VALENCE_CLEARED
    assert all(
        calc.request.basis_by_element == {"N": f"cv{evidence.cardinal}"}
        for evidence in result.evidence for calc in evidence.calculations
    )

def test_result_never_claims_complete_production_ea():
    neutral, anion = states()
    result = run_adaptive_core_valence_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        point_runner=synthetic_runner({3: 0.0040, 4: 0.0048}),
    )
    assert result.is_production_ea is False
    assert result.authorizes_pruning is False
