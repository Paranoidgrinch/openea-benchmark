from __future__ import annotations

from dataclasses import replace

import pytest

from openea_benchmark.adaptive.scalar_relativity_runner import (
    AdaptiveScalarRelativityStatus,
    ScalarRelativityBasisSpec,
    ScalarRelativityHamiltonian,
    ScalarRelativityStateSpec,
    run_adaptive_scalar_relativity_series,
)
from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionSettings,
    Stage3PointResult,
)
from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV


def states(*, r_neutral=1.00, r_anion=1.05):
    neutral = ScalarRelativityStateSpec(
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
    anion = ScalarRelativityStateSpec(
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
        ScalarRelativityBasisSpec(
            cardinal=x,
            family_id="EXPLICIT_TEST_REL_XZ",
            label=f"X:rel{x}|Y:rel{x}",
            basis_by_element={"X": f"rel{x}", "Y": f"rel{x}"},
            scalar_relativistically_recontracted=True,
        )
        for x in (3, 4, 5)
    )


def _cardinal(request):
    token = next(
        x for x in request.job_id.split("__")
        if len(x) > 1 and x[0] == "X" and x[1:].isdigit()
    )
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
        delta = delta_by_x[x]
        ea_ev = 1.0 + (delta if settings.scalar_relativistic == "SFX2C1E" else 0.0)
        neutral_energy = -100.0 - 0.001 * x
        total = neutral_energy if role == "neutral" else neutral_energy - ea_ev / HARTREE_TO_EV
        if calls is not None:
            calls.append((request, settings))
        return result_for(request, total=total)
    return run


def noop_basis_validator(element, basis_name):
    return None


def test_state_identity_is_a_hard_prerequisite():
    neutral, _ = states()
    with pytest.raises(ValueError, match="state identity"):
        replace(neutral, state_identity_validated=False)


def test_state_pair_must_describe_one_electron_attachment():
    neutral, anion = states()
    with pytest.raises(ValueError, match="minus one electron"):
        run_adaptive_scalar_relativity_series(
            neutral=neutral,
            anion=replace(anion, charge=-2),
            basis_specs=bases(),
            point_runner=synthetic_runner({3: -0.0010, 4: -0.0011}),
            basis_validator=noop_basis_validator,
        )


def test_basis_policy_is_explicit_and_relativistically_certified():
    with pytest.raises(ValueError, match="recontracted"):
        ScalarRelativityBasisSpec(
            3,
            "F",
            "bad",
            {"X": "x", "Y": "y"},
            scalar_relativistically_recontracted=False,
        )
    with pytest.raises(ValueError, match="all-electron"):
        ScalarRelativityBasisSpec(
            3,
            "F",
            "ecp",
            {"X": "ecp-x", "Y": "ecp-y"},
            scalar_relativistically_recontracted=True,
            electron_model="ECP",
        )


def test_basis_policy_maps_exact_elements_and_one_family():
    neutral, anion = states()
    wrong_elements = (
        ScalarRelativityBasisSpec(3, "F", "bad", {"X": "r3", "Z": "r3"}, True),
        ScalarRelativityBasisSpec(4, "F", "bad", {"X": "r4", "Z": "r4"}, True),
    )
    with pytest.raises(ValueError, match="map exactly"):
        run_adaptive_scalar_relativity_series(
            neutral=neutral,
            anion=anion,
            basis_specs=wrong_elements,
            point_runner=synthetic_runner({3: -0.0010, 4: -0.0011}),
            basis_validator=noop_basis_validator,
        )

    mixed = (
        ScalarRelativityBasisSpec(3, "F1", "x3", {"X": "r3", "Y": "r3"}, True),
        ScalarRelativityBasisSpec(4, "F2", "x4", {"X": "r4", "Y": "r4"}, True),
    )
    with pytest.raises(ValueError, match="one declared basis family"):
        run_adaptive_scalar_relativity_series(
            neutral=neutral,
            anion=anion,
            basis_specs=mixed,
            point_runner=synthetic_runner({3: -0.0010, 4: -0.0011}),
            basis_validator=noop_basis_validator,
        )


def test_basis_series_is_preflighted_before_any_qc_point_runs():
    neutral, anion = states()
    calls = []

    def validator(element, basis_name):
        if basis_name == "rel5":
            raise KeyError("missing basis")

    result = run_adaptive_scalar_relativity_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        point_runner=synthetic_runner({3: -0.0010, 4: -0.0011}, calls),
        basis_validator=validator,
    )
    assert result.status is AdaptiveScalarRelativityStatus.POLICY_BLOCKED
    assert result.execution_error_type == "RuntimeError"
    assert "preflight" in result.execution_error_message
    assert calls == []


def test_nr_and_x2c_requests_are_matched_except_hamiltonian():
    neutral, anion = states()
    calls = []
    result = run_adaptive_scalar_relativity_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        point_runner=synthetic_runner({3: -0.0010, 4: -0.0012}, calls),
        basis_validator=noop_basis_validator,
    )
    assert result.status is AdaptiveScalarRelativityStatus.SCALAR_RELATIVITY_CLEARED
    assert len(calls) == 8

    x3_neutral = [(req, settings) for req, settings in calls if _cardinal(req) == 3 and req.charge == 0]
    assert len(x3_neutral) == 2
    nr = next(x for x in x3_neutral if x[1].scalar_relativistic == "NONE")
    x2c = next(x for x in x3_neutral if x[1].scalar_relativistic == "SFX2C1E")
    assert nr[0].basis_by_element == x2c[0].basis_by_element == {"X": "rel3", "Y": "rel3"}
    assert nr[0].r_angstrom == x2c[0].r_angstrom == neutral.r_angstrom
    assert nr[0].source_root_id == x2c[0].source_root_id == neutral.source_root_id
    assert nr[0].source_checkpoint_path == x2c[0].source_checkpoint_path
    assert nr[1].frozen_core is False and x2c[1].frozen_core is False
    assert nr[1].checkpoint_project is True and x2c[1].checkpoint_project is True

    hams = {x.hamiltonian for x in result.evidence[0].calculations}
    assert hams == {
        ScalarRelativityHamiltonian.NONRELATIVISTIC,
        ScalarRelativityHamiltonian.SFX2C1E,
    }


def test_runner_computes_x2c_minus_nr_and_adapts_to_x5():
    neutral, anion = states()
    calls = []
    result = run_adaptive_scalar_relativity_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        point_runner=synthetic_runner({3: -0.0010, 4: -0.0020, 5: -0.0023}, calls),
        target_change_ev=0.0005,
        maximum_cardinal=5,
        basis_validator=noop_basis_validator,
    )
    assert result.status is AdaptiveScalarRelativityStatus.SCALAR_RELATIVITY_CLEARED
    assert [x.cardinal for x in result.points] == [3, 4, 5]
    assert result.assessment.highest_cardinal == 5
    assert result.assessment.central_correction_ev == pytest.approx(-0.0023)
    assert len(calls) == 12


def test_missing_requested_basis_policy_fails_closed_without_guessing():
    neutral, anion = states()
    result = run_adaptive_scalar_relativity_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases()[:2],
        point_runner=synthetic_runner({3: -0.0010, 4: -0.0020}),
        target_change_ev=0.0005,
        maximum_cardinal=5,
        basis_validator=noop_basis_validator,
    )
    assert result.status is AdaptiveScalarRelativityStatus.POLICY_BLOCKED
    assert result.execution_error_type == "MissingScalarRelativityBasisPolicy"
    assert "X=5" in result.execution_error_message


def test_scalar_runner_owns_hamiltonian_and_requires_all_electron_correlation():
    neutral, anion = states()
    with pytest.raises(ValueError, match="owns the NR/SFX2C1E"):
        run_adaptive_scalar_relativity_series(
            neutral=neutral,
            anion=anion,
            basis_specs=bases(),
            execution_settings=Stage3ExecutionSettings(scalar_relativistic="SFX2C1E"),
            point_runner=synthetic_runner({3: -0.0010, 4: -0.0011}),
            basis_validator=noop_basis_validator,
        )
    with pytest.raises(ValueError, match="all-electron"):
        run_adaptive_scalar_relativity_series(
            neutral=neutral,
            anion=anion,
            basis_specs=bases(),
            execution_settings=Stage3ExecutionSettings(frozen_core=True),
            point_runner=synthetic_runner({3: -0.0010, 4: -0.0011}),
            basis_validator=noop_basis_validator,
        )


def test_execution_failure_blocks_without_fabricating_sr_point():
    neutral, anion = states()

    def failing(request, settings):
        if _cardinal(request) == 3 and request.charge == -1 and settings.scalar_relativistic == "SFX2C1E":
            return result_for(
                request,
                total=-100.0,
                status=PointExecutionStatus.CCSD_NOT_CONVERGED,
                error="synthetic failure",
            )
        return synthetic_runner({3: -0.0010, 4: -0.0011})(request, settings)

    result = run_adaptive_scalar_relativity_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        point_runner=failing,
        basis_validator=noop_basis_validator,
    )
    assert result.status is AdaptiveScalarRelativityStatus.EXECUTION_BLOCKED
    assert result.points == ()
    assert "CCSD_NOT_CONVERGED" in result.execution_error_message


def test_subpoint_checkpoints_resume_without_recomputation(tmp_path):
    neutral, anion = states()
    calls = []
    first = run_adaptive_scalar_relativity_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        checkpoint_dir=tmp_path,
        point_runner=synthetic_runner({3: -0.0010, 4: -0.0012}, calls),
        basis_validator=noop_basis_validator,
    )
    assert first.status is AdaptiveScalarRelativityStatus.SCALAR_RELATIVITY_CLEARED
    assert len(calls) == 8

    def must_not_run(request, settings):
        raise AssertionError("completed scalar-relativity subpoint should have been reused")

    second = run_adaptive_scalar_relativity_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        checkpoint_dir=tmp_path,
        point_runner=must_not_run,
        basis_validator=noop_basis_validator,
    )
    assert second.status is AdaptiveScalarRelativityStatus.SCALAR_RELATIVITY_CLEARED
    assert all(calc.checkpoint_reused for ev in second.evidence for calc in ev.calculations)


def test_checkpoint_signature_change_fails_closed(tmp_path):
    neutral, anion = states()
    first = run_adaptive_scalar_relativity_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        checkpoint_dir=tmp_path,
        point_runner=synthetic_runner({3: -0.0010, 4: -0.0012}),
        basis_validator=noop_basis_validator,
    )
    assert first.status is AdaptiveScalarRelativityStatus.SCALAR_RELATIVITY_CLEARED

    changed_neutral = replace(neutral, r_angstrom=1.01)
    second = run_adaptive_scalar_relativity_series(
        neutral=changed_neutral,
        anion=anion,
        basis_specs=bases(),
        checkpoint_dir=tmp_path,
        point_runner=synthetic_runner({3: -0.0010, 4: -0.0012}),
        basis_validator=noop_basis_validator,
    )
    assert second.status is AdaptiveScalarRelativityStatus.EXECUTION_BLOCKED
    assert second.execution_error_type == "ValueError"
    assert "signature mismatch" in second.execution_error_message


def test_homonuclear_diatomic_uses_one_element_basis_mapping():
    neutral = ScalarRelativityStateSpec(
        "neutral", "N2", ("N", "N"), 0, 0, "n2_neutral", 1.10,
        "n2_n_root", "/validated/n2_n.chk", True,
    )
    anion = ScalarRelativityStateSpec(
        "anion", "N2", ("N", "N"), -1, 1, "n2_anion", 1.12,
        "n2_a_root", "/validated/n2_a.chk", True,
    )
    policy = tuple(
        ScalarRelativityBasisSpec(x, "N_REL_XZ", f"N:rel{x}", {"N": f"rel{x}"}, True)
        for x in (3, 4)
    )
    result = run_adaptive_scalar_relativity_series(
        neutral=neutral,
        anion=anion,
        basis_specs=policy,
        point_runner=synthetic_runner({3: -0.0010, 4: -0.0012}),
        maximum_cardinal=4,
        basis_validator=noop_basis_validator,
    )
    assert result.status is AdaptiveScalarRelativityStatus.SCALAR_RELATIVITY_CLEARED
    assert all(
        calc.request.basis_by_element == {"N": f"rel{ev.cardinal}"}
        for ev in result.evidence for calc in ev.calculations
    )


def test_result_keeps_missing_physics_explicit():
    neutral, anion = states()
    result = run_adaptive_scalar_relativity_series(
        neutral=neutral,
        anion=anion,
        basis_specs=bases(),
        point_runner=synthetic_runner({3: -0.0010, 4: -0.0012}),
        basis_validator=noop_basis_validator,
    )
    assert result.is_production_ea is False
    assert result.authorizes_pruning is False
    assert result.relativistic_scope == "SPIN_FREE_ONE_ELECTRON_X2C"
    assert result.includes_soc is False
    assert result.includes_two_electron_relativistic_terms is False
    assert result.requires_scalar_relativistic_remainder_assessment is True
