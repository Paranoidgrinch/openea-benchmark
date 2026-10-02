from dataclasses import replace

from openea_benchmark.attachment.equilibrium import (
    EquilibriumResolverSettings,
    EquilibriumStatus,
    resolve_stage3_equilibrium,
)
from openea_benchmark.attachment.model import ElectronicState, PECBranch
from openea_benchmark.attachment.stage3_bridge import (
    Stage3AttachmentBridgeStatus,
    Stage3AttachmentRecord,
)


SETTINGS = EquilibriumResolverSettings(
    max_side_points=2,
    minimum_admissible_models=2,
    minimum_curvature_hartree_per_angstrom2=0.01,
    max_model_geometry_spread_angstrom=0.002,
    max_model_energy_spread_hartree=2.0e-5,
    point_energy_tolerance_hartree=1.0e-8,
)


def record_from_function(fn, *, points=(0.98, 0.99, 1.00, 1.01, 1.02), center=1.00):
    energies = tuple(fn(r) for r in points)
    center_e = fn(center)
    branch = PECBranch(
        branch_id="job",
        state=ElectronicState(
            "state", 0, 1, 0, energy_hartree=center_e, identity_status="CLEARED"
        ),
        r_points=tuple(points),
        energies=energies,
        minimum_r=center,
        minimum_status="BRACKETED_SINGLE_MINIMUM",
        continuity_status="CLEARED",
    )
    return Stage3AttachmentRecord(
        status=Stage3AttachmentBridgeStatus.READY,
        job_id="job",
        branch=branch,
        discrete_minimum_r_angstrom=center,
        discrete_minimum_energy_hartree=center_e,
        minimum_bracket_angstrom=(0.99, 1.01),
        canonical_request_id="req",
        energy_interval_hartree=None,
        evidence=("TEST",),
        rationale="test record",
    )


def test_exact_off_grid_parabola_is_resolved():
    r0 = 1.003
    e0 = -75.123456
    rec = record_from_function(lambda r: e0 + 0.8 * (r - r0) ** 2)
    result = resolve_stage3_equilibrium(rec, settings=SETTINGS)
    assert result.status is EquilibriumStatus.RESOLVED
    assert abs(result.geometry_central_angstrom - r0) < 1.0e-10
    assert result.energy_interval_hartree.lower_hartree <= e0
    assert result.energy_interval_hartree.upper_hartree >= e0
    assert result.evidence_quality == "CONVERGENCE_ESTIMATED"


def test_weak_anharmonicity_can_resolve_when_model_ensemble_agrees():
    r0 = 1.002
    e0 = -20.0
    rec = record_from_function(
        lambda r: e0 + 0.7 * (r - r0) ** 2 + 0.05 * (r - r0) ** 3
    )
    result = resolve_stage3_equilibrium(rec, settings=SETTINGS)
    assert result.status is EquilibriumStatus.RESOLVED
    assert 0.99 <= result.geometry_central_angstrom <= 1.01


def test_three_points_are_not_enough_for_model_stability():
    rec = record_from_function(
        lambda r: -10.0 + (r - 1.0) ** 2,
        points=(0.99, 1.00, 1.01),
    )
    result = resolve_stage3_equilibrium(rec, settings=SETTINGS)
    assert result.status is EquilibriumStatus.UNRESOLVED
    assert "INSUFFICIENT_PEC_POINTS_FOR_MODEL_ENSEMBLE" in result.evidence


def test_large_model_disagreement_stays_unresolved():
    rec = record_from_function(
        lambda r: -10.0 + (r - 1.003) ** 2 + 80.0 * (r - 1.003) ** 4
    )
    strict = replace(
        SETTINGS,
        max_model_energy_spread_hartree=1.0e-9,
    )
    result = resolve_stage3_equilibrium(rec, settings=strict)
    assert result.status is EquilibriumStatus.UNRESOLVED
    assert "EQUILIBRIUM_ENERGY_MODEL_SPREAD_TOO_LARGE" in result.evidence


def test_vertex_outside_stage3_bracket_is_not_accepted():
    rec = record_from_function(
        lambda r: -10.0 + (r - 1.018) ** 2,
        center=1.00,
    )
    result = resolve_stage3_equilibrium(rec, settings=SETTINGS)
    assert result.status is EquilibriumStatus.UNRESOLVED


def test_unready_bridge_record_stays_unresolved():
    rec = record_from_function(lambda r: -10.0 + (r - 1.0) ** 2)
    rec = replace(
        rec,
        status=Stage3AttachmentBridgeStatus.UNRESOLVED,
        branch=None,
        discrete_minimum_r_angstrom=None,
        discrete_minimum_energy_hartree=None,
        minimum_bracket_angstrom=None,
        canonical_request_id=None,
    )
    result = resolve_stage3_equilibrium(rec, settings=SETTINGS)
    assert result.status is EquilibriumStatus.UNRESOLVED
    assert "STAGE3_ATTACHMENT_NOT_READY" in result.evidence


def test_resolver_never_claims_production_ea_or_ground_state():
    rec = record_from_function(lambda r: -10.0 + (r - 1.003) ** 2)
    result = resolve_stage3_equilibrium(rec, settings=SETTINGS)
    assert result.status is EquilibriumStatus.RESOLVED
    assert result.is_production_ea is False
    assert result.ground_state_assigned is False
    assert result.authorizes_pruning is False


def test_energy_upper_bound_tracks_sampled_minimum_plus_explicit_tolerance():
    rec = record_from_function(lambda r: -10.0 + (r - 1.003) ** 2)
    result = resolve_stage3_equilibrium(rec, settings=SETTINGS)
    expected = rec.discrete_minimum_energy_hartree + SETTINGS.point_energy_tolerance_hartree
    assert abs(result.energy_interval_hartree.upper_hartree - expected) < 1.0e-14
