import numpy as np
from pyscf import gto

from openea_benchmark.electronic_manifold import (
    ElectronicManifoldPoint,
    ManifoldThresholds,
)
from openea_benchmark.manifold_pec import (
    ManifoldPECRejectionReason,
    build_manifold_branch_graph,
    construct_manifold_pecs,
    scout_manifold_pec_minimum,
)
from openea_benchmark.minimum_scout import (
    MinimumScoutStatus,
    MinimumScoutThresholds,
)


THRESHOLDS = ManifoldThresholds(
    max_energy_mev=1.0,
    max_delta_s2=1.0e-3,
    max_total_spectrum=1.0e-3,
    max_spin_spectrum=1.0e-3,
    fractional_occupation_eps=1.0e-3,
    active_overlap_min=0.95,
    bridge_max_span_angstrom=0.35,
)


MOL = gto.M(
    atom=(
        "H 0 0 0; "
        "H 0 0 0.74"
    ),
    basis="def2-svp",
    unit="Angstrom",
    verbose=0,
)


S = MOL.intor(
    "int1e_ovlp"
)

eigenvalues, eigenvectors = (
    np.linalg.eigh(
        S
    )
)

ORTH_AO = (
    eigenvectors
    @ np.diag(
        eigenvalues ** -0.5
    )
    @ eigenvectors.T
)


def active(
    rank,
):
    if rank == 0:
        return np.zeros(
            (
                MOL.nao_nr(),
                0,
            )
        )

    return ORTH_AO[
        :,
        :rank,
    ].copy()


def make_manifold(
    manifold_id,
    r,
    energy,
    *,
    alpha_rank=0,
    beta_rank=2,
):
    return ElectronicManifoldPoint(
        manifold_id=manifold_id,
        molecule="XH",
        atom_a="H",
        atom_b="H",
        charge=0,
        spin_2s=1,
        r_angstrom=r,
        functional="PBE0",
        basis="def2-SVP",
        reference="UKS",
        ecp_assignments=(),
        member_root_ids=(
            f"{manifold_id}_root",
        ),
        unique_state_root_ids=(
            f"{manifold_id}_root",
        ),
        energy_center_hartree=energy,
        energy_spread_mev=0.01,
        alpha_active_rank=alpha_rank,
        beta_active_rank=beta_rank,
        alpha_fractional_occupations=tuple(
            1.0 / alpha_rank
            for _ in range(
                alpha_rank
            )
        ),
        beta_fractional_occupations=tuple(
            1.0 / beta_rank
            for _ in range(
                beta_rank
            )
        ),
        alpha_active_ao=(
            active(
                alpha_rank
            )
        ),
        beta_active_ao=(
            active(
                beta_rank
            )
        ),
        mol=MOL,
    )


def test_active_manifold_builds_one_unambiguous_pec():
    manifolds = (
        make_manifold(
            "m0",
            0.90,
            -1.00,
        ),
        make_manifold(
            "m1",
            1.00,
            -1.10,
        ),
        make_manifold(
            "m2",
            1.10,
            -1.00,
        ),
    )

    graph = (
        build_manifold_branch_graph(
            manifolds,
            thresholds=THRESHOLDS,
        )
    )

    assert (
        graph.is_fully_unambiguous
        is True
    )

    assert len(
        graph.continuous_edges
    ) == 2

    assert not (
        graph.gauge_unresolved_edges
    )

    result = (
        construct_manifold_pecs(
            manifolds,
            graph,
        )
    )

    assert len(
        result.pecs
    ) == 1

    assert not (
        result.rejected_components
    )

    scout = (
        scout_manifold_pec_minimum(
            result.pecs[0],
            thresholds=(
                MinimumScoutThresholds(
                    energy_tolerance_hartree=(
                        1.0e-8
                    )
                )
            ),
        )
    )

    assert (
        scout.status
        == MinimumScoutStatus
        .BRACKETED_SINGLE_MINIMUM
    )

    assert len(
        scout.candidates
    ) == 1

    assert (
        scout.candidates[0]
        .r_angstrom
        == 1.00
    )


def test_short_unique_all_inactive_run_is_bridged():
    manifolds = (
        make_manifold(
            "m0",
            1.00,
            -1.0,
            alpha_rank=0,
            beta_rank=0,
        ),
        make_manifold(
            "m1",
            1.10,
            -1.1,
            alpha_rank=0,
            beta_rank=0,
        ),
        make_manifold(
            "m2",
            1.20,
            -1.0,
            alpha_rank=0,
            beta_rank=0,
        ),
    )

    graph = (
        build_manifold_branch_graph(
            manifolds,
            thresholds=THRESHOLDS,
        )
    )

    assert (
        graph.is_fully_unambiguous
        is True
    )

    assert not (
        graph.continuous_edges
    )

    assert len(
        graph.bridged_edges
    ) == 2

    assert not (
        graph.gauge_unresolved_edges
    )

    result = (
        construct_manifold_pecs(
            manifolds,
            graph,
        )
    )

    assert len(
        result.pecs
    ) == 1

    assert not (
        result.rejected_components
    )


def test_rank_mismatch_separates_manifold_components():
    manifolds = (
        make_manifold(
            "m0",
            1.00,
            -1.0,
            beta_rank=1,
        ),
        make_manifold(
            "m1",
            1.10,
            -1.1,
            beta_rank=2,
        ),
    )

    graph = (
        build_manifold_branch_graph(
            manifolds,
            thresholds=THRESHOLDS,
        )
    )

    assert len(
        graph.discontinuous_edges
    ) == 1

    assert not (
        graph.continuous_edges
    )

    result = (
        construct_manifold_pecs(
            manifolds,
            graph,
        )
    )

    assert not result.pecs

    assert {
        rejected.reason
        for rejected
        in result.rejected_components
    } == {
        ManifoldPECRejectionReason
        .INSUFFICIENT_POINTS
    }

def test_long_unique_gauge_run_remains_unresolved():
    manifolds = (
        make_manifold(
            "m0",
            1.00,
            -1.0,
            alpha_rank=0,
            beta_rank=0,
        ),
        make_manifold(
            "m1",
            1.20,
            -1.1,
            alpha_rank=0,
            beta_rank=0,
        ),
        make_manifold(
            "m2",
            1.40,
            -1.0,
            alpha_rank=0,
            beta_rank=0,
        ),
    )

    graph = (
        build_manifold_branch_graph(
            manifolds,
            thresholds=THRESHOLDS,
        )
    )

    assert (
        graph.is_fully_unambiguous
        is False
    )

    assert not (
        graph.bridged_edges
    )

    assert len(
        graph.gauge_unresolved_edges
    ) == 2


def test_short_active_space_gap_is_bridged():
    manifolds = (
        make_manifold(
            "m0",
            1.00,
            -1.0,
            alpha_rank=0,
            beta_rank=2,
        ),
        make_manifold(
            "m1",
            1.10,
            -1.1,
            alpha_rank=0,
            beta_rank=0,
        ),
        make_manifold(
            "m2",
            1.20,
            -1.0,
            alpha_rank=0,
            beta_rank=2,
        ),
    )

    graph = (
        build_manifold_branch_graph(
            manifolds,
            thresholds=THRESHOLDS,
        )
    )

    assert (
        graph.is_fully_unambiguous
        is True
    )

    assert len(
        graph.bridged_edges
    ) == 2

    assert not (
        graph.gauge_unresolved_edges
    )

    result = (
        construct_manifold_pecs(
            manifolds,
            graph,
        )
    )

    assert len(
        result.pecs
    ) == 1


def test_discontinuous_rank_change_is_never_bridged():
    manifolds = (
        make_manifold(
            "m0",
            1.00,
            -1.0,
            alpha_rank=0,
            beta_rank=1,
        ),
        make_manifold(
            "m1",
            1.10,
            -1.1,
            alpha_rank=0,
            beta_rank=2,
        ),
    )

    graph = (
        build_manifold_branch_graph(
            manifolds,
            thresholds=THRESHOLDS,
        )
    )

    assert len(
        graph.discontinuous_edges
    ) == 1

    assert not (
        graph.bridged_edges
    )

    assert not (
        graph.gauge_unresolved_edges
    )
