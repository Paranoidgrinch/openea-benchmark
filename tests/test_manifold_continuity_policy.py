import numpy as np
from pyscf import gto

from openea_benchmark.electronic_manifold import (
    ElectronicManifoldPoint,
    ManifoldContinuityRelation,
    ManifoldThresholds,
    compare_electronic_manifolds,
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


def make_molecule():
    return gto.M(
        atom=(
            "H 0 0 0; "
            "H 0 0 0.74"
        ),
        basis="def2-svp",
        unit="Angstrom",
        verbose=0,
    )


MOL = make_molecule()

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


def active_space(
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


def make_point(
    manifold_id,
    r_angstrom,
    *,
    alpha_rank,
    beta_rank,
):
    return ElectronicManifoldPoint(
        manifold_id=manifold_id,
        molecule="X2",
        atom_a="H",
        atom_b="H",
        charge=0,
        spin_2s=1,
        r_angstrom=r_angstrom,
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
        energy_center_hartree=-1.0,
        energy_spread_mev=0.0,
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
            active_space(
                alpha_rank
            )
        ),
        beta_active_ao=(
            active_space(
                beta_rank
            )
        ),
        mol=MOL,
    )


def compare(
    left,
    right,
):
    return compare_electronic_manifolds(
        left,
        right,
        thresholds=THRESHOLDS,
    )


def test_inactive_alpha_and_resolved_beta_is_continuous():
    left = make_point(
        "left",
        1.00,
        alpha_rank=0,
        beta_rank=2,
    )

    right = make_point(
        "right",
        1.05,
        alpha_rank=0,
        beta_rank=2,
    )

    result = compare(
        left,
        right,
    )

    assert (
        result.relation
        == ManifoldContinuityRelation.CONTINUOUS
    )

    assert result.alpha_min_overlap is None

    assert (
        result.beta_min_overlap
        is not None
    )

    assert (
        result.beta_min_overlap
        > 0.999999
    )


def test_all_inactive_remains_gauge_unresolved():
    left = make_point(
        "left",
        1.00,
        alpha_rank=0,
        beta_rank=0,
    )

    right = make_point(
        "right",
        1.05,
        alpha_rank=0,
        beta_rank=0,
    )

    result = compare(
        left,
        right,
    )

    assert (
        result.relation
        == ManifoldContinuityRelation.GAUGE_UNRESOLVED
    )


def test_active_space_disappearance_remains_gauge_unresolved():
    left = make_point(
        "left",
        1.00,
        alpha_rank=0,
        beta_rank=2,
    )

    right = make_point(
        "right",
        1.05,
        alpha_rank=0,
        beta_rank=0,
    )

    result = compare(
        left,
        right,
    )

    assert (
        result.relation
        == ManifoldContinuityRelation.GAUGE_UNRESOLVED
    )


def test_nonzero_rank_mismatch_is_discontinuous():
    left = make_point(
        "left",
        1.00,
        alpha_rank=0,
        beta_rank=1,
    )

    right = make_point(
        "right",
        1.05,
        alpha_rank=0,
        beta_rank=2,
    )

    result = compare(
        left,
        right,
    )

    assert (
        result.relation
        == ManifoldContinuityRelation.DISCONTINUOUS
    )
