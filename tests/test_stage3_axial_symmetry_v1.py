"""Symmetry equivalence is proven by occupied-subspace rotation, not energy."""
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from types import SimpleNamespace

import numpy as np
import pytest

from openea_benchmark.adaptive.stage3_axial_symmetry import (
    AxialSymmetryPolicy, axial_subspace_alignment,
    compare_initializations_under_axial_symmetry,
)


class LinearMol:
    def __init__(self, r=1.0, axis="z", cart=False, incompatible_metric=False):
        self.cart = cart
        self.natm = 2
        self.r = r
        self.axis = axis
        self.incompatible_metric = incompatible_metric

    def atom_coords(self, unit="Angstrom"):
        v = [0., 0., self.r] if self.axis == "z" else [self.r, 0., 0.]
        return np.array([[0., 0., 0.], v])

    def intor_symmetric(self, key):
        assert key == "int1e_ovlp"
        if self.incompatible_metric:
            return np.diag([1., 1., 2., 1.])
        return np.eye(4)

    def ao_rotation_matrix(self, rotation):
        u = np.eye(4)
        u[1:, 1:] = rotation
        return u


def occ(*basis_indices):
    return np.eye(4)[:, list(basis_indices)]


def aligned(left, right, *, a=occ(0, 1), b=occ(0), right_a=None, right_b=None,
            left_mol=None, right_mol=None, policy=None):
    return axial_subspace_alignment(
        left_mol=left_mol or LinearMol(), right_mol=right_mol or LinearMol(),
        left_alpha=a, left_beta=b,
        right_alpha=right if right_a is None else right_a,
        right_beta=b if right_b is None else right_b,
        cross_overlap=np.eye(4), policy=policy or AxialSymmetryPolicy(),
    )


def test_pi_x_pi_y_components_equivalent_by_rotation():
    evidence = aligned(None, occ(0, 2))
    assert evidence.status == "EQUIVALENT"
    assert evidence.minimum_alpha_overlap > .999999
    assert evidence.minimum_beta_overlap > .999999
    assert evidence.ao_metric_residual < 1e-12


def test_unrotated_occupied_subspace_not_mistaken_for_equivalence():
    assert np.linalg.svd(occ(0, 1).T @ occ(0, 2), compute_uv=False).min() == 0.
    assert aligned(None, occ(0, 2)).status == "EQUIVALENT"


def test_equal_energy_does_not_override_wrong_orbital_space():
    evidence = aligned(None, occ(0, 3))
    assert evidence.status == "NOT_EQUIVALENT"
    assert evidence.minimum_alpha_overlap < 1.e-6


def test_same_angle_must_align_both_spin_subspaces():
    evidence = aligned(None, occ(0, 2), a=occ(0, 1), b=occ(0, 1), right_b=occ(0, 1))
    assert evidence.status == "NOT_EQUIVALENT"


def test_no_valid_ao_symmetry_fails_closed():
    evidence = aligned(None, occ(0, 2), right_mol=LinearMol(incompatible_metric=True))
    assert evidence.status == "UNAVAILABLE"


def test_cartesian_ao_fails_closed():
    evidence = aligned(None, occ(0, 2), right_mol=LinearMol(cart=True))
    assert evidence.status == "UNAVAILABLE"


def test_misaligned_molecule_axes_fail_closed():
    evidence = aligned(None, occ(0, 2), right_mol=LinearMol(axis="x"))
    assert evidence.status == "UNAVAILABLE"


def test_different_bond_lengths_allowed_for_pec_comparison():
    evidence = aligned(None, occ(0, 2), right_mol=LinearMol(r=1.05))
    assert evidence.status == "EQUIVALENT"


def test_energy_context_must_agree_even_for_identical_orbitals():
    base = dict(molecule="XY", charge=0, spin_2s=1, basis="b", reference="ROHF", r_angstrom=1., energy_hartree=-10., s2=.75)
    left = SimpleNamespace(root=SimpleNamespace(**base), mol=LinearMol(), alpha_occ_coeff=occ(0, 1), beta_occ_coeff=occ(0))
    right = SimpleNamespace(root=SimpleNamespace(**{**base, "energy_hartree": -10.01}), mol=LinearMol(), alpha_occ_coeff=occ(0, 2), beta_occ_coeff=occ(0))
    assert compare_initializations_under_axial_symmetry(left, right, AxialSymmetryPolicy()).status == "NOT_EQUIVALENT"
    right.root = SimpleNamespace(**{**base, "s2": 2.})
    assert compare_initializations_under_axial_symmetry(left, right, AxialSymmetryPolicy()).status == "NOT_EQUIVALENT"
    right.root = SimpleNamespace(**{**base, "basis": "other"})
    assert compare_initializations_under_axial_symmetry(left, right, AxialSymmetryPolicy()).status == "UNAVAILABLE"


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        AxialSymmetryPolicy(minimum_occupied_overlap=1.1)
    with pytest.raises(ValueError):
        AxialSymmetryPolicy(angle_samples=6)
    with pytest.raises(ValueError):
        AxialSymmetryPolicy(ao_metric_tolerance=float("nan"))


def test_no_occupied_beta_orbitals():
    evidence = aligned(None, occ(0, 2), b=np.empty((4, 0)), right_b=np.empty((4, 0)))
    assert evidence.status == "EQUIVALENT"
    assert evidence.minimum_beta_overlap == 1.0


def test_rotated_occupied_spaces_invariant_to_occupied_basis_rotation():
    right = occ(0, 2) @ np.array([[np.cos(.37), np.sin(.37)], [-np.sin(.37), np.cos(.37)]])
    evidence = aligned(None, right)
    assert evidence.status == "EQUIVALENT"


def test_legacy_ambiguous_initializations_automatically_clear_when_axially_equivalent(monkeypatch):
    """Prove that the existing guarded Stage-3 initialization review uses new evidence."""
    from openea_benchmark.adaptive import stage3_identity as hl
    from openea_benchmark.state_identity import StateRelation
    from test_adaptive_stage3_identity_v1 import IDENTITY, request, result, evidence

    a, b = request(1., 0, "A"), request(1., 1, "B")
    lhs, rhs = evidence(a), evidence(b)
    lhs = hl._CheckpointEvidence(**{
        **lhs.__dict__, "mol": LinearMol(),
        "alpha_occ_coeff": occ(0, 1), "beta_occ_coeff": occ(0),
    })
    rhs = hl._CheckpointEvidence(**{
        **rhs.__dict__, "mol": LinearMol(),
        "alpha_occ_coeff": occ(0, 2), "beta_occ_coeff": occ(0),
    })
    lookup = {a.request_id: lhs, b.request_id: rhs}
    monkeypatch.setattr(hl, "_load_high_level_checkpoint", lambda req, res, audit_settings: lookup[req.request_id])
    monkeypatch.setattr(hl, "compare_states", lambda *args, **kwargs: SimpleNamespace(relation=StateRelation.AMBIGUOUS))
    audit = []
    review, _, _, _ = hl.review_high_level_initializations(
        requests=[a, b], results=[result(a), result(b)], identity_thresholds=IDENTITY,
        axial_symmetry_policy=AxialSymmetryPolicy(), symmetry_evidence=audit,
    )
    assert review.status.value == "CLEARED"
    assert audit[0]["status"] == "EQUIVALENT"
    assert "hl:init:axial_orbit" in " ".join(review.evidence_ids)


def test_legacy_ambiguous_initializations_remain_unresolved_for_different_subspace(monkeypatch):
    from openea_benchmark.adaptive import stage3_identity as hl
    from openea_benchmark.state_identity import StateRelation
    from test_adaptive_stage3_identity_v1 import IDENTITY, request, result, evidence

    a, b = request(1., 0, "A"), request(1., 1, "B")
    lhs, rhs = evidence(a), evidence(b)
    lhs = hl._CheckpointEvidence(**{**lhs.__dict__, "mol": LinearMol(),
                                    "alpha_occ_coeff": occ(0, 1), "beta_occ_coeff": occ(0)})
    rhs = hl._CheckpointEvidence(**{**rhs.__dict__, "mol": LinearMol(),
                                    "alpha_occ_coeff": occ(0, 3), "beta_occ_coeff": occ(0)})
    lookup = {a.request_id: lhs, b.request_id: rhs}
    monkeypatch.setattr(hl, "_load_high_level_checkpoint", lambda req, res, audit_settings: lookup[req.request_id])
    monkeypatch.setattr(hl, "compare_states", lambda *args, **kwargs: SimpleNamespace(relation=StateRelation.AMBIGUOUS))
    audit = []
    review, _, _, _ = hl.review_high_level_initializations(
        requests=[a, b], results=[result(a), result(b)], identity_thresholds=IDENTITY,
        axial_symmetry_policy=AxialSymmetryPolicy(), symmetry_evidence=audit,
    )
    assert review.status.value == "UNRESOLVED"
    assert audit[0]["status"] == "NOT_EQUIVALENT"


def test_adjacent_pec_geometry_recovers_rotated_open_shell_continuity(monkeypatch):
    import sys
    from openea_benchmark.adaptive import stage3_identity as hl
    from openea_benchmark.branch_continuity import BranchThresholds
    from test_adaptive_stage3_identity_v1 import request, evidence

    a, b = request(.97), request(1.02)
    lhs, rhs = evidence(a), evidence(b)
    lhs = hl._CheckpointEvidence(**{**lhs.__dict__, "mol": LinearMol(.97),
                                    "alpha_occ_coeff": occ(0, 1), "beta_occ_coeff": occ(0)})
    rhs = hl._CheckpointEvidence(**{**rhs.__dict__, "mol": LinearMol(1.02),
                                    "alpha_occ_coeff": occ(0, 2), "beta_occ_coeff": occ(0)})
    gto_mock = SimpleNamespace(intor_cross=lambda key, lm, rm: np.eye(4))
    pyscf_stub = SimpleNamespace(gto=gto_mock)
    monkeypatch.setitem(sys.modules, "pyscf", pyscf_stub)
    from openea_benchmark.branch_continuity import BranchRelation
    policy = BranchThresholds(continuous_occ_min=.95, discontinuous_occ_min=.5,
                               continuous_delta_s2=.02, discontinuous_delta_s2=.2,
                               max_step_angstrom=.1)
    evidence_log = []
    comp = hl._compare_high_level_geometry(lhs, rhs, thresholds=policy,
         axial_symmetry_policy=AxialSymmetryPolicy(), symmetry_evidence=evidence_log)
    assert comp.relation is BranchRelation.CONTINUOUS
    assert evidence_log[0]["status"] == "EQUIVALENT"
