"""Axis-rotation evidence for spin-free diatomic HF determinants.

This module tests whether occupied alpha/beta spaces coincide after one *physical*
SO(2) rotation about the internuclear axis. It NEVER infers identity from equal
energies alone, and does not establish spin-orbit manifold completeness.

The actual AO operator comes from PySCF ``mol.ao_rotation_matrix``.  Molecular
AO overlaps and occupied-subspace singular values are used, not unrotated AO
coefficient arrays or labels such as ``OH`` / ``Pi``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, isfinite, pi, sin
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class AxialSymmetryPolicy:
    minimum_occupied_overlap: float = 0.999
    max_scf_energy_delta_hartree: float = 1.0e-7
    max_s2_delta: float = 1.0e-4
    ao_metric_tolerance: float = 1.0e-6
    geometry_tolerance_angstrom: float = 1.0e-6
    angle_samples: int = 48

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_occupied_overlap <= 1.0:
            raise ValueError("minimum_occupied_overlap must be within (0,1]")
        if any(not isfinite(x) or x < 0 for x in (
            self.max_scf_energy_delta_hartree, self.max_s2_delta,
            self.ao_metric_tolerance, self.geometry_tolerance_angstrom,
        )):
            raise ValueError("Axial-symmetry tolerances must be finite nonnegative")
        if self.angle_samples < 12:
            raise ValueError("At least twelve axis-rotation samples are required")


@dataclass(frozen=True)
class AxialSymmetryEvidence:
    status: str  # EQUIVALENT, NOT_EQUIVALENT, UNAVAILABLE
    angle_rad: float | None
    minimum_alpha_overlap: float | None
    minimum_beta_overlap: float | None
    ao_metric_residual: float | None
    reason: str
    alpha_singular_values: tuple[float, ...] | None = None
    beta_singular_values: tuple[float, ...] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _axis(left_mol: Any, right_mol: Any, tol: float) -> np.ndarray:
    if left_mol.cart or right_mol.cart:
        raise ValueError("Cartesian GTOs not supported by this symmetry evidence path")
    if left_mol.natm != 2 or right_mol.natm != 2:
        raise ValueError("Axis rotation requires diatomic molecules")
    first = np.asarray(left_mol.atom_coords(unit="Angstrom"), dtype=float)
    second = np.asarray(right_mol.atom_coords(unit="Angstrom"), dtype=float)
    if first.shape != (2, 3) or second.shape != (2, 3):
        raise ValueError("Unexpected nuclear coordinate shape")
    vector = first[1] - first[0]
    r = float(np.linalg.norm(vector))
    rv = second[1] - second[0]
    if r < 1.e-8 or np.linalg.norm(rv) < 1.e-8:
        raise ValueError("Degenerate internuclear axis")
    if np.linalg.norm(vector / r - rv / np.linalg.norm(rv)) > tol:
        raise ValueError("Internuclear axes disagree")
    return vector / r


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues SO(3) matrix; proper rotation leaves heteronuclear sites fixed."""
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) * cos(angle) + (1. - cos(angle)) * np.outer(axis, axis) + sin(angle) * skew


def _occupied_singular_values(left: np.ndarray, right: np.ndarray, cross: np.ndarray) -> tuple[float, ...]:
    if left.ndim != 2 or right.ndim != 2 or cross.ndim != 2:
        raise ValueError("Occupied orbital arrays must be matrices")
    if left.shape[1] != right.shape[1] or left.shape[0] != cross.shape[0] or right.shape[0] != cross.shape[1]:
        raise ValueError("Occupied spaces are incompatible")
    if left.shape[1] == 0:
        return ()
    singular_values = np.linalg.svd(left.conj().T @ cross @ right, compute_uv=False)
    if not np.all(np.isfinite(singular_values)) or np.any(singular_values > 1.00001):
        raise ValueError("Invalid occupied-space singular values")
    return tuple(float(min(1., v)) for v in sorted(singular_values, reverse=True))


def axial_subspace_alignment(
    *, left_mol: Any, right_mol: Any,
    left_alpha: np.ndarray, left_beta: np.ndarray,
    right_alpha: np.ndarray, right_beta: np.ndarray,
    cross_overlap: np.ndarray,
    policy: AxialSymmetryPolicy,
) -> AxialSymmetryEvidence:
    """Find one common axial angle maximizing the *worst* alpha/beta overlap.

    The same rotation is used for both spin populations. Overlap-metric
    invariance prevents a numerically spurious AO transformation from being
    treated as a physical symmetry. Exceptions fail closed as UNAVAILABLE.
    """
    try:
        axis = _axis(left_mol, right_mol, policy.geometry_tolerance_angstrom)
        overlap_right = np.asarray(right_mol.intor_symmetric("int1e_ovlp"))
        overlap_cross = np.asarray(cross_overlap)
        left_a, left_b = np.asarray(left_alpha), np.asarray(left_beta)
        right_a, right_b = np.asarray(right_alpha), np.asarray(right_beta)
        n = overlap_right.shape[0]
        if overlap_right.shape != (n, n) or overlap_cross.shape != (left_a.shape[0], n):
            raise ValueError("AO overlap dimensions are incompatible")
        if right_a.shape[0] != n or right_b.shape[0] != n:
            raise ValueError("Occupied coefficient dimensions are incompatible")
        if not np.all(np.isfinite(overlap_right)) or not np.all(np.isfinite(overlap_cross)):
            raise ValueError("Nonfinite AO integrals")

        def metrics(angle: float) -> tuple[float, float, float, tuple[float, ...], tuple[float, ...]]:
            transform = np.asarray(right_mol.ao_rotation_matrix(_rotation(axis, angle)))
            if transform.shape != (n, n):
                raise ValueError("AO rotation dimension mismatch")
            residual = float(np.max(np.abs(transform.conj().T @ overlap_right @ transform - overlap_right)))
            if not isfinite(residual) or residual > policy.ao_metric_tolerance:
                raise ValueError(f"AO rotation is not metric invariant (residual={residual})")
            alpha_values = _occupied_singular_values(left_a, transform @ right_a, overlap_cross)
            beta_values = _occupied_singular_values(left_b, transform @ right_b, overlap_cross)
            return (
                min(alpha_values, default=1.0), min(beta_values, default=1.0),
                residual, alpha_values, beta_values,
            )

        angles = np.linspace(0., 2. * pi, policy.angle_samples, endpoint=False)
        scores = [min(metrics(float(angle))[:2]) for angle in angles]
        idx = int(np.argmax(scores))
        spacing = 2. * pi / policy.angle_samples
        center = float(angles[idx])
        opt = minimize_scalar(
            lambda a: -min(metrics(float(a))[:2]),
            bounds=(center - spacing, center + spacing),
            method="bounded", options={"xatol": 1.e-11},
        )
        options = [(center, scores[idx])]
        if opt.success:
            options.append((float(opt.x), -float(opt.fun)))
        angle, _ = max(options, key=lambda pair: pair[1])
        a_min, b_min, metric, alpha_values, beta_values = metrics(angle)
        equivalent = min(a_min, b_min) >= policy.minimum_occupied_overlap
        return AxialSymmetryEvidence(
            status="EQUIVALENT" if equivalent else "NOT_EQUIVALENT",
            angle_rad=float(angle % (2 * pi)),
            minimum_alpha_overlap=a_min, minimum_beta_overlap=b_min,
            ao_metric_residual=metric,
            reason=("One common axial SO(2) rotation aligns both occupied spin spaces"
                    if equivalent else "No sampled/refined axial SO(2) rotation aligns both occupied spaces"),
            alpha_singular_values=alpha_values,
            beta_singular_values=beta_values,
        )
    except Exception as exc:
        return AxialSymmetryEvidence(
            status="UNAVAILABLE", angle_rad=None,
            minimum_alpha_overlap=None, minimum_beta_overlap=None,
            ao_metric_residual=None,
            reason=f"{type(exc).__name__}: {exc}",
        )


def compare_initializations_under_axial_symmetry(
    left: Any, right: Any, policy: AxialSymmetryPolicy,
) -> AxialSymmetryEvidence:
    """Additional energy/spin/context guards for SAME-GEOMETRY HF references."""
    a, b = left.root, right.root
    if (a.molecule, a.charge, a.spin_2s, a.basis, a.reference) != (
        b.molecule, b.charge, b.spin_2s, b.basis, b.reference
    ):
        return AxialSymmetryEvidence("UNAVAILABLE", None, None, None, None, "Electronic sectors disagree")
    if abs(a.r_angstrom - b.r_angstrom) > policy.geometry_tolerance_angstrom:
        return AxialSymmetryEvidence("UNAVAILABLE", None, None, None, None, "Bond lengths disagree")
    if abs(a.energy_hartree - b.energy_hartree) > policy.max_scf_energy_delta_hartree:
        return AxialSymmetryEvidence("NOT_EQUIVALENT", None, None, None, None, "HF energies disagree")
    if abs(a.s2 - b.s2) > policy.max_s2_delta:
        return AxialSymmetryEvidence("NOT_EQUIVALENT", None, None, None, None, "<S^2> values disagree")
    return axial_subspace_alignment(
        left_mol=left.mol, right_mol=right.mol,
        left_alpha=left.alpha_occ_coeff, left_beta=left.beta_occ_coeff,
        right_alpha=right.alpha_occ_coeff, right_beta=right.beta_occ_coeff,
        cross_overlap=left.mol.intor_symmetric("int1e_ovlp"),
        policy=policy,
    )
