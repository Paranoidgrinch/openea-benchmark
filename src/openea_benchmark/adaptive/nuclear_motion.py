"""Generic one-dimensional nuclear-motion layer for diatomic OpenEA v1.

The electronic-structure workflow provides state-tracked Born--Oppenheimer
potential-energy curves (PECs).  This module solves the J=0 radial nuclear
Schrodinger equation on those PECs and returns the v=0 zero-point contribution
needed for an adiabatic electron affinity::

    Delta_nuc = ZPE(neutral) - ZPE(anion)

The implementation is deliberately independent of PySCF and of any molecule-
specific isotope table.  Masses are explicit inputs with provenance.  The
solver uses a shape-preserving PCHIP interpolation as the primary potential,
an independent natural-cubic spline as an interpolation-sensitivity cross-check,
and nested finite-difference grids plus a trimmed-domain calculation as
numerical/domain checks.

Scientific invariants
---------------------
* The solver consumes identity- and continuity-cleared PECs; it never decides
  electronic-state identity from energies.
* No atomic/isotopic masses are guessed.  The isotopologue masses and their
  provenance are explicit inputs.
* A local finite box is not silently accepted as a vibrationally bound state.
  Boundary clearance and domain sensitivity must both be adequate.
* A v=0 level may invalidate an electronically bound anion if it lies above the
  lowest reviewed dissociation asymptote.
* The numerical/interpolation/domain error estimate is an evidence-based bound,
  not a statistical confidence interval.
* Rotational excitation, DBOC and non-adiabatic corrections are not included.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from typing import Iterable, Sequence

import numpy as np

from openea_benchmark.attachment.asymptote import DissociationChannel
from .model import Interval


ANGSTROM_TO_BOHR = 1.8897261254578281
HARTREE_TO_EV = 27.211386245988
U_TO_ELECTRON_MASS = 1822.888486209


class VibrationalSolveStatus(str, Enum):
    CLEARED = "CLEARED"
    PEC_REFINEMENT_REQUIRED = "PEC_REFINEMENT_REQUIRED"
    NUMERICAL_REFINEMENT_REQUIRED = "NUMERICAL_REFINEMENT_REQUIRED"
    UNRESOLVED = "UNRESOLVED"


class VibrationalBindingStatus(str, Enum):
    BOUND = "BOUND"
    UNBOUND = "UNBOUND"
    UNRESOLVED = "UNRESOLVED"
    NOT_ASSESSED = "NOT_ASSESSED"


class NuclearMotionStatus(str, Enum):
    CLEARED = "CLEARED"
    PEC_REFINEMENT_REQUIRED = "PEC_REFINEMENT_REQUIRED"
    NUMERICAL_REFINEMENT_REQUIRED = "NUMERICAL_REFINEMENT_REQUIRED"
    MODEL_CONVERGENCE_REQUIRED = "MODEL_CONVERGENCE_REQUIRED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class DiatomicMassSpecification:
    atom_a: str
    atom_b: str
    mass_a_u: float
    mass_b_u: float
    evidence_ids: tuple[str, ...]
    mass_model: str = "EXPLICIT_INPUT"

    def __post_init__(self) -> None:
        if not self.atom_a.strip() or not self.atom_b.strip():
            raise ValueError("Diatomic masses require two atom labels")
        for name in ("mass_a_u", "mass_b_u"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.evidence_ids or any(not x.strip() for x in self.evidence_ids):
            raise ValueError("Diatomic mass specification requires provenance evidence IDs")
        if not self.mass_model.strip():
            raise ValueError("mass_model must be non-empty")

    @property
    def reduced_mass_u(self) -> float:
        a = float(self.mass_a_u)
        b = float(self.mass_b_u)
        return a * b / (a + b)

    @property
    def reduced_mass_electron_masses(self) -> float:
        return self.reduced_mass_u * U_TO_ELECTRON_MASS


@dataclass(frozen=True)
class NuclearMotionPEC:
    role: str
    state_id: str
    r_angstrom: tuple[float, ...]
    energy_hartree: tuple[float, ...]
    method: str
    basis: str
    evidence_ids: tuple[str, ...]
    identity_status: str = "CLEARED"
    continuity_status: str = "CLEARED"

    def __post_init__(self) -> None:
        if self.role not in {"neutral", "anion"}:
            raise ValueError("role must be 'neutral' or 'anion'")
        if not self.state_id.strip() or not self.method.strip() or not self.basis.strip():
            raise ValueError("PEC requires state_id, method and basis provenance")
        if len(self.r_angstrom) != len(self.energy_hartree):
            raise ValueError("PEC geometry and energy arrays must align")
        if len(self.r_angstrom) < 5:
            raise ValueError("At least five PEC points are required for nuclear motion")
        if not self.evidence_ids or any(not x.strip() for x in self.evidence_ids):
            raise ValueError("PEC requires evidence IDs")
        r = tuple(float(x) for x in self.r_angstrom)
        e = tuple(float(x) for x in self.energy_hartree)
        if any(not isfinite(x) or x <= 0.0 for x in r):
            raise ValueError("PEC geometries must be positive and finite")
        if any(not isfinite(x) for x in e):
            raise ValueError("PEC energies must be finite")
        if any(b <= a for a, b in zip(r, r[1:])):
            raise ValueError("PEC geometries must be strictly increasing")


@dataclass(frozen=True)
class NuclearMotionSettings:
    coarse_grid_points: int
    fine_grid_points: int
    grid_convergence_tolerance_ev: float
    interpolation_sensitivity_tolerance_ev: float
    domain_sensitivity_tolerance_ev: float
    minimum_boundary_clearance_ev: float
    trim_fraction: float
    eigenvalue_tolerance_hartree: float

    def __post_init__(self) -> None:
        if self.coarse_grid_points < 51:
            raise ValueError("coarse_grid_points must be >= 51")
        if self.fine_grid_points <= self.coarse_grid_points:
            raise ValueError("fine_grid_points must exceed coarse_grid_points")
        if self.coarse_grid_points % 2 == 0 or self.fine_grid_points % 2 == 0:
            raise ValueError("finite-difference grid sizes must be odd")
        for name in (
            "grid_convergence_tolerance_ev",
            "interpolation_sensitivity_tolerance_ev",
            "domain_sensitivity_tolerance_ev",
            "minimum_boundary_clearance_ev",
            "eigenvalue_tolerance_hartree",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(float(self.trim_fraction)) or not (0.0 < self.trim_fraction < 0.25):
            raise ValueError("trim_fraction must lie strictly between 0 and 0.25")


@dataclass(frozen=True)
class NuclearMotionModelEvidence:
    """External convergence bound for the nuclear correction and v=0 binding.

    ``delta_zpe_half_width_ev`` bounds electronic-PEC model sensitivity of the
    *difference* ZPE(N)-ZPE(A), for example from a cardinal/method comparison.
    ``anion_binding_margin_half_width_ev`` bounds model sensitivity of the
    anion v=0-to-dissociation margin and is required only when the same run is
    used to close nuclear binding in G2.
    """

    delta_zpe_half_width_ev: float
    evidence_ids: tuple[str, ...]
    rationale: str
    anion_binding_margin_half_width_ev: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(float(self.delta_zpe_half_width_ev)) or self.delta_zpe_half_width_ev < 0.0:
            raise ValueError("delta_zpe_half_width_ev must be finite and non-negative")
        if self.anion_binding_margin_half_width_ev is not None:
            value = float(self.anion_binding_margin_half_width_ev)
            if not isfinite(value) or value < 0.0:
                raise ValueError("anion_binding_margin_half_width_ev must be finite and non-negative")
        if not self.evidence_ids or any(not x.strip() for x in self.evidence_ids):
            raise ValueError("Nuclear-motion model evidence requires provenance IDs")
        if not self.rationale.strip():
            raise ValueError("Nuclear-motion model evidence requires rationale")


@dataclass(frozen=True)
class VibrationalGroundStateResult:
    role: str
    status: VibrationalSolveStatus
    absolute_v0_energy_hartree: float | None
    potential_minimum_hartree: float | None
    zpe_hartree: float | None
    zpe_ev: float | None
    numerical_bound_ev: float | None
    grid_difference_ev: float | None
    interpolation_difference_ev: float | None
    domain_difference_ev: float | None
    left_boundary_clearance_ev: float | None
    right_boundary_clearance_ev: float | None
    requires_lower_r_extension: bool
    requires_upper_r_extension: bool
    evidence_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if self.role not in {"neutral", "anion"}:
            raise ValueError("Vibrational result role must be neutral or anion")
        if not self.rationale.strip():
            raise ValueError("Vibrational result requires a rationale")
        if self.status is VibrationalSolveStatus.CLEARED:
            required = (
                self.absolute_v0_energy_hartree,
                self.potential_minimum_hartree,
                self.zpe_hartree,
                self.zpe_ev,
                self.numerical_bound_ev,
            )
            if any(x is None for x in required):
                raise ValueError("CLEARED vibrational result is incomplete")
            if not self.evidence_ids:
                raise ValueError("CLEARED vibrational result requires evidence")

    @property
    def absolute_v0_interval_hartree(self) -> Interval | None:
        if self.absolute_v0_energy_hartree is None or self.numerical_bound_ev is None:
            return None
        half = float(self.numerical_bound_ev) / HARTREE_TO_EV
        return Interval(
            float(self.absolute_v0_energy_hartree) - half,
            float(self.absolute_v0_energy_hartree) + half,
        )


@dataclass(frozen=True)
class VibrationalBindingAssessment:
    status: VibrationalBindingStatus
    evidence_ids: tuple[str, ...]
    margin_interval_hartree: Interval | None
    rationale: str

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("Vibrational binding assessment requires rationale")


@dataclass(frozen=True)
class NuclearMotionAssessment:
    status: NuclearMotionStatus
    neutral: VibrationalGroundStateResult
    anion: VibrationalGroundStateResult
    correction_ev: float | None
    correction_half_width_ev: float | None
    anion_vibrational_binding: VibrationalBindingAssessment
    action: str | None
    evidence_ids: tuple[str, ...]
    rationale: str
    includes_rotation: bool = False
    includes_dboc: bool = False
    includes_nonadiabatic_correction: bool = False

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("Nuclear-motion assessment requires rationale")
        if self.includes_rotation or self.includes_dboc or self.includes_nonadiabatic_correction:
            raise ValueError("OpenEA-v1 nuclear-motion runner is J=0 Born-Oppenheimer vibration only")
        if self.status is NuclearMotionStatus.CLEARED:
            if self.correction_ev is None or self.correction_half_width_ev is None:
                raise ValueError("CLEARED nuclear-motion assessment requires a bounded correction")
            if self.action is not None:
                raise ValueError("CLEARED nuclear-motion assessment must not request an action")


# ---------------------------------------------------------------------------
# Shape-preserving interpolation
# ---------------------------------------------------------------------------


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n, dtype=float)

    if n == 2:
        d[:] = delta[0]
        return d

    for k in range(1, n - 1):
        left = delta[k - 1]
        right = delta[k]
        if left == 0.0 or right == 0.0 or left * right <= 0.0:
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / left + w2 / right)

    def endpoint(h0: float, h1: float, del0: float, del1: float) -> float:
        value = ((2.0 * h0 + h1) * del0 - h0 * del1) / (h0 + h1)
        if value * del0 <= 0.0:
            return 0.0
        if del0 * del1 < 0.0 and abs(value) > 3.0 * abs(del0):
            return 3.0 * del0
        return value

    d[0] = endpoint(h[0], h[1], delta[0], delta[1])
    d[-1] = endpoint(h[-1], h[-2], delta[-1], delta[-2])
    return d


def _pchip_eval(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    d = _pchip_slopes(x, y)
    idx = np.searchsorted(x, xq, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    h = x1 - x0
    t = (xq - x0) / h
    y0 = y[idx]
    y1 = y[idx + 1]
    d0 = d[idx]
    d1 = d[idx + 1]
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1


def _pchip_minimum(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    d = _pchip_slopes(x, y)
    best_x = float(x[int(np.argmin(y))])
    best_y = float(np.min(y))

    for i in range(len(x) - 1):
        h = float(x[i + 1] - x[i])
        delta = float((y[i + 1] - y[i]) / h)
        c2 = (3.0 * delta - 2.0 * d[i] - d[i + 1]) / h
        c3 = (d[i] + d[i + 1] - 2.0 * delta) / (h * h)
        candidates = [0.0, h]
        # derivative = d_i + 2*c2*s + 3*c3*s^2
        a = 3.0 * c3
        b = 2.0 * c2
        c = float(d[i])
        if abs(a) < 1.0e-18:
            if abs(b) > 1.0e-18:
                root = -c / b
                if 0.0 < root < h:
                    candidates.append(root)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                root_disc = sqrt(disc)
                for root in ((-b - root_disc) / (2.0 * a), (-b + root_disc) / (2.0 * a)):
                    if 0.0 < root < h:
                        candidates.append(root)
        for s in candidates:
            value = float(y[i] + d[i] * s + c2 * s * s + c3 * s**3)
            if value < best_y:
                best_x = float(x[i] + s)
                best_y = value
    return best_x, best_y


def _natural_cubic_second_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return natural-cubic second derivatives at the tabulated knots."""
    n = len(x)
    if n < 3:
        return np.zeros(n, dtype=float)
    h = np.diff(x)
    size = n - 2
    off = h[1:-1].copy()
    diag = 2.0 * (h[:-1] + h[1:])
    rhs = 6.0 * (
        (y[2:] - y[1:-1]) / h[1:]
        - (y[1:-1] - y[:-2]) / h[:-1]
    )

    # Thomas algorithm for the internal second derivatives.
    c = np.zeros(max(size - 1, 0), dtype=float)
    d = np.zeros(size, dtype=float)
    if size:
        denom = diag[0]
        if size > 1:
            c[0] = off[0] / denom
        d[0] = rhs[0] / denom
        for i in range(1, size):
            denom = diag[i] - off[i - 1] * c[i - 1]
            if i < size - 1:
                c[i] = off[i] / denom
            d[i] = (rhs[i] - off[i - 1] * d[i - 1]) / denom
        internal = np.zeros(size, dtype=float)
        internal[-1] = d[-1]
        for i in range(size - 2, -1, -1):
            internal[i] = d[i] - c[i] * internal[i + 1]
    else:
        internal = np.zeros(0, dtype=float)
    second = np.zeros(n, dtype=float)
    second[1:-1] = internal
    return second


def _natural_cubic_eval(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    second = _natural_cubic_second_derivatives(x, y)
    idx = np.searchsorted(x, xq, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    h = x1 - x0
    a = (x1 - xq) / h
    b = (xq - x0) / h
    return (
        a * y[idx]
        + b * y[idx + 1]
        + ((a**3 - a) * second[idx] + (b**3 - b) * second[idx + 1]) * h**2 / 6.0
    )


# ---------------------------------------------------------------------------
# Lowest eigenvalue of the radial J=0 finite-difference Hamiltonian
# ---------------------------------------------------------------------------


def _sturm_count(diag: np.ndarray, offdiag: float, energy: float) -> int:
    tiny = 1.0e-300
    q = float(diag[0] - energy)
    count = 1 if q < 0.0 else 0
    for value in diag[1:]:
        if abs(q) < tiny:
            q = -tiny if q < 0.0 else tiny
        q = float(value - energy - (offdiag * offdiag) / q)
        if q < 0.0:
            count += 1
    return count


def _lowest_tridiagonal_eigenvalue(
    diag: np.ndarray,
    offdiag: float,
    *,
    tolerance_hartree: float,
) -> float:
    edge = abs(float(offdiag))
    radii = np.full(len(diag), 2.0 * edge)
    radii[0] = edge
    radii[-1] = edge
    lower = float(np.min(diag - radii)) - 1.0
    upper = float(np.min(diag))

    # The lowest eigenvalue must lie below min(diag), but numerical safety is
    # cheap here and prevents a false bracket if future discretizations change.
    step = max(1.0, abs(upper - lower))
    while _sturm_count(diag, offdiag, upper) < 1:
        upper += step
        step *= 2.0

    for _ in range(256):
        midpoint = 0.5 * (lower + upper)
        if _sturm_count(diag, offdiag, midpoint) >= 1:
            upper = midpoint
        else:
            lower = midpoint
        if upper - lower <= tolerance_hartree:
            break
    return 0.5 * (lower + upper)


def _solve_ground_state(
    x_angstrom: np.ndarray,
    y_hartree: np.ndarray,
    *,
    reduced_mass_electron_masses: float,
    grid_points: int,
    interpolation: str,
    eigenvalue_tolerance_hartree: float,
) -> float:
    r_bohr = np.linspace(
        float(x_angstrom[0]) * ANGSTROM_TO_BOHR,
        float(x_angstrom[-1]) * ANGSTROM_TO_BOHR,
        int(grid_points),
    )
    r_angstrom = r_bohr / ANGSTROM_TO_BOHR
    if interpolation == "pchip":
        potential = _pchip_eval(x_angstrom, y_hartree, r_angstrom)
    elif interpolation == "natural_cubic":
        potential = _natural_cubic_eval(x_angstrom, y_hartree, r_angstrom)
    elif interpolation == "linear":
        potential = np.interp(r_angstrom, x_angstrom, y_hartree)
    else:
        raise ValueError(f"Unsupported interpolation {interpolation!r}")

    spacing = float(r_bohr[1] - r_bohr[0])
    mu = float(reduced_mass_electron_masses)
    kinetic_diag = 1.0 / (mu * spacing * spacing)
    kinetic_offdiag = -0.5 / (mu * spacing * spacing)
    diag = potential[1:-1] + kinetic_diag
    return _lowest_tridiagonal_eigenvalue(
        np.asarray(diag, dtype=float),
        kinetic_offdiag,
        tolerance_hartree=eigenvalue_tolerance_hartree,
    )


def _dedup(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(x) for x in items if str(x).strip()))


def solve_vibrational_ground_state(
    pec: NuclearMotionPEC,
    *,
    masses: DiatomicMassSpecification,
    settings: NuclearMotionSettings,
) -> VibrationalGroundStateResult:
    """Solve one identity-cleared J=0 diatomic v=0 state and audit convergence."""

    evidence = _dedup(pec.evidence_ids + masses.evidence_ids)
    if pec.identity_status != "CLEARED" or pec.continuity_status != "CLEARED":
        return VibrationalGroundStateResult(
            pec.role,
            VibrationalSolveStatus.UNRESOLVED,
            None, None, None, None, None, None, None, None, None, None,
            False, False,
            evidence,
            "Electronic-state identity and PEC continuity must both be CLEARED before nuclear motion.",
        )

    x = np.asarray(pec.r_angstrom, dtype=float)
    y = np.asarray(pec.energy_hartree, dtype=float)
    sampled_min_index = int(np.argmin(y))
    if sampled_min_index in {0, len(y) - 1}:
        return VibrationalGroundStateResult(
            pec.role,
            VibrationalSolveStatus.PEC_REFINEMENT_REQUIRED,
            None, None, None, None, None, None, None, None, None, None,
            sampled_min_index == 0,
            sampled_min_index == len(y) - 1,
            evidence + ("NUCLEAR_PEC_MINIMUM_AT_BOUNDARY",),
            "The supplied PEC minimum lies on a domain boundary and cannot support a finite-box v=0 claim.",
        )

    e_coarse = _solve_ground_state(
        x, y,
        reduced_mass_electron_masses=masses.reduced_mass_electron_masses,
        grid_points=settings.coarse_grid_points,
        interpolation="pchip",
        eigenvalue_tolerance_hartree=settings.eigenvalue_tolerance_hartree,
    )
    e_fine = _solve_ground_state(
        x, y,
        reduced_mass_electron_masses=masses.reduced_mass_electron_masses,
        grid_points=settings.fine_grid_points,
        interpolation="pchip",
        eigenvalue_tolerance_hartree=settings.eigenvalue_tolerance_hartree,
    )
    e_cross = _solve_ground_state(
        x, y,
        reduced_mass_electron_masses=masses.reduced_mass_electron_masses,
        grid_points=settings.fine_grid_points,
        interpolation="natural_cubic",
        eigenvalue_tolerance_hartree=settings.eigenvalue_tolerance_hartree,
    )

    span = float(x[-1] - x[0])
    trim = float(settings.trim_fraction) * span
    trim_left = float(x[0] + trim)
    trim_right = float(x[-1] - trim)
    _, vmin = _pchip_minimum(x, y)
    min_r, _ = _pchip_minimum(x, y)

    if not (trim_left < min_r < trim_right):
        return VibrationalGroundStateResult(
            pec.role,
            VibrationalSolveStatus.PEC_REFINEMENT_REQUIRED,
            e_fine, vmin, None, None, None,
            abs(e_fine - e_coarse) * HARTREE_TO_EV,
            abs(e_fine - e_cross) * HARTREE_TO_EV,
            None,
            (float(y[0]) - e_fine) * HARTREE_TO_EV,
            (float(y[-1]) - e_fine) * HARTREE_TO_EV,
            True, True,
            evidence + ("NUCLEAR_DOMAIN_TRIM_EXCLUDES_MINIMUM",),
            "The PEC domain is too narrow around the minimum for an independent domain-sensitivity check.",
        )

    # The trimmed solve evaluates the same primary PCHIP inside a smaller box;
    # this tests finite-domain sensitivity without pretending that extrapolated
    # potential values outside the electronic PEC are known.
    trim_grid = np.linspace(trim_left, trim_right, max(5, settings.fine_grid_points))
    trim_y = _pchip_eval(x, y, trim_grid)
    e_trim = _solve_ground_state(
        trim_grid,
        trim_y,
        reduced_mass_electron_masses=masses.reduced_mass_electron_masses,
        grid_points=settings.fine_grid_points,
        interpolation="pchip",
        eigenvalue_tolerance_hartree=settings.eigenvalue_tolerance_hartree,
    )

    grid_diff = abs(e_fine - e_coarse) * HARTREE_TO_EV
    interp_diff = abs(e_fine - e_cross) * HARTREE_TO_EV
    domain_diff = abs(e_fine - e_trim) * HARTREE_TO_EV
    left_clearance = (float(y[0]) - e_fine) * HARTREE_TO_EV
    right_clearance = (float(y[-1]) - e_fine) * HARTREE_TO_EV
    zpe_h = e_fine - vmin
    zpe_ev = zpe_h * HARTREE_TO_EV

    lower_extension = left_clearance < settings.minimum_boundary_clearance_ev
    upper_extension = right_clearance < settings.minimum_boundary_clearance_ev
    domain_bad = domain_diff > settings.domain_sensitivity_tolerance_ev
    interp_bad = interp_diff > settings.interpolation_sensitivity_tolerance_ev
    grid_bad = grid_diff > settings.grid_convergence_tolerance_ev

    if zpe_h <= 0.0 or not all(isfinite(v) for v in (
        e_fine, vmin, zpe_h, grid_diff, interp_diff, domain_diff,
        left_clearance, right_clearance,
    )):
        return VibrationalGroundStateResult(
            pec.role,
            VibrationalSolveStatus.UNRESOLVED,
            e_fine, vmin, zpe_h, zpe_ev, None,
            grid_diff, interp_diff, domain_diff,
            left_clearance, right_clearance,
            lower_extension, upper_extension,
            evidence + ("NUCLEAR_SOLVER_NONPHYSICAL_RESULT",),
            "The radial solver produced a non-finite or non-positive zero-point energy.",
        )

    if lower_extension or upper_extension or domain_bad or interp_bad:
        reasons = []
        if lower_extension:
            reasons.append("lower-R boundary clearance is insufficient")
        if upper_extension:
            reasons.append("upper-R boundary clearance is insufficient")
        if domain_bad:
            reasons.append("v=0 energy is sensitive to the finite PEC domain")
        if interp_bad:
            reasons.append("v=0 energy is sensitive to PEC interpolation")
        return VibrationalGroundStateResult(
            pec.role,
            VibrationalSolveStatus.PEC_REFINEMENT_REQUIRED,
            e_fine, vmin, zpe_h, zpe_ev,
            grid_diff + interp_diff + domain_diff,
            grid_diff, interp_diff, domain_diff,
            left_clearance, right_clearance,
            lower_extension or domain_bad,
            upper_extension or domain_bad,
            evidence + ("NUCLEAR_PEC_REFINEMENT_REQUIRED",),
            "; ".join(reasons) + ".",
        )

    if grid_bad:
        return VibrationalGroundStateResult(
            pec.role,
            VibrationalSolveStatus.NUMERICAL_REFINEMENT_REQUIRED,
            e_fine, vmin, zpe_h, zpe_ev,
            grid_diff + interp_diff + domain_diff,
            grid_diff, interp_diff, domain_diff,
            left_clearance, right_clearance,
            False, False,
            evidence + ("NUCLEAR_GRID_REFINEMENT_REQUIRED",),
            "Nested finite-difference grids do not yet meet the requested v=0 convergence tolerance.",
        )

    numerical_bound = grid_diff + interp_diff + domain_diff
    return VibrationalGroundStateResult(
        pec.role,
        VibrationalSolveStatus.CLEARED,
        e_fine, vmin, zpe_h, zpe_ev, numerical_bound,
        grid_diff, interp_diff, domain_diff,
        left_clearance, right_clearance,
        False, False,
        evidence + (
            "J0_RADIAL_SCHRODINGER_SOLVED",
            "PCHIP_PRIMARY_POTENTIAL",
            "NATURAL_CUBIC_INTERPOLATION_CROSSCHECK",
            "NESTED_GRID_CONVERGENCE_CLEARED",
            "FINITE_DOMAIN_SENSITIVITY_CLEARED",
            "BOUNDARY_CLEARANCE_CLEARED",
        ),
        "J=0 anharmonic v=0 level is numerically and domain converged within the active explicit settings.",
    )


def assess_vibrational_binding(
    result: VibrationalGroundStateResult,
    channels: Sequence[DissociationChannel] | None,
    *,
    safety_margin_hartree: float = 0.0,
    model_margin_half_width_ev: float | None = None,
) -> VibrationalBindingAssessment:
    """Compare an absolute v=0 interval with the lowest supplied asymptote."""

    if not isfinite(float(safety_margin_hartree)) or safety_margin_hartree < 0.0:
        raise ValueError("safety_margin_hartree must be finite and non-negative")
    if channels is None:
        return VibrationalBindingAssessment(
            VibrationalBindingStatus.NOT_ASSESSED,
            result.evidence_ids,
            None,
            "No dissociation channels were supplied for vibrational binding assessment.",
        )
    channels = tuple(channels)
    if not channels:
        return VibrationalBindingAssessment(
            VibrationalBindingStatus.UNRESOLVED,
            result.evidence_ids,
            None,
            "No dissociation channels were supplied.",
        )
    if result.status is not VibrationalSolveStatus.CLEARED:
        return VibrationalBindingAssessment(
            VibrationalBindingStatus.UNRESOLVED,
            result.evidence_ids,
            None,
            "The v=0 level is not cleared, so vibrational binding cannot be decided.",
        )
    interval = result.absolute_v0_interval_hartree
    assert interval is not None
    if any(channel.asymptotic_energy_hartree is None for channel in channels):
        return VibrationalBindingAssessment(
            VibrationalBindingStatus.UNRESOLVED,
            result.evidence_ids,
            None,
            "At least one supplied dissociation channel lacks an asymptotic energy.",
        )
    energies = tuple(float(channel.asymptotic_energy_hartree) for channel in channels)
    if any(not isfinite(value) for value in energies):
        raise ValueError("Dissociation-channel energies must be finite")
    threshold = min(energies)
    raw_margin = Interval(threshold - interval.upper, threshold - interval.lower)
    evidence = _dedup(result.evidence_ids + tuple(
        f"DISSOCIATION_CHANNEL:{channel.channel_id}" for channel in channels
    ))
    if model_margin_half_width_ev is None:
        return VibrationalBindingAssessment(
            VibrationalBindingStatus.UNRESOLVED,
            evidence,
            raw_margin,
            "A numerical v=0 binding margin is available, but final vibrational binding requires an explicit electronic-PEC/asymptote model-sensitivity bound.",
        )
    model_half_h = float(model_margin_half_width_ev) / HARTREE_TO_EV
    margin = Interval(raw_margin.lower - model_half_h, raw_margin.upper + model_half_h)
    if margin.lower > safety_margin_hartree:
        return VibrationalBindingAssessment(
            VibrationalBindingStatus.BOUND,
            evidence,
            margin,
            "The entire anion v=0 interval lies below the lowest supplied dissociation asymptote with the requested safety margin.",
        )
    if margin.upper <= 0.0:
        return VibrationalBindingAssessment(
            VibrationalBindingStatus.UNBOUND,
            evidence,
            margin,
            "The anion v=0 interval does not lie below the lowest supplied dissociation asymptote.",
        )
    return VibrationalBindingAssessment(
        VibrationalBindingStatus.UNRESOLVED,
        evidence,
        margin,
        "The anion v=0 binding-margin interval overlaps zero or the requested safety margin.",
    )




def derive_nuclear_motion_model_evidence(
    primary: NuclearMotionAssessment,
    comparison: NuclearMotionAssessment,
    *,
    evidence_ids: tuple[str, ...],
    rationale: str,
) -> NuclearMotionModelEvidence:
    """Derive a conservative PEC-model bound from two numerical D12 solves.

    Both inputs are expected to be numerically converged solves that may still
    carry ``MODEL_CONVERGENCE_REQUIRED`` because no cross-level model bound was
    available.  The primary numerical uncertainty is added later by the final
    primary assessment; therefore this model bound adds the observed central
    shift plus the comparison-level numerical uncertainty, avoiding double
    counting of the primary numerical term.
    """
    if not evidence_ids or any(not item.strip() for item in evidence_ids):
        raise ValueError("Model comparison requires evidence IDs")
    if not rationale.strip():
        raise ValueError("Model comparison requires rationale")
    for name, item in (("primary", primary), ("comparison", comparison)):
        if item.neutral.status is not VibrationalSolveStatus.CLEARED or item.anion.status is not VibrationalSolveStatus.CLEARED:
            raise ValueError(f"{name} nuclear-motion solve is not numerically cleared")
        if item.correction_ev is None:
            raise ValueError(f"{name} nuclear-motion solve lacks a DeltaZPE central value")
        if item.neutral.numerical_bound_ev is None or item.anion.numerical_bound_ev is None:
            raise ValueError(f"{name} nuclear-motion solve lacks numerical bounds")

    comparison_numerical = float(
        comparison.neutral.numerical_bound_ev + comparison.anion.numerical_bound_ev
    )
    delta_bound = abs(float(primary.correction_ev) - float(comparison.correction_ev)) + comparison_numerical

    binding_bound_ev = None
    p_margin = primary.anion_vibrational_binding.margin_interval_hartree
    c_margin = comparison.anion_vibrational_binding.margin_interval_hartree
    if p_margin is not None and c_margin is not None:
        binding_bound_ev = (
            abs(p_margin.midpoint - c_margin.midpoint)
            + c_margin.half_width
        ) * HARTREE_TO_EV

    evidence = _dedup(
        tuple(evidence_ids)
        + primary.evidence_ids
        + comparison.evidence_ids
    )
    return NuclearMotionModelEvidence(
        delta_zpe_half_width_ev=delta_bound,
        anion_binding_margin_half_width_ev=binding_bound_ev,
        evidence_ids=evidence,
        rationale=rationale,
    )

def nuclear_motion_pec_from_high_level(
    pec,
    *,
    role: str,
    state_id: str,
    evidence_ids: tuple[str, ...],
) -> NuclearMotionPEC:
    """Bind an identity-cleared Stage-3 HighLevelPEC to the nuclear solver.

    No state identity is inferred here.  The Stage-3 PEC must already have
    cleared same-geometry identity and geometry-continuity reviews, and every
    retained point must be an accepted high-level point using the same energy
    method.
    """
    from .stage3_pec import (
        HighLevelPECStatus,
        HighLevelPointStatus,
        IdentityReviewStatus,
    )

    if pec.status is not HighLevelPECStatus.READY_FOR_DISCRETE_MINIMUM_SCOUT:
        raise ValueError("Stage-3 PEC is not ready for nuclear-motion binding")
    if pec.initialization_identity_status is not IdentityReviewStatus.CLEARED:
        raise ValueError("Stage-3 initialization identity is not cleared")
    if pec.geometry_continuity_status is not IdentityReviewStatus.CLEARED:
        raise ValueError("Stage-3 geometry continuity is not cleared")
    points = tuple(sorted(pec.points, key=lambda item: float(item.r_angstrom)))
    if len(points) < 5:
        raise ValueError("At least five accepted Stage-3 PEC points are required")
    if any(point.status is not HighLevelPointStatus.ACCEPTED for point in points):
        raise ValueError("Every Stage-3 PEC point must be accepted")
    methods = {str(point.energy_method or "") for point in points}
    if len(methods) != 1 or "" in methods:
        raise ValueError("Stage-3 PEC points must share one explicit high-level energy method")
    if any(point.energy_hartree is None for point in points):
        raise ValueError("Accepted Stage-3 PEC point lacks an energy")
    point_evidence = tuple(
        evidence
        for point in points
        for evidence in point.evidence_ids
    )
    checkpoint_evidence = tuple(
        f"HIGH_LEVEL_CHECKPOINT:{path}"
        for point in points
        for path in point.high_level_checkpoint_paths
    )
    evidence = _dedup(tuple(evidence_ids) + point_evidence + checkpoint_evidence)
    if not evidence:
        raise ValueError("Nuclear-motion PEC binding requires evidence IDs")
    return NuclearMotionPEC(
        role=role,
        state_id=state_id,
        r_angstrom=tuple(float(point.r_angstrom) for point in points),
        energy_hartree=tuple(float(point.energy_hartree) for point in points),
        method=next(iter(methods)),
        basis=str(pec.basis),
        evidence_ids=evidence,
        identity_status="CLEARED",
        continuity_status="CLEARED",
    )

def run_diatomic_nuclear_motion(
    *,
    neutral_pec: NuclearMotionPEC,
    anion_pec: NuclearMotionPEC,
    masses: DiatomicMassSpecification,
    settings: NuclearMotionSettings,
    anion_dissociation_channels: Sequence[DissociationChannel] | None = None,
    dissociation_safety_margin_hartree: float = 0.0,
    model_evidence: NuclearMotionModelEvidence | None = None,
) -> NuclearMotionAssessment:
    """Solve neutral/anion v=0 levels and form the adiabatic nuclear correction."""

    if neutral_pec.role != "neutral" or anion_pec.role != "anion":
        raise ValueError("neutral_pec and anion_pec roles are fixed by the API")
    if neutral_pec.method != anion_pec.method:
        raise ValueError("Neutral and anion nuclear PECs must use the same electronic method")
    if neutral_pec.basis != anion_pec.basis:
        raise ValueError("Neutral and anion nuclear PECs must use the same orbital basis policy")

    neutral = solve_vibrational_ground_state(neutral_pec, masses=masses, settings=settings)
    anion = solve_vibrational_ground_state(anion_pec, masses=masses, settings=settings)
    binding = assess_vibrational_binding(
        anion,
        anion_dissociation_channels,
        safety_margin_hartree=dissociation_safety_margin_hartree,
        model_margin_half_width_ev=(
            None if model_evidence is None else model_evidence.anion_binding_margin_half_width_ev
        ),
    )
    evidence = _dedup(
        neutral.evidence_ids
        + anion.evidence_ids
        + binding.evidence_ids
        + (() if model_evidence is None else model_evidence.evidence_ids)
    )

    statuses = {neutral.status, anion.status}
    if VibrationalSolveStatus.UNRESOLVED in statuses:
        return NuclearMotionAssessment(
            NuclearMotionStatus.UNRESOLVED,
            neutral, anion,
            None, None,
            binding,
            "REVIEW_NUCLEAR_MOTION",
            evidence,
            "At least one v=0 solve is scientifically unresolved.",
        )
    if VibrationalSolveStatus.PEC_REFINEMENT_REQUIRED in statuses:
        return NuclearMotionAssessment(
            NuclearMotionStatus.PEC_REFINEMENT_REQUIRED,
            neutral, anion,
            None, None,
            binding,
            "REFINE_NUCLEAR_PEC",
            evidence,
            "At least one electronic PEC requires additional range/density before a stable v=0 correction is available.",
        )
    if VibrationalSolveStatus.NUMERICAL_REFINEMENT_REQUIRED in statuses:
        return NuclearMotionAssessment(
            NuclearMotionStatus.NUMERICAL_REFINEMENT_REQUIRED,
            neutral, anion,
            None, None,
            binding,
            "REFINE_NUCLEAR_MOTION_GRID",
            evidence,
            "At least one radial finite-difference solve requires a denser numerical grid.",
        )

    assert neutral.zpe_ev is not None and anion.zpe_ev is not None
    assert neutral.numerical_bound_ev is not None and anion.numerical_bound_ev is not None
    correction = float(neutral.zpe_ev - anion.zpe_ev)
    if model_evidence is None:
        return NuclearMotionAssessment(
            NuclearMotionStatus.MODEL_CONVERGENCE_REQUIRED,
            neutral, anion,
            correction, None,
            binding,
            "ASSESS_NUCLEAR_PEC_MODEL_CONVERGENCE",
            evidence,
            "Numerical v=0 solutions are converged, but DeltaZPE electronic-PEC model sensitivity is not bounded.",
        )
    # Conservative worst-case propagation; no independence assumption is made.
    half_width = float(
        neutral.numerical_bound_ev
        + anion.numerical_bound_ev
        + model_evidence.delta_zpe_half_width_ev
    )
    return NuclearMotionAssessment(
        NuclearMotionStatus.CLEARED,
        neutral, anion,
        correction, half_width,
        binding,
        None,
        evidence + ("DELTA_ZPE_NEUTRAL_MINUS_ANION",),
        "Neutral and anion J=0 v=0 levels are cleared; the nuclear-motion EA correction is ZPE(neutral)-ZPE(anion).",
    )
