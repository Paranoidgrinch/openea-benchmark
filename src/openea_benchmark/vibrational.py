from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from operator import index as as_integer

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial import polynomial as P

from .local_pec import (
    LocalPEC,
)
from .minimum_scout import (
    MinimumScoutResult,
    MinimumScoutStatus,
)


#
# CODATA 2018, identical to the constants used by the FeH local-PEC pilot.
#
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_CM = 219474.63136320
AMU_TO_ELECTRON_MASS = 1822.888486209

#
# Complex derivative roots with a relative imaginary part below this value are
# treated as real stationary points. This is deliberately loose: a nearly real
# pair indicates a near-flat shoulder, which must not pass as unimodal.
#
_STATIONARY_POINT_IMAG_TOLERANCE = 1.0e-6

_SCOUT_DELTA_TOLERANCE_HARTREE = 1.0e-12


class VibrationalAnalysisStatus(str, Enum):
    """
    Outcome of the fitted-minimum and harmonic stage.

    Only RESOLVED provides an equilibrium geometry and harmonic frequency.
    """

    RESOLVED = (
        "resolved"
    )

    NO_BRACKETED_MINIMUM = (
        "no_bracketed_minimum"
    )

    INSUFFICIENT_POINTS = (
        "insufficient_points"
    )

    FIT_RESIDUAL_EXCEEDS_TOLERANCE = (
        "fit_residual_exceeds_tolerance"
    )

    FIT_NOT_UNIMODAL = (
        "fit_not_unimodal"
    )

    FIT_MINIMUM_OUTSIDE_BRACKET = (
        "fit_minimum_outside_bracket"
    )

    NON_POSITIVE_CURVATURE = (
        "non_positive_curvature"
    )


def _positive_finite(
    value: float,
    name: str,
) -> float:
    value = float(
        value
    )

    if (
        not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(
            f"{name} must be finite and > 0"
        )

    return value


@dataclass(frozen=True)
class VibrationalFitSettings:
    """
    Explicit polynomial fit model and acceptance tolerance.

    No production default is provided deliberately.
    """

    polynomial_degree: int
    max_abs_residual_hartree: float

    def __post_init__(self) -> None:
        if isinstance(
            self.polynomial_degree,
            bool,
        ):
            raise TypeError(
                "polynomial_degree must be an integer"
            )

        try:
            degree = as_integer(
                self.polynomial_degree
            )
        except TypeError as error:
            raise TypeError(
                "polynomial_degree must be an integer"
            ) from error

        if degree < 2:
            raise ValueError(
                "polynomial_degree must be >= 2"
            )

        object.__setattr__(
            self,
            "polynomial_degree",
            degree,
        )

        object.__setattr__(
            self,
            "max_abs_residual_hartree",
            _positive_finite(
                self.max_abs_residual_hartree,
                "max_abs_residual_hartree",
            ),
        )


@dataclass(frozen=True)
class PECPolynomialFit:
    """
    Least-squares polynomial representation of one LocalPEC.

    coefficients are in Eh / Angstrom**k, in ascending powers of
    (R - reference_r_angstrom), for the energy relative to
    reference_energy_hartree. The reference is the sampled discrete minimum
    candidate.

    residuals_hartree are sampled minus fitted energies, in PEC point order.
    """

    polynomial_degree: int
    reference_r_angstrom: float
    reference_energy_hartree: float

    coefficients: tuple[
        float,
        ...,
    ]

    residuals_hartree: tuple[
        float,
        ...,
    ]

    max_abs_residual_hartree: float


@dataclass(frozen=True)
class HarmonicVibration:
    """
    Fitted equilibrium and harmonic vibration of one electronic branch.
    """

    re_angstrom: float
    emin_hartree: float

    force_constant_hartree_per_bohr2: float
    omega_e_cm: float
    zpe_hartree: float


@dataclass(frozen=True)
class VibrationalAnalysisResult:
    """
    Result of fitted harmonic analysis of one LocalPEC.
    """

    component_id: str

    status: VibrationalAnalysisStatus

    reduced_mass_amu: float

    fit: PECPolynomialFit | None
    harmonic: HarmonicVibration | None


def diatomic_reduced_mass_amu(
    mass_a_amu: float,
    mass_b_amu: float,
) -> float:
    """
    Reduced mass of an explicitly specified isotopologue.
    """
    mass_a = _positive_finite(
        mass_a_amu,
        "mass_a_amu",
    )

    mass_b = _positive_finite(
        mass_b_amu,
        "mass_b_amu",
    )

    return (
        mass_a
        * mass_b
        / (mass_a + mass_b)
    )


def _validate_scout(
    pec: LocalPEC,
    scout: MinimumScoutResult,
) -> None:
    if scout.component_id != pec.component_id:
        raise ValueError(
            "scout result belongs to a different LocalPEC component"
        )

    expected = tuple(
        right.energy_hartree
        - left.energy_hartree
        for left, right in zip(
            pec.points,
            pec.points[1:],
        )
    )

    if (
        len(expected)
        != len(scout.adjacent_delta_e_hartree)
        or any(
            abs(a - b)
            > _SCOUT_DELTA_TOLERANCE_HARTREE
            for a, b in zip(
                expected,
                scout.adjacent_delta_e_hartree,
            )
        )
    ):
        raise ValueError(
            "scout result does not match the supplied LocalPEC energies"
        )

    for candidate in scout.candidates:
        index = candidate.point_index

        if (
            not 0 < index < len(pec.points) - 1
            or pec.points[index].root_id
            != candidate.root_id
        ):
            raise ValueError(
                "scout candidate does not match the supplied LocalPEC"
            )


def _stationary_points(
    poly: Polynomial,
    lower: float,
    upper: float,
) -> tuple[
    float,
    ...,
]:
    """
    Real stationary points of poly on the closed interval [lower, upper].
    """
    tolerance = (
        _STATIONARY_POINT_IMAG_TOLERANCE
        * max(
            1.0,
            upper - lower,
        )
    )

    return tuple(
        sorted(
            float(root.real)
            for root in np.atleast_1d(
                poly.deriv().roots()
            )
            if (
                abs(root.imag)
                <= tolerance
                and lower
                <= root.real
                <= upper
            )
        )
    )


def analyze_local_pec_vibration(
    pec: LocalPEC,
    scout: MinimumScoutResult,
    *,
    reduced_mass_amu: float,
    fit_settings: VibrationalFitSettings,
) -> VibrationalAnalysisResult:
    """
    Fit one LocalPEC around its discrete minimum and analyse its vibration.

    The analysis proceeds only when the scout result for the same LocalPEC is
    BRACKETED_SINGLE_MINIMUM. The fitted polynomial must reproduce every
    sampled point within tolerance, have exactly one stationary point over the
    sampled range, and place that minimum strictly between the neighbours of
    the discrete candidate.

    The harmonic frequency and zero-point energy follow from the fitted
    curvature at Re.

    All points of the supplied LocalPEC are fitted. The caller controls the
    fit window through the LocalPEC it supplies.

    This function deliberately does not:

    - fit PECs with unresolved, multiple, or unbracketed minima;
    - choose a fit degree or tolerance;
    - estimate anharmonic zero-point corrections;
    - include rotation; results refer to J=0;
    - compare electronic branches, spin sectors, or charge states;
    - calculate an electron affinity.
    """
    if not isinstance(
        pec,
        LocalPEC,
    ):
        raise TypeError(
            "pec must be a LocalPEC"
        )

    if not isinstance(
        scout,
        MinimumScoutResult,
    ):
        raise TypeError(
            "scout must be a MinimumScoutResult"
        )

    reduced_mass = _positive_finite(
        reduced_mass_amu,
        "reduced_mass_amu",
    )

    _validate_scout(
        pec,
        scout,
    )

    def unresolved(
        status: VibrationalAnalysisStatus,
        fit: PECPolynomialFit | None,
    ) -> VibrationalAnalysisResult:
        return VibrationalAnalysisResult(
            component_id=(
                pec.component_id
            ),
            status=status,
            reduced_mass_amu=reduced_mass,
            fit=fit,
            harmonic=None,
        )

    if (
        scout.status
        is not MinimumScoutStatus.BRACKETED_SINGLE_MINIMUM
    ):
        return unresolved(
            VibrationalAnalysisStatus.NO_BRACKETED_MINIMUM,
            None,
        )

    degree = (
        fit_settings.polynomial_degree
    )

    #
    # At least one residual degree of freedom is required. An interpolating
    # polynomial reproduces every point exactly, so the residual tolerance
    # would not test anything.
    #
    if len(pec.points) < degree + 2:
        return unresolved(
            VibrationalAnalysisStatus.INSUFFICIENT_POINTS,
            None,
        )

    candidate = scout.candidates[0]

    x_angstrom = np.array(
        [
            point.r_angstrom
            - candidate.r_angstrom
            for point in pec.points
        ]
    )

    y_hartree = np.array(
        [
            point.energy_hartree
            - candidate.energy_hartree
            for point in pec.points
        ]
    )

    poly = Polynomial(
        P.polyfit(
            x_angstrom,
            y_hartree,
            degree,
        )
    )

    residuals = (
        y_hartree
        - poly(x_angstrom)
    )

    max_residual = float(
        np.max(
            np.abs(
                residuals
            )
        )
    )

    fit = PECPolynomialFit(
        polynomial_degree=degree,
        reference_r_angstrom=(
            candidate.r_angstrom
        ),
        reference_energy_hartree=(
            candidate.energy_hartree
        ),
        coefficients=tuple(
            float(value)
            for value in poly.coef
        ),
        residuals_hartree=tuple(
            float(value)
            for value in residuals
        ),
        max_abs_residual_hartree=max_residual,
    )

    if (
        max_residual
        > fit_settings.max_abs_residual_hartree
    ):
        return unresolved(
            VibrationalAnalysisStatus.FIT_RESIDUAL_EXCEEDS_TOLERANCE,
            fit,
        )

    stationary = _stationary_points(
        poly,
        float(x_angstrom[0]),
        float(x_angstrom[-1]),
    )

    if len(stationary) > 1:
        return unresolved(
            VibrationalAnalysisStatus.FIT_NOT_UNIMODAL,
            fit,
        )

    if not (
        stationary
        and candidate.left_r_angstrom
        - candidate.r_angstrom
        < stationary[0]
        < candidate.right_r_angstrom
        - candidate.r_angstrom
    ):
        return unresolved(
            VibrationalAnalysisStatus.FIT_MINIMUM_OUTSIDE_BRACKET,
            fit,
        )

    x_min = stationary[0]

    curvature_angstrom = float(
        poly.deriv(2)(x_min)
    )

    if curvature_angstrom <= 0.0:
        return unresolved(
            VibrationalAnalysisStatus.NON_POSITIVE_CURVATURE,
            fit,
        )

    #
    # d2E/dR_bohr^2 = d2E/dR_A^2 * (A / bohr)^2
    #
    force_constant = (
        curvature_angstrom
        * BOHR_TO_ANGSTROM**2
    )

    reduced_mass_me = (
        reduced_mass
        * AMU_TO_ELECTRON_MASS
    )

    omega_au = sqrt(
        force_constant
        / reduced_mass_me
    )

    harmonic = HarmonicVibration(
        re_angstrom=(
            candidate.r_angstrom
            + x_min
        ),
        emin_hartree=(
            candidate.energy_hartree
            + float(poly(x_min))
        ),
        force_constant_hartree_per_bohr2=(
            force_constant
        ),
        omega_e_cm=(
            omega_au
            * HARTREE_TO_CM
        ),
        zpe_hartree=(
            0.5
            * omega_au
        ),
    )

    return VibrationalAnalysisResult(
        component_id=(
            pec.component_id
        ),
        status=(
            VibrationalAnalysisStatus.RESOLVED
        ),
        reduced_mass_amu=reduced_mass,
        fit=fit,
        harmonic=harmonic,
    )
