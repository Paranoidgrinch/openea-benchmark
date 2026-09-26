import inspect
import json
import math
import unittest
from pathlib import Path

import numpy as np

from openea_benchmark import (
    LocalPEC,
    LocalPECPoint,
)

from openea_benchmark.minimum_scout import (
    MinimumScoutThresholds,
    scout_local_pec_minimum,
)

from openea_benchmark.vibrational import (
    AMU_TO_ELECTRON_MASS,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_CM,
    VibrationalAnalysisStatus,
    VibrationalFitSettings,
    analyze_local_pec_vibration,
    diatomic_reduced_mass_amu,
)


ENERGY_TOL = 1.0e-9

FEH_PEC_PILOT = (
    Path(__file__).resolve().parents[1]
    / "pilots"
    / "feh_high_accuracy"
    / "05_x2c_ccsdt_pec"
    / "combined.json"
)

#
# Morse reference potential in atomic units.
#
MORSE_DE = 0.1
MORSE_A = 1.0
MORSE_MU_AMU = 1.0
MORSE_RE = 1.0

MORSE_OMEGA = (
    MORSE_A
    * math.sqrt(
        2.0
        * MORSE_DE
        / (MORSE_MU_AMU * AMU_TO_ELECTRON_MASS)
    )
)


def morse_hartree(
    q_bohr,
):
    return (
        MORSE_DE
        * (
            1.0
            - np.exp(
                -MORSE_A
                * q_bohr
            )
        )
        ** 2
    )


def make_pec(
    r_values,
    energies,
) -> LocalPEC:
    return LocalPEC(
        component_id="component_0",
        points=tuple(
            LocalPECPoint(
                root_id=(
                    f"root_{index}"
                ),
                r_angstrom=float(
                    r
                ),
                energy_hartree=float(
                    energy
                ),
            )
            for index, (r, energy) in enumerate(
                zip(
                    r_values,
                    energies,
                )
            )
        ),
    )


def morse_pec(
    half_width_angstrom,
    count,
) -> LocalPEC:
    r_values = np.linspace(
        MORSE_RE - half_width_angstrom,
        MORSE_RE + half_width_angstrom,
        count,
    )

    return make_pec(
        r_values,
        -100.0
        + morse_hartree(
            (r_values - MORSE_RE)
            / BOHR_TO_ANGSTROM
        ),
    )


def analyze(
    pec,
    *,
    degree,
    max_residual=1.0e-6,
    reduced_mass_amu=MORSE_MU_AMU,
):
    scout = scout_local_pec_minimum(
        pec,
        thresholds=MinimumScoutThresholds(
            energy_tolerance_hartree=(
                ENERGY_TOL
            ),
        ),
    )

    return analyze_local_pec_vibration(
        pec,
        scout,
        reduced_mass_amu=reduced_mass_amu,
        fit_settings=VibrationalFitSettings(
            polynomial_degree=degree,
            max_abs_residual_hartree=(
                max_residual
            ),
        ),
    )


class VibrationalSettingsTests(
    unittest.TestCase
):

    def test_settings_have_no_defaults(self):
        for parameter in inspect.signature(
            VibrationalFitSettings
        ).parameters.values():
            self.assertIs(
                parameter.default,
                inspect.Parameter.empty,
            )

    def test_polynomial_degree_must_be_integer_of_at_least_two(self):
        with self.assertRaises(
            ValueError
        ):
            VibrationalFitSettings(
                polynomial_degree=1,
                max_abs_residual_hartree=1.0e-6,
            )

        for degree in (
            2.0,
            True,
        ):
            with self.assertRaises(
                TypeError
            ):
                VibrationalFitSettings(
                    polynomial_degree=degree,
                    max_abs_residual_hartree=1.0e-6,
                )

        settings = VibrationalFitSettings(
            polynomial_degree=np.int64(4),
            max_abs_residual_hartree=1.0e-6,
        )

        self.assertIs(
            type(settings.polynomial_degree),
            int,
        )

    def test_residual_tolerance_must_be_positive(self):
        for value in (
            0.0,
            -1.0e-6,
            math.nan,
        ):
            with self.assertRaises(
                ValueError
            ):
                VibrationalFitSettings(
                    polynomial_degree=2,
                    max_abs_residual_hartree=value,
                )

    def test_reduced_mass_matches_feh_pilot_isotopologue(self):
        #
        # 56Fe and 1H masses used by the FeH local-PEC pilot.
        #
        self.assertAlmostEqual(
            diatomic_reduced_mass_amu(
                55.93493633,
                1.00782503223,
            ),
            0.9899876237290641,
            places=14,
        )

        with self.assertRaises(
            ValueError
        ):
            diatomic_reduced_mass_amu(
                0.0,
                1.0,
            )


class VibrationalInputValidationTests(
    unittest.TestCase
):

    def setUp(self):
        self.pec = morse_pec(
            0.1,
            7,
        )

        self.scout = scout_local_pec_minimum(
            self.pec,
            thresholds=MinimumScoutThresholds(
                energy_tolerance_hartree=(
                    ENERGY_TOL
                ),
            ),
        )

    def call(
        self,
        pec,
        scout,
        *,
        reduced_mass_amu=1.0,
    ):
        return analyze_local_pec_vibration(
            pec,
            scout,
            reduced_mass_amu=reduced_mass_amu,
            fit_settings=VibrationalFitSettings(
                polynomial_degree=2,
                max_abs_residual_hartree=1.0,
            ),
        )

    def test_types_are_checked(self):
        with self.assertRaises(
            TypeError
        ):
            self.call(
                self.pec.points,
                self.scout,
            )

        with self.assertRaises(
            TypeError
        ):
            self.call(
                self.pec,
                self.scout.status,
            )

    def test_reduced_mass_must_be_positive(self):
        with self.assertRaises(
            ValueError
        ):
            self.call(
                self.pec,
                self.scout,
                reduced_mass_amu=0.0,
            )

    def test_scout_must_belong_to_same_pec(self):
        other = LocalPEC(
            component_id="component_1",
            points=self.pec.points,
        )

        with self.assertRaises(
            ValueError
        ):
            self.call(
                other,
                self.scout,
            )

        shifted = make_pec(
            [
                point.r_angstrom
                for point in self.pec.points
            ],
            [
                point.energy_hartree
                + 1.0e-6 * index
                for index, point in enumerate(
                    self.pec.points
                )
            ],
        )

        with self.assertRaises(
            ValueError
        ):
            self.call(
                shifted,
                self.scout,
            )


class HarmonicAnalysisTests(
    unittest.TestCase
):

    def test_exact_harmonic_pec_is_recovered(self):
        re_angstrom = 1.2345
        emin = -50.0
        k_angstrom = 0.6
        reduced_mass = 2.5

        r_values = np.linspace(
            1.15,
            1.35,
            6,
        )

        pec = make_pec(
            r_values,
            emin
            + 0.5
            * k_angstrom
            * (r_values - re_angstrom) ** 2,
        )

        result = analyze(
            pec,
            degree=2,
            max_residual=1.0e-10,
            reduced_mass_amu=reduced_mass,
        )

        self.assertEqual(
            result.status,
            VibrationalAnalysisStatus.RESOLVED,
        )

        omega = math.sqrt(
            k_angstrom
            * BOHR_TO_ANGSTROM**2
            / (reduced_mass * AMU_TO_ELECTRON_MASS)
        )

        harmonic = result.harmonic

        self.assertAlmostEqual(
            harmonic.re_angstrom,
            re_angstrom,
            places=10,
        )

        self.assertAlmostEqual(
            harmonic.emin_hartree,
            emin,
            places=10,
        )

        self.assertAlmostEqual(
            harmonic.omega_e_cm,
            omega * HARTREE_TO_CM,
            places=6,
        )

        self.assertAlmostEqual(
            harmonic.zpe_hartree,
            0.5 * omega,
            places=12,
        )

    def test_well_sampled_morse_pec_recovers_exact_constants(self):
        result = analyze(
            morse_pec(
                0.6,
                25,
            ),
            degree=10,
        )

        self.assertEqual(
            result.status,
            VibrationalAnalysisStatus.RESOLVED,
        )

        self.assertAlmostEqual(
            result.harmonic.re_angstrom,
            MORSE_RE,
            delta=1.0e-6,
        )

        self.assertAlmostEqual(
            result.harmonic.omega_e_cm,
            MORSE_OMEGA * HARTREE_TO_CM,
            delta=1.0e-2,
        )

    def test_reproduces_feh_pilot_harmonic_analysis(self):
        pilot = json.loads(
            FEH_PEC_PILOT.read_text()
        )

        for key, system in pilot[
            "systems"
        ].items():
            with self.subTest(
                system=key
            ):
                pec = make_pec(
                    [
                        point["R_angstrom"]
                        for point in system["points"]
                    ],
                    [
                        point["ccsdt_energy_hartree"]
                        for point in system["points"]
                    ],
                )

                result = analyze(
                    pec,
                    degree=2,
                    max_residual=1.0e-4,
                    reduced_mass_amu=(
                        pilot["reduced_mass_u"]
                    ),
                )

                self.assertEqual(
                    result.status,
                    VibrationalAnalysisStatus.RESOLVED,
                )

                expected = system[
                    "analysis"
                ]

                self.assertAlmostEqual(
                    result.harmonic.re_angstrom,
                    expected["Re_angstrom"],
                    places=9,
                )

                self.assertAlmostEqual(
                    result.harmonic.emin_hartree,
                    expected["Emin_hartree"],
                    places=10,
                )

                self.assertAlmostEqual(
                    result.harmonic.omega_e_cm,
                    expected["omega_e_cm-1"],
                    places=5,
                )

                self.assertAlmostEqual(
                    result.harmonic.zpe_hartree,
                    expected["zpe_hartree"],
                    places=10,
                )


class VibrationalRejectionTests(
    unittest.TestCase
):

    def test_pec_without_bracketed_minimum_is_not_fitted(self):
        r_values = np.linspace(
            1.0,
            1.5,
            6,
        )

        result = analyze(
            make_pec(
                r_values,
                -r_values,
            ),
            degree=2,
        )

        self.assertEqual(
            result.status,
            VibrationalAnalysisStatus.NO_BRACKETED_MINIMUM,
        )

        self.assertIsNone(
            result.fit
        )

        self.assertIsNone(
            result.harmonic
        )

    def test_fit_requires_residual_degree_of_freedom(self):
        result = analyze(
            morse_pec(
                0.1,
                5,
            ),
            degree=4,
        )

        self.assertEqual(
            result.status,
            VibrationalAnalysisStatus.INSUFFICIENT_POINTS,
        )

        self.assertIsNone(
            result.fit
        )

    def test_fit_residual_above_tolerance_is_rejected_but_retained(self):
        result = analyze(
            morse_pec(
                0.3,
                9,
            ),
            degree=2,
            max_residual=1.0e-6,
        )

        self.assertEqual(
            result.status,
            VibrationalAnalysisStatus.FIT_RESIDUAL_EXCEEDS_TOLERANCE,
        )

        self.assertIsNone(
            result.harmonic
        )

        self.assertEqual(
            len(result.fit.residuals_hartree),
            9,
        )

        self.assertGreater(
            result.fit.max_abs_residual_hartree,
            1.0e-6,
        )

    def test_fit_with_extra_stationary_points_is_not_unimodal(self):
        #
        # V' = 4x(x - 0.3)(x - 0.35): the samples decrease then increase, but
        # the fitted curve has a hidden maximum and second minimum.
        #
        x = np.array(
            [
                -0.4,
                -0.2,
                0.0,
                0.2,
                0.4,
                0.6,
            ]
        )

        result = analyze(
            make_pec(
                1.5 + x,
                x**4
                - (4.0 / 3.0) * 0.65 * x**3
                + 2.0 * 0.105 * x**2,
            ),
            degree=4,
            max_residual=1.0e-10,
        )

        self.assertEqual(
            result.status,
            VibrationalAnalysisStatus.FIT_NOT_UNIMODAL,
        )

    def test_fit_minimum_must_lie_between_candidate_neighbours(self):
        x = np.linspace(
            0.0,
            0.5,
            6,
        )

        result = analyze(
            make_pec(
                1.5 + x,
                np.array(
                    [
                        1.0,
                        0.0,
                        0.05,
                        0.1,
                        0.15,
                        0.2,
                    ]
                )
                * 1.0e-3,
            ),
            degree=2,
            max_residual=1.0,
        )

        self.assertEqual(
            result.status,
            VibrationalAnalysisStatus.FIT_MINIMUM_OUTSIDE_BRACKET,
        )


if __name__ == "__main__":
    unittest.main()
