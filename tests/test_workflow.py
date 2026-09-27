from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from openea_benchmark.workflow import (
    dft_scout_geometries,
    load_manifest,
    sector_plan,
    selected_system_names,
    workflow_plan,
)


def make_manifest(
    path: Path,
) -> None:
    data = {
        "benchmark": {
            "name": "test",
        },
        "systems": {
            "XY": {
                "atoms": [
                    "H",
                    "F",
                ],
                "initial_R_angstrom": (
                    1.0
                ),
                "candidate_neutral_2S": [
                    0,
                    2,
                ],
                "candidate_anion_2S": [
                    1,
                ],
                "reference": {
                    "result": "BOUND",
                    "ea_ev": 999.0,
                },
                "validation_expectation": {
                    "dummy": True,
                },
            },
        },
        "protocol": {
            "stage_1_state_discovery": {
                "method": "pbe0",
                "basis": "def2-svp",
            },
            "stage_2_pec_scout": {
                "method": "pbe0",
                "basis": "def2-svp",
                "relative_grid_angstrom": [
                    -0.1,
                    0.0,
                    0.1,
                ],
            },
        },
    }

    path.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
        )
    )


def test_manifest_and_geometry_plan():
    with TemporaryDirectory() as tmp:
        path = (
            Path(tmp)
            / "systems.yaml"
        )

        make_manifest(
            path
        )

        manifest = load_manifest(
            path
        )

        assert (
            dft_scout_geometries(
                manifest,
                "XY",
            )
            == (
                0.9,
                1.0,
                1.1,
            )
        )

        plan = sector_plan(
            manifest,
            "XY",
        )

        assert [
            (
                item["charge"],
                item["spin_2s"],
            )
            for item in plan
        ] == [
            (
                0,
                0,
            ),
            (
                0,
                2,
            ),
            (
                -1,
                1,
            ),
        ]


def test_validation_metadata_not_in_computational_plan():
    with TemporaryDirectory() as tmp:
        path = (
            Path(tmp)
            / "systems.yaml"
        )

        make_manifest(
            path
        )

        manifest = load_manifest(
            path
        )

        plan = workflow_plan(
            manifest
        )

        serialized = repr(
            plan
        )

        forbidden = (
            "999.0",
            "ea_ev",
            "uncertainty_ev",
            "validation_expectation",
            "'reference'",
            "BOUND",
            "UNBOUND",
        )

        for item in forbidden:
            assert item not in serialized

        system_plan = (
            plan["systems"]["XY"]
        )

        assert set(
            system_plan
        ) == {
            "atoms",
            "initial_R_angstrom",
            "sectors",
        }


def test_system_selection_preserves_manifest_order():
    with TemporaryDirectory() as tmp:
        path = (
            Path(tmp)
            / "systems.yaml"
        )

        make_manifest(
            path
        )

        manifest = load_manifest(
            path
        )

        assert (
            selected_system_names(
                manifest,
                None,
            )
            == (
                "XY",
            )
        )

        assert (
            selected_system_names(
                manifest,
                [
                    "XY",
                ],
            )
            == (
                "XY",
            )
        )
