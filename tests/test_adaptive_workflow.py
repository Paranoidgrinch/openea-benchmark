from openea_benchmark import workflow

from openea_benchmark.adaptive_grid import (
    AdaptiveGridPolicy,
)


INITIAL = (
    1.00,
    1.05,
    1.10,
)


def fake_geometry_result(
    r_angstrom,
):
    return (
        {
            "r_angstrom": (
                r_angstrom
            ),
            "status": (
                "processed"
            ),
        },
        (),
        (),
    )


def synthetic_sector_result(
    geometries,
    *,
    always_decreasing=False,
):
    geometries = tuple(
        sorted(
            geometries
        )
    )

    maximum = geometries[
        -1
    ]

    if (
        always_decreasing
        or maximum
        < 1.15 - 1.0e-9
    ):
        status = (
            "decreases_toward_higher_r"
        )

    else:
        status = (
            "bracketed_single_minimum"
        )

    return {
        "tracking_mode": (
            "manifold"
        ),
        "sector_status": (
            "dft_scout_complete"
        ),
        "manifold_branch_graph_unambiguous": (
            True
        ),
        "manifold_pec_scouts": [
            {
                "component_id": (
                    "component"
                ),
                "pec": {
                    "points": [
                        {
                            "r_angstrom": (
                                value
                            ),
                            "energy_center_hartree": (
                                -1.0
                            ),
                            "energy_spread_mev": (
                                0.0
                            ),
                        }
                        for value
                        in geometries
                    ]
                },
                "minimum_scout": {
                    "status": status,
                    "candidates": [],
                },
            }
        ],
    }


def test_sector_adds_only_new_geometry_until_bracketed(
    monkeypatch,
    tmp_path,
):
    calculated = []

    def fake_compute(
        **kwargs,
    ):
        r = float(
            kwargs[
                "r_angstrom"
            ]
        )

        calculated.append(
            r
        )

        return (
            fake_geometry_result(
                r
            )
        )

    def fake_analyze(
        **kwargs,
    ):
        return (
            synthetic_sector_result(
                kwargs[
                    "requested_geometries"
                ]
            )
        )

    monkeypatch.setattr(
        workflow,
        "_run_dft_scout_geometry",
        fake_compute,
    )

    monkeypatch.setattr(
        workflow,
        "_analyze_sector_dft_scout_collected",
        fake_analyze,
    )

    result = (
        workflow.run_sector_dft_scout(
            label="X",
            atom_a="H",
            atom_b="H",
            charge=0,
            spin_2s=1,
            geometries=INITIAL,
            method=object(),
            guesses=(
                "minao",
            ),
            settings=object(),
            output_dir=tmp_path,
            adaptive_policy=(
                AdaptiveGridPolicy(
                    max_extra_points_per_side=4,
                    max_extra_span_angstrom=(
                        0.20
                    ),
                )
            ),
        )
    )

    assert calculated == [
        1.00,
        1.05,
        1.10,
        1.15,
    ]

    assert (
        result[
            "adaptive_grid"
        ][
            "final_grid_angstrom"
        ]
        == (
            1.00,
            1.05,
            1.10,
            1.15,
        )
    )

    assert (
        result[
            "adaptive_grid"
        ][
            "n_added_geometries"
        ]
        == 1
    )

    assert (
        result[
            "adaptive_grid"
        ][
            "n_extension_steps"
        ]
        == 1
    )

    assert (
        result[
            "adaptive_grid"
        ][
            "stop_reason"
        ]
        == "MINIMUM_BRACKETED"
    )


def test_sector_stops_at_extension_limit_without_recomputation(
    monkeypatch,
    tmp_path,
):
    calculated = []

    def fake_compute(
        **kwargs,
    ):
        r = float(
            kwargs[
                "r_angstrom"
            ]
        )

        calculated.append(
            r
        )

        return (
            fake_geometry_result(
                r
            )
        )

    def fake_analyze(
        **kwargs,
    ):
        return (
            synthetic_sector_result(
                kwargs[
                    "requested_geometries"
                ],
                always_decreasing=True,
            )
        )

    monkeypatch.setattr(
        workflow,
        "_run_dft_scout_geometry",
        fake_compute,
    )

    monkeypatch.setattr(
        workflow,
        "_analyze_sector_dft_scout_collected",
        fake_analyze,
    )

    result = (
        workflow.run_sector_dft_scout(
            label="X",
            atom_a="H",
            atom_b="H",
            charge=0,
            spin_2s=1,
            geometries=INITIAL,
            method=object(),
            guesses=(
                "minao",
            ),
            settings=object(),
            output_dir=tmp_path,
            adaptive_policy=(
                AdaptiveGridPolicy(
                    max_extra_points_per_side=2,
                    max_extra_span_angstrom=(
                        0.10
                    ),
                )
            ),
        )
    )

    assert calculated == [
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
    ]

    assert len(
        calculated
    ) == len(
        set(
            calculated
        )
    )

    assert (
        result[
            "adaptive_grid"
        ][
            "stop_reason"
        ]
        == "EXTENSION_LIMIT_REACHED"
    )

    assert (
        result[
            "adaptive_grid"
        ][
            "n_added_geometries"
        ]
        == 2
    )


def test_sector_can_run_with_adaptive_grid_disabled(
    monkeypatch,
    tmp_path,
):
    calculated = []

    def fake_compute(
        **kwargs,
    ):
        r = float(
            kwargs[
                "r_angstrom"
            ]
        )

        calculated.append(
            r
        )

        return (
            fake_geometry_result(
                r
            )
        )

    def fake_analyze(
        **kwargs,
    ):
        return (
            synthetic_sector_result(
                kwargs[
                    "requested_geometries"
                ],
                always_decreasing=True,
            )
        )

    monkeypatch.setattr(
        workflow,
        "_run_dft_scout_geometry",
        fake_compute,
    )

    monkeypatch.setattr(
        workflow,
        "_analyze_sector_dft_scout_collected",
        fake_analyze,
    )

    result = (
        workflow.run_sector_dft_scout(
            label="X",
            atom_a="H",
            atom_b="H",
            charge=0,
            spin_2s=1,
            geometries=INITIAL,
            method=object(),
            guesses=(
                "minao",
            ),
            settings=object(),
            output_dir=tmp_path,
            adaptive_policy=None,
        )
    )

    assert calculated == list(
        INITIAL
    )

    assert (
        result[
            "adaptive_grid"
        ][
            "enabled"
        ]
        is False
    )

    assert (
        result[
            "adaptive_grid"
        ][
            "stop_reason"
        ]
        == "ADAPTIVE_DISABLED"
    )


def test_protocol_builds_explicit_adaptive_policy():
    protocol = {
        "stage_2_pec_scout": {
            "adaptive_extension": {
                "enabled": True,
                "max_extra_points_per_side": 4,
                "max_extra_span_angstrom": 0.20,
            }
        }
    }

    policy = (
        workflow
        ._adaptive_grid_policy_from_protocol(
            protocol
        )
    )

    assert policy is not None

    assert (
        policy.max_extra_points_per_side
        == 4
    )

    assert (
        policy.max_extra_span_angstrom
        == 0.20
    )
