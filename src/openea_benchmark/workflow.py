from __future__ import annotations

import argparse
from dataclasses import (
    fields,
    is_dataclass,
)
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from .adaptive_grid import (
    AdaptiveGridPolicy,
    GridDecisionAction,
    decide_adaptive_grid_extension,
)

from .branch_continuity import (
    BranchThresholds,
)
from .branch_graph import (
    build_branch_graph,
)
from .checkpoint_fingerprint import (
    deduplicate_checkpoint_roots,
)
from .electronic_manifold import (
    ManifoldThresholds,
    construct_electronic_manifolds,
)
from .local_pec import (
    construct_local_pecs,
)
from .manifold_pec import (
    build_manifold_branch_graph,
    construct_manifold_pecs,
    scout_manifold_pec_minimum,
)

from .minimum_scout import (
    MinimumScoutThresholds,
    scout_local_pec_minimum,
)
from .pyscf_backend import (
    DEFAULT_GUESSES,
    DFTMethodSpec,
    DiatomicSpec,
    SCFSettings,
    run_guess_panel,
)
from .root_record import (
    SCFRootRecord,
    SCFRunStatus,
)
from .state_identity import (
    IdentityThresholds,
)


#
# These are explicitly WEEKEND-PILOT policy values.
#
# They are not claimed to be universal production defaults.
#
WEEKEND_IDENTITY_THRESHOLDS = IdentityThresholds(
    same_energy_mev=0.5,
    same_delta_s2=1.0e-5,
    same_total_spectrum_max=1.0e-5,
    same_spin_spectrum_max=1.0e-5,
    same_total_density_rel_fro=1.0e-5,
    same_spin_density_rel_fro=1.0e-5,

    distinct_energy_mev=10.0,
    distinct_delta_s2=0.10,
    distinct_total_spectrum_max=0.10,
    distinct_spin_spectrum_max=0.10,
    distinct_total_density_rel_fro=0.10,
    distinct_spin_density_rel_fro=0.10,
)


WEEKEND_MANIFOLD_THRESHOLDS = ManifoldThresholds(
    max_energy_mev=1.0,
    max_delta_s2=1.0e-3,
    max_total_spectrum=1.0e-3,
    max_spin_spectrum=1.0e-3,
    fractional_occupation_eps=1.0e-3,
    active_overlap_min=0.95,
    bridge_max_span_angstrom=0.35,
)


WEEKEND_BRANCH_THRESHOLDS = BranchThresholds(
    continuous_occ_min=0.95,
    discontinuous_occ_min=0.50,
    continuous_delta_s2=0.02,
    discontinuous_delta_s2=0.20,
    max_step_angstrom=0.10,
)


WEEKEND_MINIMUM_THRESHOLDS = MinimumScoutThresholds(
    energy_tolerance_hartree=1.0e-6,
)


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    return value


def _jsonable(
    value: Any,
    *,
    field_name: str | None = None,
) -> Any:
    """
    Conservative serialization for provenance.

    Large AO objects and PySCF Mole objects are deliberately not serialized.
    """
    value = _enum_value(
        value
    )

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        result = {
            "type": "ndarray",
            "shape": list(
                value.shape
            ),
            "dtype": str(
                value.dtype
            ),
        }

        if value.size:
            if np.issubdtype(
                value.dtype,
                np.number,
            ):
                result[
                    "min"
                ] = float(
                    np.min(
                        value
                    )
                )

                result[
                    "max"
                ] = float(
                    np.max(
                        value
                    )
                )

        return result

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _jsonable(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            tuple,
            list,
            set,
        ),
    ):
        return [
            _jsonable(
                item
            )
            for item in value
        ]

    if is_dataclass(
        value
    ):
        output = {}

        for item in fields(
            value
        ):
            name = item.name

            if name == "mol":
                output[name] = (
                    "<PySCF Mole omitted>"
                )

                continue

            output[name] = _jsonable(
                getattr(
                    value,
                    name,
                ),
                field_name=name,
            )

        return output

    #
    # Avoid huge / unstable repr() output for PySCF-like objects.
    #
    module = getattr(
        type(value),
        "__module__",
        "",
    )

    if module.startswith(
        "pyscf"
    ):
        return (
            f"<{type(value).__name__} omitted>"
        )

    return repr(
        value
    )


def _write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            _jsonable(
                value
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def load_manifest(
    path: Path | str,
) -> dict:
    path = Path(
        path
    )

    data = yaml.safe_load(
        path.read_text()
    )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "manifest must contain a mapping"
        )

    for key in (
        "benchmark",
        "systems",
        "protocol",
    ):
        if key not in data:
            raise ValueError(
                f"manifest missing {key!r}"
            )

    if not isinstance(
        data["systems"],
        dict,
    ):
        raise ValueError(
            "systems must be a mapping"
        )

    for name, system in (
        data["systems"].items()
    ):
        atoms = system.get(
            "atoms"
        )

        if (
            not isinstance(
                atoms,
                list,
            )
            or len(
                atoms
            ) != 2
        ):
            raise ValueError(
                f"{name}: atoms must contain exactly two elements"
            )

        if float(
            system[
                "initial_R_angstrom"
            ]
        ) <= 0.0:
            raise ValueError(
                f"{name}: initial_R_angstrom must be > 0"
            )

        for key in (
            "candidate_neutral_2S",
            "candidate_anion_2S",
        ):
            values = system.get(
                key
            )

            if (
                not isinstance(
                    values,
                    list,
                )
                or not values
            ):
                raise ValueError(
                    f"{name}: {key} must be a non-empty list"
                )

            if any(
                int(value) < 0
                for value in values
            ):
                raise ValueError(
                    f"{name}: negative 2S value"
                )

    return data


def selected_system_names(
    manifest: dict,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    available = tuple(
        manifest[
            "systems"
        ].keys()
    )

    if not requested:
        return available

    unknown = sorted(
        set(
            requested
        )
        - set(
            available
        )
    )

    if unknown:
        raise ValueError(
            "unknown systems: "
            + ", ".join(
                unknown
            )
        )

    requested_set = set(
        requested
    )

    return tuple(
        name
        for name in available
        if name in requested_set
    )


def dft_scout_geometries(
    manifest: dict,
    system_name: str,
) -> tuple[float, ...]:
    system = (
        manifest[
            "systems"
        ][
            system_name
        ]
    )

    r0 = float(
        system[
            "initial_R_angstrom"
        ]
    )

    offsets = (
        manifest[
            "protocol"
        ][
            "stage_2_pec_scout"
        ][
            "relative_grid_angstrom"
        ]
    )

    values = tuple(
        round(
            r0
            + float(
                offset
            ),
            12,
        )
        for offset in offsets
    )

    if any(
        value <= 0.0
        for value in values
    ):
        raise ValueError(
            f"{system_name}: generated non-positive geometry"
        )

    if tuple(
        sorted(
            values
        )
    ) != values:
        raise ValueError(
            f"{system_name}: PEC geometry grid must be sorted"
        )

    if len(
        set(
            values
        )
    ) != len(
        values
    ):
        raise ValueError(
            f"{system_name}: duplicate PEC geometry"
        )

    return values


def sector_plan(
    manifest: dict,
    system_name: str,
) -> tuple[dict, ...]:
    system = (
        manifest[
            "systems"
        ][
            system_name
        ]
    )

    geometries = (
        dft_scout_geometries(
            manifest,
            system_name,
        )
    )

    plans = []

    for charge, key in (
        (
            0,
            "candidate_neutral_2S",
        ),
        (
            -1,
            "candidate_anion_2S",
        ),
    ):
        for spin_2s in system[
            key
        ]:
            plans.append(
                {
                    "system": system_name,
                    "charge": charge,
                    "spin_2s": int(
                        spin_2s
                    ),
                    "geometries_angstrom": (
                        list(
                            geometries
                        )
                    ),
                }
            )

    return tuple(
        plans
    )


def workflow_plan(
    manifest: dict,
    requested: Sequence[str] | None = None,
) -> dict:
    systems = (
        selected_system_names(
            manifest,
            requested,
        )
    )

    method_data = (
        manifest[
            "protocol"
        ][
            "stage_2_pec_scout"
        ]
    )

    stage1 = (
        manifest[
            "protocol"
        ][
            "stage_1_state_discovery"
        ]
    )

    if (
        stage1["method"]
        != method_data["method"]
        or stage1["basis"]
        != method_data["basis"]
    ):
        raise ValueError(
            "weekend driver currently requires identical "
            "stage-1 and stage-2 DFT model"
        )

    return {
        "benchmark": (
            manifest[
                "benchmark"
            ][
                "name"
            ]
        ),
        "method": {
            "functional": (
                method_data[
                    "method"
                ]
            ),
            "basis": (
                method_data[
                    "basis"
                ]
            ),
        },
        "systems": {
            name: {
                "atoms": (
                    manifest[
                        "systems"
                    ][
                        name
                    ][
                        "atoms"
                    ]
                ),
                "initial_R_angstrom": (
                    manifest[
                        "systems"
                    ][
                        name
                    ][
                        "initial_R_angstrom"
                    ]
                ),
                "sectors": list(
                    sector_plan(
                        manifest,
                        name,
                    )
                ),
            }
            for name in systems
        },
    }


def root_summary(
    root: SCFRootRecord,
) -> dict:
    return {
        "root_id": (
            root.root_id
        ),
        "molecule": (
            root.molecule
        ),
        "charge": (
            root.charge
        ),
        "spin_2s": (
            root.spin_2s
        ),
        "r_angstrom": (
            root.r_angstrom
        ),
        "origin_guess": (
            root.origin_guess
        ),
        "reference": (
            root.reference
        ),
        "status": (
            root.status.value
        ),
        "energy_hartree": (
            root.energy_hartree
        ),
        "s2": (
            root.s2
        ),
        "internal_stable": (
            root.internal_stable
        ),
        "checkpoint_path": (
            root.checkpoint_path
        ),
        "diagnostic_message": (
            root.diagnostic_message
        ),
    }


def _representatives_from_dedup(
    roots: Sequence[
        SCFRootRecord
    ],
    dedup,
) -> tuple[
    SCFRootRecord,
    ...,
]:
    root_by_id = {
        root.root_id: root
        for root in roots
    }

    representatives = []

    for cluster in (
        dedup
        .deduplication
        .clusters
    ):
        root_id = (
            cluster
            .representative_root_id
        )

        if root_id not in root_by_id:
            raise RuntimeError(
                "deduplication returned an unknown representative root"
            )

        representatives.append(
            root_by_id[
                root_id
            ]
        )

    return tuple(
        representatives
    )


def _adaptive_grid_policy_from_protocol(
    protocol: dict,
) -> AdaptiveGridPolicy | None:
    config = (
        protocol[
            "stage_2_pec_scout"
        ].get(
            "adaptive_extension"
        )
    )

    if config is None:
        return None

    if not bool(
        config.get(
            "enabled",
            False,
        )
    ):
        return None

    return AdaptiveGridPolicy(
        max_extra_points_per_side=int(
            config[
                "max_extra_points_per_side"
            ]
        ),
        max_extra_span_angstrom=float(
            config[
                "max_extra_span_angstrom"
            ]
        ),
    )


def _run_dft_scout_geometry(
    *,
    label: str,
    atom_a: str,
    atom_b: str,
    charge: int,
    spin_2s: int,
    r_angstrom: float,
    method: DFTMethodSpec,
    guesses: Sequence[str],
    settings: SCFSettings,
    checkpoint_dir: Path,
):
    panel = run_guess_panel(
        DiatomicSpec(
            label=label,
            atom_a=atom_a,
            atom_b=atom_b,
            r_angstrom=(
                r_angstrom
            ),
            charge=charge,
            spin_2s=spin_2s,
        ),
        method,
        guesses=guesses,
        settings=settings,
        checkpoint_dir=(
            checkpoint_dir
        ),
    )

    canonical = tuple(
        root
        for root in panel
        if (
            root.status
            == SCFRunStatus.CANONICALIZED
        )
    )

    geometry_record = {
        "r_angstrom": (
            r_angstrom
        ),
        "all_attempts": [
            root_summary(
                root
            )
            for root in panel
        ],
        "n_attempts": len(
            panel
        ),
        "n_canonical": len(
            canonical
        ),
    }

    if not canonical:
        geometry_record[
            "status"
        ] = "no_canonical_root"

        return (
            geometry_record,
            (),
            (),
        )

    dedup = (
        deduplicate_checkpoint_roots(
            canonical,
            thresholds=(
                WEEKEND_IDENTITY_THRESHOLDS
            ),
        )
    )

    representatives = (
        _representatives_from_dedup(
            canonical,
            dedup,
        )
    )

    geometry_record[
        "deduplication"
    ] = _jsonable(
        dedup
    )

    geometry_record[
        "representative_root_ids"
    ] = [
        root.root_id
        for root
        in representatives
    ]

    manifolds = ()

    try:
        manifold_result = (
            construct_electronic_manifolds(
                representatives,
                identity_thresholds=(
                    WEEKEND_IDENTITY_THRESHOLDS
                ),
                manifold_thresholds=(
                    WEEKEND_MANIFOLD_THRESHOLDS
                ),
            )
        )

        manifolds = tuple(
            manifold_result.manifolds
        )

        geometry_record[
            "manifold_construction"
        ] = _jsonable(
            manifold_result
        )

    except Exception as exc:
        geometry_record[
            "manifold_construction_error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    geometry_record[
        "status"
    ] = "processed"

    return (
        geometry_record,
        tuple(
            representatives
        ),
        manifolds,
    )


def _analyze_sector_dft_scout_collected(
    *,
    label: str,
    charge: int,
    spin_2s: int,
    method: DFTMethodSpec,
    requested_geometries: Sequence[
        float
    ],
    geometry_records,
    representative_roots,
    all_manifolds,
) -> dict:
    geometries = tuple(
        float(
            value
        )
        for value
        in requested_geometries
    )

    geometry_records = tuple(
        sorted(
            geometry_records,
            key=lambda record:
                float(
                    record[
                        "r_angstrom"
                    ]
                ),
        )
    )

    representative_roots = tuple(
        representative_roots
    )

    all_manifolds = tuple(
        all_manifolds
    )

    expected_geometry_count = len(
        geometries
    )

    covered_geometries = sorted(
        {
            float(
                root.r_angstrom
            )
            for root
            in representative_roots
        }
    )

    manifold_geometries = sorted(
        {
            float(
                manifold.r_angstrom
            )
            for manifold
            in all_manifolds
        }
    )

    result = {
        "label": label,
        "charge": charge,
        "spin_2s": spin_2s,
        "method": {
            "functional": (
                method.functional
            ),
            "basis": (
                method.basis
            ),
        },
        "geometries": (
            geometry_records
        ),
        "expected_geometry_count": (
            expected_geometry_count
        ),
        "covered_geometry_count": len(
            covered_geometries
        ),
        "covered_geometries_angstrom": (
            covered_geometries
        ),
        "manifold_geometry_count": len(
            manifold_geometries
        ),
        "manifold_geometries_angstrom": (
            manifold_geometries
        ),
        "representative_roots": [
            root_summary(
                root
            )
            for root
            in representative_roots
        ],
        "manifolds": [
            _jsonable(
                manifold
            )
            for manifold
            in all_manifolds
        ],
    }

    use_root_fallback = (
        len(
            manifold_geometries
        )
        != expected_geometry_count
    )

    if use_root_fallback:
        result[
            "tracking_mode"
        ] = "root"

        try:
            graph = build_branch_graph(
                tuple(
                    representative_roots
                ),
                thresholds=(
                    WEEKEND_BRANCH_THRESHOLDS
                ),
            )

        except Exception as exc:
            result[
                "sector_status"
            ] = "branch_graph_failed"

            result[
                "branch_graph_error"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            return result

        result[
            "branch_graph"
        ] = _jsonable(
            graph
        )

        result[
            "branch_graph_unambiguous"
        ] = bool(
            graph.is_fully_unambiguous
        )

        try:
            pec_result = (
                construct_local_pecs(
                    tuple(
                        representative_roots
                    ),
                    graph,
                )
            )

        except Exception as exc:
            result[
                "sector_status"
            ] = "local_pec_failed"

            result[
                "local_pec_error"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            return result

        result[
            "local_pec_construction"
        ] = _jsonable(
            pec_result
        )

        scouts = []

        for pec in (
            pec_result.pecs
        ):
            scout = (
                scout_local_pec_minimum(
                    pec,
                    thresholds=(
                        WEEKEND_MINIMUM_THRESHOLDS
                    ),
                )
            )

            scouts.append(
                {
                    "component_id": (
                        pec.component_id
                    ),
                    "pec": (
                        _jsonable(
                            pec
                        )
                    ),
                    "minimum_scout": (
                        _jsonable(
                            scout
                        )
                    ),
                }
            )

        result[
            "pec_scouts"
        ] = scouts

        if not pec_result.pecs:
            result[
                "sector_status"
            ] = (
                "no_resolved_local_pec"
            )

        elif (
            not graph
            .is_fully_unambiguous
        ):
            result[
                "sector_status"
            ] = (
                "pec_present_but_graph_ambiguous"
            )

        else:
            result[
                "sector_status"
            ] = "dft_scout_complete"

        return result

    result[
        "tracking_mode"
    ] = "manifold"

    if len(
        manifold_geometries
    ) < 2:
        result[
            "sector_status"
        ] = (
            "insufficient_manifold_geometry_coverage"
        )

        return result

    try:
        manifold_graph = (
            build_manifold_branch_graph(
                tuple(
                    all_manifolds
                ),
                thresholds=(
                    WEEKEND_MANIFOLD_THRESHOLDS
                ),
            )
        )

    except Exception as exc:
        result[
            "sector_status"
        ] = (
            "manifold_branch_graph_failed"
        )

        result[
            "manifold_branch_graph_error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return result

    result[
        "manifold_branch_graph"
    ] = _jsonable(
        manifold_graph
    )

    result[
        "manifold_branch_graph_unambiguous"
    ] = bool(
        manifold_graph
        .is_fully_unambiguous
    )

    try:
        manifold_pec_result = (
            construct_manifold_pecs(
                tuple(
                    all_manifolds
                ),
                manifold_graph,
            )
        )

    except Exception as exc:
        result[
            "sector_status"
        ] = (
            "manifold_pec_failed"
        )

        result[
            "manifold_pec_error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return result

    result[
        "manifold_pec_construction"
    ] = _jsonable(
        manifold_pec_result
    )

    manifold_scouts = []

    for pec in (
        manifold_pec_result.pecs
    ):
        scout = (
            scout_manifold_pec_minimum(
                pec,
                thresholds=(
                    WEEKEND_MINIMUM_THRESHOLDS
                ),
            )
        )

        manifold_scouts.append(
            {
                "component_id": (
                    pec.component_id
                ),
                "pec": (
                    _jsonable(
                        pec
                    )
                ),
                "minimum_scout": (
                    _jsonable(
                        scout
                    )
                ),
            }
        )

    result[
        "manifold_pec_scouts"
    ] = manifold_scouts

    if not (
        manifold_pec_result.pecs
    ):
        result[
            "sector_status"
        ] = (
            "no_resolved_manifold_pec"
        )

    elif (
        not manifold_graph
        .is_fully_unambiguous
    ):
        result[
            "sector_status"
        ] = (
            "manifold_pec_present_but_graph_ambiguous"
        )

    else:
        result[
            "sector_status"
        ] = "dft_scout_complete"

    return result


def run_sector_dft_scout(
    *,
    label: str,
    atom_a: str,
    atom_b: str,
    charge: int,
    spin_2s: int,
    geometries: Sequence[float],
    method: DFTMethodSpec,
    guesses: Sequence[str],
    settings: SCFSettings,
    output_dir: Path,
    adaptive_policy: AdaptiveGridPolicy | None = None,
) -> dict:
    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    initial_geometries = tuple(
        float(
            value
        )
        for value
        in geometries
    )

    if len(
        initial_geometries
    ) < 2:
        raise ValueError(
            "DFT scout requires at least two "
            "initial geometries"
        )

    if tuple(
        sorted(
            initial_geometries
        )
    ) != initial_geometries:
        raise ValueError(
            "initial geometries must be sorted"
        )

    if len(
        set(
            initial_geometries
        )
    ) != len(
        initial_geometries
    ):
        raise ValueError(
            "initial geometries contain duplicates"
        )

    current_geometries = list(
        initial_geometries
    )

    geometry_records_by_r = {}

    representative_roots = []
    all_manifolds = []

    decision_history = []

    n_extension_steps = 0

    def write_progress():
        _write_json(
            output_dir
            / "progress.json",
            {
                "geometries": [
                    geometry_records_by_r[
                        r_angstrom
                    ]
                    for r_angstrom
                    in sorted(
                        geometry_records_by_r
                    )
                ],
                "adaptive_grid": {
                    "enabled": (
                        adaptive_policy
                        is not None
                    ),
                    "initial_grid_angstrom": (
                        initial_geometries
                    ),
                    "current_requested_grid_angstrom": (
                        tuple(
                            current_geometries
                        )
                    ),
                    "computed_geometries_angstrom": (
                        tuple(
                            sorted(
                                geometry_records_by_r
                            )
                        )
                    ),
                    "n_extension_steps": (
                        n_extension_steps
                    ),
                    "decision_history": (
                        decision_history
                    ),
                },
            },
        )

    def calculate_geometry(
        r_angstrom: float,
        *,
        phase: str,
        index: int,
        total: int,
    ) -> None:
        r_angstrom = float(
            r_angstrom
        )

        if r_angstrom in (
            geometry_records_by_r
        ):
            raise RuntimeError(
                "attempted duplicate DFT geometry: "
                f"{r_angstrom:.12f} A"
            )

        print()
        print(
            f"[{label}] q={charge:+d} "
            f"2S={spin_2s} "
            f"R={r_angstrom:.6f} A "
            f"({phase} {index}/{total})"
        )

        (
            geometry_record,
            representatives,
            manifolds,
        ) = _run_dft_scout_geometry(
            label=label,
            atom_a=atom_a,
            atom_b=atom_b,
            charge=charge,
            spin_2s=spin_2s,
            r_angstrom=(
                r_angstrom
            ),
            method=method,
            guesses=guesses,
            settings=settings,
            checkpoint_dir=(
                checkpoint_dir
            ),
        )

        geometry_records_by_r[
            r_angstrom
        ] = geometry_record

        representative_roots.extend(
            representatives
        )

        all_manifolds.extend(
            manifolds
        )

        write_progress()

    #
    # Initial fixed scout grid.
    #
    for index, r_angstrom in enumerate(
        initial_geometries,
        start=1,
    ):
        calculate_geometry(
            r_angstrom,
            phase="initial",
            index=index,
            total=len(
                initial_geometries
            ),
        )

    def analyze_current():
        return (
            _analyze_sector_dft_scout_collected(
                label=label,
                charge=charge,
                spin_2s=spin_2s,
                method=method,
                requested_geometries=tuple(
                    current_geometries
                ),
                geometry_records=tuple(
                    geometry_records_by_r.values()
                ),
                representative_roots=tuple(
                    representative_roots
                ),
                all_manifolds=tuple(
                    all_manifolds
                ),
            )
        )

    result = analyze_current()

    if adaptive_policy is None:
        result[
            "adaptive_grid"
        ] = {
            "enabled": False,
            "policy": None,
            "initial_grid_angstrom": (
                initial_geometries
            ),
            "final_grid_angstrom": tuple(
                current_geometries
            ),
            "n_extension_steps": 0,
            "n_added_geometries": 0,
            "decision_history": (),
            "stop_reason": (
                "ADAPTIVE_DISABLED"
            ),
        }

        return result

    #
    # Hard guard is deliberately larger than the policy can ever use.
    # It protects against a programming error in the decision loop.
    #
    loop_guard = (
        2
        * adaptive_policy
        .max_extra_points_per_side
        + 4
    )

    final_stop_reason = None

    for _iteration in range(
        loop_guard
    ):
        decision = (
            decide_adaptive_grid_extension(
                sector_result=result,
                current_geometries=tuple(
                    current_geometries
                ),
                initial_geometries=(
                    initial_geometries
                ),
                policy=(
                    adaptive_policy
                ),
            )
        )

        decision_history.append(
            _jsonable(
                decision
            )
        )

        if (
            decision.action
            == GridDecisionAction.STOP
        ):
            if (
                decision.stop_reason
                is None
            ):
                raise RuntimeError(
                    "adaptive STOP decision "
                    "has no stop reason"
                )

            final_stop_reason = (
                decision
                .stop_reason
                .value
            )

            break

        new_geometries = tuple(
            float(
                value
            )
            for value
            in decision
            .new_geometries_angstrom
        )

        if not new_geometries:
            raise RuntimeError(
                "adaptive EXTEND decision "
                "contains no geometries"
            )

        n_extension_steps += 1

        for index, r_angstrom in enumerate(
            new_geometries,
            start=1,
        ):
            if r_angstrom in (
                geometry_records_by_r
            ):
                raise RuntimeError(
                    "adaptive policy requested an "
                    "already-computed geometry: "
                    f"{r_angstrom:.12f} A"
                )

            current_geometries.append(
                r_angstrom
            )

            current_geometries.sort()

            calculate_geometry(
                r_angstrom,
                phase=(
                    "adaptive "
                    f"step {n_extension_steps}"
                ),
                index=index,
                total=len(
                    new_geometries
                ),
            )

        result = analyze_current()

    else:
        raise RuntimeError(
            "adaptive grid loop guard reached "
            "before the policy produced STOP"
        )

    if final_stop_reason is None:
        raise RuntimeError(
            "adaptive grid terminated without "
            "a stop reason"
        )

    result[
        "adaptive_grid"
    ] = {
        "enabled": True,
        "policy": _jsonable(
            adaptive_policy
        ),
        "initial_grid_angstrom": (
            initial_geometries
        ),
        "final_grid_angstrom": tuple(
            current_geometries
        ),
        "n_extension_steps": (
            n_extension_steps
        ),
        "n_added_geometries": (
            len(
                current_geometries
            )
            - len(
                initial_geometries
            )
        ),
        "decision_history": tuple(
            decision_history
        ),
        "stop_reason": (
            final_stop_reason
        ),
    }

    write_progress()

    return result


def _sector_directory_name(
    charge: int,
    spin_2s: int,
) -> str:
    charge_tag = (
        f"q{charge:+d}"
        .replace(
            "+",
            "p",
        )
        .replace(
            "-",
            "m",
        )
    )

    return (
        f"{charge_tag}"
        f"__2S{spin_2s}"
    )


def run_system_dft_scout(
    *,
    manifest: dict,
    system_name: str,
    output_root: Path,
    guesses: Sequence[str],
    settings: SCFSettings,
    resume: bool,
) -> dict:
    system = (
        manifest[
            "systems"
        ][
            system_name
        ]
    )

    atom_a, atom_b = (
        system[
            "atoms"
        ]
    )

    protocol = (
        manifest[
            "protocol"
        ]
    )

    adaptive_policy = (
        _adaptive_grid_policy_from_protocol(
            protocol
        )
    )

    method = DFTMethodSpec(
        functional=(
            protocol[
                "stage_2_pec_scout"
            ][
                "method"
            ]
        ),
        basis=(
            protocol[
                "stage_2_pec_scout"
            ][
                "basis"
            ]
        ),
    )

    system_dir = (
        output_root
        / system_name
    )

    system_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sector_results = []

    for plan in sector_plan(
        manifest,
        system_name,
    ):
        charge = int(
            plan[
                "charge"
            ]
        )

        spin_2s = int(
            plan[
                "spin_2s"
            ]
        )

        sector_dir = (
            system_dir
            / _sector_directory_name(
                charge,
                spin_2s,
            )
        )

        final_path = (
            sector_dir
            / "sector.json"
        )

        if (
            resume
            and final_path.is_file()
        ):
            print(
                f"[{system_name}] reusing "
                f"q={charge:+d} 2S={spin_2s}"
            )

            sector_result = (
                json.loads(
                    final_path.read_text()
                )
            )

        else:
            sector_result = (
                run_sector_dft_scout(
                    label=system_name,
                    atom_a=atom_a,
                    atom_b=atom_b,
                    charge=charge,
                    spin_2s=spin_2s,
                    geometries=(
                        plan[
                            "geometries_angstrom"
                        ]
                    ),
                    method=method,
                    guesses=guesses,
                    settings=settings,
                    output_dir=sector_dir,
                    adaptive_policy=adaptive_policy,
                )
            )

            _write_json(
                final_path,
                sector_result,
            )

        sector_results.append(
            sector_result
        )

    summary = {
        "system": (
            system_name
        ),
        "atoms": [
            atom_a,
            atom_b,
        ],
        "computational_input": {
            "initial_R_angstrom": (
                system[
                    "initial_R_angstrom"
                ]
            ),
            "candidate_neutral_2S": (
                system[
                    "candidate_neutral_2S"
                ]
            ),
            "candidate_anion_2S": (
                system[
                    "candidate_anion_2S"
                ]
            ),
        },
        "sectors": (
            sector_results
        ),
        "validation_metadata_used_in_computation": (
            False
        ),
    }

    _write_json(
        system_dir
        / "summary.json",
        summary,
    )

    return summary


def run_dft_scout(
    *,
    manifest: dict,
    requested: Sequence[str] | None,
    output_root: Path,
    guesses: Sequence[str],
    settings: SCFSettings,
    resume: bool,
) -> dict:
    names = selected_system_names(
        manifest,
        requested,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "benchmark": (
            manifest[
                "benchmark"
            ][
                "name"
            ]
        ),
        "mode": (
            "dft-scout"
        ),
        "systems": {},
        "validation_metadata_used_in_computation": (
            False
        ),
    }

    for index, name in enumerate(
        names,
        start=1,
    ):
        print()
        print("=" * 120)
        print(
            f"SYSTEM {index}/{len(names)}: {name}"
        )
        print("=" * 120)

        result[
            "systems"
        ][
            name
        ] = run_system_dft_scout(
            manifest=manifest,
            system_name=name,
            output_root=output_root,
            guesses=guesses,
            settings=settings,
            resume=resume,
        )

        _write_json(
            output_root
            / "summary.json",
            result,
        )

    return result


def _default_output_root(
    manifest: dict,
) -> Path:
    return (
        Path("runs")
        / str(
            manifest[
                "benchmark"
            ][
                "name"
            ]
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generic OpenEA diatomic workflow orchestrator"
        )
    )

    parser.add_argument(
        "manifest",
        type=Path,
    )

    parser.add_argument(
        "--mode",
        choices=(
            "plan",
            "dft-scout",
            "benchmark-auto",
        ),
        default="plan",
    )

    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help=(
            "Restrict to one system. "
            "May be supplied repeatedly."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--grid-level",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--guess",
        action="append",
        dest="guesses",
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--max-stage3-points", type=int, default=100,
        help="Bounded per-invocation CCSD(T) point budget for benchmark-auto",
    )
    parser.add_argument(
        "--stage3-memory-mb", type=int, default=12000,
    )
    parser.add_argument(
        "--no-provisional-stage3", action="store_true",
        help="Execute Stage-3 only when adaptive release gates are cleared",
    )
    parser.add_argument(
        "--auto-dry-run", action="store_true",
        help="Plan/select automatically without executing Stage-3 points",
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = (
        build_parser()
        .parse_args(
            argv
        )
    )

    manifest = load_manifest(
        args.manifest
    )

    if args.mode == "plan":
        print(
            json.dumps(
                workflow_plan(
                    manifest,
                    args.systems,
                ),
                indent=2,
            )
        )

        return 0

    if args.threads < 1:
        raise ValueError(
            "--threads must be >= 1"
        )

    guesses = tuple(
        args.guesses
        if args.guesses
        else DEFAULT_GUESSES
    )

    settings = SCFSettings(
        grid_level=(
            args.grid_level
        ),
        num_threads=(
            args.threads
        ),
    )

    output_root = (
        args.output
        if args.output is not None
        else _default_output_root(
            manifest
        )
    )

    if args.mode == "benchmark-auto":
        from .benchmark_auto import AcquisitionSettings, run_benchmark_acquisition
        records = {}
        for name in selected_system_names(manifest, args.systems):
            summary_path = output_root / name / "summary.json"
            if summary_path.exists():
                with summary_path.open(encoding="utf-8") as f:
                    summary = json.load(f)
                print(f"[BENCHMARK AUTO] Reusing completed scout for {name}", flush=True)
            else:
                print(f"[BENCHMARK AUTO] Running DFT scout for {name}", flush=True)
                summary = run_system_dft_scout(
                    manifest=manifest, system_name=name, output_root=output_root,
                    guesses=guesses, settings=settings, resume=args.resume,
                )
            records[name] = run_benchmark_acquisition(
                summary=summary, manifest=manifest,
                output_dir=output_root / name / "benchmark_auto",
                settings=AcquisitionSettings(
                    max_points=args.max_stage3_points,
                    threads=args.threads,
                    memory_mb=args.stage3_memory_mb,
                    enable_provisional_points=not args.no_provisional_stage3,
                    compute_points=not args.auto_dry_run,
                ),
            )
            print(f"[BENCHMARK AUTO] {name}: " + json.dumps({
                "outcome": records[name]["outcome"],
                "stage3_release": records[name]["stage3_release"],
                "completed": records[name]["completed_point_count"],
                "failed": records[name]["failed_point_count"],
                "pending": records[name]["pending_point_count"],
                "blocked_jobs": len(records[name]["blocked_jobs"]),
            }), flush=True)
        print("BENCHMARK AUTO RUN COMPLETE: results are evidence, not a certified EA")
        return 0

    result = run_dft_scout(
        manifest=manifest,
        requested=args.systems,
        output_root=output_root,
        guesses=guesses,
        settings=settings,
        resume=args.resume,
    )

    print()
    print("=" * 120)
    print("DFT SCOUT COMPLETE")
    print("=" * 120)
    print(
        "output:",
        output_root,
    )

    print(
        "systems:",
        ", ".join(
            result[
                "systems"
            ].keys()
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
