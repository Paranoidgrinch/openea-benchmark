from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from itertools import combinations
from math import isfinite
from pathlib import Path

import numpy as np

from pyscf import gto
from pyscf.scf import chkfile

from .checkpoint_fingerprint import (
    StateFingerprint,
    fingerprint_from_checkpoint,
)
from .root_record import (
    SCFRootRecord,
    SCFRunStatus,
)
from .state_identity import (
    IdentityThresholds,
    compare_states,
)


HARTREE_TO_MEV = (
    27.211386245988
    * 1000.0
)


class ManifoldContinuityRelation(
    str,
    Enum,
):
    CONTINUOUS = "CONTINUOUS"
    GAUGE_UNRESOLVED = (
        "GAUGE_UNRESOLVED"
    )
    DISCONTINUOUS = "DISCONTINUOUS"


@dataclass(frozen=True)
class ManifoldThresholds:
    """
    Explicit thresholds for grouping and tracking electronic manifolds.

    No production defaults are provided deliberately.
    """

    max_energy_mev: float
    max_delta_s2: float
    max_total_spectrum: float
    max_spin_spectrum: float

    fractional_occupation_eps: float

    active_overlap_min: float

    bridge_max_span_angstrom: float

    def __post_init__(
        self,
    ) -> None:
        positive = (
            "max_energy_mev",
            "max_delta_s2",
            "max_total_spectrum",
            "max_spin_spectrum",
            "fractional_occupation_eps",
            "bridge_max_span_angstrom",
        )

        for name in positive:
            value = float(
                getattr(
                    self,
                    name,
                )
            )

            if (
                not isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and > 0"
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        overlap = float(
            self.active_overlap_min
        )

        if (
            not isfinite(overlap)
            or overlap <= 0.0
            or overlap > 1.0
        ):
            raise ValueError(
                "active_overlap_min must be "
                "finite and in (0, 1]"
            )

        object.__setattr__(
            self,
            "active_overlap_min",
            overlap,
        )

        if (
            self.fractional_occupation_eps
            >= 0.5
        ):
            raise ValueError(
                "fractional_occupation_eps "
                "must be < 0.5"
            )


@dataclass(frozen=True)
class ElectronicManifoldPoint:
    manifold_id: str

    molecule: str
    atom_a: str
    atom_b: str

    charge: int
    spin_2s: int

    r_angstrom: float

    functional: str
    basis: str
    reference: str
    ecp_assignments: tuple[
        tuple[str, str],
        ...,
    ]

    member_root_ids: tuple[
        str,
        ...,
    ]

    unique_state_root_ids: tuple[
        str,
        ...,
    ]

    energy_center_hartree: float
    energy_spread_mev: float

    alpha_active_rank: int
    beta_active_rank: int

    alpha_fractional_occupations: tuple[
        float,
        ...,
    ]

    beta_fractional_occupations: tuple[
        float,
        ...,
    ]

    #
    # AO coefficient matrices spanning the active
    # ensemble-density eigenspaces.
    #
    alpha_active_ao: np.ndarray
    beta_active_ao: np.ndarray

    #
    # PySCF molecule object is required only for cross-geometry
    # AO overlap evaluation.
    #
    mol: object


@dataclass(frozen=True)
class RejectedManifoldComponent:
    member_root_ids: tuple[
        str,
        ...,
    ]

    reason: str


@dataclass(frozen=True)
class ManifoldConstructionResult:
    manifolds: tuple[
        ElectronicManifoldPoint,
        ...,
    ]

    rejected_components: tuple[
        RejectedManifoldComponent,
        ...,
    ]


@dataclass(frozen=True)
class ManifoldContinuity:
    left_manifold_id: str
    right_manifold_id: str

    left_r_angstrom: float
    right_r_angstrom: float

    relation: (
        ManifoldContinuityRelation
    )

    alpha_rank_left: int
    alpha_rank_right: int

    beta_rank_left: int
    beta_rank_right: int

    alpha_min_overlap: (
        float | None
    )

    beta_min_overlap: (
        float | None
    )


def _find_fingerprint(
    value,
) -> StateFingerprint | None:
    if isinstance(
        value,
        StateFingerprint,
    ):
        return value

    if is_dataclass(
        value
    ):
        for field in fields(
            value
        ):
            found = (
                _find_fingerprint(
                    getattr(
                        value,
                        field.name,
                    )
                )
            )

            if found is not None:
                return found

    if isinstance(
        value,
        dict,
    ):
        for item in value.values():
            found = (
                _find_fingerprint(
                    item
                )
            )

            if found is not None:
                return found

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        for item in value:
            found = (
                _find_fingerprint(
                    item
                )
            )

            if found is not None:
                return found

    return None


def _fingerprint(
    root: SCFRootRecord,
) -> StateFingerprint:
    result = (
        fingerprint_from_checkpoint(
            root
        )
    )

    found = _find_fingerprint(
        result
    )

    if found is None:
        raise ValueError(
            "checkpoint did not yield "
            f"a StateFingerprint: {root.root_id}"
        )

    return found


def _same_context(
    roots: tuple[
        SCFRootRecord,
        ...,
    ],
) -> None:
    if not roots:
        raise ValueError(
            "at least one root is required"
        )

    first = roots[0]

    expected = (
        first.molecule,
        first.atom_a,
        first.atom_b,
        first.charge,
        first.spin_2s,
        first.r_angstrom,
        first.functional,
        first.basis,
        first.reference,
        first.ecp_assignments,
    )

    for root in roots[1:]:
        actual = (
            root.molecule,
            root.atom_a,
            root.atom_b,
            root.charge,
            root.spin_2s,
            root.r_angstrom,
            root.functional,
            root.basis,
            root.reference,
            root.ecp_assignments,
        )

        if actual != expected:
            raise ValueError(
                "all roots must belong to "
                "the same geometry and "
                "scientific calculation context"
            )


def _manifold_compatible(
    comparison,
    thresholds: ManifoldThresholds,
) -> bool:
    return (
        abs(
            comparison.delta_energy_mev
        )
        <= thresholds.max_energy_mev

        and abs(
            comparison.delta_s2
        )
        <= thresholds.max_delta_s2

        and (
            comparison.total_spectrum_max
            <= thresholds.max_total_spectrum
        )

        and (
            comparison.spin_spectrum_max
            <= thresholds.max_spin_spectrum
        )
    )


def _components(
    root_ids: tuple[
        str,
        ...,
    ],
    edges: set[
        tuple[str, str]
    ],
) -> tuple[
    tuple[str, ...],
    ...,
]:
    neighbors = {
        root_id: set()
        for root_id in root_ids
    }

    for left, right in edges:
        neighbors[left].add(
            right
        )
        neighbors[right].add(
            left
        )

    seen = set()
    result = []

    for start in root_ids:
        if start in seen:
            continue

        stack = [
            start
        ]

        component = []

        while stack:
            current = (
                stack.pop()
            )

            if current in seen:
                continue

            seen.add(
                current
            )

            component.append(
                current
            )

            stack.extend(
                sorted(
                    neighbors[
                        current
                    ]
                    - seen
                )
            )

        result.append(
            tuple(
                sorted(
                    component
                )
            )
        )

    return tuple(
        result
    )


def _is_complete_component(
    component: tuple[
        str,
        ...,
    ],
    edges: set[
        tuple[str, str]
    ],
) -> bool:
    for left, right in combinations(
        component,
        2,
    ):
        edge = tuple(
            sorted(
                (
                    left,
                    right,
                )
            )
        )

        if edge not in edges:
            return False

    return True


def _same_state_representatives(
    roots: tuple[
        SCFRootRecord,
        ...,
    ],
    fingerprints: dict[
        str,
        StateFingerprint,
    ],
    identity_thresholds: (
        IdentityThresholds
    ),
) -> tuple[
    SCFRootRecord,
    ...,
]:
    parent = {
        root.root_id:
            root.root_id
        for root in roots
    }

    def find(
        value,
    ):
        while (
            parent[value]
            != value
        ):
            parent[value] = (
                parent[
                    parent[value]
                ]
            )

            value = parent[
                value
            ]

        return value

    def union(
        left,
        right,
    ):
        a = find(
            left
        )

        b = find(
            right
        )

        if a == b:
            return

        if a < b:
            parent[b] = a
        else:
            parent[a] = b

    for left, right in combinations(
        roots,
        2,
    ):
        comparison = (
            compare_states(
                left,
                fingerprints[
                    left.root_id
                ],
                right,
                fingerprints[
                    right.root_id
                ],
                thresholds=(
                    identity_thresholds
                ),
            )
        )

        if (
            comparison.relation.value
            == "SAME_STATE"
        ):
            union(
                left.root_id,
                right.root_id,
            )

    groups = {}

    for root in roots:
        key = find(
            root.root_id
        )

        groups.setdefault(
            key,
            [],
        ).append(
            root
        )

    representatives = []

    for group in groups.values():
        representatives.append(
            sorted(
                group,
                key=lambda root:
                    root.root_id,
            )[0]
        )

    return tuple(
        sorted(
            representatives,
            key=lambda root:
                root.root_id,
        )
    )


def _sqrt_and_invsqrt(
    matrix: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    values, vectors = (
        np.linalg.eigh(
            matrix
        )
    )

    if np.min(
        values
    ) <= 0.0:
        raise ValueError(
            "AO overlap matrix is not "
            "positive definite"
        )

    sqrt = (
        vectors
        @ np.diag(
            np.sqrt(
                values
            )
        )
        @ vectors.T
    )

    invsqrt = (
        vectors
        @ np.diag(
            1.0
            / np.sqrt(
                values
            )
        )
        @ vectors.T
    )

    return (
        sqrt,
        invsqrt,
    )


def _occupied_coefficients(
    root: SCFRootRecord,
):
    if (
        root.checkpoint_path
        is None
    ):
        raise ValueError(
            "canonical root has no "
            "checkpoint path"
        )

    mol, data = (
        chkfile.load_scf(
            str(
                root.checkpoint_path
            )
        )
    )

    coeff = data[
        "mo_coeff"
    ]

    occ = data[
        "mo_occ"
    ]

    if (
        isinstance(
            coeff,
            np.ndarray,
        )
        and coeff.ndim == 3
    ):
        ca = np.asarray(
            coeff[0]
        )

        cb = np.asarray(
            coeff[1]
        )

        oa = np.asarray(
            occ[0]
        )

        ob = np.asarray(
            occ[1]
        )

        return (
            mol,
            ca[
                :,
                oa > 0.5,
            ],
            cb[
                :,
                ob > 0.5,
            ],
        )

    if isinstance(
        coeff,
        (
            tuple,
            list,
        ),
    ):
        ca = np.asarray(
            coeff[0]
        )

        cb = np.asarray(
            coeff[1]
        )

        oa = np.asarray(
            occ[0]
        )

        ob = np.asarray(
            occ[1]
        )

        return (
            mol,
            ca[
                :,
                oa > 0.5,
            ],
            cb[
                :,
                ob > 0.5,
            ],
        )

    c = np.asarray(
        coeff
    )

    o = np.asarray(
        occ
    )

    return (
        mol,
        c[
            :,
            o > 0.0,
        ],
        c[
            :,
            o > 1.0,
        ],
    )


def _orth_projector(
    mol,
    coeff,
) -> np.ndarray:
    overlap = mol.intor(
        "int1e_ovlp"
    )

    sqrt_s, _ = (
        _sqrt_and_invsqrt(
            overlap
        )
    )

    q = (
        sqrt_s
        @ coeff
    )

    q, _ = np.linalg.qr(
        q
    )

    return (
        q @ q.T
    )


def _ensemble_active_space(
    roots: tuple[
        SCFRootRecord,
        ...,
    ],
    *,
    fractional_eps: float,
):
    alpha_projectors = []
    beta_projectors = []

    mol = None

    for root in roots:
        (
            mol_here,
            alpha_occ,
            beta_occ,
        ) = (
            _occupied_coefficients(
                root
            )
        )

        if mol is None:
            mol = mol_here

        alpha_projectors.append(
            _orth_projector(
                mol_here,
                alpha_occ,
            )
        )

        beta_projectors.append(
            _orth_projector(
                mol_here,
                beta_occ,
            )
        )

    overlap = mol.intor(
        "int1e_ovlp"
    )

    _, invsqrt_s = (
        _sqrt_and_invsqrt(
            overlap
        )
    )

    def analyze(
        projectors,
    ):
        average = np.mean(
            projectors,
            axis=0,
        )

        values, vectors = (
            np.linalg.eigh(
                average
            )
        )

        order = np.argsort(
            values
        )[::-1]

        values = values[
            order
        ]

        vectors = vectors[
            :,
            order
        ]

        mask = (
            (
                values
                > fractional_eps
            )
            & (
                values
                < (
                    1.0
                    - fractional_eps
                )
            )
        )

        active_values = (
            values[
                mask
            ]
        )

        active_vectors = (
            vectors[
                :,
                mask
            ]
        )

        active_ao = (
            invsqrt_s
            @ active_vectors
        )

        return (
            active_values,
            active_ao,
        )

    (
        alpha_values,
        alpha_ao,
    ) = analyze(
        alpha_projectors
    )

    (
        beta_values,
        beta_ao,
    ) = analyze(
        beta_projectors
    )

    return (
        mol,
        alpha_values,
        alpha_ao,
        beta_values,
        beta_ao,
    )


def _manifold_id(
    root_ids: tuple[
        str,
        ...,
    ],
) -> str:
    raw = "\n".join(
        sorted(
            root_ids
        )
    )

    digest = sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[
        :16
    ]

    return (
        f"manifold_{digest}"
    )


def construct_electronic_manifolds(
    roots,
    *,
    identity_thresholds: (
        IdentityThresholds
    ),
    manifold_thresholds: (
        ManifoldThresholds
    ),
) -> ManifoldConstructionResult:
    eligible = tuple(
        sorted(
            (
                root
                for root in roots
                if (
                    root.status
                    == SCFRunStatus
                    .CANONICALIZED
                    and (
                        root.checkpoint_path
                        is not None
                    )
                )
            ),
            key=lambda root:
                root.root_id,
        )
    )

    if not eligible:
        return (
            ManifoldConstructionResult(
                manifolds=(),
                rejected_components=(),
            )
        )

    _same_context(
        eligible
    )

    fingerprints = {
        root.root_id:
            _fingerprint(
                root
            )
        for root in eligible
    }

    compatible_edges = set()

    for left, right in combinations(
        eligible,
        2,
    ):
        comparison = (
            compare_states(
                left,
                fingerprints[
                    left.root_id
                ],
                right,
                fingerprints[
                    right.root_id
                ],
                thresholds=(
                    identity_thresholds
                ),
            )
        )

        if _manifold_compatible(
            comparison,
            manifold_thresholds,
        ):
            compatible_edges.add(
                tuple(
                    sorted(
                        (
                            left.root_id,
                            right.root_id,
                        )
                    )
                )
            )

    root_ids = tuple(
        root.root_id
        for root in eligible
    )

    components = _components(
        root_ids,
        compatible_edges,
    )

    root_by_id = {
        root.root_id: root
        for root in eligible
    }

    manifolds = []
    rejected = []

    for component in components:
        if not _is_complete_component(
            component,
            compatible_edges,
        ):
            rejected.append(
                RejectedManifoldComponent(
                    member_root_ids=(
                        component
                    ),
                    reason=(
                        "MANIFOLD_COMPONENT_NOT_"
                        "PAIRWISE_COMPATIBLE"
                    ),
                )
            )

            continue

        component_roots = tuple(
            root_by_id[
                root_id
            ]
            for root_id
            in component
        )

        unique_roots = (
            _same_state_representatives(
                component_roots,
                fingerprints,
                identity_thresholds,
            )
        )

        (
            mol,
            alpha_values,
            alpha_ao,
            beta_values,
            beta_ao,
        ) = (
            _ensemble_active_space(
                unique_roots,
                fractional_eps=(
                    manifold_thresholds
                    .fractional_occupation_eps
                ),
            )
        )

        energies = np.asarray(
            [
                root.energy_hartree
                for root
                in unique_roots
            ],
            dtype=float,
        )

        #
        # Exact members of a degenerate electronic manifold should
        # have the same energy.  Small orientation-dependent SCF
        # splittings are therefore treated as a numerical/manifold
        # energy envelope rather than as states to be energetically
        # ranked.
        #
        # The midpoint is invariant to duplicate sampling of one
        # particular gauge representative.
        #
        energy_min = float(
            np.min(
                energies
            )
        )

        energy_max = float(
            np.max(
                energies
            )
        )

        center = (
            0.5
            * (
                energy_min
                + energy_max
            )
        )

        spread_mev = (
            (
                energy_max
                - energy_min
            )
            * HARTREE_TO_MEV
        )

        first = component_roots[
            0
        ]

        manifolds.append(
            ElectronicManifoldPoint(
                manifold_id=(
                    _manifold_id(
                        component
                    )
                ),

                molecule=(
                    first.molecule
                ),
                atom_a=(
                    first.atom_a
                ),
                atom_b=(
                    first.atom_b
                ),

                charge=first.charge,
                spin_2s=first.spin_2s,

                r_angstrom=(
                    first.r_angstrom
                ),

                functional=(
                    first.functional
                ),
                basis=first.basis,
                reference=(
                    first.reference
                ),
                ecp_assignments=(
                    first.ecp_assignments
                ),

                member_root_ids=(
                    component
                ),

                unique_state_root_ids=tuple(
                    root.root_id
                    for root
                    in unique_roots
                ),

                energy_center_hartree=(
                    center
                ),
                energy_spread_mev=(
                    spread_mev
                ),

                alpha_active_rank=int(
                    alpha_ao.shape[
                        1
                    ]
                ),
                beta_active_rank=int(
                    beta_ao.shape[
                        1
                    ]
                ),

                alpha_fractional_occupations=tuple(
                    float(value)
                    for value
                    in alpha_values
                ),

                beta_fractional_occupations=tuple(
                    float(value)
                    for value
                    in beta_values
                ),

                alpha_active_ao=(
                    alpha_ao
                ),
                beta_active_ao=(
                    beta_ao
                ),

                mol=mol,
            )
        )

    return (
        ManifoldConstructionResult(
            manifolds=tuple(
                sorted(
                    manifolds,
                    key=lambda item:
                        item.manifold_id,
                )
            ),
            rejected_components=tuple(
                rejected
            ),
        )
    )


def _active_overlap(
    left_mol,
    left_ao,
    right_mol,
    right_ao,
):
    """
    Compare one spin-channel active manifold across two geometries.

    INACTIVE
        Both geometries have rank zero in this spin channel. There is no
        active subspace to compare, but this is not evidence against
        continuity in another resolved spin channel.

    UNRESOLVED
        The active subspace exists at only one of the two geometries.
        Continuity cannot be decided from this channel alone.

    RANK_MISMATCH
        Both geometries have non-zero active spaces, but their dimensions
        differ.

    RESOLVED
        Equal non-zero active ranks; continuity is quantified through the
        minimum singular value of the cross-geometry active-space overlap.
    """
    left_rank = int(
        left_ao.shape[1]
    )

    right_rank = int(
        right_ao.shape[1]
    )

    if (
        left_rank == 0
        and right_rank == 0
    ):
        return (
            "INACTIVE",
            None,
        )

    if (
        left_rank == 0
        or right_rank == 0
    ):
        return (
            "UNRESOLVED",
            None,
        )

    if (
        left_rank
        != right_rank
    ):
        return (
            "RANK_MISMATCH",
            None,
        )

    cross = gto.intor_cross(
        "int1e_ovlp",
        left_mol,
        right_mol,
    )

    matrix = (
        left_ao.T
        @ cross
        @ right_ao
    )

    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
    )

    minimum = float(
        np.min(
            singular_values
        )
    )

    return (
        "RESOLVED",
        minimum,
    )


def compare_electronic_manifolds(
    left: ElectronicManifoldPoint,
    right: ElectronicManifoldPoint,
    *,
    thresholds: ManifoldThresholds,
) -> ManifoldContinuity:
    left_context = (
        left.molecule,
        left.atom_a,
        left.atom_b,
        left.charge,
        left.spin_2s,
        left.functional,
        left.basis,
        left.reference,
        left.ecp_assignments,
    )

    right_context = (
        right.molecule,
        right.atom_a,
        right.atom_b,
        right.charge,
        right.spin_2s,
        right.functional,
        right.basis,
        right.reference,
        right.ecp_assignments,
    )

    if (
        left_context
        != right_context
    ):
        raise ValueError(
            "manifold comparison requires "
            "the same scientific context"
        )

    (
        alpha_status,
        alpha_overlap,
    ) = _active_overlap(
        left.mol,
        left.alpha_active_ao,
        right.mol,
        right.alpha_active_ao,
    )

    (
        beta_status,
        beta_overlap,
    ) = _active_overlap(
        left.mol,
        left.beta_active_ao,
        right.mol,
        right.beta_active_ao,
    )

    statuses = (
        alpha_status,
        beta_status,
    )

    resolved_overlaps = [
        overlap
        for status, overlap
        in (
            (
                alpha_status,
                alpha_overlap,
            ),
            (
                beta_status,
                beta_overlap,
            ),
        )
        if (
            status
            == "RESOLVED"
            and overlap
            is not None
        )
    ]

    if (
        "RANK_MISMATCH"
        in statuses
    ):
        relation = (
            ManifoldContinuityRelation
            .DISCONTINUOUS
        )

    elif any(
        overlap
        < thresholds
        .active_overlap_min
        for overlap
        in resolved_overlaps
    ):
        relation = (
            ManifoldContinuityRelation
            .DISCONTINUOUS
        )

    elif (
        "UNRESOLVED"
        in statuses
    ):
        relation = (
            ManifoldContinuityRelation
            .GAUGE_UNRESOLVED
        )

    elif (
        not resolved_overlaps
    ):
        #
        # No active manifold was observed at either geometry.
        # This is not evidence of a discontinuity.
        #
        relation = (
            ManifoldContinuityRelation
            .GAUGE_UNRESOLVED
        )

    else:
        relation = (
            ManifoldContinuityRelation
            .CONTINUOUS
        )

    return ManifoldContinuity(
        left_manifold_id=(
            left.manifold_id
        ),
        right_manifold_id=(
            right.manifold_id
        ),

        left_r_angstrom=(
            left.r_angstrom
        ),
        right_r_angstrom=(
            right.r_angstrom
        ),

        relation=relation,

        alpha_rank_left=(
            left.alpha_active_rank
        ),
        alpha_rank_right=(
            right.alpha_active_rank
        ),

        beta_rank_left=(
            left.beta_active_rank
        ),
        beta_rank_right=(
            right.beta_active_rank
        ),

        alpha_min_overlap=(
            alpha_overlap
        ),

        beta_min_overlap=(
            beta_overlap
        ),
    )
