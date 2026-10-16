from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

from .root_record import (
    SCFRootRecord,
    SCFRunStatus,
)


HARTREE_TO_EV = 27.211386245988


class StateRelation(str, Enum):
    """
    Conservative pairwise electronic-state relation.

    SAME_STATE:
        All configured same-state criteria agree.

    DISTINCT_STATE:
        The roots are clearly distinct according to the configured
        scientific evidence.

    AMBIGUOUS:
        Available evidence is insufficient for either conclusion.

    AMBIGUOUS is an intentional scientific result and must not be silently
    converted into SAME_STATE or DISTINCT_STATE.
    """

    SAME_STATE = "SAME_STATE"
    DISTINCT_STATE = "DISTINCT_STATE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class IdentityThresholds:
    """
    Explicit state-identity decision thresholds.

    OpenEA deliberately provides no production defaults here.

    Threshold values must be supplied by the workflow and validated before
    prediction calculations.  This prevents preliminary development values
    from silently becoming frozen scientific policy.
    """

    same_energy_mev: float
    same_delta_s2: float
    same_total_spectrum_max: float
    same_spin_spectrum_max: float
    same_total_density_rel_fro: float
    same_spin_density_rel_fro: float

    distinct_energy_mev: float
    distinct_delta_s2: float
    distinct_total_spectrum_max: float
    distinct_spin_spectrum_max: float
    distinct_total_density_rel_fro: float
    distinct_spin_density_rel_fro: float

    def __post_init__(self) -> None:
        values = (
            self.same_energy_mev,
            self.same_delta_s2,
            self.same_total_spectrum_max,
            self.same_spin_spectrum_max,
            self.same_total_density_rel_fro,
            self.same_spin_density_rel_fro,
            self.distinct_energy_mev,
            self.distinct_delta_s2,
            self.distinct_total_spectrum_max,
            self.distinct_spin_spectrum_max,
            self.distinct_total_density_rel_fro,
            self.distinct_spin_density_rel_fro,
        )

        if any(
            not isfinite(float(x))
            or float(x) < 0.0
            for x in values
        ):
            raise ValueError(
                "identity thresholds must be finite and >= 0"
            )

        pairs = (
            (
                "energy",
                self.same_energy_mev,
                self.distinct_energy_mev,
            ),
            (
                "delta_s2",
                self.same_delta_s2,
                self.distinct_delta_s2,
            ),
            (
                "total_spectrum",
                self.same_total_spectrum_max,
                self.distinct_total_spectrum_max,
            ),
            (
                "spin_spectrum",
                self.same_spin_spectrum_max,
                self.distinct_spin_spectrum_max,
            ),
            (
                "total_density_rel_fro",
                self.same_total_density_rel_fro,
                self.distinct_total_density_rel_fro,
            ),
            (
                "spin_density_rel_fro",
                self.same_spin_density_rel_fro,
                self.distinct_spin_density_rel_fro,
            ),
        )

        for name, same, distinct in pairs:
            if same >= distinct:
                raise ValueError(
                    f"{name}: same-state threshold must be "
                    "strictly below distinct-state threshold"
                )


@dataclass(frozen=True)
class StateFingerprint:
    """
    Rotation-insensitive density evidence for one SCF root.

    Spectra are obtained from the total and spin density matrices in an
    orthonormal AO representation.

    Separate alpha/beta occupation eigenvalue spectra are intentionally not
    used because for a single determinant they largely reproduce the imposed
    occupations rather than useful state identity.

    This fingerprint is designed for roots calculated in the same molecular
    geometry and AO basis.  Branch continuity between different R values or
    different basis sets is a separate workflow problem.
    """

    total_spectrum: tuple[float, ...]
    spin_spectrum: tuple[float, ...]

    total_density_orth: tuple[
        tuple[float, ...],
        ...,
    ]
    spin_density_orth: tuple[
        tuple[float, ...],
        ...,
    ]

    total_trace: float
    spin_trace: float

    def __post_init__(self) -> None:
        if not self.total_spectrum:
            raise ValueError(
                "total_spectrum must be non-empty"
            )

        if (
            len(self.total_spectrum)
            != len(self.spin_spectrum)
        ):
            raise ValueError(
                "total and spin spectra must have equal length"
            )

        n = len(
            self.total_spectrum
        )

        if len(
            self.total_density_orth
        ) != n:
            raise ValueError(
                "total density dimension must match spectrum"
            )

        if len(
            self.spin_density_orth
        ) != n:
            raise ValueError(
                "spin density dimension must match spectrum"
            )

        if any(
            len(row) != n
            for row in self.total_density_orth
        ):
            raise ValueError(
                "total_density_orth must be square"
            )

        if any(
            len(row) != n
            for row in self.spin_density_orth
        ):
            raise ValueError(
                "spin_density_orth must be square"
            )

        numbers = (
            *self.total_spectrum,
            *self.spin_spectrum,
            *(
                value
                for row in self.total_density_orth
                for value in row
            ),
            *(
                value
                for row in self.spin_density_orth
                for value in row
            ),
            self.total_trace,
            self.spin_trace,
        )

        if any(
            not isfinite(float(x))
            for x in numbers
        ):
            raise ValueError(
                "fingerprint values must be finite"
            )


@dataclass(frozen=True)
class StateComparison:
    root_a: str
    root_b: str

    delta_energy_mev: float
    delta_s2: float

    total_spectrum_max: float
    total_spectrum_l2: float

    spin_spectrum_max: float
    spin_spectrum_l2: float

    total_density_rel_fro: float
    spin_density_rel_fro: float

    relation: StateRelation


def fingerprint_from_orthonormal_density(
    dm_orth,
) -> StateFingerprint:
    """
    Build rotation-insensitive total/spin density spectra.

    dm_orth must be spin resolved with shape

        (2, n, n)

    in an orthonormal AO representation.

    The function symmetrizes numerical noise before diagonalization.
    """
    dm = np.asarray(
        dm_orth,
        dtype=float,
    )

    if (
        dm.ndim != 3
        or dm.shape[0] != 2
        or dm.shape[1] != dm.shape[2]
    ):
        raise ValueError(
            "expected spin-resolved density with shape (2, n, n)"
        )

    if not np.all(
        np.isfinite(dm)
    ):
        raise ValueError(
            "density contains non-finite values"
        )

    da = 0.5 * (
        dm[0] + dm[0].T
    )

    db = 0.5 * (
        dm[1] + dm[1].T
    )

    total = da + db
    spin = da - db

    total_spectrum = np.sort(
        np.linalg.eigvalsh(total)
    )[::-1]

    spin_spectrum = np.sort(
        np.linalg.eigvalsh(spin)
    )[::-1]

    return StateFingerprint(
        total_spectrum=tuple(
            float(x)
            for x in total_spectrum
        ),
        spin_spectrum=tuple(
            float(x)
            for x in spin_spectrum
        ),
        total_density_orth=tuple(
            tuple(
                float(x)
                for x in row
            )
            for row in total
        ),
        spin_density_orth=tuple(
            tuple(
                float(x)
                for x in row
            )
            for row in spin
        ),
        total_trace=float(
            np.trace(total)
        ),
        spin_trace=float(
            np.trace(spin)
        ),
    )


def _spectrum_distances(
    a: tuple[float, ...],
    b: tuple[float, ...],
) -> tuple[float, float]:
    if len(a) != len(b):
        raise ValueError(
            "fingerprint spectrum dimensions differ"
        )

    delta = (
        np.asarray(a, dtype=float)
        - np.asarray(b, dtype=float)
    )

    return (
        float(
            np.max(
                np.abs(delta)
            )
        ),
        float(
            np.linalg.norm(delta)
        ),
    )


def _relative_frobenius_distance(
    a,
    b,
) -> float:
    """
    Symmetric relative Frobenius distance.

    For matrices A and B,

        d = 2 ||A-B||_F / (||A||_F + ||B||_F).

    If both matrices have zero Frobenius norm, d is defined as zero.

    The metric is invariant under a common orthogonal transformation of the
    orthonormal AO representation.  Unlike an eigenvalue spectrum alone, it
    remains sensitive to differently oriented occupied/density subspaces.
    """
    aa = np.asarray(
        a,
        dtype=float,
    )

    bb = np.asarray(
        b,
        dtype=float,
    )

    if aa.shape != bb.shape:
        raise ValueError(
            "density matrix dimensions differ"
        )

    if aa.ndim != 2:
        raise ValueError(
            "density matrices must be two-dimensional"
        )

    numerator = 2.0 * float(
        np.linalg.norm(
            aa - bb,
            ord="fro",
        )
    )

    denominator = float(
        np.linalg.norm(
            aa,
            ord="fro",
        )
        + np.linalg.norm(
            bb,
            ord="fro",
        )
    )

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def classify_metrics(
    *,
    same_spin: bool,
    delta_energy_mev: float,
    delta_s2: float,
    total_spectrum_max: float,
    spin_spectrum_max: float,
    total_density_rel_fro: float,
    spin_density_rel_fro: float,
    thresholds: IdentityThresholds,
) -> StateRelation:
    """
    Conservative three-way state-equivalence decision.

    Different requested spin sectors are always DISTINCT_STATE.

    SAME_STATE requires every configured same-state descriptor to agree.

    DISTINCT_STATE requires both:

    - a meaningful energy separation; and
    - at least one clearly different spin/density descriptor.

    Energy separation alone is therefore not enough to declare two numerical
    solutions different electronic states.

    Everything between these limits remains AMBIGUOUS.
    """
    if not same_spin:
        return StateRelation.DISTINCT_STATE

    values = (
        delta_energy_mev,
        delta_s2,
        total_spectrum_max,
        spin_spectrum_max,
        total_density_rel_fro,
        spin_density_rel_fro,
    )

    if any(
        not isfinite(float(x))
        for x in values
    ):
        raise ValueError(
            "comparison metrics must be finite"
        )

    de = abs(
        float(delta_energy_mev)
    )

    ds2 = abs(
        float(delta_s2)
    )

    dtotal = abs(
        float(total_spectrum_max)
    )

    dspin = abs(
        float(spin_spectrum_max)
    )

    dtotal_density = abs(
        float(total_density_rel_fro)
    )

    dspin_density = abs(
        float(spin_density_rel_fro)
    )

    if (
        de <= thresholds.same_energy_mev
        and ds2 <= thresholds.same_delta_s2
        and dtotal
        <= thresholds.same_total_spectrum_max
        and dspin
        <= thresholds.same_spin_spectrum_max
        and dtotal_density
        <= thresholds.same_total_density_rel_fro
        and dspin_density
        <= thresholds.same_spin_density_rel_fro
    ):
        return StateRelation.SAME_STATE

    structurally_distinct = (
        ds2
        >= thresholds.distinct_delta_s2
        or dtotal
        >= thresholds.distinct_total_spectrum_max
        or dspin
        >= thresholds.distinct_spin_spectrum_max
        or dtotal_density
        >= thresholds.distinct_total_density_rel_fro
        or dspin_density
        >= thresholds.distinct_spin_density_rel_fro
    )

    if (
        de >= thresholds.distinct_energy_mev
        and structurally_distinct
    ):
        return StateRelation.DISTINCT_STATE

    return StateRelation.AMBIGUOUS


def _assert_same_comparison_context(
    a: SCFRootRecord,
    b: SCFRootRecord,
) -> None:
    """
    Same-state deduplication is only defined at identical geometry/method.

    Different R values are handled later by branch-following logic.
    Different basis sets are handled by explicit projection/continuity logic.
    """
    fields = (
        ("molecule", a.molecule, b.molecule),
        ("atom_a", a.atom_a, b.atom_a),
        ("atom_b", a.atom_b, b.atom_b),
        ("charge", a.charge, b.charge),
        ("functional", a.functional, b.functional),
        ("basis", a.basis, b.basis),
        ("reference", a.reference, b.reference),
        (
            "ecp_assignments",
            a.ecp_assignments,
            b.ecp_assignments,
        ),
    )

    for name, x, y in fields:
        if x != y:
            raise ValueError(
                f"cannot compare state identity across different "
                f"{name}: {x!r} != {y!r}"
            )

    if abs(
        a.r_angstrom - b.r_angstrom
    ) > 1.0e-10:
        raise ValueError(
            "cannot deduplicate roots from different geometries"
        )


def compare_states(
    a: SCFRootRecord,
    fingerprint_a: StateFingerprint,
    b: SCFRootRecord,
    fingerprint_b: StateFingerprint,
    *,
    thresholds: IdentityThresholds,
) -> StateComparison:
    """
    Compare two canonicalized roots at one geometry and one method.
    """
    _assert_same_comparison_context(
        a,
        b,
    )

    for root in (
        a,
        b,
    ):
        # A converged root whose internal instability was *tested and recorded*
        # (seed-continuous reference branch) is admitted as well: the comparison
        # uses energy, <S^2> and the density fingerprint, not stability.
        if not (
            root.status == SCFRunStatus.CANONICALIZED
            or (root.status == SCFRunStatus.CONVERGED and root.internal_stable is False)
        ):
            raise ValueError(
                "state identity requires canonicalized roots"
            )

        if root.energy_hartree is None:
            raise ValueError(
                "state identity requires root energies"
            )

        if root.s2 is None:
            raise ValueError(
                "state identity requires <S^2>"
            )

    total_max, total_l2 = (
        _spectrum_distances(
            fingerprint_a.total_spectrum,
            fingerprint_b.total_spectrum,
        )
    )

    spin_max, spin_l2 = (
        _spectrum_distances(
            fingerprint_a.spin_spectrum,
            fingerprint_b.spin_spectrum,
        )
    )

    total_density_rel_fro = (
        _relative_frobenius_distance(
            fingerprint_a.total_density_orth,
            fingerprint_b.total_density_orth,
        )
    )

    spin_density_rel_fro = (
        _relative_frobenius_distance(
            fingerprint_a.spin_density_orth,
            fingerprint_b.spin_density_orth,
        )
    )

    de_mev = abs(
        float(
            b.energy_hartree
            - a.energy_hartree
        )
    ) * HARTREE_TO_EV * 1000.0

    ds2 = abs(
        float(
            b.s2 - a.s2
        )
    )

    relation = classify_metrics(
        same_spin=(
            a.spin_2s
            == b.spin_2s
        ),
        delta_energy_mev=de_mev,
        delta_s2=ds2,
        total_spectrum_max=total_max,
        spin_spectrum_max=spin_max,
        total_density_rel_fro=(
            total_density_rel_fro
        ),
        spin_density_rel_fro=(
            spin_density_rel_fro
        ),
        thresholds=thresholds,
    )

    return StateComparison(
        root_a=a.root_id,
        root_b=b.root_id,
        delta_energy_mev=de_mev,
        delta_s2=ds2,
        total_spectrum_max=total_max,
        total_spectrum_l2=total_l2,
        spin_spectrum_max=spin_max,
        spin_spectrum_l2=spin_l2,
        total_density_rel_fro=(
            total_density_rel_fro
        ),
        spin_density_rel_fro=(
            spin_density_rel_fro
        ),
        relation=relation,
    )


