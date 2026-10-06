"""Component-resolved CBS extrapolation for OpenEA validation evidence.

This module performs no electronic-structure calculation.  It consumes
completed QZ/5Z component evidence and extrapolates SCF, CCSD correlation,
and perturbative triples separately.

Primary AV{Q,5}Z exponents follow component-specific literature values:
    SCF  8.7042
    CCSD 3.2711
    (T)  3.6018

Sensitivity models deliberately use distinct, literature-motivated
alternatives rather than silently averaging extrapolations:
    SCF  10.3626
    CCSD 3.0
    (T)  3.3723

A combined-correlation inverse-cubic extrapolation is also retained as an
independent sanity check.

No result from this layer is a production adiabatic electron affinity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping

HARTREE_TO_EV = 27.211386245988

PRIMARY_EXPONENTS = {
    "scf": 8.7042,
    "ccsd_correlation": 3.2711,
    "triples": 3.6018,
}

SENSITIVITY_EXPONENTS = {
    "scf": 10.3626,
    "ccsd_correlation": 3.0,
    "triples": 3.3723,
}


class CBSStatus(str, Enum):
    CLEARED = "CLEARED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ComponentPair:
    role: str
    basis_qz: str
    basis_5z: str
    scf_qz: float
    scf_5z: float
    ccsd_correlation_qz: float
    ccsd_correlation_5z: float
    triples_qz: float
    triples_5z: float


@dataclass(frozen=True)
class SpeciesCBSComponents:
    role: str
    model: str
    scf_cbs_hartree: float
    ccsd_correlation_cbs_hartree: float
    triples_cbs_hartree: float
    total_cbs_hartree: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EACBSModel:
    model: str
    scf_contribution_ev: float
    ccsd_correlation_contribution_ev: float
    triples_contribution_ev: float
    ea_cbs_aug_ev: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CBSResolvedResult:
    status: CBSStatus
    primary_model: EACBSModel
    sensitivity_model: EACBSModel
    combined_correlation_sanity_ev: float
    cbs_model_sensitivity_bound_ev: float
    diffuse_correction_ev: float
    diffuse_residual_estimate_ev: float | None
    fixed_geometry_transfer_check_ev: float | None
    ea_cbs_plus_diffuse_primary_ev: float
    conservative_intermediate_half_width_ev: float
    lower_intermediate_ev: float
    upper_intermediate_ev: float
    evidence: tuple[str, ...]
    next_actions: tuple[str, ...]
    is_production_ea: bool = False
    includes_zpe: bool = False
    includes_core_valence: bool = False
    includes_scalar_relativity: bool = False
    includes_soc: bool = False
    includes_post_ccsd_t: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["primary_model"] = self.primary_model.to_dict()
        data["sensitivity_model"] = self.sensitivity_model.to_dict()
        return data


def inverse_power_cbs(
    e_low: float,
    e_high: float,
    *,
    low_cardinal: int,
    high_cardinal: int,
    exponent: float,
) -> float:
    if high_cardinal <= low_cardinal:
        raise ValueError("high_cardinal must exceed low_cardinal")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    values = (e_low, e_high, float(exponent))
    if not all(isfinite(float(x)) for x in values):
        raise ValueError("CBS inputs must be finite")
    lo = float(low_cardinal) ** exponent
    hi = float(high_cardinal) ** exponent
    return (hi * e_high - lo * e_low) / (hi - lo)


def _finite(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"missing {label}")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"nonfinite {label}")
    return number


def build_component_pairs(
    points: list[Mapping[str, Any]],
    *,
    qz_basis: str = "aug-cc-pvqz",
    z5_basis: str = "aug-cc-pv5z",
) -> dict[str, ComponentPair]:
    by_key = {(str(p["role"]), str(p["basis"])): p for p in points}
    result = {}
    for role in ("neutral", "anion"):
        q = by_key.get((role, qz_basis))
        z = by_key.get((role, z5_basis))
        if q is None or z is None:
            raise ValueError(f"missing QZ/5Z evidence for {role}")
        if not bool(q.get("reusable")) or not bool(z.get("reusable")):
            raise ValueError(f"nonreusable QZ/5Z evidence for {role}")
        result[role] = ComponentPair(
            role=role,
            basis_qz=qz_basis,
            basis_5z=z5_basis,
            scf_qz=_finite(q.get("scf_energy_hartree"), f"{role} QZ SCF"),
            scf_5z=_finite(z.get("scf_energy_hartree"), f"{role} 5Z SCF"),
            ccsd_correlation_qz=_finite(
                q.get("ccsd_correlation_hartree"), f"{role} QZ CCSD corr"
            ),
            ccsd_correlation_5z=_finite(
                z.get("ccsd_correlation_hartree"), f"{role} 5Z CCSD corr"
            ),
            triples_qz=_finite(
                q.get("triples_correction_hartree"), f"{role} QZ (T)"
            ),
            triples_5z=_finite(
                z.get("triples_correction_hartree"), f"{role} 5Z (T)"
            ),
        )
    return result


def extrapolate_species(
    pair: ComponentPair,
    *,
    model_name: str,
    exponents: Mapping[str, float],
) -> SpeciesCBSComponents:
    scf = inverse_power_cbs(
        pair.scf_qz, pair.scf_5z,
        low_cardinal=4, high_cardinal=5,
        exponent=float(exponents["scf"]),
    )
    ccsd = inverse_power_cbs(
        pair.ccsd_correlation_qz, pair.ccsd_correlation_5z,
        low_cardinal=4, high_cardinal=5,
        exponent=float(exponents["ccsd_correlation"]),
    )
    triples = inverse_power_cbs(
        pair.triples_qz, pair.triples_5z,
        low_cardinal=4, high_cardinal=5,
        exponent=float(exponents["triples"]),
    )
    return SpeciesCBSComponents(
        role=pair.role,
        model=model_name,
        scf_cbs_hartree=scf,
        ccsd_correlation_cbs_hartree=ccsd,
        triples_cbs_hartree=triples,
        total_cbs_hartree=scf + ccsd + triples,
    )


def ea_from_species(
    neutral: SpeciesCBSComponents,
    anion: SpeciesCBSComponents,
    *,
    model_name: str,
) -> EACBSModel:
    return EACBSModel(
        model=model_name,
        scf_contribution_ev=(
            neutral.scf_cbs_hartree - anion.scf_cbs_hartree
        ) * HARTREE_TO_EV,
        ccsd_correlation_contribution_ev=(
            neutral.ccsd_correlation_cbs_hartree
            - anion.ccsd_correlation_cbs_hartree
        ) * HARTREE_TO_EV,
        triples_contribution_ev=(
            neutral.triples_cbs_hartree - anion.triples_cbs_hartree
        ) * HARTREE_TO_EV,
        ea_cbs_aug_ev=(
            neutral.total_cbs_hartree - anion.total_cbs_hartree
        ) * HARTREE_TO_EV,
    )


def combined_correlation_sanity(
    pairs: Mapping[str, ComponentPair],
    *,
    scf_exponent: float = PRIMARY_EXPONENTS["scf"],
    correlation_exponent: float = 3.0,
) -> float:
    species = {}
    for role, p in pairs.items():
        scf = inverse_power_cbs(
            p.scf_qz, p.scf_5z,
            low_cardinal=4, high_cardinal=5,
            exponent=scf_exponent,
        )
        corr4 = p.ccsd_correlation_qz + p.triples_qz
        corr5 = p.ccsd_correlation_5z + p.triples_5z
        corr = inverse_power_cbs(
            corr4, corr5,
            low_cardinal=4, high_cardinal=5,
            exponent=correlation_exponent,
        )
        species[role] = scf + corr
    return (species["neutral"] - species["anion"]) * HARTREE_TO_EV


def fixed_basis_ea_from_points(
    points: list[Mapping[str, Any]],
    *,
    basis: str,
) -> float:
    by_role = {
        str(p["role"]): p
        for p in points
        if str(p["basis"]) == basis
    }
    if set(by_role) != {"neutral", "anion"}:
        raise ValueError(f"need neutral and anion for {basis}")
    n = _finite(
        by_role["neutral"].get("ccsd_t_total_hartree"),
        f"neutral {basis} total",
    )
    a = _finite(
        by_role["anion"].get("ccsd_t_total_hartree"),
        f"anion {basis} total",
    )
    return (n - a) * HARTREE_TO_EV


def resolve_cbs(
    *,
    points: list[Mapping[str, Any]],
    daug_fixed_ea_ev: float,
    full_pec_aug5_ea_ev: float | None,
    diffuse_residual_estimate_ev: float | None,
    model_spread_target_ev: float = 0.005,
) -> CBSResolvedResult:
    pairs = build_component_pairs(points)

    primary_species = {
        role: extrapolate_species(
            pair,
            model_name="COMPONENT_SPECIFIC_AVQ5_PRIMARY",
            exponents=PRIMARY_EXPONENTS,
        )
        for role, pair in pairs.items()
    }
    primary = ea_from_species(
        primary_species["neutral"],
        primary_species["anion"],
        model_name="COMPONENT_SPECIFIC_AVQ5_PRIMARY",
    )

    sensitivity_species = {
        role: extrapolate_species(
            pair,
            model_name="COMPONENT_SPECIFIC_SENSITIVITY",
            exponents=SENSITIVITY_EXPONENTS,
        )
        for role, pair in pairs.items()
    }
    sensitivity = ea_from_species(
        sensitivity_species["neutral"],
        sensitivity_species["anion"],
        model_name="COMPONENT_SPECIFIC_SENSITIVITY",
    )

    combined = combined_correlation_sanity(pairs)
    model_bound = max(
        abs(primary.ea_cbs_aug_ev - sensitivity.ea_cbs_aug_ev),
        abs(primary.ea_cbs_aug_ev - combined),
    )

    aug5_fixed = fixed_basis_ea_from_points(
        points,
        basis="aug-cc-pv5z",
    )
    diffuse_correction = float(daug_fixed_ea_ev) - aug5_fixed

    geometry_check = None
    if full_pec_aug5_ea_ev is not None:
        geometry_check = abs(
            float(full_pec_aug5_ea_ev) - aug5_fixed
        )

    diffuse_residual = (
        None if diffuse_residual_estimate_ev is None
        else abs(float(diffuse_residual_estimate_ev))
    )

    half_width = model_bound
    evidence = [
        "QZ_5Z_COMPONENT_EVIDENCE_COMPLETE",
        "SCF_CCSD_TRIPLES_EXTRAPOLATED_SEPARATELY",
        "INDEPENDENT_COMBINED_CORRELATION_SANITY_CHECK",
        "DAUG_MINUS_AUG_DIFFUSE_CORRECTION_SEPARATE",
        "NO_BASIS_AVERAGING",
        "VALIDATION_ONLY",
    ]
    if diffuse_residual is not None:
        half_width += diffuse_residual
        evidence.append("DIFFUSE_RESIDUAL_CONVERGENCE_ESTIMATED")
    else:
        evidence.append("DIFFUSE_RESIDUAL_UNKNOWN")
    if geometry_check is not None:
        half_width += geometry_check
        evidence.append("FIXED_GEOMETRY_TRANSFER_DIRECTLY_CHECKED")
    else:
        evidence.append("FIXED_GEOMETRY_TRANSFER_UNKNOWN")

    center = primary.ea_cbs_aug_ev + diffuse_correction

    if model_bound <= model_spread_target_ev:
        status = CBSStatus.CLEARED
        evidence.append("CBS_MODEL_SENSITIVITY_WITHIN_TARGET")
        next_actions = (
            "ASSESS_CORE_VALENCE",
            "ASSESS_SCALAR_RELATIVITY",
            "ASSESS_POST_CCSD_T",
            "ASSESS_SOC",
            "SOLVE_NUCLEAR_MOTION",
        )
    else:
        status = CBSStatus.NEED_MORE_EVIDENCE
        evidence.append("CBS_MODEL_SENSITIVITY_EXCEEDS_TARGET")
        next_actions = (
            "COMPUTE_ADDITIONAL_CARDINAL_EVIDENCE_OR_REVISE_CBS_MODEL",
            "DO_NOT_ADVANCE_CBS_AS_CLEARED",
        )

    return CBSResolvedResult(
        status=status,
        primary_model=primary,
        sensitivity_model=sensitivity,
        combined_correlation_sanity_ev=combined,
        cbs_model_sensitivity_bound_ev=model_bound,
        diffuse_correction_ev=diffuse_correction,
        diffuse_residual_estimate_ev=diffuse_residual,
        fixed_geometry_transfer_check_ev=geometry_check,
        ea_cbs_plus_diffuse_primary_ev=center,
        conservative_intermediate_half_width_ev=half_width,
        lower_intermediate_ev=center - half_width,
        upper_intermediate_ev=center + half_width,
        evidence=tuple(evidence),
        next_actions=next_actions,
        is_production_ea=False,
        includes_zpe=False,
        includes_core_valence=False,
        includes_scalar_relativity=False,
        includes_soc=False,
        includes_post_ccsd_t=False,
    )
