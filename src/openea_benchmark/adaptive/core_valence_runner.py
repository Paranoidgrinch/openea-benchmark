"""Generic adaptive core-valence execution for the OpenEA single-reference branch.

The runner deliberately separates *basis-policy selection* from execution.  It
never guesses a core-valence basis family from an element symbol.  Instead the
caller binds an explicit cardinal-by-cardinal ``basis_by_element`` policy that
has already been judged appropriate for the molecular system.

For every cardinal X and both electron-attachment partners, exactly matched
CCSD(T) calculations are performed in two correlation spaces:

    all-electron (AE)
    frozen-core (FC; PySCF ``CCSD.set_frozen()`` chemical-core policy)

The additive correction is then

    Delta_CV(X) = EA_AE(X) - EA_FC(X)

with identical geometry, one-electron basis, charge/spin sector, HF reference
policy and numerical settings inside each AE/FC pair.  Neutral and anion may
use different equilibrium geometries, as required for an adiabatic EA.

This module does not choose electronic states, basis families, ECPs, or a
multireference method.  It requires already validated state provenance and
fails closed if the requested basis policy or checkpoint evidence is missing.
It also never promotes its result to a complete production EA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV
from openea_benchmark.attachment.core_valence_correction import (
    CVCardinalPoint,
    CoreValenceAssessment,
    CoreValenceStatus,
    assess_core_valence,
    cv_point_from_results,
)

from .stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3ExecutionSettings,
    Stage3PointResult,
    run_stage3_point,
)


class CoreValenceCorrelationSpace(str, Enum):
    ALL_ELECTRON = "ALL_ELECTRON"
    FROZEN_CORE = "FROZEN_CORE"


class AdaptiveCoreValenceStatus(str, Enum):
    CORE_VALENCE_CLEARED = "CORE_VALENCE_CLEARED"
    CORE_VALENCE_UNRESOLVED = "CORE_VALENCE_UNRESOLVED"
    EXECUTION_BLOCKED = "EXECUTION_BLOCKED"
    POLICY_BLOCKED = "POLICY_BLOCKED"


@dataclass(frozen=True)
class CoreValenceStateSpec:
    """One already resolved molecular state used in the CV correction."""

    role: str
    system: str
    atoms: tuple[str, str]
    charge: int
    spin_2s: int
    state_id: str
    r_angstrom: float
    source_root_id: str
    source_checkpoint_path: str
    state_identity_validated: bool

    def __post_init__(self) -> None:
        if self.role not in {"neutral", "anion"}:
            raise ValueError("Core-valence state role must be neutral or anion")
        if not self.system.strip() or not self.state_id.strip() or not self.source_root_id.strip():
            raise ValueError("Core-valence state requires system/state/source provenance")
        if len(self.atoms) != 2 or any(not str(x).strip() for x in self.atoms):
            raise ValueError("Core-valence v1 runner requires exactly two atoms")
        if self.spin_2s < 0:
            raise ValueError("spin_2s must be non-negative")
        if not isfinite(float(self.r_angstrom)) or float(self.r_angstrom) <= 0.0:
            raise ValueError("Core-valence geometry must be finite and positive")
        if not str(self.source_checkpoint_path).strip():
            raise ValueError("Core-valence state requires a source checkpoint path")
        if not self.state_identity_validated:
            raise ValueError(
                "Core-valence execution requires state identity to be validated before production correction work"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoreValenceBasisSpec:
    """Explicit all-electron orbital-basis assignment for one cardinal X."""

    cardinal: int
    family_id: str
    label: str
    basis_by_element: Mapping[str, str]
    electron_model: str = "ALL_ELECTRON"

    def __post_init__(self) -> None:
        if self.cardinal < 2:
            raise ValueError("Core-valence cardinal must be >= 2")
        if not self.family_id.strip() or not self.label.strip():
            raise ValueError("Core-valence basis specification requires family_id and label")
        if self.electron_model != "ALL_ELECTRON":
            raise ValueError(
                "Generic OpenEA-v1 core-valence runner currently supports only explicit all-electron orbital-basis policies; ECP/core-replacement policies require a dedicated treatment"
            )
        if not self.basis_by_element:
            raise ValueError("Core-valence basis_by_element must not be empty")
        if any(not str(k).strip() or not str(v).strip() for k, v in self.basis_by_element.items()):
            raise ValueError("Core-valence basis_by_element contains an empty element or basis name")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["basis_by_element"] = dict(self.basis_by_element)
        return d


@dataclass(frozen=True)
class CoreValenceSubcalculation:
    role: str
    correlation_space: CoreValenceCorrelationSpace
    request: Stage3ExecutionRequest
    result: Stage3PointResult
    checkpoint_reused: bool
    frozen_core_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "correlation_space": self.correlation_space.value,
            "request": self.request.to_dict(),
            "result": self.result.to_dict(),
            "checkpoint_reused": self.checkpoint_reused,
            "frozen_core_policy": self.frozen_core_policy,
        }


@dataclass(frozen=True)
class CoreValenceCardinalEvidence:
    cardinal: int
    basis: CoreValenceBasisSpec
    point: CVCardinalPoint
    calculations: tuple[CoreValenceSubcalculation, ...]

    def __post_init__(self) -> None:
        if self.cardinal != self.basis.cardinal or self.cardinal != self.point.cardinal:
            raise ValueError("Core-valence cardinal evidence is internally inconsistent")
        keys = {(x.role, x.correlation_space.value) for x in self.calculations}
        expected = {
            ("neutral", CoreValenceCorrelationSpace.ALL_ELECTRON.value),
            ("neutral", CoreValenceCorrelationSpace.FROZEN_CORE.value),
            ("anion", CoreValenceCorrelationSpace.ALL_ELECTRON.value),
            ("anion", CoreValenceCorrelationSpace.FROZEN_CORE.value),
        }
        if keys != expected:
            raise ValueError("Core-valence cardinal evidence requires exactly neutral/anion x AE/FC")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cardinal": self.cardinal,
            "basis": self.basis.to_dict(),
            "point": self.point.to_dict(),
            "calculations": [x.to_dict() for x in self.calculations],
        }


@dataclass(frozen=True)
class AdaptiveCoreValenceResult:
    status: AdaptiveCoreValenceStatus
    evidence: tuple[CoreValenceCardinalEvidence, ...]
    assessment: CoreValenceAssessment | None
    execution_error_type: str | None = None
    execution_error_message: str | None = None
    is_production_ea: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if self.is_production_ea or self.authorizes_pruning:
            raise ValueError("Core-valence runner cannot promote a complete production EA or prune states")

    @property
    def points(self) -> tuple[CVCardinalPoint, ...]:
        return tuple(x.point for x in self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "evidence": [x.to_dict() for x in self.evidence],
            "assessment": None if self.assessment is None else self.assessment.to_dict(),
            "execution_error_type": self.execution_error_type,
            "execution_error_message": self.execution_error_message,
            "is_production_ea": self.is_production_ea,
            "authorizes_pruning": self.authorizes_pruning,
        }


PointRunner = Callable[[Stage3ExecutionRequest, Stage3ExecutionSettings], Stage3PointResult]


def _validate_state_pair(neutral: CoreValenceStateSpec, anion: CoreValenceStateSpec) -> None:
    if neutral.role != "neutral" or anion.role != "anion":
        raise ValueError("Core-valence runner requires neutral then anion state specifications")
    if neutral.system != anion.system:
        raise ValueError("Neutral and anion must belong to the same molecular system")
    if neutral.atoms != anion.atoms:
        raise ValueError("Neutral and anion atom ordering must match")
    if anion.charge != neutral.charge - 1:
        raise ValueError("Anion charge must equal neutral charge minus one electron")


def _basis_map(
    basis_specs: tuple[CoreValenceBasisSpec, ...], atoms: tuple[str, str]
) -> dict[int, CoreValenceBasisSpec]:
    by_x: dict[int, CoreValenceBasisSpec] = {}
    expected_elements = set(atoms)
    for spec in basis_specs:
        if spec.cardinal in by_x:
            raise ValueError(f"Duplicate core-valence basis specification X={spec.cardinal}")
        if set(spec.basis_by_element) != expected_elements:
            raise ValueError(
                "Core-valence basis specification must map exactly the molecular elements: "
                f"X={spec.cardinal} expected={sorted(expected_elements)} "
                f"provided={sorted(spec.basis_by_element)}"
            )
        by_x[spec.cardinal] = spec
    if not by_x:
        raise ValueError("At least one core-valence basis specification is required")
    family_ids = {x.family_id for x in by_x.values()}
    if len(family_ids) != 1:
        raise ValueError(
            "All cardinal points in one core-valence convergence series must use one declared basis family"
        )
    return by_x


def _safe_token(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def _request_for(
    state: CoreValenceStateSpec,
    basis: CoreValenceBasisSpec,
    space: CoreValenceCorrelationSpace,
) -> Stage3ExecutionRequest:
    token = _safe_token(state.state_id)
    job_id = f"core_valence__{_safe_token(state.system)}__{state.role}__{token}__X{basis.cardinal}__{space.value}"
    return Stage3ExecutionRequest(
        request_id=f"{job_id}__fixed_state_geometry",
        job_id=job_id,
        system=state.system,
        atoms=state.atoms,
        charge=state.charge,
        spin_2s=state.spin_2s,
        component_id=f"{job_id}__delta_cv_component",
        r_angstrom=float(state.r_angstrom),
        basis=basis.label,
        methods=("CCSD", "CCSD(T)"),
        requested_reference="ROHF",
        scf_reference="RHF" if state.spin_2s == 0 else "ROHF",
        source_link_status="SINGLE_DFT_INITIALIZATION",
        source_root_id=state.source_root_id,
        source_checkpoint_path=state.source_checkpoint_path,
        source_origin_guess=f"CORE_VALENCE_FROM_VALIDATED_STATE:{state.state_id}",
        grid_index=0,
        initialization_index=0,
        dft_center_r_angstrom=float(state.r_angstrom),
        dft_center_energy_hartree=0.0,
        requires_independent_state_identity_validation=True,
        basis_by_element=dict(basis.basis_by_element),
        authorizes_pruning=False,
    )


def _settings_for(
    base: Stage3ExecutionSettings,
    space: CoreValenceCorrelationSpace,
) -> Stage3ExecutionSettings:
    if not base.run_ccsd_t:
        raise ValueError("Core-valence correction requires CCSD(T) totals")
    if base.scalar_relativistic != "NONE":
        raise ValueError(
            "Core-valence correction and scalar relativity are separate OpenEA corrections; the generic CV runner requires scalar_relativistic='NONE'"
        )
    # Explicitly project the validated source checkpoint when the CV basis differs
    # from the state-resolution basis.  The source remains an initial guess only.
    return replace(
        base,
        checkpoint_project=True,
        frozen_core=(space is CoreValenceCorrelationSpace.FROZEN_CORE),
    )


def _checkpoint_signature(
    request: Stage3ExecutionRequest,
    settings: Stage3ExecutionSettings,
    space: CoreValenceCorrelationSpace,
) -> str:
    payload = {
        "schema": "OPENEA_CORE_VALENCE_SUBPOINT_V1",
        "request": request.to_dict(),
        "correlation_space": space.value,
        "frozen_core": settings.frozen_core,
        "frozen_core_policy": (
            "PYSCF_CCSD_SET_FROZEN_CHEMCORE"
            if settings.frozen_core
            else "ALL_ELECTRON_CCSD"
        ),
        "run_ccsd_t": settings.run_ccsd_t,
        "scalar_relativistic": settings.scalar_relativistic,
        "scf_conv_tol": settings.scf_conv_tol,
        "scf_conv_tol_grad": settings.scf_conv_tol_grad,
        "scf_max_cycle": settings.scf_max_cycle,
        "cc_conv_tol": settings.cc_conv_tol,
        "cc_conv_tol_normt": settings.cc_conv_tol_normt,
        "cc_max_cycle": settings.cc_max_cycle,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def _result_from_dict(raw: Mapping[str, Any]) -> Stage3PointResult:
    data = dict(raw)
    data["status"] = PointExecutionStatus(str(data["status"]))
    return Stage3PointResult(**data)


def _checkpoint_path(
    checkpoint_dir: Path,
    *,
    cardinal: int,
    role: str,
    space: CoreValenceCorrelationSpace,
) -> Path:
    return checkpoint_dir / f"X{cardinal}" / f"{role}__{space.value}.json"


def _load_subpoint_checkpoint(
    path: Path,
    *,
    expected_signature: str,
    request: Stage3ExecutionRequest,
) -> Stage3PointResult | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != expected_signature:
        raise ValueError(
            f"Core-valence checkpoint signature mismatch for {path}; refusing stale/incompatible evidence"
        )
    result = _result_from_dict(payload["result"])
    if result.request_id != request.request_id or result.job_id != request.job_id:
        raise ValueError("Core-valence checkpoint belongs to another request/job")
    if result.status is not PointExecutionStatus.COMPLETED:
        return None
    if result.ccsd_t_total_hartree is None:
        return None
    return result


def _save_subpoint_checkpoint(
    path: Path,
    *,
    signature: str,
    space: CoreValenceCorrelationSpace,
    result: Stage3PointResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "OPENEA_CORE_VALENCE_SUBPOINT_V1",
        "signature": signature,
        "correlation_space": space.value,
        "result": result.to_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _execute_subpoint(
    *,
    state: CoreValenceStateSpec,
    basis: CoreValenceBasisSpec,
    space: CoreValenceCorrelationSpace,
    base_settings: Stage3ExecutionSettings,
    point_runner: PointRunner,
    checkpoint_dir: Path | None,
) -> CoreValenceSubcalculation:
    request = _request_for(state, basis, space)
    settings = _settings_for(base_settings, space)
    signature = _checkpoint_signature(request, settings, space)
    result = None
    reused = False
    path = None
    if checkpoint_dir is not None:
        path = _checkpoint_path(
            checkpoint_dir, cardinal=basis.cardinal, role=state.role, space=space
        )
        result = _load_subpoint_checkpoint(
            path, expected_signature=signature, request=request
        )
        reused = result is not None

    if result is None:
        result = point_runner(request, settings)
        if not isinstance(result, Stage3PointResult):
            raise TypeError("Core-valence point runner must return Stage3PointResult")
        if result.request_id != request.request_id or result.job_id != request.job_id:
            raise ValueError("Core-valence point runner returned another request/job")
        if path is not None:
            _save_subpoint_checkpoint(path, signature=signature, space=space, result=result)

    policy = (
        "PYSCF_CCSD_SET_FROZEN_CHEMCORE"
        if space is CoreValenceCorrelationSpace.FROZEN_CORE
        else "ALL_ELECTRON_CCSD"
    )
    return CoreValenceSubcalculation(
        role=state.role,
        correlation_space=space,
        request=request,
        result=result,
        checkpoint_reused=reused,
        frozen_core_policy=policy,
    )


def _evaluate_cardinal(
    *,
    neutral: CoreValenceStateSpec,
    anion: CoreValenceStateSpec,
    basis: CoreValenceBasisSpec,
    base_settings: Stage3ExecutionSettings,
    point_runner: PointRunner,
    checkpoint_dir: Path | None,
) -> CoreValenceCardinalEvidence:
    calculations: list[CoreValenceSubcalculation] = []
    totals: dict[tuple[str, CoreValenceCorrelationSpace], float] = {}

    for state in (neutral, anion):
        for space in (
            CoreValenceCorrelationSpace.ALL_ELECTRON,
            CoreValenceCorrelationSpace.FROZEN_CORE,
        ):
            calc = _execute_subpoint(
                state=state,
                basis=basis,
                space=space,
                base_settings=base_settings,
                point_runner=point_runner,
                checkpoint_dir=checkpoint_dir,
            )
            calculations.append(calc)
            result = calc.result
            if result.status is not PointExecutionStatus.COMPLETED:
                raise RuntimeError(
                    f"Core-valence subcalculation failed: X={basis.cardinal} "
                    f"role={state.role} space={space.value} status={result.status.value} "
                    f"error={result.error_type}:{result.error_message}"
                )
            if result.ccsd_t_total_hartree is None:
                raise RuntimeError(
                    f"Core-valence subcalculation has no CCSD(T) total: X={basis.cardinal} "
                    f"role={state.role} space={space.value}"
                )
            if result.basis != basis.label:
                raise RuntimeError("Core-valence result basis label differs from authorized basis")
            if abs(float(result.r_angstrom) - float(state.r_angstrom)) > 1.0e-10:
                raise RuntimeError("Core-valence result geometry differs from authorized state geometry")
            if result.charge != state.charge or result.spin_2s != state.spin_2s:
                raise RuntimeError("Core-valence result charge/spin sector differs from authorized state")
            totals[(state.role, space)] = float(result.ccsd_t_total_hartree)

    point = cv_point_from_results(
        cardinal=basis.cardinal,
        basis=basis.label,
        ae_neutral_h=totals[("neutral", CoreValenceCorrelationSpace.ALL_ELECTRON)],
        ae_anion_h=totals[("anion", CoreValenceCorrelationSpace.ALL_ELECTRON)],
        fc_neutral_h=totals[("neutral", CoreValenceCorrelationSpace.FROZEN_CORE)],
        fc_anion_h=totals[("anion", CoreValenceCorrelationSpace.FROZEN_CORE)],
        hartree_to_ev=HARTREE_TO_EV,
    )
    return CoreValenceCardinalEvidence(
        cardinal=basis.cardinal,
        basis=basis,
        point=point,
        calculations=tuple(calculations),
    )


def run_adaptive_core_valence_series(
    *,
    neutral: CoreValenceStateSpec,
    anion: CoreValenceStateSpec,
    basis_specs: tuple[CoreValenceBasisSpec, ...],
    initial_cardinals: tuple[int, ...] = (3, 4),
    maximum_cardinal: int = 5,
    target_change_ev: float = 0.002,
    max_contraction_ratio: float = 0.8,
    execution_settings: Stage3ExecutionSettings | None = None,
    checkpoint_dir: Path | str | None = None,
    point_runner: PointRunner | None = None,
) -> AdaptiveCoreValenceResult:
    """Run an adaptive, checkpointable matched AE/FC core-valence series.

    ``point_runner`` is an injection seam for unit tests/backends.  The default
    executes the already established PySCF Stage-3 CCSD(T) point runner.
    """

    _validate_state_pair(neutral, anion)
    by_x = _basis_map(basis_specs, neutral.atoms)
    if len(initial_cardinals) < 2:
        raise ValueError("Core-valence convergence requires at least two initial cardinals")
    if tuple(sorted(initial_cardinals)) != initial_cardinals or len(set(initial_cardinals)) != len(initial_cardinals):
        raise ValueError("initial_cardinals must be sorted and unique")
    if maximum_cardinal < max(initial_cardinals):
        raise ValueError("maximum_cardinal is below the initial core-valence series")
    missing_initial = [x for x in initial_cardinals if x not in by_x]
    if missing_initial:
        raise ValueError(f"Missing explicit core-valence basis policy for initial cardinals {missing_initial}")
    if not isfinite(float(target_change_ev)) or target_change_ev <= 0.0:
        raise ValueError("target_change_ev must be finite and positive")
    if not isfinite(float(max_contraction_ratio)) or max_contraction_ratio <= 0.0:
        raise ValueError("max_contraction_ratio must be finite and positive")

    settings = execution_settings or Stage3ExecutionSettings()
    # Validate method-separation invariants before any expensive work starts.
    _settings_for(settings, CoreValenceCorrelationSpace.ALL_ELECTRON)
    runner = point_runner or (lambda req, cfg: run_stage3_point(req, settings=cfg))
    cp_dir = None if checkpoint_dir is None else Path(checkpoint_dir)

    evidence: list[CoreValenceCardinalEvidence] = []

    def execute_x(x: int) -> Exception | None:
        spec = by_x.get(x)
        if spec is None:
            return ValueError(
                f"No explicit core-valence basis policy is available for requested X={x}"
            )
        try:
            ev = _evaluate_cardinal(
                neutral=neutral,
                anion=anion,
                basis=spec,
                base_settings=settings,
                point_runner=runner,
                checkpoint_dir=cp_dir,
            )
            evidence.append(ev)
            return None
        except Exception as exc:
            return exc

    for x in initial_cardinals:
        error = execute_x(x)
        if error is not None:
            return AdaptiveCoreValenceResult(
                status=AdaptiveCoreValenceStatus.EXECUTION_BLOCKED,
                evidence=tuple(evidence),
                assessment=(
                    None
                    if not evidence
                    else assess_core_valence(
                        tuple(e.point for e in evidence),
                        target_change_ev=target_change_ev,
                        max_contraction_ratio=max_contraction_ratio,
                        max_cardinal=maximum_cardinal,
                    )
                ),
                execution_error_type=type(error).__name__,
                execution_error_message=str(error),
            )

    while True:
        assessment = assess_core_valence(
            tuple(e.point for e in evidence),
            target_change_ev=target_change_ev,
            max_contraction_ratio=max_contraction_ratio,
            max_cardinal=maximum_cardinal,
        )
        if assessment.status is CoreValenceStatus.CLEARED:
            return AdaptiveCoreValenceResult(
                AdaptiveCoreValenceStatus.CORE_VALENCE_CLEARED,
                tuple(evidence),
                assessment,
            )
        if assessment.status is CoreValenceStatus.BLOCKED:
            return AdaptiveCoreValenceResult(
                AdaptiveCoreValenceStatus.POLICY_BLOCKED,
                tuple(evidence),
                assessment,
                "CoreValenceAssessmentBlocked",
                assessment.action,
            )
        if assessment.action == "MAX_CARDINAL_REACHED_UNRESOLVED":
            return AdaptiveCoreValenceResult(
                AdaptiveCoreValenceStatus.CORE_VALENCE_UNRESOLVED,
                tuple(evidence),
                assessment,
            )
        if not assessment.action.startswith("COMPUTE_X"):
            return AdaptiveCoreValenceResult(
                AdaptiveCoreValenceStatus.POLICY_BLOCKED,
                tuple(evidence),
                assessment,
                "UnknownCoreValenceAction",
                assessment.action,
            )
        try:
            requested = int(assessment.action.removeprefix("COMPUTE_X"))
        except ValueError:
            return AdaptiveCoreValenceResult(
                AdaptiveCoreValenceStatus.POLICY_BLOCKED,
                tuple(evidence),
                assessment,
                "InvalidCoreValenceAction",
                assessment.action,
            )
        if requested > maximum_cardinal:
            return AdaptiveCoreValenceResult(
                AdaptiveCoreValenceStatus.CORE_VALENCE_UNRESOLVED,
                tuple(evidence),
                assessment,
            )
        if requested in {x.cardinal for x in evidence}:
            return AdaptiveCoreValenceResult(
                AdaptiveCoreValenceStatus.POLICY_BLOCKED,
                tuple(evidence),
                assessment,
                "DuplicateCoreValenceRequest",
                f"Assessment requested already evaluated X={requested}",
            )
        if requested not in by_x:
            return AdaptiveCoreValenceResult(
                AdaptiveCoreValenceStatus.POLICY_BLOCKED,
                tuple(evidence),
                assessment,
                "MissingCoreValenceBasisPolicy",
                f"No explicit core-valence basis policy is available for requested X={requested}",
            )
        error = execute_x(requested)
        if error is not None:
            return AdaptiveCoreValenceResult(
                AdaptiveCoreValenceStatus.EXECUTION_BLOCKED,
                tuple(evidence),
                assessment,
                type(error).__name__,
                str(error),
            )
