"""Generic adaptive scalar-relativity execution for OpenEA single-reference production.

This runner evaluates the spin-free one-electron X2C correction as a matched
Hamiltonian difference at already validated neutral/anion states::

    Delta_SR(X) = EA_SFX2C1E(X) - EA_NR(X)

For every cardinal number X, the NR and SFX2C1E members use exactly the same
molecular geometry, electronic state, all-electron correlation treatment,
orbital-basis assignment, SCF/CC numerical settings and source-state
provenance.  The only intended physical difference is the one-electron
Hamiltonian selected through ``Stage3ExecutionSettings.scalar_relativistic``.

The basis policy is deliberately explicit.  The runner never guesses a
relativistic/recontracted basis family from an element symbol and never swaps
basis families between the NR and X2C members.  ECP/core-replacement policies
are outside this v1 runner.

Scope limits are part of the result contract: SFX2C1E is a spin-free
*one-electron* X2C correction.  Spin-orbit coupling and the missing
two-electron/picture-change relativistic remainder remain separate OpenEA
corrections and are never silently assigned zero.
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
from openea_benchmark.attachment.scalar_relativity import (
    ScalarRelativityAssessment,
    ScalarRelativityPoint,
    assess_scalar_relativity,
)

from .stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3ExecutionSettings,
    Stage3PointResult,
    run_stage3_point,
)


class ScalarRelativityHamiltonian(str, Enum):
    NONRELATIVISTIC = "NR"
    SFX2C1E = "SFX2C1E"


class AdaptiveScalarRelativityStatus(str, Enum):
    SCALAR_RELATIVITY_CLEARED = "SCALAR_RELATIVITY_CLEARED"
    SCALAR_RELATIVITY_UNRESOLVED = "SCALAR_RELATIVITY_UNRESOLVED"
    EXECUTION_BLOCKED = "EXECUTION_BLOCKED"
    POLICY_BLOCKED = "POLICY_BLOCKED"


@dataclass(frozen=True)
class ScalarRelativityStateSpec:
    """One already resolved molecular state used in the SR correction."""

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
            raise ValueError("Scalar-relativity state role must be neutral or anion")
        if not self.system.strip() or not self.state_id.strip() or not self.source_root_id.strip():
            raise ValueError("Scalar-relativity state requires system/state/source provenance")
        if len(self.atoms) != 2 or any(not str(x).strip() for x in self.atoms):
            raise ValueError("Scalar-relativity v1 runner requires exactly two atoms")
        if self.spin_2s < 0:
            raise ValueError("spin_2s must be non-negative")
        if not isfinite(float(self.r_angstrom)) or float(self.r_angstrom) <= 0.0:
            raise ValueError("Scalar-relativity geometry must be finite and positive")
        if not str(self.source_checkpoint_path).strip():
            raise ValueError("Scalar-relativity state requires a source checkpoint path")
        if not self.state_identity_validated:
            raise ValueError(
                "Scalar-relativity execution requires state identity to be validated before production correction work"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScalarRelativityBasisSpec:
    """Explicit relativistically suitable orbital-basis assignment for one X."""

    cardinal: int
    family_id: str
    label: str
    basis_by_element: Mapping[str, str]
    scalar_relativistically_recontracted: bool
    electron_model: str = "ALL_ELECTRON"

    def __post_init__(self) -> None:
        if self.cardinal < 2:
            raise ValueError("Scalar-relativity cardinal must be >= 2")
        if not self.family_id.strip() or not self.label.strip():
            raise ValueError("Scalar-relativity basis specification requires family_id and label")
        if self.electron_model != "ALL_ELECTRON":
            raise ValueError(
                "Generic OpenEA-v1 scalar-relativity runner currently supports only explicit all-electron orbital-basis policies; ECP/core-replacement policies require a dedicated relativistic treatment"
            )
        if not self.scalar_relativistically_recontracted:
            raise ValueError(
                "Scalar-relativity basis policy must explicitly certify a scalar-relativistically appropriate/recontracted orbital basis"
            )
        if not self.basis_by_element:
            raise ValueError("Scalar-relativity basis_by_element must not be empty")
        if any(not str(k).strip() or not str(v).strip() for k, v in self.basis_by_element.items()):
            raise ValueError("Scalar-relativity basis_by_element contains an empty element or basis name")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["basis_by_element"] = dict(self.basis_by_element)
        return data


@dataclass(frozen=True)
class ScalarRelativitySubcalculation:
    role: str
    hamiltonian: ScalarRelativityHamiltonian
    request: Stage3ExecutionRequest
    result: Stage3PointResult
    checkpoint_reused: bool
    correlation_space: str = "ALL_ELECTRON"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "hamiltonian": self.hamiltonian.value,
            "request": self.request.to_dict(),
            "result": self.result.to_dict(),
            "checkpoint_reused": self.checkpoint_reused,
            "correlation_space": self.correlation_space,
        }


@dataclass(frozen=True)
class ScalarRelativityCardinalEvidence:
    cardinal: int
    basis: ScalarRelativityBasisSpec
    point: ScalarRelativityPoint
    calculations: tuple[ScalarRelativitySubcalculation, ...]

    def __post_init__(self) -> None:
        if self.cardinal != self.basis.cardinal or self.cardinal != self.point.cardinal:
            raise ValueError("Scalar-relativity cardinal evidence is internally inconsistent")
        keys = {(x.role, x.hamiltonian.value) for x in self.calculations}
        expected = {
            ("neutral", ScalarRelativityHamiltonian.NONRELATIVISTIC.value),
            ("neutral", ScalarRelativityHamiltonian.SFX2C1E.value),
            ("anion", ScalarRelativityHamiltonian.NONRELATIVISTIC.value),
            ("anion", ScalarRelativityHamiltonian.SFX2C1E.value),
        }
        if keys != expected:
            raise ValueError("Scalar-relativity cardinal evidence requires neutral/anion x NR/SFX2C1E")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cardinal": self.cardinal,
            "basis": self.basis.to_dict(),
            "point": self.point.to_dict(),
            "calculations": [x.to_dict() for x in self.calculations],
        }


@dataclass(frozen=True)
class AdaptiveScalarRelativityResult:
    status: AdaptiveScalarRelativityStatus
    evidence: tuple[ScalarRelativityCardinalEvidence, ...]
    assessment: ScalarRelativityAssessment | None
    execution_error_type: str | None = None
    execution_error_message: str | None = None
    relativistic_scope: str = "SPIN_FREE_ONE_ELECTRON_X2C"
    correlation_space: str = "ALL_ELECTRON"
    includes_soc: bool = False
    includes_two_electron_relativistic_terms: bool = False
    requires_scalar_relativistic_remainder_assessment: bool = True
    is_production_ea: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if self.relativistic_scope != "SPIN_FREE_ONE_ELECTRON_X2C":
            raise ValueError("Generic scalar-relativity runner has a fixed SFX2C1E scope")
        if self.correlation_space != "ALL_ELECTRON":
            raise ValueError("Generic scalar-relativity runner uses matched all-electron CCSD(T)")
        if self.includes_soc or self.includes_two_electron_relativistic_terms:
            raise ValueError("SFX2C1E runner must not claim SOC or two-electron relativistic terms")
        if not self.requires_scalar_relativistic_remainder_assessment:
            raise ValueError("SFX2C1E cannot close the missing relativistic remainder by itself")
        if self.is_production_ea or self.authorizes_pruning:
            raise ValueError("Scalar-relativity runner cannot promote a complete production EA or prune states")

    @property
    def points(self) -> tuple[ScalarRelativityPoint, ...]:
        return tuple(x.point for x in self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "evidence": [x.to_dict() for x in self.evidence],
            "assessment": None if self.assessment is None else self.assessment.to_dict(),
            "execution_error_type": self.execution_error_type,
            "execution_error_message": self.execution_error_message,
            "relativistic_scope": self.relativistic_scope,
            "correlation_space": self.correlation_space,
            "includes_soc": self.includes_soc,
            "includes_two_electron_relativistic_terms": self.includes_two_electron_relativistic_terms,
            "requires_scalar_relativistic_remainder_assessment": self.requires_scalar_relativistic_remainder_assessment,
            "is_production_ea": self.is_production_ea,
            "authorizes_pruning": self.authorizes_pruning,
        }


PointRunner = Callable[[Stage3ExecutionRequest, Stage3ExecutionSettings], Stage3PointResult]
BasisValidator = Callable[[str, str], None]


def _validate_state_pair(
    neutral: ScalarRelativityStateSpec,
    anion: ScalarRelativityStateSpec,
) -> None:
    if neutral.role != "neutral" or anion.role != "anion":
        raise ValueError("Scalar-relativity runner requires neutral then anion state specifications")
    if neutral.system != anion.system:
        raise ValueError("Neutral and anion must belong to the same molecular system")
    if neutral.atoms != anion.atoms:
        raise ValueError("Neutral and anion atom ordering must match")
    if anion.charge != neutral.charge - 1:
        raise ValueError("Anion charge must equal neutral charge minus one electron")


def _basis_map(
    basis_specs: tuple[ScalarRelativityBasisSpec, ...],
    atoms: tuple[str, str],
) -> dict[int, ScalarRelativityBasisSpec]:
    by_x: dict[int, ScalarRelativityBasisSpec] = {}
    expected_elements = set(atoms)
    for spec in basis_specs:
        if spec.cardinal in by_x:
            raise ValueError(f"Duplicate scalar-relativity basis specification X={spec.cardinal}")
        if set(spec.basis_by_element) != expected_elements:
            raise ValueError(
                "Scalar-relativity basis specification must map exactly the molecular elements: "
                f"X={spec.cardinal} expected={sorted(expected_elements)} "
                f"provided={sorted(spec.basis_by_element)}"
            )
        by_x[spec.cardinal] = spec
    if not by_x:
        raise ValueError("At least one scalar-relativity basis specification is required")
    family_ids = {x.family_id for x in by_x.values()}
    if len(family_ids) != 1:
        raise ValueError(
            "All cardinal points in one scalar-relativity convergence series must use one declared basis family"
        )
    return by_x


def _default_basis_validator(element: str, basis_name: str) -> None:
    from pyscf import gto

    gto.basis.load(basis_name, element)


def _preflight_basis_policy(
    basis_specs: tuple[ScalarRelativityBasisSpec, ...],
    validator: BasisValidator,
) -> None:
    for spec in basis_specs:
        for element, basis_name in spec.basis_by_element.items():
            try:
                validator(element, basis_name)
            except Exception as exc:
                raise RuntimeError(
                    "Scalar-relativity basis preflight failed before QC execution: "
                    f"X={spec.cardinal} element={element} basis={basis_name}"
                ) from exc


def _safe_token(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def _request_for(
    state: ScalarRelativityStateSpec,
    basis: ScalarRelativityBasisSpec,
    hamiltonian: ScalarRelativityHamiltonian,
) -> Stage3ExecutionRequest:
    token = _safe_token(state.state_id)
    job_id = (
        f"scalar_relativity__{_safe_token(state.system)}__{state.role}__{token}"
        f"__X{basis.cardinal}__{hamiltonian.value}"
    )
    return Stage3ExecutionRequest(
        request_id=f"{job_id}__fixed_state_geometry",
        job_id=job_id,
        system=state.system,
        atoms=state.atoms,
        charge=state.charge,
        spin_2s=state.spin_2s,
        component_id=f"{job_id}__delta_sr_component",
        r_angstrom=float(state.r_angstrom),
        basis=basis.label,
        methods=("CCSD", "CCSD(T)"),
        requested_reference="ROHF",
        scf_reference="RHF" if state.spin_2s == 0 else "ROHF",
        source_link_status="SINGLE_DFT_INITIALIZATION",
        source_root_id=state.source_root_id,
        source_checkpoint_path=state.source_checkpoint_path,
        source_origin_guess=f"SCALAR_RELATIVITY_FROM_VALIDATED_STATE:{state.state_id}",
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
    hamiltonian: ScalarRelativityHamiltonian,
) -> Stage3ExecutionSettings:
    if not base.run_ccsd_t:
        raise ValueError("Scalar-relativity correction requires CCSD(T) totals")
    if base.frozen_core:
        raise ValueError(
            "Generic scalar-relativity correction is defined here at matched all-electron CCSD(T); frozen-core scalar-relativity would leave a distinct core-correlation relativistic contribution unaccounted"
        )
    if base.scalar_relativistic != "NONE":
        raise ValueError(
            "Scalar-relativity runner owns the NR/SFX2C1E Hamiltonian switch; supply scalar_relativistic='NONE' in the base settings"
        )
    return replace(
        base,
        checkpoint_project=True,
        frozen_core=False,
        scalar_relativistic=(
            "NONE"
            if hamiltonian is ScalarRelativityHamiltonian.NONRELATIVISTIC
            else "SFX2C1E"
        ),
    )


def _checkpoint_signature(
    request: Stage3ExecutionRequest,
    settings: Stage3ExecutionSettings,
    hamiltonian: ScalarRelativityHamiltonian,
) -> str:
    payload = {
        "schema": "OPENEA_SCALAR_RELATIVITY_SUBPOINT_V1",
        "request": request.to_dict(),
        "hamiltonian": hamiltonian.value,
        "correlation_space": "ALL_ELECTRON",
        "frozen_core": settings.frozen_core,
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
    hamiltonian: ScalarRelativityHamiltonian,
) -> Path:
    return checkpoint_dir / f"X{cardinal}" / f"{role}__{hamiltonian.value}.json"


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
            f"Scalar-relativity checkpoint signature mismatch for {path}; refusing stale/incompatible evidence"
        )
    result = _result_from_dict(payload["result"])
    if result.request_id != request.request_id or result.job_id != request.job_id:
        raise ValueError("Scalar-relativity checkpoint belongs to another request/job")
    if result.status is not PointExecutionStatus.COMPLETED:
        return None
    if result.ccsd_t_total_hartree is None:
        return None
    return result


def _save_subpoint_checkpoint(
    path: Path,
    *,
    signature: str,
    hamiltonian: ScalarRelativityHamiltonian,
    result: Stage3PointResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "OPENEA_SCALAR_RELATIVITY_SUBPOINT_V1",
        "signature": signature,
        "hamiltonian": hamiltonian.value,
        "correlation_space": "ALL_ELECTRON",
        "result": result.to_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _execute_subpoint(
    *,
    state: ScalarRelativityStateSpec,
    basis: ScalarRelativityBasisSpec,
    hamiltonian: ScalarRelativityHamiltonian,
    base_settings: Stage3ExecutionSettings,
    point_runner: PointRunner,
    checkpoint_dir: Path | None,
) -> ScalarRelativitySubcalculation:
    request = _request_for(state, basis, hamiltonian)
    settings = _settings_for(base_settings, hamiltonian)
    signature = _checkpoint_signature(request, settings, hamiltonian)
    result = None
    reused = False
    path = None
    if checkpoint_dir is not None:
        path = _checkpoint_path(
            checkpoint_dir,
            cardinal=basis.cardinal,
            role=state.role,
            hamiltonian=hamiltonian,
        )
        result = _load_subpoint_checkpoint(
            path,
            expected_signature=signature,
            request=request,
        )
        reused = result is not None

    if result is None:
        result = point_runner(request, settings)
        if not isinstance(result, Stage3PointResult):
            raise TypeError("Scalar-relativity point runner must return Stage3PointResult")
        if result.request_id != request.request_id or result.job_id != request.job_id:
            raise ValueError("Scalar-relativity point runner returned another request/job")
        if path is not None:
            _save_subpoint_checkpoint(
                path,
                signature=signature,
                hamiltonian=hamiltonian,
                result=result,
            )

    return ScalarRelativitySubcalculation(
        role=state.role,
        hamiltonian=hamiltonian,
        request=request,
        result=result,
        checkpoint_reused=reused,
    )


def _evaluate_cardinal(
    *,
    neutral: ScalarRelativityStateSpec,
    anion: ScalarRelativityStateSpec,
    basis: ScalarRelativityBasisSpec,
    base_settings: Stage3ExecutionSettings,
    point_runner: PointRunner,
    checkpoint_dir: Path | None,
) -> ScalarRelativityCardinalEvidence:
    calculations: list[ScalarRelativitySubcalculation] = []
    totals: dict[tuple[str, ScalarRelativityHamiltonian], float] = {}

    for state in (neutral, anion):
        for hamiltonian in (
            ScalarRelativityHamiltonian.NONRELATIVISTIC,
            ScalarRelativityHamiltonian.SFX2C1E,
        ):
            calc = _execute_subpoint(
                state=state,
                basis=basis,
                hamiltonian=hamiltonian,
                base_settings=base_settings,
                point_runner=point_runner,
                checkpoint_dir=checkpoint_dir,
            )
            calculations.append(calc)
            result = calc.result
            if result.status is not PointExecutionStatus.COMPLETED:
                raise RuntimeError(
                    f"Scalar-relativity subcalculation failed: X={basis.cardinal} "
                    f"role={state.role} hamiltonian={hamiltonian.value} "
                    f"status={result.status.value} error={result.error_type}:{result.error_message}"
                )
            if result.ccsd_t_total_hartree is None:
                raise RuntimeError(
                    f"Scalar-relativity subcalculation has no CCSD(T) total: X={basis.cardinal} "
                    f"role={state.role} hamiltonian={hamiltonian.value}"
                )
            if result.basis != basis.label:
                raise RuntimeError("Scalar-relativity result basis label differs from authorized basis")
            if abs(float(result.r_angstrom) - float(state.r_angstrom)) > 1.0e-10:
                raise RuntimeError("Scalar-relativity result geometry differs from authorized state geometry")
            if result.charge != state.charge or result.spin_2s != state.spin_2s:
                raise RuntimeError("Scalar-relativity result charge/spin sector differs from authorized state")
            totals[(state.role, hamiltonian)] = float(result.ccsd_t_total_hartree)

    ea_nr = (
        totals[("neutral", ScalarRelativityHamiltonian.NONRELATIVISTIC)]
        - totals[("anion", ScalarRelativityHamiltonian.NONRELATIVISTIC)]
    ) * HARTREE_TO_EV
    ea_x2c = (
        totals[("neutral", ScalarRelativityHamiltonian.SFX2C1E)]
        - totals[("anion", ScalarRelativityHamiltonian.SFX2C1E)]
    ) * HARTREE_TO_EV
    point = ScalarRelativityPoint(
        cardinal=basis.cardinal,
        basis=basis.label,
        ea_nr_ev=ea_nr,
        ea_sfx2c1e_ev=ea_x2c,
        delta_sr_ev=ea_x2c - ea_nr,
    )
    return ScalarRelativityCardinalEvidence(
        cardinal=basis.cardinal,
        basis=basis,
        point=point,
        calculations=tuple(calculations),
    )


def run_adaptive_scalar_relativity_series(
    *,
    neutral: ScalarRelativityStateSpec,
    anion: ScalarRelativityStateSpec,
    basis_specs: tuple[ScalarRelativityBasisSpec, ...],
    initial_cardinals: tuple[int, ...] = (3, 4),
    maximum_cardinal: int = 5,
    target_change_ev: float = 0.0005,
    execution_settings: Stage3ExecutionSettings | None = None,
    checkpoint_dir: Path | str | None = None,
    point_runner: PointRunner | None = None,
    basis_validator: BasisValidator | None = None,
) -> AdaptiveScalarRelativityResult:
    """Run a matched, adaptive, checkpointable NR/SFX2C1E CCSD(T) series.

    The default backend uses the generic PySCF Stage-3 point executor.  Tests or
    other backends may inject ``point_runner``.  ``basis_validator`` can be
    injected for non-PySCF tests; in production the default preflights every
    explicitly supplied relativistic basis before any expensive QC point runs.
    """

    _validate_state_pair(neutral, anion)
    by_x = _basis_map(basis_specs, neutral.atoms)
    if len(initial_cardinals) < 2:
        raise ValueError("Scalar-relativity convergence requires at least two initial cardinals")
    if tuple(sorted(initial_cardinals)) != initial_cardinals or len(set(initial_cardinals)) != len(initial_cardinals):
        raise ValueError("initial_cardinals must be sorted and unique")
    if maximum_cardinal < max(initial_cardinals):
        raise ValueError("maximum_cardinal is below the initial scalar-relativity series")
    missing_initial = [x for x in initial_cardinals if x not in by_x]
    if missing_initial:
        raise ValueError(
            f"Missing explicit scalar-relativity basis policy for initial cardinals {missing_initial}"
        )
    if not isfinite(float(target_change_ev)) or target_change_ev <= 0.0:
        raise ValueError("target_change_ev must be finite and positive")

    settings = execution_settings or Stage3ExecutionSettings()
    # Validate method-separation invariants before basis loading or expensive work.
    _settings_for(settings, ScalarRelativityHamiltonian.NONRELATIVISTIC)
    _settings_for(settings, ScalarRelativityHamiltonian.SFX2C1E)

    validator = basis_validator or _default_basis_validator
    try:
        _preflight_basis_policy(
            tuple(spec for spec in basis_specs if spec.cardinal <= maximum_cardinal),
            validator,
        )
    except Exception as exc:
        return AdaptiveScalarRelativityResult(
            status=AdaptiveScalarRelativityStatus.POLICY_BLOCKED,
            evidence=(),
            assessment=None,
            execution_error_type=type(exc).__name__,
            execution_error_message=str(exc),
        )

    runner = point_runner or (lambda req, cfg: run_stage3_point(req, settings=cfg))
    cp_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    evidence: list[ScalarRelativityCardinalEvidence] = []

    def execute_x(x: int) -> Exception | None:
        spec = by_x.get(x)
        if spec is None:
            return ValueError(
                f"No explicit scalar-relativity basis policy is available for requested X={x}"
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
            return AdaptiveScalarRelativityResult(
                status=AdaptiveScalarRelativityStatus.EXECUTION_BLOCKED,
                evidence=tuple(evidence),
                assessment=(
                    None
                    if not evidence
                    else assess_scalar_relativity(
                        list(e.point for e in evidence),
                        target_change_ev=target_change_ev,
                        max_cardinal=maximum_cardinal,
                    )
                ),
                execution_error_type=type(error).__name__,
                execution_error_message=str(error),
            )

    while True:
        assessment = assess_scalar_relativity(
            list(e.point for e in evidence),
            target_change_ev=target_change_ev,
            max_cardinal=maximum_cardinal,
        )
        if assessment.status == "CLEARED":
            return AdaptiveScalarRelativityResult(
                AdaptiveScalarRelativityStatus.SCALAR_RELATIVITY_CLEARED,
                tuple(evidence),
                assessment,
            )
        if assessment.status == "UNRESOLVED":
            return AdaptiveScalarRelativityResult(
                AdaptiveScalarRelativityStatus.SCALAR_RELATIVITY_UNRESOLVED,
                tuple(evidence),
                assessment,
            )
        if not assessment.action.startswith("COMPUTE_X"):
            return AdaptiveScalarRelativityResult(
                AdaptiveScalarRelativityStatus.POLICY_BLOCKED,
                tuple(evidence),
                assessment,
                "UnexpectedScalarRelativityAction",
                assessment.action,
            )
        next_x = int(assessment.action.removeprefix("COMPUTE_X"))
        if next_x > maximum_cardinal:
            return AdaptiveScalarRelativityResult(
                AdaptiveScalarRelativityStatus.SCALAR_RELATIVITY_UNRESOLVED,
                tuple(evidence),
                assessment,
                "MaximumScalarRelativityCardinalReached",
                f"Assessment requested X={next_x} beyond configured maximum X={maximum_cardinal}",
            )
        if next_x not in by_x:
            return AdaptiveScalarRelativityResult(
                AdaptiveScalarRelativityStatus.POLICY_BLOCKED,
                tuple(evidence),
                assessment,
                "MissingScalarRelativityBasisPolicy",
                f"No explicit scalar-relativity basis policy is available for requested X={next_x}",
            )
        error = execute_x(next_x)
        if error is not None:
            return AdaptiveScalarRelativityResult(
                AdaptiveScalarRelativityStatus.EXECUTION_BLOCKED,
                tuple(evidence),
                assessment,
                type(error).__name__,
                str(error),
            )
