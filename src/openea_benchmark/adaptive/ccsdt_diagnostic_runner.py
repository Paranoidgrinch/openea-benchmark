"""Generic CCSDT triples-reliability diagnostic runner for OpenEA v1.

This module turns an *already authorized* post-CCSD(T) diagnostic into matched
CCSD(T)/CCSDT calculations for the resolved neutral and anion states.

Scientific role
---------------
The runner evaluates

    Delta_T3(X) = EA_CCSDT(X) - EA_CCSD(T)(X)

at fixed, already validated state geometries.  It is diagnostic evidence for
the reliability of perturbative triples; it is not a default production rung
and it never escalates to CCSDTQ or higher coupled-cluster rank.

Execution contract
------------------
* neutral/anion state identity must already be validated;
* the diagnostic must be explicitly authorized by upstream planning;
* all calculations use an all-electron orbital basis but an explicitly bound
  frozen-core *correlation* space;
* neutral and anion use the same declared basis family at each cardinal;
* CCSD(T) and CCSDT for a given state/cardinal use matched geometry, basis,
  charge, spin, frozen-core count, nonrelativistic Hamiltonian, C1 metadata,
  SCF settings and source-checkpoint initialization policy;
* C1 is metadata for the CCpy/PySCF bridge, not an attempt to exploit spatial
  symmetry;
* basis policies and frozen-core counts are supplied explicitly and are never
  guessed from element symbols;
* checkpoints are signature-validated before reuse;
* the result is passed to :func:`assess_post_ccsd_t`, which may authorize a
  small/stable optional Delta_T3 correction or trigger reference-character
  reassessment.  This module never requests CCSDTQ.

CCpy is imported lazily so the rest of OpenEA remains importable when the
optional post-CC backend is absent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import importlib.metadata
import json
import math
from math import isfinite
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV
from openea_benchmark.attachment.post_ccsd_t import PostCCAssessment, PostCCPoint, assess_post_ccsd_t


class CCSDTDiagnosticMethod(str, Enum):
    CCSD_T = "CCSD(T)"
    CCSDT = "CCSDT"


class CCSDTMethodExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class AdaptiveCCSDTDiagnosticStatus(str, Enum):
    TRIPLES_RELIABILITY_CLEARED = "TRIPLES_RELIABILITY_CLEARED"
    TRIPLES_RELIABILITY_WARNING = "TRIPLES_RELIABILITY_WARNING"
    TRIPLES_RELIABILITY_UNRESOLVED = "TRIPLES_RELIABILITY_UNRESOLVED"
    EXECUTION_BLOCKED = "EXECUTION_BLOCKED"
    POLICY_BLOCKED = "POLICY_BLOCKED"


@dataclass(frozen=True)
class CCSDTDiagnosticAuthorization:
    """Explicit upstream permission to spend CCSDT resources.

    The runner intentionally does not infer authorization merely because CCpy
    is installed.  An upstream planner/reference-character decision must bind
    provenance for why this diagnostic is scientifically justified.
    """

    authorized: bool
    reason: str
    evidence_ids: tuple[str, ...] = ()
    authorized_cardinals: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("CCSDT diagnostic authorization requires a reason")
        if self.authorized and not self.evidence_ids:
            raise ValueError("Authorized CCSDT diagnostic requires provenance evidence IDs")
        if self.authorized and not self.authorized_cardinals:
            raise ValueError("Authorized CCSDT diagnostic requires explicit authorized_cardinals")
        if any(x < 2 for x in self.authorized_cardinals):
            raise ValueError("Authorized CCSDT diagnostic cardinals must be >= 2")
        if tuple(sorted(set(self.authorized_cardinals))) != self.authorized_cardinals:
            raise ValueError("authorized_cardinals must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CCSDTDiagnosticStateSpec:
    """One already resolved state entering the triples diagnostic."""

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
    nfrozen_spatial_orbitals: int

    def __post_init__(self) -> None:
        if self.role not in {"neutral", "anion"}:
            raise ValueError("CCSDT diagnostic state role must be neutral or anion")
        if not self.system.strip() or not self.state_id.strip() or not self.source_root_id.strip():
            raise ValueError("CCSDT diagnostic state requires system/state/source provenance")
        if len(self.atoms) != 2 or any(not str(x).strip() for x in self.atoms):
            raise ValueError("CCSDT diagnostic v1 runner requires exactly two atoms")
        if self.spin_2s < 0:
            raise ValueError("spin_2s must be non-negative")
        if not isfinite(float(self.r_angstrom)) or float(self.r_angstrom) <= 0.0:
            raise ValueError("CCSDT diagnostic geometry must be finite and positive")
        if not str(self.source_checkpoint_path).strip():
            raise ValueError("CCSDT diagnostic state requires a source checkpoint path")
        if not self.state_identity_validated:
            raise ValueError(
                "CCSDT diagnostic requires state identity to be validated before post-CC work"
            )
        if self.nfrozen_spatial_orbitals < 0:
            raise ValueError("nfrozen_spatial_orbitals must be non-negative")

    @property
    def scf_reference(self) -> str:
        return "RHF" if self.spin_2s == 0 else "ROHF"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CCSDTDiagnosticBasisSpec:
    """Explicit all-electron orbital-basis assignment for one cardinal."""

    cardinal: int
    family_id: str
    label: str
    basis_by_element: Mapping[str, str]
    electron_model: str = "ALL_ELECTRON"

    def __post_init__(self) -> None:
        if self.cardinal < 2:
            raise ValueError("CCSDT diagnostic cardinal must be >= 2")
        if not self.family_id.strip() or not self.label.strip():
            raise ValueError("CCSDT diagnostic basis specification requires family_id and label")
        if self.electron_model != "ALL_ELECTRON":
            raise ValueError(
                "Generic OpenEA-v1 CCSDT diagnostic currently supports only explicit all-electron orbital-basis policies; ECP/core-replacement post-CC diagnostics require a dedicated validated treatment"
            )
        if not self.basis_by_element:
            raise ValueError("CCSDT diagnostic basis_by_element must not be empty")
        if any(not str(k).strip() or not str(v).strip() for k, v in self.basis_by_element.items()):
            raise ValueError("CCSDT diagnostic basis_by_element contains an empty element or basis name")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["basis_by_element"] = dict(self.basis_by_element)
        return d


@dataclass(frozen=True)
class CCSDTDiagnosticExecutionSettings:
    scf_conv_tol: float = 1.0e-10
    scf_conv_tol_grad: float = 1.0e-7
    scf_max_cycle: int = 150
    cc_energy_convergence: float = 1.0e-8
    cc_amp_convergence: float = 1.0e-8
    cc_max_iterations: int = 100
    max_memory_mb: int = 12000
    checkpoint_project: bool = True
    require_internal_stability: bool = True
    require_rhf_external_stability: bool = True
    reference_energy_match_tolerance_hartree: float = 1.0e-10
    symmetry_group: str = "C1"
    scalar_relativistic: str = "NONE"

    def __post_init__(self) -> None:
        for name in (
            "scf_conv_tol",
            "scf_conv_tol_grad",
            "cc_energy_convergence",
            "cc_amp_convergence",
            "reference_energy_match_tolerance_hartree",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.scf_max_cycle <= 0 or self.cc_max_iterations <= 0:
            raise ValueError("SCF/CC iteration limits must be positive")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.symmetry_group.upper() != "C1":
            raise ValueError("Generic CCpy bridge is deliberately restricted to explicit C1 metadata")
        if self.scalar_relativistic != "NONE":
            raise ValueError(
                "CCSDT triples diagnostic is a nonrelativistic frozen-core valence-correlation diagnostic; scalar relativity is a separate correction"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CCSDTMethodRequest:
    request_id: str
    role: str
    system: str
    atoms: tuple[str, str]
    charge: int
    spin_2s: int
    state_id: str
    r_angstrom: float
    source_root_id: str
    source_checkpoint_path: str
    cardinal: int
    basis_label: str
    basis_by_element: Mapping[str, str]
    family_id: str
    method: CCSDTDiagnosticMethod
    nfrozen_spatial_orbitals: int
    scf_reference: str
    symmetry_group: str = "C1"
    correlation_space: str = "FROZEN_CORE_VALENCE"
    hamiltonian: str = "NONRELATIVISTIC"

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("CCSDT method request requires request_id")
        if self.role not in {"neutral", "anion"}:
            raise ValueError("CCSDT method request role must be neutral or anion")
        if self.scf_reference not in {"RHF", "ROHF"}:
            raise ValueError("CCSDT method request supports RHF or ROHF references")
        expected_reference = "RHF" if self.spin_2s == 0 else "ROHF"
        if self.scf_reference != expected_reference:
            raise ValueError(
                f"spin_2s={self.spin_2s} requires {expected_reference}, not {self.scf_reference}"
            )
        if self.symmetry_group.upper() != "C1":
            raise ValueError("CCSDT method request requires explicit C1 metadata")
        if self.correlation_space != "FROZEN_CORE_VALENCE":
            raise ValueError("CCSDT diagnostic request must use frozen-core valence correlation")
        if self.hamiltonian != "NONRELATIVISTIC":
            raise ValueError("CCSDT diagnostic request must remain nonrelativistic")
        if self.nfrozen_spatial_orbitals < 0:
            raise ValueError("nfrozen_spatial_orbitals must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["basis_by_element"] = dict(self.basis_by_element)
        d["method"] = self.method.value
        return d


@dataclass(frozen=True)
class CCSDTMethodResult:
    request_id: str
    status: CCSDTMethodExecutionStatus
    method: CCSDTDiagnosticMethod
    reference_energy_hartree: float | None
    correlation_energy_hartree: float | None
    total_energy_hartree: float | None
    ccpy_version: str | None
    scf_converged: bool
    internal_stable: bool | None
    external_stable: bool | None
    external_stability_available: bool
    symmetry_group: str
    nfrozen_spatial_orbitals: int
    error_type: str | None = None
    error_message: str | None = None
    is_production_ea: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if self.symmetry_group.upper() != "C1":
            raise ValueError("CCSDT method result must retain explicit C1 provenance")
        if self.nfrozen_spatial_orbitals < 0:
            raise ValueError("nfrozen_spatial_orbitals must be non-negative")
        if self.is_production_ea or self.authorizes_pruning:
            raise ValueError("CCSDT diagnostic subcalculation cannot be a production EA or prune states")
        if self.status is CCSDTMethodExecutionStatus.COMPLETED:
            for name in (
                "reference_energy_hartree",
                "correlation_energy_hartree",
                "total_energy_hartree",
            ):
                value = getattr(self, name)
                if value is None or not isfinite(float(value)):
                    raise ValueError(f"Completed CCSDT method result requires finite {name}")
            if not self.scf_converged:
                raise ValueError("Completed CCSDT method result requires converged SCF")
        if self.status is CCSDTMethodExecutionStatus.ERROR and not (self.error_type or self.error_message):
            raise ValueError("Error CCSDT method result requires error provenance")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["method"] = self.method.value
        return data


@dataclass(frozen=True)
class CCSDTSubcalculation:
    request: CCSDTMethodRequest
    result: CCSDTMethodResult
    checkpoint_reused: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "result": self.result.to_dict(),
            "checkpoint_reused": self.checkpoint_reused,
        }


@dataclass(frozen=True)
class CCSDTCardinalEvidence:
    cardinal: int
    basis: CCSDTDiagnosticBasisSpec
    point: PostCCPoint
    calculations: tuple[CCSDTSubcalculation, ...]

    def __post_init__(self) -> None:
        if self.cardinal != self.basis.cardinal or self.cardinal != self.point.cardinal:
            raise ValueError("CCSDT cardinal evidence is internally inconsistent")
        keys = {(x.request.role, x.request.method.value) for x in self.calculations}
        expected = {
            ("neutral", CCSDTDiagnosticMethod.CCSD_T.value),
            ("neutral", CCSDTDiagnosticMethod.CCSDT.value),
            ("anion", CCSDTDiagnosticMethod.CCSD_T.value),
            ("anion", CCSDTDiagnosticMethod.CCSDT.value),
        }
        if keys != expected:
            raise ValueError("CCSDT cardinal evidence requires neutral/anion x CCSD(T)/CCSDT")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cardinal": self.cardinal,
            "basis": self.basis.to_dict(),
            "point": self.point.to_dict(),
            "calculations": [x.to_dict() for x in self.calculations],
        }


@dataclass(frozen=True)
class AdaptiveCCSDTDiagnosticResult:
    status: AdaptiveCCSDTDiagnosticStatus
    evidence: tuple[CCSDTCardinalEvidence, ...]
    assessment: PostCCAssessment | None
    authorization: CCSDTDiagnosticAuthorization
    execution_error_type: str | None = None
    execution_error_message: str | None = None
    method_role: str = "DIAGNOSTIC"
    correlation_space: str = "FROZEN_CORE_VALENCE"
    hamiltonian: str = "NONRELATIVISTIC"
    includes_scalar_relativity: bool = False
    includes_soc: bool = False
    is_production_ea: bool = False
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if self.method_role != "DIAGNOSTIC":
            raise ValueError("CCSDT runner has a fixed DIAGNOSTIC role")
        if self.correlation_space != "FROZEN_CORE_VALENCE":
            raise ValueError("CCSDT runner has a fixed frozen-core valence-correlation scope")
        if self.hamiltonian != "NONRELATIVISTIC":
            raise ValueError("CCSDT runner has a fixed nonrelativistic scope")
        if self.includes_scalar_relativity or self.includes_soc:
            raise ValueError("CCSDT diagnostic must not absorb relativistic or SOC corrections")
        if self.is_production_ea or self.authorizes_pruning:
            raise ValueError("CCSDT diagnostic cannot be a complete production EA or prune states")

    @property
    def points(self) -> tuple[PostCCPoint, ...]:
        return tuple(x.point for x in self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "evidence": [x.to_dict() for x in self.evidence],
            "assessment": None if self.assessment is None else self.assessment.to_dict(),
            "authorization": self.authorization.to_dict(),
            "execution_error_type": self.execution_error_type,
            "execution_error_message": self.execution_error_message,
            "method_role": self.method_role,
            "correlation_space": self.correlation_space,
            "hamiltonian": self.hamiltonian,
            "includes_scalar_relativity": self.includes_scalar_relativity,
            "includes_soc": self.includes_soc,
            "is_production_ea": self.is_production_ea,
            "authorizes_pruning": self.authorizes_pruning,
        }


MethodRunner = Callable[[CCSDTMethodRequest, CCSDTDiagnosticExecutionSettings], CCSDTMethodResult]
BasisValidator = Callable[[str, str], None]


def _validate_state_pair(neutral: CCSDTDiagnosticStateSpec, anion: CCSDTDiagnosticStateSpec) -> None:
    if neutral.role != "neutral" or anion.role != "anion":
        raise ValueError("CCSDT diagnostic requires neutral then anion state specifications")
    if neutral.system != anion.system:
        raise ValueError("Neutral and anion must belong to the same molecular system")
    if neutral.atoms != anion.atoms:
        raise ValueError("Neutral and anion atom ordering must match")
    if anion.charge != neutral.charge - 1:
        raise ValueError("Anion charge must equal neutral charge minus one electron")
    if neutral.nfrozen_spatial_orbitals != anion.nfrozen_spatial_orbitals:
        raise ValueError(
            "Neutral and anion must use the same explicitly declared frozen-core orbital count for Delta_T3"
        )


def _basis_map(
    basis_specs: tuple[CCSDTDiagnosticBasisSpec, ...], atoms: tuple[str, str]
) -> dict[int, CCSDTDiagnosticBasisSpec]:
    by_x: dict[int, CCSDTDiagnosticBasisSpec] = {}
    expected_elements = set(atoms)
    for spec in basis_specs:
        if spec.cardinal in by_x:
            raise ValueError(f"Duplicate CCSDT diagnostic basis specification X={spec.cardinal}")
        if set(spec.basis_by_element) != expected_elements:
            raise ValueError(
                "CCSDT diagnostic basis specification must map exactly the molecular elements: "
                f"X={spec.cardinal} expected={sorted(expected_elements)} "
                f"provided={sorted(spec.basis_by_element)}"
            )
        by_x[spec.cardinal] = spec
    if not by_x:
        raise ValueError("At least one CCSDT diagnostic basis specification is required")
    family_ids = {x.family_id for x in by_x.values()}
    if len(family_ids) != 1:
        raise ValueError(
            "All cardinal points in one CCSDT diagnostic series must use one declared basis family"
        )
    return by_x


def _default_basis_validator(element: str, basis_name: str) -> None:
    from pyscf import gto

    gto.basis.load(basis_name, element)


def _preflight_basis_policy(
    basis_specs: tuple[CCSDTDiagnosticBasisSpec, ...], validator: BasisValidator
) -> None:
    for spec in basis_specs:
        for element, basis_name in spec.basis_by_element.items():
            try:
                validator(element, basis_name)
            except Exception as exc:
                raise RuntimeError(
                    "CCSDT diagnostic basis preflight failed before QC execution: "
                    f"X={spec.cardinal} element={element} basis={basis_name}"
                ) from exc


def _safe_token(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def _method_token(method: CCSDTDiagnosticMethod) -> str:
    if method is CCSDTDiagnosticMethod.CCSD_T:
        return "CCSD_pT"
    if method is CCSDTDiagnosticMethod.CCSDT:
        return "CCSDT"
    raise ValueError(f"Unsupported CCSDT diagnostic method: {method}")


def _request_for(
    state: CCSDTDiagnosticStateSpec,
    basis: CCSDTDiagnosticBasisSpec,
    method: CCSDTDiagnosticMethod,
) -> CCSDTMethodRequest:
    token = _safe_token(state.state_id)
    method_token = _method_token(method)
    request_id = (
        f"ccsdt_diagnostic__{_safe_token(state.system)}__{state.role}__{token}"
        f"__X{basis.cardinal}__{method_token}"
    )
    return CCSDTMethodRequest(
        request_id=request_id,
        role=state.role,
        system=state.system,
        atoms=state.atoms,
        charge=state.charge,
        spin_2s=state.spin_2s,
        state_id=state.state_id,
        r_angstrom=float(state.r_angstrom),
        source_root_id=state.source_root_id,
        source_checkpoint_path=state.source_checkpoint_path,
        cardinal=basis.cardinal,
        basis_label=basis.label,
        basis_by_element=dict(basis.basis_by_element),
        family_id=basis.family_id,
        method=method,
        nfrozen_spatial_orbitals=state.nfrozen_spatial_orbitals,
        scf_reference=state.scf_reference,
    )


def _checkpoint_signature(
    request: CCSDTMethodRequest,
    settings: CCSDTDiagnosticExecutionSettings,
) -> str:
    payload = {
        "schema": "OPENEA_CCSDT_DIAGNOSTIC_SUBPOINT_V1",
        "request": request.to_dict(),
        "settings": settings.to_dict(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def _checkpoint_path(
    checkpoint_dir: Path,
    *,
    cardinal: int,
    role: str,
    method: CCSDTDiagnosticMethod,
) -> Path:
    return checkpoint_dir / f"X{cardinal}" / f"{role}__{_method_token(method)}.json"


def _result_from_dict(raw: Mapping[str, Any]) -> CCSDTMethodResult:
    data = dict(raw)
    data["status"] = CCSDTMethodExecutionStatus(str(data["status"]))
    data["method"] = CCSDTDiagnosticMethod(str(data["method"]))
    return CCSDTMethodResult(**data)


def _load_subpoint_checkpoint(
    path: Path,
    *,
    expected_signature: str,
    request: CCSDTMethodRequest,
) -> CCSDTMethodResult | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != expected_signature:
        raise ValueError(
            f"CCSDT diagnostic checkpoint signature mismatch for {path}; refusing stale/incompatible evidence"
        )
    result = _result_from_dict(payload["result"])
    if result.request_id != request.request_id or result.method is not request.method:
        raise ValueError("CCSDT diagnostic checkpoint belongs to another request/method")
    if result.status is not CCSDTMethodExecutionStatus.COMPLETED:
        return None
    return result


def _save_subpoint_checkpoint(
    path: Path,
    *,
    signature: str,
    request: CCSDTMethodRequest,
    result: CCSDTMethodResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "OPENEA_CCSDT_DIAGNOSTIC_SUBPOINT_V1",
        "signature": signature,
        "request": request.to_dict(),
        "result": result.to_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _ccpy_version() -> str:
    for name in ("coupled-cluster-py", "ccpy"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "UNKNOWN"


def _extract_ccsd_t_correction(driver: Any) -> float:
    raw = getattr(driver, "deltap3", None)
    if raw is None or len(raw) == 0:
        raise RuntimeError("CCpy CCSD(T) did not populate driver.deltap3")
    item = raw[0]
    if not isinstance(item, dict):
        raise RuntimeError(f"Unexpected CCpy deltap3 structure: {type(item)!r}")
    finite = {
        str(k): float(v)
        for k, v in item.items()
        if v is not None and math.isfinite(float(v))
    }
    if not finite:
        raise RuntimeError(f"No finite CCSD(T) correction in deltap3: {item!r}")
    if "A" in finite:
        return finite["A"]
    distinct = {round(v, 14) for v in finite.values()}
    if len(distinct) == 1 or len(finite) == 1:
        return next(iter(finite.values()))
    raise RuntimeError(
        "Ambiguous CCpy CCSD(T) deltap3 payload; refusing to guess: "
        f"{finite}"
    )


def _validate_c1_metadata(mol: Any) -> None:
    groupname = str(getattr(mol, "groupname", "")).upper()
    irrep_name = tuple(str(x) for x in getattr(mol, "irrep_name", ()))
    symm_orb = getattr(mol, "symm_orb", None)
    if groupname != "C1":
        raise RuntimeError(f"CCpy bridge requires explicit C1 metadata, got group={groupname!r}")
    if not irrep_name or any(name.upper() != "A" for name in irrep_name):
        raise RuntimeError(f"CCpy C1 metadata is incomplete/ambiguous: irrep_name={irrep_name!r}")
    if symm_orb is None:
        raise RuntimeError("CCpy C1 metadata is missing mol.symm_orb")


def _run_ccpy_method(
    request: CCSDTMethodRequest,
    settings: CCSDTDiagnosticExecutionSettings,
) -> CCSDTMethodResult:
    """Execute one matched CCSD(T) or CCSDT subpoint with PySCF + CCpy."""

    import pyscf
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver

    checkpoint = Path(request.source_checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {checkpoint}")

    version = _ccpy_version()
    if version == "UNKNOWN":
        raise RuntimeError("CCpy/coupled-cluster-py is not installed as a discoverable distribution")

    atom_spec = [
        (request.atoms[0], (0.0, 0.0, 0.0)),
        (request.atoms[1], (0.0, 0.0, float(request.r_angstrom))),
    ]
    mol = gto.M(
        atom=atom_spec,
        basis=dict(request.basis_by_element),
        charge=request.charge,
        spin=request.spin_2s,
        unit="Angstrom",
        symmetry="C1",
        cart=False,
        verbose=0,
        max_memory=settings.max_memory_mb,
    )
    _validate_c1_metadata(mol)

    if request.scf_reference == "RHF":
        mf = scf.RHF(mol)
        dm0 = scf.hf.init_guess_by_chkfile(
            mol,
            request.source_checkpoint_path,
            project=settings.checkpoint_project,
        )
    else:
        mf = scf.ROHF(mol)
        dm0 = scf.rohf.init_guess_by_chkfile(
            mol,
            request.source_checkpoint_path,
            project=settings.checkpoint_project,
        )

    mf.conv_tol = settings.scf_conv_tol
    mf.conv_tol_grad = settings.scf_conv_tol_grad
    mf.max_cycle = settings.scf_max_cycle
    mf.max_memory = settings.max_memory_mb
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError(
            f"SCF did not converge for {request.role} X={request.cardinal} {request.method.value}"
        )

    if request.scf_reference == "RHF":
        _, _, stable_i, stable_e = mf.stability(
            internal=True,
            external=settings.require_rhf_external_stability,
            return_status=True,
        )
        internal_stable = bool(stable_i)
        external_available = bool(settings.require_rhf_external_stability)
        external_stable = bool(stable_e) if external_available else None
    else:
        _, _, stable_i, _ = mf.stability(
            internal=True,
            external=False,
            return_status=True,
        )
        internal_stable = bool(stable_i)
        external_available = False
        external_stable = None

    if settings.require_internal_stability and not internal_stable:
        raise RuntimeError(
            f"SCF internal stability failed for {request.role} X={request.cardinal} {request.method.value}"
        )
    if (
        request.scf_reference == "RHF"
        and settings.require_rhf_external_stability
        and external_stable is False
    ):
        raise RuntimeError(
            f"RHF external stability failed for {request.role} X={request.cardinal} {request.method.value}"
        )

    driver = Driver.from_pyscf(mf, nfrozen=request.nfrozen_spatial_orbitals)
    driver.options["energy_convergence"] = settings.cc_energy_convergence
    driver.options["amp_convergence"] = settings.cc_amp_convergence
    driver.options["maximum_iterations"] = settings.cc_max_iterations

    if request.method is CCSDTDiagnosticMethod.CCSD_T:
        driver.run_cc(method="ccsd")
        ccsd_corr = float(driver.correlation_energy)
        driver.run_ccp3(method="ccsd(t)")
        triples = _extract_ccsd_t_correction(driver)
        correlation = ccsd_corr + triples
    elif request.method is CCSDTDiagnosticMethod.CCSDT:
        driver.run_cc(method="ccsdt")
        correlation = float(driver.correlation_energy)
    else:  # pragma: no cover - enum exhaustiveness
        raise ValueError(request.method)

    reference = float(mf.e_tot)
    total = reference + correlation
    if not all(isfinite(x) for x in (reference, correlation, total)):
        raise RuntimeError("CCpy returned a non-finite post-CC energy")

    return CCSDTMethodResult(
        request_id=request.request_id,
        status=CCSDTMethodExecutionStatus.COMPLETED,
        method=request.method,
        reference_energy_hartree=reference,
        correlation_energy_hartree=correlation,
        total_energy_hartree=total,
        ccpy_version=version,
        scf_converged=True,
        internal_stable=internal_stable,
        external_stable=external_stable,
        external_stability_available=external_available,
        symmetry_group="C1",
        nfrozen_spatial_orbitals=request.nfrozen_spatial_orbitals,
    )


def _execute_subpoint(
    *,
    state: CCSDTDiagnosticStateSpec,
    basis: CCSDTDiagnosticBasisSpec,
    method: CCSDTDiagnosticMethod,
    settings: CCSDTDiagnosticExecutionSettings,
    method_runner: MethodRunner,
    checkpoint_dir: Path | None,
    allow_compute: bool,
) -> CCSDTSubcalculation:
    request = _request_for(state, basis, method)
    signature = _checkpoint_signature(request, settings)
    result = None
    reused = False
    path = None
    if checkpoint_dir is not None:
        path = _checkpoint_path(
            checkpoint_dir,
            cardinal=basis.cardinal,
            role=state.role,
            method=method,
        )
        result = _load_subpoint_checkpoint(
            path,
            expected_signature=signature,
            request=request,
        )
        reused = result is not None

    if result is None:
        if not allow_compute:
            raise PermissionError(
                f"CCSDT diagnostic X={basis.cardinal} is not authorized for new computation and no compatible completed checkpoint exists"
            )
        try:
            result = method_runner(request, settings)
        except Exception as exc:
            result = CCSDTMethodResult(
                request_id=request.request_id,
                status=CCSDTMethodExecutionStatus.ERROR,
                method=request.method,
                reference_energy_hartree=None,
                correlation_energy_hartree=None,
                total_energy_hartree=None,
                ccpy_version=None,
                scf_converged=False,
                internal_stable=None,
                external_stable=None,
                external_stability_available=(request.scf_reference == "RHF"),
                symmetry_group="C1",
                nfrozen_spatial_orbitals=request.nfrozen_spatial_orbitals,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        if not isinstance(result, CCSDTMethodResult):
            raise TypeError("CCSDT method runner must return CCSDTMethodResult")
        if result.request_id != request.request_id or result.method is not request.method:
            raise ValueError("CCSDT method runner returned another request/method")
        if path is not None:
            _save_subpoint_checkpoint(
                path,
                signature=signature,
                request=request,
                result=result,
            )

    return CCSDTSubcalculation(
        request=request,
        result=result,
        checkpoint_reused=reused,
    )


def _evaluate_cardinal(
    *,
    neutral: CCSDTDiagnosticStateSpec,
    anion: CCSDTDiagnosticStateSpec,
    basis: CCSDTDiagnosticBasisSpec,
    settings: CCSDTDiagnosticExecutionSettings,
    method_runner: MethodRunner,
    checkpoint_dir: Path | None,
    allow_compute: bool,
) -> CCSDTCardinalEvidence:
    calculations: list[CCSDTSubcalculation] = []
    totals: dict[tuple[str, CCSDTDiagnosticMethod], float] = {}
    refs: dict[tuple[str, CCSDTDiagnosticMethod], float] = {}

    for state in (neutral, anion):
        for method in (CCSDTDiagnosticMethod.CCSD_T, CCSDTDiagnosticMethod.CCSDT):
            calc = _execute_subpoint(
                state=state,
                basis=basis,
                method=method,
                settings=settings,
                method_runner=method_runner,
                checkpoint_dir=checkpoint_dir,
                allow_compute=allow_compute,
            )
            calculations.append(calc)
            result = calc.result
            if result.status is not CCSDTMethodExecutionStatus.COMPLETED:
                raise RuntimeError(
                    "CCSDT diagnostic subcalculation failed: "
                    f"X={basis.cardinal} role={state.role} method={method.value} "
                    f"error={result.error_type}:{result.error_message}"
                )
            if result.total_energy_hartree is None or result.reference_energy_hartree is None:
                raise RuntimeError("Completed CCSDT diagnostic subcalculation lacks total/reference energy")
            if result.nfrozen_spatial_orbitals != state.nfrozen_spatial_orbitals:
                raise RuntimeError("CCSDT diagnostic result changed the authorized frozen-core definition")
            if result.symmetry_group.upper() != "C1":
                raise RuntimeError("CCSDT diagnostic result lost the explicit C1 bridge metadata")
            totals[(state.role, method)] = float(result.total_energy_hartree)
            refs[(state.role, method)] = float(result.reference_energy_hartree)

    for role in ("neutral", "anion"):
        ref_delta = abs(
            refs[(role, CCSDTDiagnosticMethod.CCSD_T)]
            - refs[(role, CCSDTDiagnosticMethod.CCSDT)]
        )
        if ref_delta > settings.reference_energy_match_tolerance_hartree:
            raise RuntimeError(
                "CCSD(T) and CCSDT did not use a matched SCF reference energy: "
                f"X={basis.cardinal} role={role} |dE_ref|={ref_delta:.3e} Eh"
            )

    ea_ccsd_t = (
        totals[("neutral", CCSDTDiagnosticMethod.CCSD_T)]
        - totals[("anion", CCSDTDiagnosticMethod.CCSD_T)]
    ) * HARTREE_TO_EV
    ea_ccsdt = (
        totals[("neutral", CCSDTDiagnosticMethod.CCSDT)]
        - totals[("anion", CCSDTDiagnosticMethod.CCSDT)]
    ) * HARTREE_TO_EV
    point = PostCCPoint(
        cardinal=basis.cardinal,
        basis=basis.label,
        ea_ccsd_t_ev=ea_ccsd_t,
        ea_ccsdt_ev=ea_ccsdt,
        delta_t3_ev=ea_ccsdt - ea_ccsd_t,
        ea_ccsdtq_ev=None,
        delta_t4_ev=None,
    )
    return CCSDTCardinalEvidence(
        cardinal=basis.cardinal,
        basis=basis,
        point=point,
        calculations=tuple(calculations),
    )


def run_adaptive_ccsdt_diagnostic_series(
    *,
    authorization: CCSDTDiagnosticAuthorization,
    neutral: CCSDTDiagnosticStateSpec,
    anion: CCSDTDiagnosticStateSpec,
    basis_specs: tuple[CCSDTDiagnosticBasisSpec, ...],
    initial_cardinals: tuple[int, ...] = (2, 3),
    maximum_cardinal: int = 3,
    triples_target_ev: float = 0.001,
    execution_settings: CCSDTDiagnosticExecutionSettings | None = None,
    checkpoint_dir: Path | str | None = None,
    method_runner: MethodRunner | None = None,
    basis_validator: BasisValidator | None = None,
) -> AdaptiveCCSDTDiagnosticResult:
    """Run an explicitly authorized, adaptive CCSDT triples diagnostic.

    The default backend is PySCF + CCpy.  Tests or future validated backends may
    inject ``method_runner``.  The function never runs CCSDTQ and never decides
    on its own that CCSDT is worth the cost; that decision is represented by
    ``authorization``.
    """

    _validate_state_pair(neutral, anion)
    by_x = _basis_map(basis_specs, neutral.atoms)
    if not authorization.authorized:
        return AdaptiveCCSDTDiagnosticResult(
            status=AdaptiveCCSDTDiagnosticStatus.POLICY_BLOCKED,
            evidence=(),
            assessment=None,
            authorization=authorization,
            execution_error_type="CCSDTDiagnosticNotAuthorized",
            execution_error_message=authorization.reason,
        )
    if not initial_cardinals:
        raise ValueError("CCSDT diagnostic requires at least one initial cardinal")
    if tuple(sorted(initial_cardinals)) != initial_cardinals or len(set(initial_cardinals)) != len(initial_cardinals):
        raise ValueError("initial_cardinals must be sorted and unique")
    if maximum_cardinal < max(initial_cardinals):
        raise ValueError("maximum_cardinal is below the initial CCSDT diagnostic series")
    missing_initial = [x for x in initial_cardinals if x not in by_x]
    if missing_initial:
        raise ValueError(
            f"Missing explicit CCSDT diagnostic basis policy for initial cardinals {missing_initial}"
        )
    if not isfinite(float(triples_target_ev)) or triples_target_ev <= 0.0:
        raise ValueError("triples_target_ev must be finite and positive")

    settings = execution_settings or CCSDTDiagnosticExecutionSettings()
    validator = basis_validator or _default_basis_validator
    try:
        _preflight_basis_policy(
            tuple(spec for spec in basis_specs if spec.cardinal <= maximum_cardinal),
            validator,
        )
    except Exception as exc:
        return AdaptiveCCSDTDiagnosticResult(
            status=AdaptiveCCSDTDiagnosticStatus.POLICY_BLOCKED,
            evidence=(),
            assessment=None,
            authorization=authorization,
            execution_error_type=type(exc).__name__,
            execution_error_message=str(exc),
        )

    runner = method_runner or _run_ccpy_method
    cp_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    evidence: list[CCSDTCardinalEvidence] = []

    def execute_x(x: int) -> Exception | None:
        spec = by_x.get(x)
        if spec is None:
            return ValueError(
                f"No explicit CCSDT diagnostic basis policy is available for requested X={x}"
            )
        try:
            ev = _evaluate_cardinal(
                neutral=neutral,
                anion=anion,
                basis=spec,
                settings=settings,
                method_runner=runner,
                checkpoint_dir=cp_dir,
                allow_compute=(x in authorization.authorized_cardinals),
            )
            evidence.append(ev)
            return None
        except Exception as exc:
            return exc

    for x in initial_cardinals:
        error = execute_x(x)
        if error is not None:
            partial = None
            if evidence:
                partial = assess_post_ccsd_t(
                    list(e.point for e in evidence),
                    triples_target_ev=triples_target_ev,
                )
            return AdaptiveCCSDTDiagnosticResult(
                status=AdaptiveCCSDTDiagnosticStatus.EXECUTION_BLOCKED,
                evidence=tuple(evidence),
                assessment=partial,
                authorization=authorization,
                execution_error_type=type(error).__name__,
                execution_error_message=str(error),
            )

    while True:
        assessment = assess_post_ccsd_t(
            list(e.point for e in evidence),
            triples_target_ev=triples_target_ev,
        )
        if assessment.status == "CLEARED":
            return AdaptiveCCSDTDiagnosticResult(
                AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_CLEARED,
                tuple(evidence),
                assessment,
                authorization,
            )
        if assessment.status == "POST_CC_WARNING":
            return AdaptiveCCSDTDiagnosticResult(
                AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_WARNING,
                tuple(evidence),
                assessment,
                authorization,
            )
        if assessment.status != "NEED_MORE_EVIDENCE" or not assessment.action.startswith("COMPUTE_T3_X"):
            return AdaptiveCCSDTDiagnosticResult(
                AdaptiveCCSDTDiagnosticStatus.POLICY_BLOCKED,
                tuple(evidence),
                assessment,
                authorization,
                "UnexpectedPostCCAction",
                assessment.action,
            )

        next_x = int(assessment.action.removeprefix("COMPUTE_T3_X"))
        if next_x > maximum_cardinal:
            return AdaptiveCCSDTDiagnosticResult(
                AdaptiveCCSDTDiagnosticStatus.TRIPLES_RELIABILITY_UNRESOLVED,
                tuple(evidence),
                assessment,
                authorization,
                "MaximumCCSDTDiagnosticCardinalReached",
                f"Diagnostic requested X={next_x} but maximum_cardinal={maximum_cardinal}",
            )
        if next_x in {e.cardinal for e in evidence}:
            return AdaptiveCCSDTDiagnosticResult(
                AdaptiveCCSDTDiagnosticStatus.POLICY_BLOCKED,
                tuple(evidence),
                assessment,
                authorization,
                "RepeatedCCSDTDiagnosticAction",
                assessment.action,
            )
        error = execute_x(next_x)
        if error is not None:
            return AdaptiveCCSDTDiagnosticResult(
                AdaptiveCCSDTDiagnosticStatus.EXECUTION_BLOCKED,
                tuple(evidence),
                assessment,
                authorization,
                type(error).__name__,
                str(error),
            )
