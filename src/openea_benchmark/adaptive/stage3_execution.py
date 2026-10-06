"""Auditable single-point execution for adaptively authorized Stage-3 jobs.

This module is deliberately narrower than a full high-level PEC engine.  It
turns *authorized* jobs emitted by the existing Stage-3 planner into explicit
single-geometry/single-initialization requests and can execute one request with
PySCF.  It does not choose electronic states, prune candidates, fit PECs, pick
ground states, or evaluate an electron affinity.

Scientific invariants
---------------------
* Only explicitly authorized Stage-3 job IDs can become execution requests.
* Every DFT initialization retained by Stage-3 provenance is retained as an
  independent high-level reference check; none is selected arbitrarily.
* A Stage-2 checkpoint is used only as an SCF initial guess.  The high-level
  HF reference is reconverged at each requested geometry.
* Closed-shell references use RHF -> RCCSD(T).
* Open-shell references use ROHF for the SCF state, then an explicitly
  semicanonicalized UHF representation for UCCSD(T).  This is recorded in the
  result because PySCF converts ROHF CC to UCCSD and (T) is sensitive to the
  orbital representation.
* ROHF external stability is never fabricated: PySCF does not currently
  provide that test.  Internal stability is checked.
* The CC correlation space is explicit through ``settings.frozen_core``;
  legacy/default execution is all-electron, while frozen-core requests use
  PySCF ``CCSD.set_frozen()``.
* A numerical result is not a production EA, not a ground-state assignment,
  and not permission to prune another state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .stage3_naming import checkpoint_artifact_basename


class PointExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    SCF_NOT_CONVERGED = "SCF_NOT_CONVERGED"
    SCF_UNSTABLE = "SCF_UNSTABLE"
    CCSD_NOT_CONVERGED = "CCSD_NOT_CONVERGED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Stage3ExecutionSettings:
    scf_conv_tol: float = 1.0e-10
    scf_conv_tol_grad: float = 1.0e-7
    scf_max_cycle: int = 100
    cc_conv_tol: float = 1.0e-8
    cc_conv_tol_normt: float = 1.0e-6
    cc_max_cycle: int = 100
    max_memory_mb: int = 12000
    checkpoint_project: bool | None = None
    require_internal_stability: bool = True
    require_rhf_external_stability: bool = True
    run_ccsd_t: bool = True
    frozen_core: bool = False
    verbose: int = 4
    artifact_dir: str | None = None

    def __post_init__(self) -> None:
        for name in ("scf_conv_tol", "scf_conv_tol_grad", "cc_conv_tol", "cc_conv_tol_normt"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.scf_max_cycle <= 0 or self.cc_max_cycle <= 0:
            raise ValueError("SCF/CC cycle limits must be positive")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.artifact_dir is not None and not str(self.artifact_dir).strip():
            raise ValueError("artifact_dir must be non-empty when supplied")


@dataclass(frozen=True)
class Stage3ExecutionRequest:
    request_id: str
    job_id: str
    system: str
    atoms: tuple[str, str]
    charge: int
    spin_2s: int
    component_id: str
    r_angstrom: float
    basis: str
    methods: tuple[str, ...]
    requested_reference: str
    scf_reference: str
    source_link_status: str
    source_root_id: str
    source_checkpoint_path: str
    source_origin_guess: str | None
    grid_index: int
    initialization_index: int
    dft_center_r_angstrom: float
    dft_center_energy_hartree: float
    requires_independent_state_identity_validation: bool
    authorizes_pruning: bool = False

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.job_id.strip():
            raise ValueError("Execution request requires stable IDs")
        if len(self.atoms) != 2 or any(not str(x).strip() for x in self.atoms):
            raise ValueError("Stage-3 v1 execution currently requires exactly two atoms")
        if self.spin_2s < 0:
            raise ValueError("spin_2s must be non-negative")
        if not isfinite(float(self.r_angstrom)) or self.r_angstrom <= 0.0:
            raise ValueError("Execution geometry must be positive and finite")
        if not self.basis.strip():
            raise ValueError("Execution request requires a basis")
        if "CCSD" not in self.methods:
            raise ValueError("Stage-3 execution requires CCSD")
        if self.requested_reference.upper() != "ROHF":
            raise ValueError("Current Stage-3 contract must request ROHF")
        expected = "RHF" if self.spin_2s == 0 else "ROHF"
        if self.scf_reference != expected:
            raise ValueError(f"spin_2s={self.spin_2s} requires {expected}, not {self.scf_reference}")
        if not self.source_root_id.strip() or not self.source_checkpoint_path.strip():
            raise ValueError("Execution request requires explicit checkpoint provenance")
        if self.authorizes_pruning:
            raise ValueError("Stage-3 execution never authorizes state pruning")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage3PointResult:
    request_id: str
    job_id: str
    status: PointExecutionStatus
    system: str
    charge: int
    spin_2s: int
    component_id: str
    r_angstrom: float
    basis: str
    source_root_id: str
    source_checkpoint_path: str
    scf_reference: str
    cc_reference: str | None
    pyscf_version: str | None
    scf_converged: bool
    scf_energy_hartree: float | None
    s2: float | None
    multiplicity: float | None
    internal_stable: bool | None
    external_stable: bool | None
    external_stability_available: bool
    semicanonicalization: str | None
    cc_class: str | None
    ccsd_converged: bool | None
    ccsd_correlation_hartree: float | None
    ccsd_total_hartree: float | None
    triples_correction_hartree: float | None
    ccsd_t_total_hartree: float | None
    t1_diagnostic: float | None
    t1_diagnostic_definition: str | None
    error_type: str | None
    error_message: str | None
    is_production_ea: bool = False
    ground_state_assigned: bool = False
    authorizes_pruning: bool = False
    high_level_checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


_ALLOWED_SOURCE_LINKS = {
    "SINGLE_DFT_INITIALIZATION",
    "MULTIPLE_DFT_INITIALIZATIONS",
}


def _manifest_atoms(manifest: Mapping[str, Any], system: str) -> tuple[str, str]:
    systems = manifest.get("systems")
    if not isinstance(systems, Mapping) or system not in systems:
        raise ValueError(f"System absent from manifest: {system}")
    entry = systems[system]
    if not isinstance(entry, Mapping):
        raise TypeError(f"Manifest entry for {system} must be a mapping")
    atoms = tuple(str(x) for x in entry.get("atoms", ()))
    if len(atoms) != 2:
        raise ValueError("Stage-3 v1 execution currently supports diatomics only")
    return atoms[0], atoms[1]


def build_stage3_execution_requests(
    *,
    stage3_plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authorized_job_ids: Iterable[str],
) -> tuple[Stage3ExecutionRequest, ...]:
    """Expand authorized Stage-3 jobs into deterministic point requests.

    Authorization is explicit.  This function intentionally refuses to infer
    permission from the mere presence of a job in ``stage3_plan``.
    """
    if bool(stage3_plan.get("automatic_pruning_performed", False)):
        raise ValueError("Refusing Stage-3 plan that reports automatic pruning")

    authorized = tuple(str(x) for x in authorized_job_ids)
    if len(authorized) != len(set(authorized)):
        raise ValueError("Duplicate authorized Stage-3 job IDs")

    jobs = stage3_plan.get("jobs", ())
    if not isinstance(jobs, Sequence):
        raise TypeError("stage3_plan.jobs must be a sequence")

    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in jobs:
        if not isinstance(raw, Mapping):
            raise TypeError("Stage-3 jobs must be mappings")
        job_id = str(raw.get("job_id", ""))
        if not job_id:
            raise ValueError("Stage-3 job missing job_id")
        if job_id in by_id:
            raise ValueError(f"Duplicate Stage-3 job ID: {job_id}")
        by_id[job_id] = raw

    missing = sorted(set(authorized) - set(by_id))
    if missing:
        raise ValueError(f"Authorized jobs absent from Stage-3 plan: {missing}")

    requests: list[Stage3ExecutionRequest] = []

    for job_id in authorized:
        job = by_id[job_id]
        if job.get("execution_status") != "PLANNED_NOT_RUN":
            raise ValueError(f"{job_id}: execution_status is not PLANNED_NOT_RUN")

        system = str(job.get("system", stage3_plan.get("system", "")))
        atoms = _manifest_atoms(manifest, system)
        spin_2s = int(job["spin_2s"])
        requested_reference = str(job.get("reference", ""))
        scf_reference = "RHF" if spin_2s == 0 else "ROHF"
        methods = tuple(str(x) for x in job.get("methods", ()))
        grid = tuple(float(x) for x in job.get("local_grid_angstrom", ()))
        if not grid:
            raise ValueError(f"{job_id}: empty local Stage-3 grid")
        if any((not isfinite(r) or r <= 0.0) for r in grid):
            raise ValueError(f"{job_id}: invalid geometry in local grid")

        source = job.get("source_provenance")
        if not isinstance(source, Mapping):
            raise ValueError(f"{job_id}: missing source_provenance")
        link_status = str(source.get("link_status", ""))
        if link_status not in _ALLOWED_SOURCE_LINKS:
            raise ValueError(f"{job_id}: unresolved checkpoint provenance ({link_status or 'MISSING'})")

        checkpoints = source.get("checkpoint_candidates", ())
        if not isinstance(checkpoints, Sequence) or not checkpoints:
            raise ValueError(f"{job_id}: no checkpoint candidates")

        center_r = float(job["dft_center_r_angstrom"])
        center_e = float(job["dft_center_energy_hartree"])

        for init_index, candidate in enumerate(checkpoints):
            if not isinstance(candidate, Mapping):
                raise TypeError(f"{job_id}: checkpoint candidate must be a mapping")
            root_id = str(candidate.get("root_id", ""))
            checkpoint = str(candidate.get("checkpoint_path", ""))
            if not root_id or not checkpoint:
                raise ValueError(f"{job_id}: checkpoint candidate lacks root/path")
            origin_guess_raw = candidate.get("origin_guess")
            origin_guess = None if origin_guess_raw is None else str(origin_guess_raw)

            for grid_index, r_angstrom in enumerate(grid):
                safe_root = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in root_id)
                request_id = f"{job_id}__init{init_index:02d}_{safe_root}__g{grid_index:03d}"
                requests.append(Stage3ExecutionRequest(
                    request_id=request_id,
                    job_id=job_id,
                    system=system,
                    atoms=atoms,
                    charge=int(job["charge"]),
                    spin_2s=spin_2s,
                    component_id=str(job["component_id"]),
                    r_angstrom=r_angstrom,
                    basis=str(job["basis"]),
                    methods=methods,
                    requested_reference=requested_reference,
                    scf_reference=scf_reference,
                    source_link_status=link_status,
                    source_root_id=root_id,
                    source_checkpoint_path=checkpoint,
                    source_origin_guess=origin_guess,
                    grid_index=grid_index,
                    initialization_index=init_index,
                    dft_center_r_angstrom=center_r,
                    dft_center_energy_hartree=center_e,
                    requires_independent_state_identity_validation=bool(
                        job.get("requires_independent_state_identity_validation", True)
                    ),
                ))

    return tuple(requests)


def _empty_result(request: Stage3ExecutionRequest, status: PointExecutionStatus, **updates: Any) -> Stage3PointResult:
    data: dict[str, Any] = dict(
        request_id=request.request_id,
        job_id=request.job_id,
        status=status,
        system=request.system,
        charge=request.charge,
        spin_2s=request.spin_2s,
        component_id=request.component_id,
        r_angstrom=request.r_angstrom,
        basis=request.basis,
        source_root_id=request.source_root_id,
        source_checkpoint_path=request.source_checkpoint_path,
        scf_reference=request.scf_reference,
        cc_reference=None,
        pyscf_version=None,
        scf_converged=False,
        scf_energy_hartree=None,
        s2=None,
        multiplicity=None,
        internal_stable=None,
        external_stable=None,
        external_stability_available=(request.scf_reference == "RHF"),
        semicanonicalization=None,
        cc_class=None,
        ccsd_converged=None,
        ccsd_correlation_hartree=None,
        ccsd_total_hartree=None,
        triples_correction_hartree=None,
        ccsd_t_total_hartree=None,
        t1_diagnostic=None,
        t1_diagnostic_definition=None,
        error_type=None,
        error_message=None,
    )
    data.update(updates)
    return Stage3PointResult(**data)


def _run_stage3_point_pyscf(
    request: Stage3ExecutionRequest,
    settings: Stage3ExecutionSettings,
) -> Stage3PointResult:
    """Execute one authorized request with PySCF; imported lazily for testability."""
    import pyscf
    from pyscf import cc, gto, scf

    checkpoint = Path(request.source_checkpoint_path)
    if not checkpoint.is_file():
        return _empty_result(
            request,
            PointExecutionStatus.ERROR,
            pyscf_version=getattr(pyscf, "__version__", None),
            error_type="CheckpointNotFound",
            error_message=str(checkpoint),
        )

    atom_spec = [
        (request.atoms[0], (0.0, 0.0, 0.0)),
        (request.atoms[1], (0.0, 0.0, float(request.r_angstrom))),
    ]
    mol = gto.M(
        atom=atom_spec,
        basis=request.basis,
        charge=request.charge,
        spin=request.spin_2s,
        unit="Angstrom",
        symmetry=False,
        verbose=settings.verbose,
        max_memory=settings.max_memory_mb,
    )

    if request.scf_reference == "RHF":
        mf = scf.RHF(mol)
        dm0 = scf.hf.init_guess_by_chkfile(
            mol, request.source_checkpoint_path, project=settings.checkpoint_project
        )
    else:
        mf = scf.ROHF(mol)
        dm0 = scf.rohf.init_guess_by_chkfile(
            mol, request.source_checkpoint_path, project=settings.checkpoint_project
        )

    high_level_checkpoint: Path | None = None
    if settings.artifact_dir is not None:
        artifact_dir = Path(settings.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        high_level_checkpoint = artifact_dir / checkpoint_artifact_basename(
            request.request_id
        )
        mf.chkfile = str(high_level_checkpoint)

    mf.conv_tol = settings.scf_conv_tol
    mf.conv_tol_grad = settings.scf_conv_tol_grad
    mf.max_cycle = settings.scf_max_cycle
    mf.max_memory = settings.max_memory_mb
    mf.kernel(dm0=dm0)

    try:
        s2, multiplicity = mf.spin_square()
        s2 = float(s2)
        multiplicity = float(multiplicity)
    except Exception:
        s2 = multiplicity = None

    common = dict(
        pyscf_version=getattr(pyscf, "__version__", None),
        scf_converged=bool(mf.converged),
        scf_energy_hartree=float(mf.e_tot) if mf.e_tot is not None else None,
        s2=s2,
        multiplicity=multiplicity,
        high_level_checkpoint_path=(
            str(high_level_checkpoint)
            if high_level_checkpoint is not None and high_level_checkpoint.is_file()
            else None
        ),
    )

    if not mf.converged:
        return _empty_result(request, PointExecutionStatus.SCF_NOT_CONVERGED, **common)

    # Stability is deliberately checked on the actual high-level HF reference.
    # ROHF external stability is not available in PySCF and remains explicit None.
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

    common.update(
        internal_stable=internal_stable,
        external_stability_available=external_available,
        external_stable=external_stable,
    )

    if settings.require_internal_stability and not internal_stable:
        return _empty_result(request, PointExecutionStatus.SCF_UNSTABLE, **common)
    if request.scf_reference == "RHF" and settings.require_rhf_external_stability and external_stable is False:
        return _empty_result(request, PointExecutionStatus.SCF_UNSTABLE, **common)

    if request.scf_reference == "ROHF":
        # PySCF converts ROHF CCSD to UCCSD.  Canonicalize alpha/beta occupied
        # and virtual spaces explicitly first so the non-iterative triples
        # correction is not evaluated on a non-semicanonical representation.
        cc_mf = mf.to_uhf()
        mo_energy, mo_coeff = cc_mf.canonicalize(cc_mf.mo_coeff, cc_mf.mo_occ)
        cc_mf.mo_energy = mo_energy
        cc_mf.mo_coeff = mo_coeff
        cc_reference = "SEMICANONICAL_UHF_FROM_ROHF"
        semicanonicalization = "UHF_OCCUPIED_VIRTUAL_CANONICALIZATION"
    else:
        cc_mf = mf
        cc_reference = "RHF"
        semicanonicalization = "RHF_CANONICAL_REFERENCE"

    mycc = cc.CCSD(cc_mf)
    # Correlation space must be explicit. PySCF correlates all electrons
    # when frozen is None. For a valence frozen-core calculation use
    # PySCF's documented automatic chemical-core rule.
    if settings.frozen_core:
        mycc.set_frozen()
    mycc.conv_tol = settings.cc_conv_tol
    mycc.conv_tol_normt = settings.cc_conv_tol_normt
    mycc.max_cycle = settings.cc_max_cycle
    mycc.max_memory = settings.max_memory_mb

    # Reuse the same transformed integrals for CCSD and (T).
    eris = mycc.ao2mo()
    ecorr, t1, t2 = mycc.kernel(eris=eris)
    ccsd_converged = bool(mycc.converged)
    ccsd_total = float(mycc.e_tot) if mycc.e_tot is not None else None

    t1_diagnostic = None
    t1_definition = None
    # PySCF's standard T1 diagnostic is a restricted closed-shell definition.
    if request.scf_reference == "RHF" and hasattr(mycc, "get_t1_diagnostic"):
        try:
            t1_diagnostic = float(mycc.get_t1_diagnostic(t1))
            t1_definition = "PYSCF_RCCSD_T1_NORMALIZED_BY_CORRELATED_ELECTRONS"
        except Exception:
            t1_diagnostic = None
            t1_definition = None

    common.update(
        cc_reference=cc_reference,
        semicanonicalization=semicanonicalization,
        cc_class=type(mycc).__name__,
        ccsd_converged=ccsd_converged,
        ccsd_correlation_hartree=float(ecorr),
        ccsd_total_hartree=ccsd_total,
        t1_diagnostic=t1_diagnostic,
        t1_diagnostic_definition=t1_definition,
    )

    if not ccsd_converged:
        return _empty_result(request, PointExecutionStatus.CCSD_NOT_CONVERGED, **common)

    triples = None
    ccsd_t_total = None
    if settings.run_ccsd_t and "CCSD(T)" in request.methods:
        triples = float(mycc.ccsd_t(t1=t1, t2=t2, eris=eris))
        if ccsd_total is not None:
            ccsd_t_total = ccsd_total + triples

    common.update(
        triples_correction_hartree=triples,
        ccsd_t_total_hartree=ccsd_t_total,
    )
    return _empty_result(request, PointExecutionStatus.COMPLETED, **common)


def run_stage3_point(
    request: Stage3ExecutionRequest,
    *,
    settings: Stage3ExecutionSettings | None = None,
    runner: Callable[[Stage3ExecutionRequest, Stage3ExecutionSettings], Stage3PointResult] | None = None,
) -> Stage3PointResult:
    """Run one Stage-3 point with a fail-closed result record.

    ``runner`` is an injection seam used by unit tests and future alternative
    open-source backends.  Production calls normally leave it ``None``.
    """
    settings = settings or Stage3ExecutionSettings()
    implementation = runner or _run_stage3_point_pyscf
    try:
        result = implementation(request, settings)
    except Exception as exc:  # preserve evidence rather than losing the job
        return _empty_result(
            request,
            PointExecutionStatus.ERROR,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    if not isinstance(result, Stage3PointResult):
        raise TypeError("Stage-3 runner must return Stage3PointResult")
    if result.request_id != request.request_id or result.job_id != request.job_id:
        raise ValueError("Stage-3 runner returned result for another request/job")
    return result
