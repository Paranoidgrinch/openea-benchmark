"""G2 EA-EOM diagnostic execution: raw roots, explicit basis policies and resume.

This is NOT a boundness oracle. In particular, EA-EOM eigenvalues and a
stationary energy across finite Gaussian bases do not independently prove
continuum exclusion. Root identity across changing orbital spaces must be
reviewed outside the executor before emitting G2-cleared evidence.

Calculations are restricted to a previously approved neutral single-reference
state (RHF/RCCSD or UHF/UCCSD). They are not a substitute for an MR branch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping
import copy
import importlib.metadata
import json
import os

from .attachment_continuum import (
    EOMEAAttachmentPoint,
    StabilizationPoint,
    pyscf_eom_eigenvalue_to_attachment_ea_ev,
)
from .model import Review, ReviewStatus


class G2EOMStatus(str, Enum):
    COMPLETE_ROOT_REVIEW_REQUIRED = "COMPLETE_ROOT_REVIEW_REQUIRED"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    EXECUTION_BLOCKED = "EXECUTION_BLOCKED"


@dataclass(frozen=True)
class G2EOMNeutralState:
    system: str
    atoms: tuple[str, str]
    charge: int
    spin_2s: int
    state_id: str
    r_angstrom: float
    source_root_id: str
    source_checkpoint_path: str
    state_identity_validated: bool
    reference_character_validated: bool
    scf_reference: str  # RHF (closed shell) or UHF (open shell)
    nfrozen_orbitals: int = 0

    def __post_init__(self) -> None:
        if not all((self.system.strip(), self.state_id.strip(), self.source_root_id.strip())):
            raise ValueError("G2 EOM neutral requires system/state/root provenance")
        if len(self.atoms) != 2 or any(not a.strip() for a in self.atoms):
            raise ValueError("G2 EOM v1 requires two defined atoms")
        if self.spin_2s < 0 or not isfinite(float(self.r_angstrom)) or self.r_angstrom <= 0:
            raise ValueError("Invalid neutral geometry/spin")
        if (self.spin_2s == 0 and self.scf_reference != "RHF") or (
            self.spin_2s > 0 and self.scf_reference != "UHF"
        ):
            raise ValueError("RHF for singlet, UHF for open-shell neutral required")
        if self.nfrozen_orbitals < 0:
            raise ValueError("Frozen-orbital count cannot be negative")
        if not (self.state_identity_validated and self.reference_character_validated):
            raise ValueError("G2 EOM requires neutral state and SR reference validation")
        if not self.source_checkpoint_path.strip():
            raise ValueError("Neutral source checkpoint is required")


@dataclass(frozen=True)
class G2EOMBasis:
    augmentation_level: int
    label: str
    family_id: str
    basis_by_element: Mapping[str, str]
    electron_model: str = "ALL_ELECTRON"

    def __post_init__(self) -> None:
        if self.augmentation_level < 0 or not self.label.strip() or not self.family_id.strip():
            raise ValueError("Invalid explicit G2 augmentation/basis metadata")
        if not self.basis_by_element or any(not k.strip() or not v.strip() for k, v in self.basis_by_element.items()):
            raise ValueError("G2 basis_by_element must be explicit")
        if self.electron_model != "ALL_ELECTRON":
            raise ValueError("ECP/core-replacement requires a separate validated G2 implementation")


@dataclass(frozen=True)
class DiffuseShellSelector:
    """One reviewed, *uncontracted* diffuse shell, indexed in PySCF load order.

    This is deliberately NOT an automatic 'most diffuse shell' heuristic.
    A reviewer provides shell indices after inspecting the chosen basis family.
    The shell must consist of exactly one exponent/contraction coefficient.
    """
    element: str
    shell_index: int

    def __post_init__(self) -> None:
        if not self.element.strip() or self.shell_index < 0:
            raise ValueError("Invalid diffuse shell selector")


@dataclass(frozen=True)
class G2EOMStabilization:
    reference_augmentation_level: int
    scale_factor: float
    selectors: tuple[DiffuseShellSelector, ...]

    def __post_init__(self) -> None:
        if self.reference_augmentation_level < 0:
            raise ValueError("Invalid stabilization reference augmentation level")
        if not isfinite(float(self.scale_factor)) or self.scale_factor <= 0:
            raise ValueError("Diffuse scaling must be finite and positive")
        if not self.selectors:
            raise ValueError("Stabilization scaling requires explicit shell selectors")
        keys = [(x.element, x.shell_index) for x in self.selectors]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate diffuse shell selectors")


@dataclass(frozen=True)
class G2EOMAuthorization:
    authorized: bool
    reason: str
    evidence_ids: tuple[str, ...]
    authorized_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("G2 EOM authorization requires an explicit rationale")
        if self.authorized and (not self.evidence_ids or not self.authorized_keys):
            raise ValueError("Authorized G2 calculation requires provenance and explicit point keys")
        if len(self.authorized_keys) != len(set(self.authorized_keys)):
            raise ValueError("Duplicate authorization keys")


@dataclass(frozen=True)
class G2EOMSettings:
    nroots: int = 3
    max_memory_mb: int = 12000
    scf_conv_tol: float = 1.0e-10
    scf_max_cycle: int = 150
    cc_conv_tol: float = 1.0e-8
    cc_max_cycle: int = 120
    project_checkpoint: bool = True
    require_internal_stability: bool = True
    require_rhf_external_stability: bool = True

    def __post_init__(self) -> None:
        if self.nroots < 2 or self.max_memory_mb <= 0 or self.scf_max_cycle <= 0 or self.cc_max_cycle <= 0:
            raise ValueError("Invalid EOM root count / computation limits")
        if not (isfinite(self.scf_conv_tol) and self.scf_conv_tol > 0 and
                isfinite(self.cc_conv_tol) and self.cc_conv_tol > 0):
            raise ValueError("Invalid G2 numerical convergence tolerances")


@dataclass(frozen=True)
class G2EOMRequest:
    key: str
    state: G2EOMNeutralState
    basis: G2EOMBasis
    scale_factor: float | None
    selectors: tuple[DiffuseShellSelector, ...]
    source_checkpoint_sha256: str

    def __post_init__(self) -> None:
        if not self.key.strip() or len(self.source_checkpoint_sha256) != 64:
            raise ValueError("G2 request requires key and fingerprinted source checkpoint")


@dataclass(frozen=True)
class G2EOMRoot:
    root_index: int
    omega_hartree: float
    attachment_ea_ev: float
    # Raw EOM roots have no cross-basis state approval and no calibrated 1p weight.
    def __post_init__(self) -> None:
        if self.root_index < 0 or not isfinite(self.omega_hartree) or not isfinite(self.attachment_ea_ev):
            raise ValueError("Invalid EOM root")
        if abs(self.attachment_ea_ev - pyscf_eom_eigenvalue_to_attachment_ea_ev(self.omega_hartree)) > 1e-8:
            raise ValueError("EOM root sign/energy mismatch")


@dataclass(frozen=True)
class G2EOMRawResult:
    key: str
    roots: tuple[G2EOMRoot, ...]
    reference_kind: str
    pyscf_version: str
    scf_converged: bool
    ccsd_converged: bool
    eom_converged: bool

    def __post_init__(self) -> None:
        if not self.key.strip() or self.reference_kind not in {"RHF", "UHF"}:
            raise ValueError("Invalid EOM backend result reference/key")
        if not self.roots or len({r.root_index for r in self.roots}) != len(self.roots):
            raise ValueError("EOM results require unique roots")
        if not (self.scf_converged and self.ccsd_converged and self.eom_converged):
            raise ValueError("Unconverged EOM results cannot enter G2 evidence")


@dataclass(frozen=True)
class G2EOMSubpoint:
    request: G2EOMRequest
    result: G2EOMRawResult
    checkpoint_reused: bool
    evidence_id: str


@dataclass(frozen=True)
class G2EOMSeries:
    status: G2EOMStatus
    subpoints: tuple[G2EOMSubpoint, ...]
    notes: tuple[str, ...]
    method_role: str = "DIAGNOSTIC"
    is_production_ea: bool = False

    def __post_init__(self) -> None:
        if self.is_production_ea or self.method_role != "DIAGNOSTIC":
            raise ValueError("G2 EA-EOM must remain diagnostic")


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _safe(s: str) -> str:
    return "".join(x if x.isalnum() or x in "-_" else "_" for x in s)


def _key_for_level(level: int) -> str:
    return f"aug:{level}"


def _key_for_scale(level: int, scale: float) -> str:
    return f"scale:{level}:{float(scale):.12g}"


def _parse_basis_from_pyscf(element: str, name: str) -> Any:
    from pyscf import gto
    return gto.basis.load(name, element)


def scaled_diffuse_basis(
    basis_by_element: Mapping[str, str],
    selectors: tuple[DiffuseShellSelector, ...],
    scale_factor: float,
    *,
    loader: Callable[[str, str], Any] = _parse_basis_from_pyscf,
) -> dict[str, Any]:
    """Scale ONLY explicitly indexed single-primitive uncontracted shells.

    A contracted shell cannot be partially scaled without changing the
    contraction meaning. Unsupported shell shapes fail before any computation.
    """
    if not isfinite(float(scale_factor)) or scale_factor <= 0:
        raise ValueError("Invalid diffuse scale factor")
    if not selectors:
        raise ValueError("Explicit diffuse selectors required")
    if len({(s.element, s.shell_index) for s in selectors}) != len(selectors):
        raise ValueError("Duplicate shell selector")
    if not set(s.element for s in selectors).issubset(basis_by_element):
        raise ValueError("Diffuse selector element absent from basis policy")
    expanded: dict[str, Any] = {
        element: copy.deepcopy(loader(element, name))
        for element, name in basis_by_element.items()
    }
    for sel in selectors:
        shells = expanded[sel.element]
        if sel.shell_index >= len(shells):
            raise ValueError("Diffuse shell selector is out of range")
        shell = shells[sel.shell_index]
        if len(shell) != 2 or not isinstance(shell[0], int) or len(shell[1]) != 2:
            raise ValueError("Diffuse scaling only supports simple uncontracted single-primitive shells")
        exponent = float(shell[1][0])
        if not isfinite(exponent) or exponent <= 0:
            raise ValueError("Diffuse exponent must be positive")
        shell[1][0] = exponent * scale_factor
    return expanded


def _backend_pyscf(request: G2EOMRequest, settings: G2EOMSettings, basis: Mapping[str, Any]) -> G2EOMRawResult:
    """Actual PySCF RHF/RCCSD or UHF/UCCSD EA-EOM backend (lazy import)."""
    import numpy as np
    import pyscf
    from pyscf import cc, gto, scf

    st = request.state
    mol = gto.M(
        atom=[(st.atoms[0], (0.0, 0.0, 0.0)),
              (st.atoms[1], (0.0, 0.0, st.r_angstrom))],
        charge=st.charge, spin=st.spin_2s,
        basis=dict(basis), unit="Angstrom", symmetry=False,
        max_memory=settings.max_memory_mb, verbose=0,
    )
    mf = scf.RHF(mol) if st.scf_reference == "RHF" else scf.UHF(mol)
    if st.scf_reference == "RHF":
        dm0 = scf.hf.init_guess_by_chkfile(mol, st.source_checkpoint_path,
                                           project=settings.project_checkpoint)
    else:
        dm0 = scf.uhf.init_guess_by_chkfile(mol, st.source_checkpoint_path,
                                            project=settings.project_checkpoint)
    mf.conv_tol = settings.scf_conv_tol
    mf.max_cycle = settings.scf_max_cycle
    mf.max_memory = settings.max_memory_mb
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("Neutral-reference SCF did not converge")
    _, _, internal, external = mf.stability(
        internal=True,
        external=bool(settings.require_rhf_external_stability and st.scf_reference == "RHF"),
        return_status=True,
    )
    if settings.require_internal_stability and not bool(internal):
        raise RuntimeError("Neutral reference has an internal SCF instability")
    if settings.require_rhf_external_stability and st.scf_reference == "RHF" and not bool(external):
        raise RuntimeError("Neutral reference has an external SCF instability")

    mycc = cc.RCCSD(mf) if st.scf_reference == "RHF" else cc.UCCSD(mf)
    if st.nfrozen_orbitals:
        mycc.frozen = st.nfrozen_orbitals
    mycc.conv_tol = settings.cc_conv_tol
    mycc.max_cycle = settings.cc_max_cycle
    mycc.max_memory = settings.max_memory_mb
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("Neutral-reference CCSD did not converge")
    eom = mycc.eomea_method()
    energies, vectors = eom.kernel(nroots=settings.nroots, koopmans=False)
    values = np.atleast_1d(energies)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("EA-EOM returned a non-finite eigenvalue")
    cv = getattr(eom, "converged", None)
    if cv is None or not np.all(cv):
        raise RuntimeError("EA-EOM root convergence was not established")
    if len(values) != settings.nroots:
        raise RuntimeError("EA-EOM returned fewer roots than requested")
    roots = tuple(
        G2EOMRoot(int(i), float(omega), pyscf_eom_eigenvalue_to_attachment_ea_ev(float(omega)))
        for i, omega in enumerate(values)
    )
    return G2EOMRawResult(request.key, roots, st.scf_reference,
                          str(pyscf.__version__), True, True, True)


Backend = Callable[[G2EOMRequest, G2EOMSettings, Mapping[str, Any]], G2EOMRawResult]
BasisLoader = Callable[[str, str], Any]


def _checkpoint_signature(request: G2EOMRequest, settings: G2EOMSettings,
                          basis_material: Mapping[str, Any], backend_id: str) -> str:
    payload = {"request": asdict(request), "settings": asdict(settings),
               "resolved_basis": basis_material, "backend_id": backend_id,
               "schema": "G2_EOM_RAW_V1"}
    return sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _load_checkpoint(path: Path, signature: str, request: G2EOMRequest) -> G2EOMRawResult | None:
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("signature") != signature or saved.get("schema") != "G2_EOM_RAW_V1":
        raise ValueError(f"Incompatible G2 EOM checkpoint {path}; refusing to reuse")
    data = saved["result"]
    roots = tuple(G2EOMRoot(**item) for item in data["roots"])
    result = G2EOMRawResult(
        key=data["key"], roots=roots, reference_kind=data["reference_kind"],
        pyscf_version=data["pyscf_version"], scf_converged=data["scf_converged"],
        ccsd_converged=data["ccsd_converged"], eom_converged=data["eom_converged"],
    )
    if result.key != request.key or result.reference_kind != request.state.scf_reference:
        raise ValueError("Incompatible G2 checkpoint state/reference")
    return result


def _save_checkpoint(path: Path, signature: str, result: G2EOMRawResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema": "G2_EOM_RAW_V1", "signature": signature, "result": asdict(result)}
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def run_g2_eom_diagnostics(
    *,
    authorization: G2EOMAuthorization,
    neutral: G2EOMNeutralState,
    basis_specs: tuple[G2EOMBasis, ...],
    stabilization_specs: tuple[G2EOMStabilization, ...] = (),
    settings: G2EOMSettings = G2EOMSettings(),
    checkpoint_dir: str | Path | None = None,
    backend: Backend = _backend_pyscf,
    basis_loader: BasisLoader = _parse_basis_from_pyscf,
    backend_id: str | None = None,
) -> G2EOMSeries:
    """Run/restore explicitly permitted raw G2 diagnostics.

    No raw root is automatically marked as state-identical across bases.
    Authorization is per point. An unauthorized point may be read from a
    compatible checkpoint but never computed afresh.
    """
    if not authorization.authorized:
        return G2EOMSeries(G2EOMStatus.POLICY_BLOCKED, (), (authorization.reason,))
    if not basis_specs:
        raise ValueError("No G2 EOM basis levels supplied")
    levels = [x.augmentation_level for x in basis_specs]
    if levels != sorted(set(levels)):
        raise ValueError("G2 EOM augmentation levels must be unique and sorted")
    family_ids = {x.family_id for x in basis_specs}
    if len(family_ids) != 1:
        raise ValueError("One diffuse basis family per G2 EOM series")
    for bs in basis_specs:
        if set(bs.basis_by_element) != set(neutral.atoms):
            raise ValueError("Each basis level must map exactly all molecular elements")
    by_level = {x.augmentation_level: x for x in basis_specs}
    if not Path(neutral.source_checkpoint_path).is_file():
        raise FileNotFoundError("Neutral source SCF checkpoint does not exist")
    source_hash = sha256()
    with Path(neutral.source_checkpoint_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            source_hash.update(chunk)
    source_digest = source_hash.hexdigest()
    if backend_id is None:
        if backend is not _backend_pyscf:
            raise ValueError("Injected G2 backend requires explicit backend_id for checkpoint provenance")
        try:
            version = importlib.metadata.version("pyscf")
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError("Cannot fingerprint the installed PySCF backend")
        backend_id = f"PySCF-{version}-G2-EOM-RCCSD-UCCSD-v1"
    if not backend_id.strip():
        raise ValueError("G2 checkpoint backend_id must be nonempty")

    # Full basis resolution and scaling preflight before ANY expensive EOM job.
    prepared: list[tuple[G2EOMRequest, dict[str, Any]]] = []
    for bs in basis_specs:
        req = G2EOMRequest(_key_for_level(bs.augmentation_level), neutral, bs,
                           None, (), source_digest)
        material = {k: copy.deepcopy(basis_loader(k, v)) for k, v in bs.basis_by_element.items()}
        prepared.append((req, material))
    if len({_canonical(material) for _, material in prepared}) != len(prepared):
        raise ValueError("Different augmentation levels resolved to identical orbital bases; no diffuse convergence evidence")
    if len({spec.reference_augmentation_level for spec in stabilization_specs}) > 1:
        raise ValueError("A stabilization series must use one fixed augmentation-level baseline")
    keys_seen = set(req.key for req, _ in prepared)
    for spec in stabilization_specs:
        if spec.reference_augmentation_level not in by_level:
            raise ValueError("Stabilization basis augmentation level missing")
        bs = by_level[spec.reference_augmentation_level]
        key = _key_for_scale(bs.augmentation_level, spec.scale_factor)
        if key in keys_seen:
            raise ValueError("Duplicate stabilization key")
        keys_seen.add(key)
        material = scaled_diffuse_basis(bs.basis_by_element, spec.selectors,
                                        spec.scale_factor, loader=basis_loader)
        req = G2EOMRequest(key, neutral, bs, spec.scale_factor, spec.selectors, source_digest)
        prepared.append((req, material))
    invalid = set(authorization.authorized_keys) - keys_seen
    if invalid:
        raise ValueError(f"G2 authorization references undefined keys: {sorted(invalid)}")

    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    output = []
    for req, basis_material in prepared:
        signature = _checkpoint_signature(req, settings, basis_material, backend_id)
        path = directory / f"{_safe(req.key)}.json" if directory else None
        previous = _load_checkpoint(path, signature, req) if path else None
        reused = previous is not None
        if previous is None:
            if req.key not in authorization.authorized_keys:
                return G2EOMSeries(G2EOMStatus.POLICY_BLOCKED, tuple(output),
                                   (f"No compatible checkpoint and no authorization for {req.key}",))
            try:
                result = backend(req, settings, basis_material)
            except Exception as exc:
                return G2EOMSeries(G2EOMStatus.EXECUTION_BLOCKED, tuple(output),
                                   (f"{req.key}: {type(exc).__name__}: {exc}",))
            if not isinstance(result, G2EOMRawResult) or result.key != req.key:
                raise ValueError("G2 backend returned a result for a different request")
            if result.reference_kind != neutral.scf_reference or len(result.roots) != settings.nroots:
                raise ValueError("G2 backend violated the declared reference/root contract")
            if path:
                _save_checkpoint(path, signature, result)
        else:
            result = previous
        output.append(G2EOMSubpoint(req, result, reused, f"G2_EOM:{signature}"))
    return G2EOMSeries(
        G2EOMStatus.COMPLETE_ROOT_REVIEW_REQUIRED,
        tuple(output),
        ("Raw EOM spectra complete. Cross-basis/state identity, spin-sector identity, and continuum character remain unreviewed.",),
    )


def evidence_points_from_review(
    series: G2EOMSeries,
    *,
    selections: Mapping[str, tuple[int, Review]],
) -> tuple[tuple[EOMEAAttachmentPoint, ...], tuple[StabilizationPoint, ...]]:
    """Adapt raw roots to Patch-11 G2 evidence after OUT-OF-BAND identity review.

    A selected energy root is not itself a verified root identity. For that
    reason a CLEARED review must have its own evidence ID unrelated to the
    raw EOM signature; a pending review remains pending rather than being
    promoted to CLEARED by the execution layer.
    """
    out_eom = []
    out_stab = []
    for sub in series.subpoints:
        if sub.request.key not in selections:
            raise ValueError(f"Missing external root selection for {sub.request.key}")
        root_index, review = selections[sub.request.key]
        roots = {x.root_index: x for x in sub.result.roots}
        if root_index not in roots:
            raise ValueError("Requested EOM root index does not exist")
        if review.status is ReviewStatus.NOT_APPLICABLE:
            raise ValueError("Root identity may not be marked NOT_APPLICABLE")
        if review.status is ReviewStatus.CLEARED and (
            not review.evidence_ids or set(review.evidence_ids).issubset({sub.evidence_id})
        ):
            raise ValueError("Cleared root identity requires separate reviewed evidence")
        root = roots[root_index]
        ids = (sub.evidence_id,)
        if sub.request.scale_factor is None:
            out_eom.append(EOMEAAttachmentPoint(
                augmentation_level=sub.request.basis.augmentation_level,
                attachment_ea_ev=root.attachment_ea_ev,
                state_identity=review, evidence_ids=ids,
                one_particle_weight=None, basis_label=sub.request.basis.label,
            ))
        else:
            out_stab.append(StabilizationPoint(
                scale_factor=sub.request.scale_factor,
                attachment_ea_ev=root.attachment_ea_ev,
                state_identity=review, evidence_ids=ids,
            ))
    return tuple(out_eom), tuple(out_stab)
