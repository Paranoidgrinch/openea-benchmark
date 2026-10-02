"""Auditable atomic-fragment CCSD(T) execution for dissociation thresholds.

This module exists to provide *computed* fragment thresholds to the attachment
binding gate instead of hand-entered energies or arbitrary large-R molecular
points.

The first version is intentionally narrow:
- one atomic fragment per request,
- explicitly supplied charge and spin,
- RHF -> RCCSD(T) for closed shells,
- ROHF -> semicanonical UHF -> UCCSD(T) for open shells,
- the same numerical Stage-3 execution settings are reused.

The requested atomic state is still user/workflow supplied. This layer does
not discover or prove the atomic ground-state manifold and therefore does not
turn a smoke-test threshold into production dissociation evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Any, Callable

from openea_benchmark.adaptive.stage3_execution import Stage3ExecutionSettings

from .asymptote import DissociationChannel


class FragmentExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    SCF_NOT_CONVERGED = "SCF_NOT_CONVERGED"
    SCF_UNSTABLE = "SCF_UNSTABLE"
    CCSD_NOT_CONVERGED = "CCSD_NOT_CONVERGED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class AtomicFragmentRequest:
    fragment_id: str
    element: str
    charge: int
    spin_2s: int
    basis: str
    methods: tuple[str, ...] = ("CCSD", "CCSD(T)")
    state_label: str | None = None

    def __post_init__(self) -> None:
        if not self.fragment_id.strip():
            raise ValueError("fragment_id must be non-empty")
        if not self.element.strip():
            raise ValueError("element must be non-empty")
        if self.spin_2s < 0:
            raise ValueError("spin_2s must be non-negative")
        if not self.basis.strip():
            raise ValueError("basis must be non-empty")
        if "CCSD" not in self.methods:
            raise ValueError("fragment execution requires CCSD")


@dataclass(frozen=True)
class AtomicFragmentResult:
    fragment_id: str
    status: FragmentExecutionStatus
    element: str
    charge: int
    spin_2s: int
    basis: str
    state_label: str | None
    scf_reference: str
    cc_reference: str | None
    scf_converged: bool
    scf_energy_hartree: float | None
    internal_stable: bool | None
    external_stable: bool | None
    external_stability_available: bool
    ccsd_converged: bool | None
    ccsd_total_hartree: float | None
    triples_correction_hartree: float | None
    ccsd_t_total_hartree: float | None
    s2: float | None
    multiplicity: float | None
    pyscf_version: str | None
    error_type: str | None
    error_message: str | None
    state_assignment_source: str = "EXPLICIT_REQUEST"
    state_manifold_resolved: bool = False
    is_production_threshold: bool = False
    authorizes_pruning: bool = False
    external_instability_waived: bool = False
    external_instability_waiver_reason: str | None = None
    two_electron_ccsd_exact_space: bool = False

    def __post_init__(self) -> None:
        if self.state_manifold_resolved:
            raise ValueError(
                "atomic fragment v0.1 does not establish the complete atomic state manifold"
            )
        if self.is_production_threshold or self.authorizes_pruning:
            raise ValueError(
                "atomic fragment v0.1 cannot claim production threshold or authorize pruning"
            )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = self.status.value
        return out


def _empty_result(
    request: AtomicFragmentRequest,
    status: FragmentExecutionStatus,
    **updates: Any,
) -> AtomicFragmentResult:
    data: dict[str, Any] = dict(
        fragment_id=request.fragment_id,
        status=status,
        element=request.element,
        charge=request.charge,
        spin_2s=request.spin_2s,
        basis=request.basis,
        state_label=request.state_label,
        scf_reference="RHF" if request.spin_2s == 0 else "ROHF",
        cc_reference=None,
        scf_converged=False,
        scf_energy_hartree=None,
        internal_stable=None,
        external_stable=None,
        external_stability_available=False,
        ccsd_converged=None,
        ccsd_total_hartree=None,
        triples_correction_hartree=None,
        ccsd_t_total_hartree=None,
        s2=None,
        multiplicity=None,
        pyscf_version=None,
        error_type=None,
        error_message=None,
    )
    data.update(updates)
    return AtomicFragmentResult(**data)


def _run_atomic_fragment_pyscf(
    request: AtomicFragmentRequest,
    settings: Stage3ExecutionSettings,
) -> AtomicFragmentResult:
    import pyscf
    from pyscf import cc, gto, scf

    mol = gto.M(
        atom=[(request.element, (0.0, 0.0, 0.0))],
        basis=request.basis,
        charge=request.charge,
        spin=request.spin_2s,
        unit="Angstrom",
        symmetry=False,
        verbose=settings.verbose,
        max_memory=settings.max_memory_mb,
    )

    if request.spin_2s == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)

    mf.conv_tol = settings.scf_conv_tol
    mf.conv_tol_grad = settings.scf_conv_tol_grad
    mf.max_cycle = settings.scf_max_cycle
    mf.max_memory = settings.max_memory_mb
    mf.kernel()

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
    )
    if not mf.converged:
        return _empty_result(
            request,
            FragmentExecutionStatus.SCF_NOT_CONVERGED,
            **common,
        )

    if request.spin_2s == 0:
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
        return _empty_result(
            request,
            FragmentExecutionStatus.SCF_UNSTABLE,
            **common,
        )

    # Narrow atomic two-electron exception:
    # For a closed-shell *atomic* two-electron fragment (e.g. H-), RHF may be
    # externally unstable to an unrestricted determinant.  CCSD nevertheless
    # spans the complete excitation space for two electrons in the chosen
    # one-particle basis.  We therefore retain external_stable=False as
    # diagnostic evidence but do not block the correlated energy.  This waiver
    # is not applied to molecules, open shells, or systems with != 2 electrons.
    two_electron_closed_shell_atom = (
        request.spin_2s == 0
        and int(mol.nelectron) == 2
        and int(mol.natm) == 1
    )
    waive_external_instability = (
        settings.require_rhf_external_stability
        and external_stable is False
        and two_electron_closed_shell_atom
    )

    common.update(
        external_instability_waived=waive_external_instability,
        external_instability_waiver_reason=(
            "ATOMIC_CLOSED_SHELL_TWO_ELECTRON_CCSD_COMPLETE_EXCITATION_SPACE"
            if waive_external_instability
            else None
        ),
        two_electron_ccsd_exact_space=two_electron_closed_shell_atom,
    )

    if (
        request.spin_2s == 0
        and settings.require_rhf_external_stability
        and external_stable is False
        and not waive_external_instability
    ):
        return _empty_result(
            request,
            FragmentExecutionStatus.SCF_UNSTABLE,
            **common,
        )

    if request.spin_2s == 0:
        cc_mf = mf
        cc_reference = "RHF"
    else:
        cc_mf = mf.to_uhf()
        mo_energy, mo_coeff = cc_mf.canonicalize(
            cc_mf.mo_coeff,
            cc_mf.mo_occ,
        )
        cc_mf.mo_energy = mo_energy
        cc_mf.mo_coeff = mo_coeff
        cc_reference = "SEMICANONICAL_UHF_FROM_ROHF"

    mycc = cc.CCSD(cc_mf)
    mycc.conv_tol = settings.cc_conv_tol
    mycc.conv_tol_normt = settings.cc_conv_tol_normt
    mycc.max_cycle = settings.cc_max_cycle
    mycc.max_memory = settings.max_memory_mb

    eris = mycc.ao2mo()
    ecorr, t1, t2 = mycc.kernel(eris=eris)
    ccsd_converged = bool(mycc.converged)
    ccsd_total = float(mycc.e_tot) if mycc.e_tot is not None else None

    common.update(
        cc_reference=cc_reference,
        ccsd_converged=ccsd_converged,
        ccsd_total_hartree=ccsd_total,
    )

    if not ccsd_converged:
        return _empty_result(
            request,
            FragmentExecutionStatus.CCSD_NOT_CONVERGED,
            **common,
        )

    triples = None
    total = ccsd_total
    if settings.run_ccsd_t and "CCSD(T)" in request.methods:
        triples = float(mycc.ccsd_t(t1=t1, t2=t2, eris=eris))
        if ccsd_total is not None:
            total = ccsd_total + triples

    return _empty_result(
        request,
        FragmentExecutionStatus.COMPLETED,
        **common,
        triples_correction_hartree=triples,
        ccsd_t_total_hartree=total,
    )


def run_atomic_fragment(
    request: AtomicFragmentRequest,
    *,
    settings: Stage3ExecutionSettings | None = None,
    runner: Callable[
        [AtomicFragmentRequest, Stage3ExecutionSettings],
        AtomicFragmentResult,
    ] | None = None,
) -> AtomicFragmentResult:
    settings = settings or Stage3ExecutionSettings()
    implementation = runner or _run_atomic_fragment_pyscf
    try:
        result = implementation(request, settings)
    except Exception as exc:
        return _empty_result(
            request,
            FragmentExecutionStatus.ERROR,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    if not isinstance(result, AtomicFragmentResult):
        raise TypeError("fragment runner must return AtomicFragmentResult")
    if result.fragment_id != request.fragment_id:
        raise ValueError("fragment runner returned another fragment")
    return result


def build_validation_dissociation_channel(
    *,
    channel_id: str,
    fragment_a: AtomicFragmentResult,
    fragment_b: AtomicFragmentResult,
) -> DissociationChannel:
    """Build a non-production scalar threshold from two completed fragments."""
    if not channel_id.strip():
        raise ValueError("channel_id must be non-empty")
    for result in (fragment_a, fragment_b):
        if result.status is not FragmentExecutionStatus.COMPLETED:
            raise ValueError(
                f"{result.fragment_id}: fragment calculation is not COMPLETED"
            )
        if result.ccsd_t_total_hartree is None:
            raise ValueError(
                f"{result.fragment_id}: missing CCSD(T) total energy"
            )
        if not isfinite(float(result.ccsd_t_total_hartree)):
            raise ValueError(
                f"{result.fragment_id}: non-finite CCSD(T) total energy"
            )

    threshold = (
        float(fragment_a.ccsd_t_total_hartree)
        + float(fragment_b.ccsd_t_total_hartree)
    )
    return DissociationChannel(
        channel_id=channel_id,
        fragment_a=fragment_a.fragment_id,
        fragment_b=fragment_b.fragment_id,
        asymptotic_energy_hartree=threshold,
        source=(
            "VALIDATION_ONLY_CCSD(T)_ATOMIC_FRAGMENTS;"
            "EXPLICIT_ATOMIC_STATES;STATE_MANIFOLD_NOT_RESOLVED"
        ),
    )
