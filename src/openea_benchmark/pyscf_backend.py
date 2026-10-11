from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re
from typing import Iterable

from .context import (
    calculation_context,
    canonical_json,
    checkpoint_context,
    context_digest,
)
from .geometry import (
    GEOMETRY_DECIMALS,
    canonicalize_r_angstrom,
)
from .root_record import (
    SCFRootRecord,
    SCFRunStatus,
)


EH_TO_EV = 27.211386245988

DEFAULT_GUESSES = (
    "minao",
    "atom",
    "hcore",
)


@dataclass(frozen=True)
class DiatomicSpec:
    """
    One fixed-geometry diatomic electronic-structure problem.
    """

    label: str
    atom_a: str
    atom_b: str

    r_angstrom: float
    charge: int
    spin_2s: int

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError(
                "label must be non-empty"
            )

        if not self.atom_a.strip():
            raise ValueError(
                "atom_a must be non-empty"
            )

        if not self.atom_b.strip():
            raise ValueError(
                "atom_b must be non-empty"
            )

        object.__setattr__(
            self,
            "r_angstrom",
            canonicalize_r_angstrom(
                self.r_angstrom
            ),
        )

        if self.spin_2s < 0:
            raise ValueError(
                "spin_2s must be >= 0"
            )


@dataclass(frozen=True)
class DFTMethodSpec:
    """
    DFT model and one-electron basis specification.

    ECPs are assigned element by element. This avoids applying a def2 ECP
    globally to atoms for which no ECP is intended.
    """

    functional: str
    basis: str

    ecp_assignments: tuple[
        tuple[str, str],
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not self.functional.strip():
            raise ValueError(
                "functional must be non-empty"
            )

        if not self.basis.strip():
            raise ValueError(
                "basis must be non-empty"
            )

        seen: set[str] = set()

        for element, ecp_name in self.ecp_assignments:
            if not element.strip():
                raise ValueError(
                    "ECP element must be non-empty"
                )

            if not ecp_name.strip():
                raise ValueError(
                    "ECP name must be non-empty"
                )

            if element in seen:
                raise ValueError(
                    f"duplicate ECP assignment for {element}"
                )

            seen.add(element)

        object.__setattr__(
            self,
            "ecp_assignments",
            tuple(
                sorted(
                    self.ecp_assignments
                )
            ),
        )


@dataclass(frozen=True)
class SCFSettings:
    """
    Numerical settings for the Tier-0 SCF backend.

    The defaults reproduce the validated development settings. They are
    workflow defaults, not claims of universal scientific optimality.

    The rescue shift and damping are temporary convergence aids. A rescued
    root is accepted only after an unshifted, undamped cleanup SCF.
    """

    conv_tol: float = 1.0e-10
    max_cycle: int = 200

    grid_level: int = 5

    stability_tol: float = 1.0e-6
    stability_nroots: int = 3
    stability_max_rounds: int = 8

    stability_conv_tol: float = 1.0e-11
    stability_max_cycle: int = 100

    rescue_level_shift: float = 0.20
    rescue_damp: float = 0.20
    rescue_max_cycle: int = 400

    num_threads: int = 1

    def __post_init__(self) -> None:
        if (
            not isfinite(self.conv_tol)
            or self.conv_tol <= 0.0
        ):
            raise ValueError(
                "conv_tol must be finite and > 0"
            )

        if self.max_cycle < 1:
            raise ValueError(
                "max_cycle must be >= 1"
            )

        if self.grid_level < 0:
            raise ValueError(
                "grid_level must be >= 0"
            )

        if (
            not isfinite(self.stability_tol)
            or self.stability_tol <= 0.0
        ):
            raise ValueError(
                "stability_tol must be finite and > 0"
            )

        if self.stability_nroots < 1:
            raise ValueError(
                "stability_nroots must be >= 1"
            )

        if self.stability_max_rounds < 0:
            raise ValueError(
                "stability_max_rounds must be >= 0"
            )

        if (
            not isfinite(
                self.stability_conv_tol
            )
            or self.stability_conv_tol <= 0.0
        ):
            raise ValueError(
                "stability_conv_tol must be finite and > 0"
            )

        if self.stability_max_cycle < 1:
            raise ValueError(
                "stability_max_cycle must be >= 1"
            )

        if (
            not isfinite(
                self.rescue_level_shift
            )
            or self.rescue_level_shift < 0.0
        ):
            raise ValueError(
                "rescue_level_shift must be finite and >= 0"
            )

        if (
            not isfinite(self.rescue_damp)
            or not 0.0 <= self.rescue_damp < 1.0
        ):
            raise ValueError(
                "rescue_damp must satisfy 0 <= damp < 1"
            )

        if self.rescue_max_cycle < 1:
            raise ValueError(
                "rescue_max_cycle must be >= 1"
            )

        if self.num_threads < 1:
            raise ValueError(
                "num_threads must be >= 1"
            )


def build_molecule(
    spec: DiatomicSpec,
    method: DFTMethodSpec,
):
    """
    Construct the PySCF Mole object for one fixed-R diatomic problem.
    """
    from pyscf import gto

    atom = (
        f"{spec.atom_a} 0.0 0.0 0.0; "
        f"{spec.atom_b} 0.0 0.0 {spec.r_angstrom:.12f}"
    )

    kwargs = dict(
        atom=atom,
        basis=method.basis,
        charge=spec.charge,
        spin=spec.spin_2s,
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )

    if method.ecp_assignments:
        kwargs["ecp"] = dict(
            method.ecp_assignments
        )

    return gto.M(**kwargs)


def _reference_for(
    spec: DiatomicSpec,
) -> str:
    if spec.spin_2s == 0:
        return "RKS"

    return "UKS"


def _slug(value: str) -> str:
    value = re.sub(
        r"[^A-Za-z0-9._+-]+",
        "_",
        value,
    )

    return value.strip("_") or "root"


def _calculation_context(
    spec: DiatomicSpec,
    method: DFTMethodSpec,
    guess: str,
) -> dict:
    return calculation_context(
        molecule=spec.label,
        atom_a=spec.atom_a,
        atom_b=spec.atom_b,
        charge=spec.charge,
        spin_2s=spec.spin_2s,
        r_angstrom=spec.r_angstrom,
        functional=method.functional,
        basis=method.basis,
        reference=_reference_for(
            spec
        ),
        origin_guess=guess,
        ecp_assignments=(
            method.ecp_assignments
        ),
    )


def _ecp_context_tag(
    method: DFTMethodSpec,
) -> str:
    if not method.ecp_assignments:
        return "none"

    return context_digest(
        {
            "ecp_assignments": [
                list(item)
                for item
                in method.ecp_assignments
            ]
        }
    )


def _root_id(
    spec: DiatomicSpec,
    method: DFTMethodSpec,
    guess: str,
) -> str:
    context = _calculation_context(
        spec,
        method,
        guess,
    )

    full_context_tag = (
        context_digest(
            context
        )
    )

    raw = (
        f"{spec.label}"
        f"__{spec.atom_a}-{spec.atom_b}"
        f"__q{spec.charge:+d}"
        f"__s{spec.spin_2s}"
        f"__R{spec.r_angstrom:.{GEOMETRY_DECIMALS}f}"
        f"__{method.functional}"
        f"__{method.basis}"
        f"__ecp-{_ecp_context_tag(method)}"
        f"__{guess}"
        f"__ctx-{full_context_tag}"
    )

    return _slug(raw)


def _checkpoint_path(
    root_id: str,
    checkpoint_dir: Path | None,
) -> Path | None:
    if checkpoint_dir is None:
        return None

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        checkpoint_dir
        / f"{root_id}.chk"
    )


def _write_final_checkpoint(
    mf,
    checkpoint: Path | None,
    *,
    spec: DiatomicSpec,
    method: DFTMethodSpec,
    root_id: str,
    guess: str,
) -> None:
    """
    Explicitly persist the final SCF solution and OpenEA scientific context.

    The OpenEA metadata is written into the same HDF5 checkpoint as the
    final SCF solution. A checkpoint is therefore self-describing with
    respect to the calculation context from which its SCF root originated.
    """
    if checkpoint is None:
        return

    from pyscf import lib
    from pyscf.scf import chkfile

    chkfile.dump_scf(
        mf.mol,
        str(checkpoint),
        float(mf.e_tot),
        mf.mo_energy,
        mf.mo_coeff,
        mf.mo_occ,
        overwrite_mol=True,
    )

    payload = checkpoint_context(
        root_id=root_id,
        calculation=(
            _calculation_context(
                spec,
                method,
                guess,
            )
        ),
    )

    lib.chkfile.dump(
        str(checkpoint),
        "openea/context_json",
        canonical_json(
            payload
        ),
    )


def _new_mean_field(
    mol,
    method: DFTMethodSpec,
    settings: SCFSettings,
    checkpoint: Path | None,
):
    from pyscf import dft

    if mol.spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)

    mf.xc = method.functional
    mf.conv_tol = settings.conv_tol
    mf.max_cycle = settings.max_cycle

    mf.grids.level = (
        settings.grid_level
    )

    if hasattr(mf, "nlcgrids"):
        mf.nlcgrids.level = (
            settings.grid_level
        )

    if checkpoint is not None:
        mf.chkfile = str(checkpoint)

    return mf


def _run_standard_or_rescue(
    mol,
    method: DFTMethodSpec,
    settings: SCFSettings,
    guess: str,
    checkpoint: Path | None,
):
    diagnostics: list[str] = []

    standard = _new_mean_field(
        mol,
        method,
        settings,
        checkpoint,
    )

    standard.init_guess = guess

    try:
        standard.kernel()
    except Exception as exc:
        diagnostics.append(
            "standard SCF raised "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        if standard.converged:
            return (
                standard,
                "standard",
                diagnostics,
            )

        diagnostics.append(
            "standard SCF did not converge"
        )

    rescue = _new_mean_field(
        mol,
        method,
        settings,
        checkpoint,
    )

    rescue.init_guess = guess
    rescue.max_cycle = (
        settings.rescue_max_cycle
    )
    rescue.level_shift = (
        settings.rescue_level_shift
    )
    rescue.damp = (
        settings.rescue_damp
    )

    try:
        rescue.kernel()
    except Exception as exc:
        diagnostics.append(
            "shifted rescue raised "
            f"{type(exc).__name__}: {exc}"
        )

        return (
            None,
            "failed",
            diagnostics,
        )

    if not rescue.converged:
        diagnostics.append(
            "shifted rescue did not converge"
        )

        return (
            None,
            "failed",
            diagnostics,
        )

    dm_rescue = rescue.make_rdm1()

    clean = _new_mean_field(
        mol,
        method,
        settings,
        checkpoint,
    )

    clean.max_cycle = (
        settings.rescue_max_cycle
    )

    try:
        clean.kernel(
            dm0=dm_rescue
        )
    except Exception as exc:
        diagnostics.append(
            "unshifted cleanup raised "
            f"{type(exc).__name__}: {exc}"
        )

        return (
            None,
            "failed",
            diagnostics,
        )

    if not clean.converged:
        diagnostics.append(
            "unshifted cleanup did not converge"
        )

        return (
            None,
            "failed",
            diagnostics,
        )

    diagnostics.append(
        "standard SCF failed; "
        "temporary shifted/damped rescue "
        "followed by unshifted cleanup succeeded"
    )

    return (
        clean,
        "shifted_rescue_clean_restart",
        diagnostics,
    )


def _internal_stability(
    mf,
    reference: str,
    settings: SCFSettings,
):
    from pyscf.scf import stability

    initial_energy = float(
        mf.e_tot
    )

    current = mf
    diagnostics: list[str] = []

    for round_index in range(
        settings.stability_max_rounds + 1
    ):
        try:
            if reference == "RKS":
                (
                    mo_internal,
                    _,
                    stable_internal,
                    _,
                ) = stability.rhf_stability(
                    current,
                    internal=True,
                    external=False,
                    return_status=True,
                    nroots=(
                        settings.stability_nroots
                    ),
                    tol=settings.stability_tol,
                )

            elif reference == "UKS":
                (
                    mo_internal,
                    _,
                    stable_internal,
                    _,
                ) = stability.uhf_stability(
                    current,
                    internal=True,
                    external=False,
                    return_status=True,
                    nroots=(
                        settings.stability_nroots
                    ),
                    tol=settings.stability_tol,
                )

            else:
                raise ValueError(
                    f"unsupported reference {reference}"
                )

        except Exception as exc:
            diagnostics.append(
                "internal stability analysis raised "
                f"{type(exc).__name__}: {exc}"
            )

            shift_ev = (
                float(current.e_tot)
                - initial_energy
            ) * EH_TO_EV

            return (
                current,
                False,
                shift_ev,
                diagnostics,
            )

        if bool(stable_internal):
            shift_ev = (
                float(current.e_tot)
                - initial_energy
            ) * EH_TO_EV

            return (
                current,
                True,
                shift_ev,
                diagnostics,
            )

        if (
            round_index
            >= settings.stability_max_rounds
        ):
            diagnostics.append(
                "internal stability not reached "
                "within configured rotation limit"
            )

            shift_ev = (
                float(current.e_tot)
                - initial_energy
            ) * EH_TO_EV

            return (
                current,
                False,
                shift_ev,
                diagnostics,
            )

        newton = current.newton()

        newton.conv_tol = (
            settings.stability_conv_tol
        )

        newton.max_cycle = (
            settings.stability_max_cycle
        )

        try:
            energy = newton.kernel(
                mo_coeff=mo_internal,
                mo_occ=current.mo_occ,
            )
        except Exception as exc:
            diagnostics.append(
                "stability-follow Newton SCF raised "
                f"{type(exc).__name__}: {exc}"
            )

            shift_ev = (
                float(current.e_tot)
                - initial_energy
            ) * EH_TO_EV

            return (
                current,
                False,
                shift_ev,
                diagnostics,
            )

        if (
            not newton.converged
            or not isfinite(float(energy))
        ):
            diagnostics.append(
                "stability-follow Newton SCF "
                "did not converge"
            )

            shift_ev = (
                float(current.e_tot)
                - initial_energy
            ) * EH_TO_EV

            return (
                current,
                False,
                shift_ev,
                diagnostics,
            )

        current = newton

    raise RuntimeError(
        "unreachable stability loop state"
    )


def _external_rks_stability(
    mf,
    settings: SCFSettings,
):
    from pyscf.scf import stability

    diagnostics: list[str] = []

    try:
        (
            _,
            _,
            stable_internal,
            stable_external,
        ) = stability.rhf_stability(
            mf,
            internal=True,
            external=True,
            return_status=True,
            nroots=settings.stability_nroots,
            tol=settings.stability_tol,
        )

    except Exception as exc:
        diagnostics.append(
            "RKS external stability analysis raised "
            f"{type(exc).__name__}: {exc}"
        )

        return (
            None,
            diagnostics,
        )

    if not bool(stable_internal):
        diagnostics.append(
            "external-stability recheck unexpectedly "
            "reported internal instability"
        )

    return (
        bool(stable_external),
        diagnostics,
    )


def _spin_data(
    mf,
    reference: str,
):
    diagnostics: list[str] = []

    if reference == "RKS":
        return (
            0.0,
            1.0,
            diagnostics,
        )

    try:
        s2, multiplicity = (
            mf.spin_square()
        )

        return (
            float(s2),
            float(multiplicity),
            diagnostics,
        )

    except Exception as exc:
        diagnostics.append(
            "spin_square raised "
            f"{type(exc).__name__}: {exc}"
        )

        return (
            None,
            None,
            diagnostics,
        )


def _failed_record(
    *,
    root_id: str,
    spec: DiatomicSpec,
    method: DFTMethodSpec,
    guess: str,
    reference: str,
    checkpoint: Path | None,
    scf_path: str,
    diagnostics: Iterable[str],
) -> SCFRootRecord:
    return SCFRootRecord(
        root_id=root_id,
        molecule=spec.label,
        atom_a=spec.atom_a,
        atom_b=spec.atom_b,
        charge=spec.charge,
        spin_2s=spec.spin_2s,
        r_angstrom=spec.r_angstrom,
        functional=method.functional,
        basis=method.basis,
        origin_guess=guess,
        scf_path=scf_path,
        reference=reference,
        status=SCFRunStatus.FAILED,
        checkpoint_path=(
            str(checkpoint)
            if checkpoint is not None
            else None
        ),
        diagnostic_message="; ".join(
            diagnostics
        ),
        ecp_assignments=(
            method.ecp_assignments
        ),
    )


def run_scf_attempt(
    spec: DiatomicSpec,
    method: DFTMethodSpec,
    guess: str,
    *,
    settings: SCFSettings | None = None,
    checkpoint_dir: Path | str | None = None,
) -> SCFRootRecord:
    """
    Run one generic SCF-root attempt.

    Sequence:

    1. standard RKS/UKS SCF;
    2. if needed, temporary shifted/damped rescue;
    3. mandatory unshifted cleanup after rescue;
    4. internal stability canonicalization;
    5. RKS external-stability diagnostic;
    6. spin diagnostic.

    No root is discarded because of spin contamination or external
    instability. Those are recorded as scientific diagnostics.
    """
    from pyscf import lib

    if settings is None:
        settings = SCFSettings()

    from .compute_threads import set_compute_threads
    set_compute_threads(settings.num_threads)

    guess = str(guess).strip()

    if not guess:
        raise ValueError(
            "guess must be non-empty"
        )

    reference = _reference_for(
        spec
    )

    root_id = _root_id(
        spec,
        method,
        guess,
    )

    checkpoint_root = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else None
    )

    checkpoint = _checkpoint_path(
        root_id,
        checkpoint_root,
    )

    diagnostics: list[str] = []

    try:
        mol = build_molecule(
            spec,
            method,
        )
    except Exception as exc:
        diagnostics.append(
            "molecule construction raised "
            f"{type(exc).__name__}: {exc}"
        )

        return _failed_record(
            root_id=root_id,
            spec=spec,
            method=method,
            guess=guess,
            reference=reference,
            checkpoint=checkpoint,
            scf_path="not_started",
            diagnostics=diagnostics,
        )

    (
        mf,
        scf_path,
        convergence_diagnostics,
    ) = _run_standard_or_rescue(
        mol,
        method,
        settings,
        guess,
        checkpoint,
    )

    diagnostics.extend(
        convergence_diagnostics
    )

    if mf is None:
        return _failed_record(
            root_id=root_id,
            spec=spec,
            method=method,
            guess=guess,
            reference=reference,
            checkpoint=checkpoint,
            scf_path=scf_path,
            diagnostics=diagnostics,
        )

    (
        final_mf,
        internal_stable,
        stability_shift_ev,
        stability_diagnostics,
    ) = _internal_stability(
        mf,
        reference,
        settings,
    )

    diagnostics.extend(
        stability_diagnostics
    )

    external_stable: bool | None = None

    if (
        reference == "RKS"
        and internal_stable
    ):
        (
            external_stable,
            external_diagnostics,
        ) = _external_rks_stability(
            final_mf,
            settings,
        )

        diagnostics.extend(
            external_diagnostics
        )

    (
        s2,
        observed_multiplicity,
        spin_diagnostics,
    ) = _spin_data(
        final_mf,
        reference,
    )

    diagnostics.extend(
        spin_diagnostics
    )

    if internal_stable:
        status = (
            SCFRunStatus.CANONICALIZED
        )
    else:
        status = (
            SCFRunStatus.CONVERGED
        )

    checkpoint_for_record = checkpoint

    if checkpoint is not None:
        try:
            _write_final_checkpoint(
                final_mf,
                checkpoint,
                spec=spec,
                method=method,
                root_id=root_id,
                guess=guess,
            )

        except Exception as exc:
            diagnostics.append(
                "final checkpoint write raised "
                f"{type(exc).__name__}: {exc}"
            )

            #
            # Never expose a possibly stale checkpoint as if it represented
            # the final root.
            #
            checkpoint_for_record = None

    return SCFRootRecord(
        root_id=root_id,
        molecule=spec.label,
        atom_a=spec.atom_a,
        atom_b=spec.atom_b,
        charge=spec.charge,
        spin_2s=spec.spin_2s,
        r_angstrom=spec.r_angstrom,
        functional=method.functional,
        basis=method.basis,
        origin_guess=guess,
        scf_path=scf_path,
        reference=reference,
        status=status,
        energy_hartree=float(
            final_mf.e_tot
        ),
        internal_stable=(
            bool(internal_stable)
        ),
        external_stable=(
            external_stable
        ),
        stability_shift_ev=(
            float(stability_shift_ev)
        ),
        s2=s2,
        observed_multiplicity=(
            observed_multiplicity
        ),
        checkpoint_path=(
            str(checkpoint_for_record)
            if checkpoint_for_record is not None
            else None
        ),
        diagnostic_message="; ".join(
            diagnostics
        ),
        ecp_assignments=(
            method.ecp_assignments
        ),
    )


def run_guess_panel(
    spec: DiatomicSpec,
    method: DFTMethodSpec,
    *,
    guesses: Iterable[str] = DEFAULT_GUESSES,
    settings: SCFSettings | None = None,
    checkpoint_dir: Path | str | None = None,
) -> tuple[SCFRootRecord, ...]:
    """
    Run independent generic SCF attempts for one fixed geometry/spin sector.

    This function deliberately performs no energetic root selection and no
    deduplication. Every attempted root is returned for later state-identity
    analysis.
    """
    guess_tuple = tuple(
        str(x).strip()
        for x in guesses
    )

    if not guess_tuple:
        raise ValueError(
            "at least one guess is required"
        )

    if any(
        not x
        for x in guess_tuple
    ):
        raise ValueError(
            "guesses must be non-empty"
        )

    if (
        len(set(guess_tuple))
        != len(guess_tuple)
    ):
        raise ValueError(
            "duplicate guesses are not allowed"
        )

    records = []

    for guess in guess_tuple:
        records.append(
            run_scf_attempt(
                spec,
                method,
                guess,
                settings=settings,
                checkpoint_dir=(
                    checkpoint_dir
                ),
            )
        )

    return tuple(records)
