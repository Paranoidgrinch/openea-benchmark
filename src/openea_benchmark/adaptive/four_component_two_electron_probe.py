"""Optional four-component DHF Coulomb/Gaunt/Breit operator sensitivity pilot.

These are *total-energy mean-field Hamiltonian comparisons*, not the omitted
spin-free two-electron X2C picture-change correction and not a production EA/SOC
term.  The Breit variant includes Gaunt.  It MUST NOT be added to the existing
SFX2C1E or FCI-SISO corrections without a separately validated no-double-counting
construction.  Exactly three real Dirac-HF solves are required per point.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from typing import Callable, Mapping

from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV


class FourComponentStatus(str, Enum):
    COMPLETE_REVIEW_REQUIRED = 'COMPLETE_REVIEW_REQUIRED'
    POLICY_BLOCKED = 'POLICY_BLOCKED'
    ERROR = 'ERROR'


class FourComponentHamiltonian(str, Enum):
    DIRAC_COULOMB = 'DIRAC_COULOMB'
    DIRAC_COULOMB_GAUNT = 'DIRAC_COULOMB_GAUNT'
    DIRAC_COULOMB_BREIT = 'DIRAC_COULOMB_BREIT'


HAMILTONIANS = tuple(FourComponentHamiltonian)
METHOD_SCOPE = '4C_DHF_DC_DCG_DCB_MEAN_FIELD_OPERATOR_SENSITIVITY_ONLY'


@dataclass(frozen=True)
class FourComponentPointRequest:
    request_id: str
    role: str
    atoms: tuple[str, str]
    charge: int
    spin_2s: int
    r_angstrom: float
    basis_by_element: Mapping[str, str]
    state_id: str
    source_state_review_id: str
    state_identity_reviewed: bool

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.state_id.strip() or not self.source_state_review_id.strip():
            raise ValueError('Request, state and independent source-review identifiers are required')
        if self.role not in {'neutral', 'anion'} or self.charge != (0 if self.role == 'neutral' else -1):
            raise ValueError('Charge must match neutral/anion role')
        if len(self.atoms) != 2 or any(not x.strip() for x in self.atoms):
            raise ValueError('Exactly two ordered atoms are required')
        if set(self.basis_by_element) != set(self.atoms) or any(not str(x).strip() for x in self.basis_by_element.values()):
            raise ValueError('Explicit nonempty all-electron basis for every element required')
        if not isfinite(float(self.r_angstrom)) or self.r_angstrom <= 0 or self.spin_2s < 0:
            raise ValueError('Invalid geometry or spin')
        if not self.state_identity_reviewed:
            raise ValueError('An independent spin-free reference-state review is required before a costly diagnostic')

    def to_dict(self):
        d = asdict(self)
        d['basis_by_element'] = dict(self.basis_by_element)
        return d


@dataclass(frozen=True)
class FourComponentSettings:
    authorized: bool = False
    scf_conv_tol: float = 1e-10
    max_cycle: int = 120
    max_memory_mb: int = 4000
    max_calculations: int = 3
    nuclear_model: str = 'POINT_NUCLEUS'
    with_ssss: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.scf_conv_tol) or not 0 < self.scf_conv_tol <= 1e-7:
            raise ValueError('Invalid DHF convergence tolerance')
        if self.max_cycle <= 0 or self.max_memory_mb <= 0 or self.max_calculations < 0:
            raise ValueError('Invalid resource budget')
        if self.nuclear_model != 'POINT_NUCLEUS':
            raise ValueError('The pilot only implements an explicit point-nucleus model')
        if not self.with_ssss:
            raise ValueError('DHF operator comparison requires all small-small Coulomb integrals')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FourComponentPointResult:
    request_id: str
    role: str
    status: FourComponentStatus
    reason: str
    energies_hartree: tuple[tuple[str, float], ...] = ()
    gaunt_minus_coulomb_hartree: float | None = None
    breit_minus_coulomb_hartree: float | None = None
    breit_minus_gaunt_hartree: float | None = None
    pyscf_version: str | None = None
    source_signature: str | None = None
    calculations_completed: int = 0
    method_scope: str = METHOD_SCOPE
    two_electron_x2c_picture_change_computed: bool = False
    spin_free_scalar_remainder_computed: bool = False
    independent_soc_correction_computed: bool = False
    correlated_relativistic_correction_computed: bool = False
    production_correction_validated: bool = False
    uncertainty_bounded: bool = False

    def __post_init__(self) -> None:
        if self.method_scope != METHOD_SCOPE or any((
            self.two_electron_x2c_picture_change_computed,
            self.spin_free_scalar_remainder_computed,
            self.independent_soc_correction_computed,
            self.correlated_relativistic_correction_computed,
            self.production_correction_validated,
            self.uncertainty_bounded,
        )):
            raise ValueError('DHF pilot cannot claim production/picture-change/spin-free/correlated evidence')
        if self.status is FourComponentStatus.COMPLETE_REVIEW_REQUIRED:
            if tuple(name for name, _ in self.energies_hartree) != tuple(x.value for x in HAMILTONIANS):
                raise ValueError('Completed DHF comparison needs all three distinct Hamiltonians in order')
            if any(not isfinite(float(e)) for _, e in self.energies_hartree):
                raise ValueError('Nonfinite DHF energy')
            e = [v for _, v in self.energies_hartree]
            if any(x is None or not isfinite(float(x)) for x in (
                self.gaunt_minus_coulomb_hartree,
                self.breit_minus_coulomb_hartree,
                self.breit_minus_gaunt_hartree,
            )) or (abs(self.gaunt_minus_coulomb_hartree - (e[1] - e[0])) > 1e-10
                   or abs(self.breit_minus_coulomb_hartree - (e[2] - e[0])) > 1e-10
                   or abs(self.breit_minus_gaunt_hartree - (e[2] - e[1])) > 1e-10):
                raise ValueError('Inconsistent Dirac-Coulomb/Gaunt/Breit differences')
            if self.calculations_completed != 3 or not self.pyscf_version or not self.source_signature:
                raise ValueError('Incomplete DHF provenance')
        else:
            if self.energies_hartree or any(x is not None for x in (
                self.gaunt_minus_coulomb_hartree,
                self.breit_minus_coulomb_hartree,
                self.breit_minus_gaunt_hartree,
            )):
                raise ValueError('Failed/blocked DHF cannot publish partial correction candidates')

    def to_dict(self):
        d = asdict(self)
        d['status'] = self.status.value
        d['energies_hartree'] = dict(self.energies_hartree)
        return d


@dataclass(frozen=True)
class FourComponentPairResult:
    status: FourComponentStatus
    neutral: FourComponentPointResult
    anion: FourComponentPointResult
    gaunt_ea_sensitivity_ev: float | None = None
    breit_ea_sensitivity_ev: float | None = None
    gauge_increment_ea_sensitivity_ev: float | None = None
    method_scope: str = METHOD_SCOPE
    production_correction_validated: bool = False
    uncertainty_bounded: bool = False
    independent_soc_correction_computed: bool = False
    scalar_two_electron_x2c_remainder_computed: bool = False

    def __post_init__(self):
        if self.method_scope != METHOD_SCOPE or any((
            self.production_correction_validated, self.uncertainty_bounded,
            self.independent_soc_correction_computed,
            self.scalar_two_electron_x2c_remainder_computed,
        )):
            raise ValueError('4c-DHF sensitivity cannot be promoted to physical correction')
        if self.status is FourComponentStatus.COMPLETE_REVIEW_REQUIRED:
            if any(x.status is not FourComponentStatus.COMPLETE_REVIEW_REQUIRED for x in (self.neutral, self.anion)):
                raise ValueError('Pair requires completed points')
            for name in ('gaunt_ea_sensitivity_ev', 'breit_ea_sensitivity_ev', 'gauge_increment_ea_sensitivity_ev'):
                value = getattr(self, name)
                if value is None or not isfinite(float(value)):
                    raise ValueError('Completed pair needs finite observed sensitivities')
        elif any(getattr(self, x) is not None for x in (
            'gaunt_ea_sensitivity_ev', 'breit_ea_sensitivity_ev', 'gauge_increment_ea_sensitivity_ev'
        )):
            raise ValueError('Unresolved pair cannot publish EA sensitivity')

    def to_dict(self):
        d = asdict(self)
        d['status'] = self.status.value
        d['neutral'] = self.neutral.to_dict()
        d['anion'] = self.anion.to_dict()
        return d


# Backend injection exists for deterministic failure/provenance tests;
# production uses PySCF's actual four-component DHF implementation.
EnergyBackend = Callable[[FourComponentPointRequest, FourComponentSettings, FourComponentHamiltonian], tuple[float, bool, str]]


def _pyscf_dhf_energy(request: FourComponentPointRequest, settings: FourComponentSettings,
                      hamiltonian: FourComponentHamiltonian) -> tuple[float, bool, str]:
    import pyscf
    from pyscf import gto, scf

    mol = gto.M(
        atom=[(request.atoms[0], (0.0, 0.0, 0.0)),
              (request.atoms[1], (0.0, 0.0, request.r_angstrom))],
        basis=dict(request.basis_by_element), charge=request.charge,
        spin=request.spin_2s, unit='Angstrom', symmetry=False,
        max_memory=settings.max_memory_mb, verbose=0,
    )
    mf = scf.DHF(mol)
    mf.with_ssss = True
    mf.with_gaunt = hamiltonian is FourComponentHamiltonian.DIRAC_COULOMB_GAUNT
    # Breit *includes* Gaunt; do not enable both toggles as two independent terms.
    mf.with_breit = hamiltonian is FourComponentHamiltonian.DIRAC_COULOMB_BREIT
    mf.conv_tol = settings.scf_conv_tol
    mf.max_cycle = settings.max_cycle
    mf.max_memory = settings.max_memory_mb
    mf.verbose = 0
    e = float(mf.kernel())
    return e, bool(mf.converged), str(pyscf.__version__)


def run_four_component_point(request: FourComponentPointRequest,
                             settings: FourComponentSettings,
                             *, backend: EnergyBackend | None = None) -> FourComponentPointResult:
    def blocked(reason: str, status=FourComponentStatus.POLICY_BLOCKED, count=0):
        return FourComponentPointResult(request_id=request.request_id, role=request.role,
                                        status=status, reason=reason, calculations_completed=count)

    if not settings.authorized:
        return blocked('Explicit expensive-DHF authorization required')
    if settings.max_calculations < len(HAMILTONIANS):
        return blocked('Budget must cover three complete DHF Hamiltonian runs')
    backend = _pyscf_dhf_energy if backend is None else backend
    out = []
    versions = set()
    for ham in HAMILTONIANS:
        try:
            e, converged, version = backend(request, settings, ham)
            if not isfinite(float(e)) or not converged or not str(version).strip():
                return blocked(f'{ham.value}: unconverged/nonfinite energy or missing PySCF provenance',
                               FourComponentStatus.ERROR, len(out) + 1)
            out.append((ham.value, float(e)))
            versions.add(str(version))
        except Exception as exc:
            return blocked(f'{ham.value}: {type(exc).__name__}: {exc}', FourComponentStatus.ERROR, len(out) + 1)
    if len(versions) != 1:
        return blocked('Backend version drift within the matched three-run series', FourComponentStatus.ERROR, 3)
    energies = [e for _, e in out]
    signature = sha256(json.dumps({
        'request': request.to_dict(), 'settings': settings.to_dict(),
        'hamiltonians': [x.value for x in HAMILTONIANS], 'backend_version': next(iter(versions)),
    }, sort_keys=True).encode()).hexdigest()
    return FourComponentPointResult(
        request_id=request.request_id, role=request.role,
        status=FourComponentStatus.COMPLETE_REVIEW_REQUIRED,
        reason='4c-DHF operator sensitivity only; no spin-free or SOC correction certified',
        energies_hartree=tuple(out), gaunt_minus_coulomb_hartree=energies[1] - energies[0],
        breit_minus_coulomb_hartree=energies[2] - energies[0],
        breit_minus_gaunt_hartree=energies[2] - energies[1],
        pyscf_version=next(iter(versions)), source_signature=signature, calculations_completed=3,
    )


def assess_four_component_pair(neutral_request: FourComponentPointRequest,
                              anion_request: FourComponentPointRequest,
                              neutral: FourComponentPointResult,
                              anion: FourComponentPointResult) -> FourComponentPairResult:
    if (neutral_request.role != 'neutral' or anion_request.role != 'anion'
            or neutral_request.atoms != anion_request.atoms
            or dict(neutral_request.basis_by_element) != dict(anion_request.basis_by_element)
            or neutral_request.source_state_review_id == anion_request.source_state_review_id
            or neutral_request.request_id == anion_request.request_id
            or neutral.request_id != neutral_request.request_id or neutral.role != 'neutral'
            or anion.request_id != anion_request.request_id or anion.role != 'anion'):
        raise ValueError('DHF pair provenance, atom ordering, basis and roles must match')
    if (neutral.status is not FourComponentStatus.COMPLETE_REVIEW_REQUIRED
            or anion.status is not FourComponentStatus.COMPLETE_REVIEW_REQUIRED
            or neutral.pyscf_version != anion.pyscf_version):
        return FourComponentPairResult(status=FourComponentStatus.ERROR, neutral=neutral, anion=anion)
    # These are *not* independent scalar/SOC EA correction terms: mixed spin
    # Breit/Gaunt mean-field sensitivities are never automatically additive.
    return FourComponentPairResult(
        status=FourComponentStatus.COMPLETE_REVIEW_REQUIRED,
        neutral=neutral, anion=anion,
        gaunt_ea_sensitivity_ev=(neutral.gaunt_minus_coulomb_hartree - anion.gaunt_minus_coulomb_hartree) * HARTREE_TO_EV,
        breit_ea_sensitivity_ev=(neutral.breit_minus_coulomb_hartree - anion.breit_minus_coulomb_hartree) * HARTREE_TO_EV,
        gauge_increment_ea_sensitivity_ev=(neutral.breit_minus_gaunt_hartree - anion.breit_minus_gaunt_hartree) * HARTREE_TO_EV,
    )
