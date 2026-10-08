"""Finite-difference Hartree-Fock diagonal Born-Oppenheimer correction pilot.

The gauge-invariant Slater-determinant fidelity at nuclear displacements +/-h
estimates the diagonal *HF* quantum metric along each Cartesian nuclear axis.

This code computes no CCSD/NEVPT2 correlation to DBOC, no off-diagonal
nonadiabatic coupling, and no production correction or certified uncertainty.
All nuclei, isotopic nuclear masses and numerical step sizes are explicit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from math import exp, isfinite
from typing import Any, Callable, Mapping

import numpy as np

from openea_benchmark.attachment.component_resolved_cbs import HARTREE_TO_EV

AMU_TO_ELECTRON_MASS = 1822.888486209
BOHR_TO_ANGSTROM = 0.529177210903


class DBOCStatus(str, Enum):
    COMPLETE_REVIEW_REQUIRED = 'COMPLETE_REVIEW_REQUIRED'
    UNRESOLVED = 'UNRESOLVED'
    ERROR = 'ERROR'


@dataclass(frozen=True)
class HFDBOCRequest:
    request_id: str
    role: str
    atoms: tuple[str, str]
    charge: int
    spin_2s: int
    r_angstrom: float
    basis_by_element: Mapping[str, str]
    state_id: str
    state_identity_reviewed: bool
    nuclear_masses_amu: tuple[float, float]
    nuclear_mass_source: str
    scf_reference: str

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.state_id.strip():
            raise ValueError('Request and state provenance IDs required')
        if self.role not in {'neutral', 'anion'}:
            raise ValueError('Role must be neutral or anion')
        if (self.role == 'neutral' and self.charge != 0) or (self.role == 'anion' and self.charge != -1):
            raise ValueError('Neutral/anion charge mismatch')
        if len(self.atoms) != 2 or any(not a.strip() for a in self.atoms):
            raise ValueError('Diatomic requires two ordered atoms')
        if set(self.basis_by_element) != set(self.atoms) or any(not str(v).strip() for v in self.basis_by_element.values()):
            raise ValueError('Explicit basis for every element required')
        if not isfinite(self.r_angstrom) or self.r_angstrom <= 0:
            raise ValueError('Invalid bond length')
        if self.spin_2s < 0:
            raise ValueError('Invalid spin_2s')
        if self.scf_reference not in {'RHF', 'ROHF', 'UHF'}:
            raise ValueError('Unsupported reference')
        if self.scf_reference == 'RHF' and self.spin_2s != 0:
            raise ValueError('RHF cannot represent open-shell spin')
        if len(self.nuclear_masses_amu) != 2 or any(not isfinite(m) or m <= 0 for m in self.nuclear_masses_amu):
            raise ValueError('Explicit positive nuclear masses in amu required')
        if not self.nuclear_mass_source.strip():
            raise ValueError('Nuclear mass provenance required')
        if not self.state_identity_reviewed:
            raise ValueError('HF state identity must be reviewed before DBOC diagnostic')


@dataclass(frozen=True)
class HFDBOCSettings:
    steps_bohr: tuple[float, ...] = (0.008, 0.004)
    scf_conv_tol: float = 1e-11
    max_cycle: int = 150
    max_memory_mb: int = 4000
    max_scf_calculations: int = 24
    min_determinant_fidelity: float = 0.70
    verbose: int = 0

    def __post_init__(self) -> None:
        if len(self.steps_bohr) != 2 or not (self.steps_bohr[0] > self.steps_bohr[1] > 0):
            raise ValueError('Exactly two descending positive steps required')
        if any(not isfinite(h) or h < 1e-4 or h > 0.1 for h in self.steps_bohr):
            raise ValueError('Unsupported finite-difference step in bohr')
        if not isfinite(self.scf_conv_tol) or not 0 < self.scf_conv_tol <= 1e-7:
            raise ValueError('SCF tolerance too loose or invalid')
        if min(self.max_cycle, self.max_memory_mb, self.max_scf_calculations) <= 0:
            raise ValueError('Invalid resource budgets')
        if self.max_scf_calculations < 12 * len(self.steps_bohr):
            raise ValueError('SCF budget insufficient for all two-atom Cartesian displacements')
        if not 0 < self.min_determinant_fidelity < 1:
            raise ValueError('Invalid minimum determinant fidelity')


@dataclass(frozen=True)
class DBOCAxis:
    atom_index: int
    cartesian_axis: int
    step_bohr: float
    fidelity: float
    metric_inverse_bohr2: float
    contribution_hartree: float


@dataclass(frozen=True)
class HFDBOCResult:
    request_id: str
    role: str
    status: DBOCStatus
    reason: str
    steps_bohr: tuple[float, ...]
    dboc_by_step_hartree: tuple[float, ...] = ()
    axes: tuple[DBOCAxis, ...] = ()
    max_step_difference_ev: float | None = None
    source_signature: str | None = None
    scf_count: int = 0
    method_scope: str = 'NONRELATIVISTIC_HF_DETERMINANT_FINITE_DIFFERENCE'
    correlated_dboc_included: bool = False
    nonadiabatic_off_diagonal_included: bool = False
    scientific_uncertainty_bounded: bool = False
    is_production_correction: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        return d


@dataclass(frozen=True)
class HFDBOCPair:
    status: DBOCStatus
    neutral: HFDBOCResult
    anion: HFDBOCResult
    delta_ea_hartree_candidate: float | None
    observed_step_sensitivity_ev: float | None
    is_production_correction: bool = False
    correlated_remainder_bounded: bool = False
    nonadiabatic_remainder_bounded: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        d['neutral'] = self.neutral.to_dict()
        d['anion'] = self.anion.to_dict()
        return d


def determinant_fidelity(
    minus_alpha: np.ndarray, minus_beta: np.ndarray,
    plus_alpha: np.ndarray, plus_beta: np.ndarray, cross_overlap: np.ndarray,
) -> float:
    """Gauge-independent |<D_-|D_+>|^2 using *cross-geometry* AO metric.

    Orbital coefficient matrices contain occupied spin orbitals as columns.
    The two spin sectors are multiplied, not added. Zero-electron sectors
    contribute unit overlap. Uses slogdet to avoid determinant overflow.
    """
    s = np.asarray(cross_overlap)
    if s.ndim != 2 or not np.isfinite(s).all():
        raise ValueError('Cross-AO overlap must be a finite matrix')
    log_abs = 0.0
    for a, b in ((minus_alpha, plus_alpha), (minus_beta, plus_beta)):
        a, b = np.asarray(a), np.asarray(b)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != s.shape[0] or b.shape[0] != s.shape[1]:
            raise ValueError('Occupied spin-orbital / AO metric dimension mismatch')
        if a.shape[1] != b.shape[1] or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Electron count changed or invalid orbital coefficients')
        if a.shape[1]:
            sign, logdet = np.linalg.slogdet(a.conj().T @ s @ b)
            if abs(sign) < 1e-12 or not isfinite(float(logdet)):
                raise ValueError('Singular occupied-orbital cross overlap')
            log_abs += float(logdet)
    log_fidelity = 2.0 * log_abs
    if log_fidelity > 1e-7:
        raise ValueError('Unphysical determinant fidelity > 1: AO normalization mismatch')
    return min(1., exp(log_fidelity))


def quantum_metric_from_fidelity(fidelity: float, h_bohr: float) -> float:
    """Central difference: 1-|<Psi(R-h)|Psi(R+h)>|^2 = 4 h^2 g + O(h^4)."""
    if not isfinite(fidelity) or not 0 <= fidelity <= 1:
        raise ValueError('Invalid wavefunction fidelity')
    if not isfinite(h_bohr) or h_bohr <= 0:
        raise ValueError('Invalid step')
    return (1.-fidelity)/(4*h_bohr*h_bohr)


def nuclear_cartesian_displacement(r_angstrom: float, atom_index: int, axis: int, signed_step_bohr: float) -> tuple[tuple[float, float, float], ...]:
    if not isfinite(r_angstrom) or r_angstrom <= 0 or atom_index not in (0, 1) or axis not in (0, 1, 2):
        raise ValueError('Invalid diatomic geometry or nuclear displacement')
    xyz = np.zeros((2, 3), dtype=float)
    xyz[1, 2] = r_angstrom
    xyz[atom_index, axis] += signed_step_bohr*BOHR_TO_ANGSTROM
    return tuple(tuple(float(c) for c in atom) for atom in xyz)


def _occupied_spin_mos(mf: Any, reference: str) -> tuple[np.ndarray, np.ndarray]:
    occ = mf.mo_occ
    coeff = mf.mo_coeff
    if reference == 'UHF':
        if len(occ) != 2 or len(coeff) != 2:
            raise ValueError('UHF alpha/beta orbitals missing')
        out = []
        for c, o in zip(coeff, occ):
            c, o = np.asarray(c), np.asarray(o)
            if c.ndim != 2 or o.ndim != 1 or c.shape[1] != len(o):
                raise ValueError('Invalid UHF orbital coefficient shape')
            if not np.all(np.isclose(o, 0., atol=1e-6) | np.isclose(o, 1., atol=1e-6)):
                raise ValueError('Invalid UHF spin occupancies')
            out.append(c[:, o > .5])
        return out[0], out[1]
    c, o = np.asarray(coeff), np.asarray(occ)
    if c.ndim != 2 or o.ndim != 1 or c.shape[1] != len(o):
        raise ValueError('Invalid restricted orbital coefficient shape')
    if not np.all(np.isclose(o, 0., atol=1e-6) | np.isclose(o, 1., atol=1e-6) | np.isclose(o, 2., atol=1e-6)):
        raise ValueError('Invalid restricted occupancies')
    if reference == 'RHF' and np.any(np.isclose(o, 1., atol=1e-6)):
        raise ValueError('RHF may not contain singly occupied orbitals')
    return c[:, o > .5], c[:, o > 1.5]


def run_hf_dboc_point(request: HFDBOCRequest, settings: HFDBOCSettings = HFDBOCSettings(), *, backend: Callable[..., Any] | None = None,
                      ao_overlap: Callable[[Any, Any], np.ndarray] | None = None) -> HFDBOCResult:
    """Execute +/- displaced converged SCFs, then nuclear-mass-weighted HF DBOC.

    backend if provided is a test-only injection with the same (request,xyz,
    settings)->(mol,mf) interface, *not* a production method capability.
    """
    signature = sha256(json.dumps({'request': asdict(request), 'settings': asdict(settings)}, sort_keys=True).encode()).hexdigest()
    scf_count = 0
    if backend is None:
        from pyscf import gto, scf
        def backend(req: HFDBOCRequest, xyz: tuple[tuple[float, float, float], ...], config: HFDBOCSettings) -> tuple[Any, Any]:
            mol = gto.M(atom=[(a, v) for a, v in zip(req.atoms, xyz)],
                        unit='Angstrom', charge=req.charge, spin=req.spin_2s,
                        basis=dict(req.basis_by_element), symmetry=False,
                        verbose=config.verbose, max_memory=config.max_memory_mb)
            if getattr(mol, '_ecp', None):
                raise ValueError('HF DBOC pilot cannot use ECPs')
            mf = {'RHF': scf.RHF, 'ROHF': scf.ROHF, 'UHF': scf.UHF}[req.scf_reference](mol)
            mf.conv_tol = config.scf_conv_tol
            mf.max_cycle = config.max_cycle
            mf.kernel()
            if not mf.converged:
                raise RuntimeError('Displaced SCF not converged')
            return mol, mf
    if ao_overlap is None:
        from pyscf import gto
        ao_overlap = lambda m_minus, m_plus: gto.intor_cross('int1e_ovlp', m_minus, m_plus)
    axes = []
    by_step = []
    try:
        for h in settings.steps_bohr:
            total = 0.
            for atom in (0, 1):
                for axis in (0, 1, 2):
                    minus_mol, minus_mf = backend(request, nuclear_cartesian_displacement(request.r_angstrom, atom, axis, -h), settings)
                    scf_count += 1
                    plus_mol, plus_mf = backend(request, nuclear_cartesian_displacement(request.r_angstrom, atom, axis, +h), settings)
                    scf_count += 1
                    if scf_count > settings.max_scf_calculations:
                        raise ValueError('SCF resource budget exceeded')
                    if not bool(getattr(minus_mf, 'converged', False)) or not bool(getattr(plus_mf, 'converged', False)):
                        raise ValueError('Displaced SCF not converged')
                    s = np.asarray(ao_overlap(minus_mol, plus_mol))
                    a_m, b_m = _occupied_spin_mos(minus_mf, request.scf_reference)
                    a_p, b_p = _occupied_spin_mos(plus_mf, request.scf_reference)
                    fid = determinant_fidelity(a_m, b_m, a_p, b_p, s)
                    if fid < settings.min_determinant_fidelity:
                        raise ValueError('Displaced HF roots lost continuous determinant overlap')
                    metric = quantum_metric_from_fidelity(fid, h)
                    contribution = metric/(2. * request.nuclear_masses_amu[atom] * AMU_TO_ELECTRON_MASS)
                    axes.append(DBOCAxis(atom, axis, h, fid, metric, contribution))
                    total += contribution
            by_step.append(total)
        if not all(isfinite(v) and v >= 0 for v in by_step):
            raise ValueError('Nonphysical HF DBOC values')
        diff_ev = abs(by_step[1] - by_step[0])*HARTREE_TO_EV
        return HFDBOCResult(request.request_id, request.role, DBOCStatus.COMPLETE_REVIEW_REQUIRED,
                            'HF-only diagonal nuclear quantum metric; step sensitivity is not a scientific bound',
                            settings.steps_bohr, tuple(by_step), tuple(axes), diff_ev,
                            signature, scf_count)
    except Exception as exc:
        return HFDBOCResult(request.request_id, request.role, DBOCStatus.UNRESOLVED,
                            f'{type(exc).__name__}: {exc}', settings.steps_bohr,
                            source_signature=signature, scf_count=scf_count)


def assess_hf_dboc_pair(neutral_request: HFDBOCRequest, anion_request: HFDBOCRequest,
                        neutral: HFDBOCResult, anion: HFDBOCResult) -> HFDBOCPair:
    if neutral_request.role != 'neutral' or anion_request.role != 'anion':
        raise ValueError('Require neutral and anion request')
    if neutral_request.atoms != anion_request.atoms or neutral_request.nuclear_masses_amu != anion_request.nuclear_masses_amu or neutral_request.nuclear_mass_source != anion_request.nuclear_mass_source:
        raise ValueError('Different isotope/mass provenance across EA pair')
    if neutral_request.basis_by_element != anion_request.basis_by_element:
        raise ValueError('DBOC EA pilot requires matched basis')
    if neutral.request_id != neutral_request.request_id or anion.request_id != anion_request.request_id or neutral.role != 'neutral' or anion.role != 'anion':
        raise ValueError('Mismatched point result provenance')
    if neutral.status is not DBOCStatus.COMPLETE_REVIEW_REQUIRED or anion.status is not DBOCStatus.COMPLETE_REVIEW_REQUIRED:
        return HFDBOCPair(DBOCStatus.UNRESOLVED, neutral, anion, None, None)
    if neutral.steps_bohr != anion.steps_bohr or len(neutral.dboc_by_step_hartree) != 2 or len(anion.dboc_by_step_hartree) != 2:
        raise ValueError('Unmatched HF DBOC finite-difference grid')
    corrections = [n-a for n,a in zip(neutral.dboc_by_step_hartree, anion.dboc_by_step_hartree)]
    return HFDBOCPair(DBOCStatus.COMPLETE_REVIEW_REQUIRED, neutral, anion, corrections[-1], abs(corrections[-1]-corrections[0])*HARTREE_TO_EV)
