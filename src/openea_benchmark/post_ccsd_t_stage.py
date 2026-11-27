"""Automated post-CCSD(T) layer: iterative triples and connected quadruples (Patch 49).

    Delta_T = EA[CCSDT]  - EA[CCSD(T)]   in aug-cc-pVDZ and aug-cc-pVTZ
    Delta_Q = EA[CCSDTQ] - EA[CCSDT]     in aug-cc-pVDZ, else cc-pVDZ (exact quadruples)

frozen (noble-gas) core, nonrelativistic, at the TZ equilibrium of the selected
states.  All three methods run in CCpy on one reference per state and basis:
the ROHF/RHF reference is first converged in the state's symmetry block (point
group, irrep occupation, seeded from its high-level checkpoint), then carried
into C1 (CCpy needs C1 metadata) and must keep its energy there.

(T) definition (2026-11-24): the CBS energy it corrects is semicanonical ROHF-UCCSD(T)
(PySCF, Stage 3).  CCpy's ROHF (T) is evaluated in the non-semicanonical ROHF orbitals
and differs from it (CH 0.5 meV, CuO 1.2 meV on the EA).  CCSD, CCSDT and CCSDTQ are
invariant under the occupied-occupied / virtual-virtual rotations of the
semicanonicalization, (T) is not; the open-shell CCSD(T) baseline is therefore the
semicanonical one (``semicanonical_ccsd_t``), so Delta_T removes exactly the (T) the CBS
energy contains.  CCpy's CCSD of the same point checks that both codes correlate the
same space (``CCSD_INVARIANCE_TOLERANCE_HARTREE``).  The composite is then CCSDT(Q) in the
small basis plus a basis-set increment evaluated entirely in the Stage-3 CCSD(T) definition.

Central values: Delta_T(aug-cc-pVTZ), bound |Delta_T(TZ) - Delta_T(DZ)| (only the
aDZ value when aTZ CCSDT exceeds the memory limit: bound 50 % of it);
Delta_Q in the basis chosen at the start of the run (``quadruples_basis``, user decision 2026-11-27:
both options kept): "aug-cc-pvdz" = first affordable of aug-cc-pVDZ, cc-pVDZ; "cc-pvdz" = cc-pVDZ only
(aug-cc-pVDZ changes Delta_Q by meV only, at ~10x the memory and days of CCSDTQ).  Bound 50 % of |Delta_Q|
(pilot; DZ quadruples are basis-incomplete), plus Q_NO_DIFFUSE_ALLOWANCE_EV for a cc-pVDZ Delta_Q.  CR-CC(2,4) is not used: the OH study (runs/oh_crcc24_transfer_*)
showed its DZ->TZ transfer disagreeing with exact CCSDTQ by 0.1 eV.

Exact CCSDTQ amplitudes grow as (o v)^4 and CCpy keeps them in RAM (only the DIIS
vectors can go to scratch).  When neither quadruples basis fits the machine (RAM
estimate above the limit, OPENEA_Q_MEMORY_LIMIT_MB overrides the policy value, or
the out-of-core DIIS files above half the free scratch), Delta_Q is not computed
(user decision 2026-11-03): the layer applies Delta_T and widens its bound by a
quadruples allowance max(Q_ALLOWANCE_MIN_EV, |Delta_T|), status
CLEARED_WITH_QUADRUPLES_ALLOWANCE, reported as a warning in the final verdict.
Exact Delta_Q(DZ) in verify_v1 stayed below 0.011 eV (NH, PO, SS, SH, CH, BO, SiH,
BrO, IO); CR-CC(2,4)-CR-CC(2,3) is no substitute (NH, SH, OH: 0.05-0.1 eV off with
the wrong sign; CCpy's CR-CC(3,4) is closed-shell only).  On a machine with enough
memory the exact CCSDTQ runs as before.

Route A (2026-11-06): before falling back to that allowance, the quadruples increment
E(CCSDTQ) - E(CCSDT) is computed in frozen-natural-orbital spaces of the last quadruples
basis at two truncation levels (``fno_increment``; the larger space is as large as the
CCSDTQ memory limit allows); status CLEARED_FNO_CCSDTQ, bound = level change + the
relative quadruples bound.  PO cc-pVDZ: 8 / 12 virtual NOs (DMRG prototype, same
spaces) gave -0.0083 / -0.0091 eV against exact CCSDTQ - CCSDT -0.0084 eV.  The
allowance remains the fallback when an FNO point fails.
"""
from __future__ import annotations

from .heavy_elements import apply_hamiltonian, mole_basis_kwargs

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time
import uuid
from typing import Any, Mapping, Sequence

import numpy as np

from .core_valence_stage import selected_state_specs
from .electronic_ea_stage import HARTREE_TO_EV, _save_json
from .point_pool import run_parallel
from .scratch import sweep_dead

METHODS = ("CCSD(T)", "CCSDT", "CCSDTQ")
# Open-shell CCSD(T) baseline: semicanonical ROHF-UCCSD(T) as in the CBS layer (module docstring).
TRIPLES_BASELINE = "SEMICANONICAL_UHF_FROM_ROHF"
SEMICANONICAL_CONV_TOL_NORMT = 1.0e-6
# CCSD is invariant under the semicanonical rotations, so CCpy and PySCF agree on it up to the
# frozen-core definition: Stage 3 freezes the lowest semicanonical orbital(s) of each spin, CCpy
# the ROHF core (CH, OH, NH- aug-cc-pVDZ: 4-6 microhartree; NH X3Sigma- 21).  Another reference
# or correlation space differs by many millihartree; the guard leaves room for larger cores.
# Measured on CH/CH- (axial CCpy, every term semicanonical as reference): Delta_T(EA) off by
# -0.19 / -0.24 meV (aDZ / aTZ) with this baseline, +0.64 / +0.95 meV with CCpy's ROHF (T).
CCSD_INVARIANCE_TOLERANCE_HARTREE = 1.0e-3
# File tokens; "CCSD(T)" without parentheses would collide with "CCSDT".
METHOD_TOKEN = {"CCSD(T)": "CCSD_pT", "CCSDT": "CCSDT", "CCSDTQ": "CCSDTQ"}


@dataclass(frozen=True)
class PostCCSDTStagePolicy:
    """Explicit pilot policy, not a validated production threshold set."""
    triples_bases: tuple[str, ...] = ("aug-cc-pvdz", "aug-cc-pvtz")
    # First affordable basis is used; cc-pVDZ (W4 practice for the quadruples
    # increment) when aug-cc-pVDZ CCSDTQ exceeds the storage limit.
    quadruples_bases: tuple[str, ...] = ("aug-cc-pvdz", "cc-pvdz")
    quadruples_relative_bound: float = 0.5
    # Per-point memory limit; every point reserves its estimate in the point-pool
    # memory gate, and points above point_pool.BIG_TASK_MB (150 GB) run one at a
    # time machine-wide (user decision 2026-10-13: NO, O2, ClO CCSDTQ/cc-pVDZ need
    # 130-200 GB; run them one after the other).
    q_memory_limit_mb: int = 210000
    max_memory_mb: int = 48000
    scf_conv_tol: float = 1.0e-10
    cc_convergence: float = 1.0e-8
    cc_max_iterations: int = 200
    c1_energy_tolerance_hartree: float = 1.0e-7


def _mol(spec: Any, basis: str, symmetry: Any, max_memory_mb: int):
    from pyscf import gto
    from .heavy_elements import basis_name_for, routed
    if any(routed(el) for el in spec.atoms):  # PP elements take the -PP member, X2C ones the X2C set
        basis = {el: basis_name_for(el, basis) for el in dict.fromkeys(spec.atoms)}
    return gto.M(atom=[(spec.atoms[0], (0.0, 0.0, 0.0)), (spec.atoms[1], (0.0, 0.0, float(spec.r_angstrom)))],
                 **mole_basis_kwargs(spec.atoms, basis), charge=spec.charge, spin=spec.spin_2s, unit="Angstrom", symmetry=symmetry,
                 verbose=0, max_memory=max_memory_mb, dump_input=False)


def _kernel_with_soscf_rescue(mf, dm0, conv_tol: float) -> bool:
    """Run the SCF from dm0; a stalled DIIS run is continued by second-order SCF from its
    last orbitals (same occupations and symmetry block), as in Stage 3.  Returns True when
    the rescue was needed and converged.

    MnO- and MoO- (verify_v2, 2026-11-13..18): the symmetry-block SCF from the Stage-3
    checkpoint did not converge in 200 DIIS cycles in aug-cc-pVDZ and aug-cc-pVTZ, so every
    anion post-CC point failed while the neutral points ran for days."""
    mf.kernel(dm0=dm0)
    if mf.converged:
        return False
    soscf = mf.newton()
    soscf.conv_tol, soscf.max_cycle = conv_tol, mf.max_cycle
    soscf.kernel(mf.mo_coeff, mf.mo_occ)
    if not soscf.converged:
        return False
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = soscf.mo_coeff, soscf.mo_occ, soscf.mo_energy
    mf.e_tot, mf.converged = soscf.e_tot, True
    return True


def reference_c1(spec: Any, basis: str, policy: PostCCSDTStagePolicy):
    """Reference converged in the symmetry block, then carried into C1 unchanged."""
    return _references(spec, basis, policy)[1]


def reference_symmetric(spec: Any, basis: str, policy: PostCCSDTStagePolicy):
    """The symmetry-block reference itself (orbitals carry irrep labels), after the same C1
    check as ``reference_c1``; None without a point group."""
    if not spec.point_group or str(spec.point_group).upper() == "C1":
        return None
    return occupation_ordered(_references(spec, basis, policy)[0])


def _references(spec: Any, basis: str, policy: PostCCSDTStagePolicy):
    from pyscf import scf
    open_shell = spec.spin_2s > 0
    mol = _mol(spec, basis, spec.point_group or False, policy.max_memory_mb)
    mf = apply_hamiltonian(scf.ROHF(mol) if open_shell else scf.RHF(mol))
    if spec.irrep_nelec is not None:
        mf.irrep_nelec = ({k: (int(v[0]), int(v[1])) for k, v in spec.irrep_nelec.items()} if open_shell
                          else {k: int(v[0]) + int(v[1]) for k, v in spec.irrep_nelec.items()})
    guess = scf.rohf.init_guess_by_chkfile if open_shell else scf.hf.init_guess_by_chkfile
    mf.conv_tol, mf.max_cycle, mf.max_memory = policy.scf_conv_tol, 200, policy.max_memory_mb
    rescued = _kernel_with_soscf_rescue(mf, guess(mol, spec.source_checkpoint_path, project=True),
                                        policy.scf_conv_tol)
    if not mf.converged:
        raise RuntimeError(f"symmetry-block SCF not converged ({basis})")
    mol1 = _mol(spec, basis, "C1", policy.max_memory_mb)
    mf1 = apply_hamiltonian(scf.ROHF(mol1) if open_shell else scf.RHF(mol1))
    mf1.conv_tol, mf1.max_cycle, mf1.max_memory = policy.scf_conv_tol, 50, policy.max_memory_mb
    rescued |= _kernel_with_soscf_rescue(mf1, mf.make_rdm1(), policy.scf_conv_tol)
    if not mf1.converged or abs(mf1.e_tot - mf.e_tot) > policy.c1_energy_tolerance_hartree:
        raise RuntimeError(f"C1 reference left the symmetry-block solution ({mf1.e_tot} vs {mf.e_tot})")
    mf1 = occupation_ordered(mf1)
    mf1.openea_soscf_rescue = rescued
    mf.openea_soscf_rescue = rescued
    return mf, mf1


def occupation_ordered(mf):
    """``mf`` with its orbitals in the order doubly, singly occupied, virtual (energy order within).

    CCpy takes the first orbitals as occupied.  A ROHF orbital energy does not have to follow
    the occupation: CH- 3Sigma- in aug-cc-pVTZ has a virtual between the doubly and the singly
    occupied orbitals, and CCpy stopped with "Integrals don't match mean field energy"
    (verify_v1, 2026-10-26).  Reordering leaves the determinant and its energy unchanged."""
    import numpy as np
    order = np.argsort(-np.asarray(mf.mo_occ), kind="stable")
    if np.array_equal(order, np.arange(order.size)):
        return mf
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = mf.mo_coeff[:, order], mf.mo_occ[order], mf.mo_energy[order]
    return mf


# CCpy keeps diis_size (6) amplitude and residual vectors, plus the amplitudes, the
# residual and intermediates: ~16 copies of the highest-rank amplitude array.
# 2026-10-13: CCSDTQ/cc-pVDZ of NO, ClO and N2 together with CCSDT/aug-cc-pVTZ
# points took ~450 GB; the earlier 3-copy estimate was 5x low.
DIIS_SIZE = 6
DIIS_COPIES = 2 * DIIS_SIZE
IN_CORE_COPIES = 4
AMPLITUDE_COPIES = DIIS_COPIES + IN_CORE_COPIES
# With DIIS on disk: amplitudes, residual, intermediates plus the flattened vector
# and one row read back in push/extrapolate (benchmark 2026-10-14: peak 5.1 copies).
OUT_OF_CORE_COPIES = 6
# Out-of-core DIIS (user decision 2026-10-14): when the DIIS vectors of a point
# exceed this size, CCpy writes them to an HDF5 file in a per-point directory on
# GPFS scratch, removed when the point ends.  OPENEA_DIIS_OUT_OF_CORE=0/1 forces
# the choice (benchmark in-core vs scratch).
DIIS_OUT_OF_CORE_MB = 10000  # superseded by IN_CORE_LIMIT_MB (2026-11-17), kept for reference
# In-core first (user decision 2026-11-17): out-of-core DIIS on GPFS made CC points
# 8-15x slower (BO aTZ CCSDT neutral 8.0 h -> 1.0 h in-core at 28 GB; VO anion aDZ
# CCSDT 43.5 h -> 2.9 h at 29 GB; energies equal to 5e-13 Eh).  A point stays in core
# with DIIS_SIZE vectors when its 16 copies fit IN_CORE_LIMIT_MB, else with
# REDUCED_DIIS_SIZE vectors (10 copies); only beyond that the DIIS vectors go to
# scratch, and at most MAX_OUT_OF_CORE_POINTS such points run at a time machine-wide.
IN_CORE_LIMIT_ENV = "OPENEA_CC_IN_CORE_LIMIT_MB"
IN_CORE_LIMIT_MB = 120000
REDUCED_DIIS_SIZE = 3
MAX_OUT_OF_CORE_POINTS = 2
OUT_OF_CORE_LIMIT_GROUP = ("cc_out_of_core", MAX_OUT_OF_CORE_POINTS)
# Resource limits of the quadruples decision; kept out of PostCCSDTStagePolicy because
# the policy is part of every post-CC point signature (aTZ CCSDT points take days).
Q_MEMORY_LIMIT_ENV = "OPENEA_Q_MEMORY_LIMIT_MB"
SCRATCH_SHARE = 0.5
Q_ALLOWANCE_MIN_EV = 0.02
# Delta_Q without diffuse functions underestimates the anion's quadruples: symcc 2026-11-27,
# vertical EA at one bond length, Delta_Q(aug-cc-pVDZ) - Delta_Q(cc-pVDZ) = OH +0.0071, NH +0.0066,
# CH +0.0037, SH +0.0038, SiH +0.0023 eV (cc-pVDZ gives 1/4 - 1/2 of the aug value), so a bound
# relative to Delta_Q(cc-pVDZ) is not conservative.  A cc-pVDZ Delta_Q (exact or FNO) adds the
# largest measured difference to its bound.  Not a policy field (point signature).
Q_NO_DIFFUSE_ALLOWANCE_EV = 0.0075


def has_diffuse(basis: str) -> bool:
    return str(basis).lower().startswith(("aug-", "d-aug-", "t-aug-"))


def q_diffuse_allowance_ev(basis: str) -> float:
    return 0.0 if has_diffuse(basis) else Q_NO_DIFFUSE_ALLOWANCE_EV


# Quadruples basis option, chosen per run at the start (workflow --quadruples-basis, Q_BASIS in
# scripts/run_one.sh).  Not a policy field: the policy is part of every post-CC point signature,
# and the choice only selects among the policy's bases (cached points of either basis stay valid).
QUADRUPLES_BASIS_CHOICES = ("cc-pvdz", "aug-cc-pvdz")


def quadruples_candidates(policy: "PostCCSDTStagePolicy", choice: str) -> tuple[str, ...]:
    if choice not in QUADRUPLES_BASIS_CHOICES:
        raise ValueError(f"quadruples basis {choice!r}: choose one of {QUADRUPLES_BASIS_CHOICES}")
    if choice == "aug-cc-pvdz":
        return tuple(policy.quadruples_bases)
    return tuple(b for b in policy.quadruples_bases if not has_diffuse(b))
# No triples basis fits either: no post-CCSD(T) correction, this allowance instead
# (verify_v1 |Delta_T| + |Delta_Q| <= 0.042 eV: IO 0.041, BrO 0.032, CuO aDZ 0.027).
POST_CC_ALLOWANCE_EV = 0.05
# Route A: when CCSDTQ does not fit, E(CCSDTQ) - E(CCSDT) in frozen natural orbitals
# (fno_increment) replaces the quadruples allowance.  Not a policy field: the policy is
# part of every post-CC point signature.  OPENEA_FNO=0 disables it.
FNO_BEYOND_TRIPLES = os.environ.get("OPENEA_FNO", "1") != "0"
# CCSDT as CCpy CC(P) over every symmetry-allowed triple (cc_pspace.py): the same equations
# as the dense ``ccsdt`` (CN aDZ/cc-pVDZ to 1e-10 Eh), amplitudes packed (a<b<c, i<j<k) and
# only triples of the reference symmetry, so the point stays in core.  CN 2Sigma+ aTZ C2v,
# 8 threads: dense 263 s/iteration 29.9 GB, CC(P) 276 s 5.5 GB (Fortran serial), 94 s with
# CCpy built with OpenMP.  Not part of the point signature (same energies): cached dense
# points stay valid.  OPENEA_CCSDT_ALGORITHM=dense restores the dense code.
CCSDT_ALGORITHM = os.environ.get("OPENEA_CCSDT_ALGORITHM", "symmetric_ccp")
GROUP_ORDER = {"C1": 1, "CS": 2, "C2": 2, "C2V": 4, "C2H": 4, "D2": 4, "D2H": 8}
CCP_BYTES_PER_TRIPLE = 400          # T3, residual, 6 DIIS pairs, excitation list, loop copies (CN aTZ fit)
CCP_INTEGRAL_COPIES = 2             # sorted H blocks and the CCSD-like intermediates
CCP_BASE_MB = 1000
CC_SCRATCH_ROOT = Path(os.environ.get("OPENEA_CC_SCRATCH", f"/scratch/{os.environ.get('USER', 'openea')}/openea_cc"))


def _amplitude_mb(spec: Any, basis: str, method: str) -> float:
    """One copy of the highest-rank amplitude array: 8 bytes x sum over spin blocks of (o v)^n."""
    from .adaptive.stage3_execution import valence_frozen_core_orbitals
    rank = {"CCSD(T)": 2, "CCSDT": 3, "CCSDTQ": 4}[method]
    mol = _mol(spec, basis, False, 1000)
    nfz = valence_frozen_core_orbitals(mol)
    na, nb = mol.nelec
    nmo = mol.nao_nr()
    x, y = (na - nfz) * (nmo - na), (nb - nfz) * (nmo - nb)
    return 8 * sum(x ** (rank - k) * y ** k for k in range(rank + 1)) / 1.0e6


# CCSDTQ on symmetry-blocked tensors (symcc: CCpy's CCSDTQ equations, blocks of the reference
# point group only).  OH aug-cc-pVDZ (axial m blocks): 2103 s vs 10106 s CCpy dense, E equal to
# 3e-11 Eh (1489-1639 s depending on load); C2v keeps ~1/4 of every tensor.  Used when symcc is importable and the state has a
# point group; OPENEA_CCSDTQ_ALGORITHM=dense keeps CCpy.  Not in the point signature (same energy).
CCSDTQ_ALGORITHM = os.environ.get("OPENEA_CCSDTQ_ALGORITHM", "symcc")
# T, residual and the T4-update intermediates in full blocks; DIIS keeps only the unique
# amplitudes (symcc >= 99919c3).  OH aug-cc-pVDZ CCSDTQ, m blocks: peak 3.8 GB = ~3 blocked
# copies (CCpy dense: 12 GB); 5 leaves room for the integral blocks.
SYMCC_COPIES = 5


def symcc_available() -> bool:
    try:
        import symcc.cc  # noqa: F401
        return True
    except ImportError:
        return False


def symcc_applies(spec: Any, method: str) -> bool:
    group = str(getattr(spec, "point_group", None) or "C1").upper()
    return (method == "CCSDTQ" and CCSDTQ_ALGORITHM == "symcc" and group in GROUP_ORDER and group != "C1"
            and symcc_available())


def symmetric_ccp_applies(spec: Any, method: str) -> bool:
    group = str(getattr(spec, "point_group", None) or "C1").upper()
    return method == "CCSDT" and CCSDT_ALGORITHM == "symmetric_ccp" and group in GROUP_ORDER and group != "C1"


def ccp_storage_mb(spec: Any, basis: str) -> float:
    """RAM of a symmetric CC(P) CCSDT point: packed triples (all unique ones / group order)
    plus the dense integral blocks CCpy keeps (aa, ab, bb over the correlated orbitals)."""
    from math import comb
    from .adaptive.stage3_execution import valence_frozen_core_orbitals
    mol = _mol(spec, basis, False, 1000)
    nfz = valence_frozen_core_orbitals(mol)
    na, nb = mol.nelec
    nmo = mol.nao_nr()
    oa, ob, va, vb = na - nfz, nb - nfz, nmo - na, nmo - nb
    unique = (comb(oa, 3) * comb(va, 3) + comb(oa, 2) * ob * comb(va, 2) * vb
              + oa * comb(ob, 2) * va * comb(vb, 2) + comb(ob, 3) * comb(vb, 3))
    triples = unique / GROUP_ORDER[str(spec.point_group).upper()]
    n = nmo - nfz
    return CCP_BASE_MB + (CCP_BYTES_PER_TRIPLE * triples + CCP_INTEGRAL_COPIES * 3 * 8 * n ** 4) / 1.0e6


def in_core_copies(diis_size: int) -> int:
    return 2 * diis_size + IN_CORE_COPIES


def cc_mode(amplitude_mb: float) -> tuple[bool, int]:
    """(DIIS out of core, DIIS size) of a point whose amplitude array has ``amplitude_mb``."""
    forced = os.environ.get("OPENEA_DIIS_OUT_OF_CORE")
    if forced in ("0", "1"):
        return forced == "1", DIIS_SIZE
    limit = float(os.environ.get(IN_CORE_LIMIT_ENV, IN_CORE_LIMIT_MB))
    for size in (DIIS_SIZE, REDUCED_DIIS_SIZE):
        if in_core_copies(size) * amplitude_mb <= limit:
            return False, size
    return True, DIIS_SIZE


def diis_out_of_core(amplitude_mb: float) -> bool:
    return cc_mode(amplitude_mb)[0]


def cc_limit_group(spec: Any, basis: str, method: str):
    """Machine-wide concurrency group of an out-of-core point (None for in-core points)."""
    if symmetric_ccp_applies(spec, method) or symcc_applies(spec, method):
        return None
    return OUT_OF_CORE_LIMIT_GROUP if diis_out_of_core(_amplitude_mb(spec, basis, method)) else None


def q_memory_limit_mb(policy: "PostCCSDTStagePolicy") -> int:
    return int(os.environ.get(Q_MEMORY_LIMIT_ENV, policy.q_memory_limit_mb))


def cc_scratch_need_mb(spec: Any, basis: str, method: str) -> float:
    """Scratch for the DIIS file of an out-of-core point (amplitude and residual vectors)."""
    if symmetric_ccp_applies(spec, method) or symcc_applies(spec, method):
        return 0.0
    one = _amplitude_mb(spec, basis, method)
    return DIIS_COPIES * one if diis_out_of_core(one) else 0.0


def scratch_free_mb() -> float:
    path = CC_SCRATCH_ROOT
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1.0e6


def cc_storage_mb(spec: Any, basis: str, method: str) -> float:
    """Memory (RAM) estimate of a CCpy point; DIIS copies count only when kept in core."""
    if symmetric_ccp_applies(spec, method):
        return ccp_storage_mb(spec, basis)
    if symcc_applies(spec, method):   # dense amplitudes / group order, all copies in core
        return SYMCC_COPIES * _amplitude_mb(spec, basis, method) / GROUP_ORDER[str(spec.point_group).upper()]
    one = _amplitude_mb(spec, basis, method)
    out_of_core, size = cc_mode(one)
    return (OUT_OF_CORE_COPIES if out_of_core else in_core_copies(size)) * one


def cached_diis_class():
    """CCpy's DIIS with the residual overlap matrix kept in memory.

    CCpy's ``extrapolate`` rebuilds the full B matrix every iteration: with the
    vectors on GPFS that re-reads ~2 x 21 residual vectors (2026-10-14 benchmark,
    CN CCSDTQ/cc-pVDZ: 506 s of a 930 s iteration).  Here ``push`` computes the new
    residual's overlaps with the stored ones once (diis_size reads), and
    ``extrapolate`` only reads the diis_size amplitude vectors.  Same equations,
    same Gaussian elimination.
    """
    import ccpy.drivers.diis as diis_mod

    class CachedDIIS(diis_mod.DIIS):
        def __init__(self, T, diis_size, out_of_core):
            super().__init__(T, diis_size, out_of_core)
            self.overlap = np.zeros((diis_size, diis_size), dtype=self.data_type)

        def push(self, T, T_residuum, iteration):
            slot = iteration % self.diis_size
            resid = T_residuum.flatten()
            self.T_list[slot, :] = T.flatten()
            self.T_residuum_list[slot, :] = resid
            for j in range(min(iteration + 1, self.diis_size)):
                value = np.dot(resid.conj(), resid) if j == slot else np.dot(
                    resid.conj(), self.T_residuum_list[j, :])
                self.overlap[slot, j] = self.overlap[j, slot] = value

        def extrapolate(self):
            n = self.diis_size
            B = -1.0 * np.ones((n + 1, n + 1), self.data_type)
            B[:n, :n] = self.overlap
            B[-1, -1] = 0.0
            rhs = np.zeros(n + 1, dtype=self.data_type)
            rhs[-1] = -1.0
            coeff = self.solve_gauss(B, rhs)
            x_xtrap = np.zeros(self.ndim, dtype=self.data_type)
            for i in range(n):
                x_xtrap += coeff[i] * self.T_list[i, :]
            return x_xtrap

    return CachedDIIS


@contextmanager
def cc_scratch_dir():
    """Per-point directory for CCpy's DIIS file (tempfile-based), always removed afterwards.

    Points run one per pool worker process, so redirecting ``tempfile.tempdir`` is safe.
    """
    CC_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    sweep_dead(CC_SCRATCH_ROOT)
    path = CC_SCRATCH_ROOT / f"{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    path.mkdir()
    previous, tempfile.tempdir = tempfile.tempdir, str(path)
    try:
        yield path
    finally:
        tempfile.tempdir = previous
        shutil.rmtree(path, ignore_errors=True)


def _point_signature(spec: Any, basis: str, method: str, policy: PostCCSDTStagePolicy) -> str:
    payload = {"state": {k: v for k, v in asdict(spec).items()}, "basis": basis, "method": method,
               "policy": asdict(policy)}
    if method == "CCSD(T)" and spec.spin_2s > 0:
        payload["triples_baseline"] = TRIPLES_BASELINE  # CCpy-(T) points before 2026-11-24 are recomputed
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def semicanonical_ccsd_t(mf: Any, nfrozen: int, policy: PostCCSDTStagePolicy) -> tuple[float, float]:
    """(CCSD, CCSD(T)) correlation energy of a ROHF reference exactly as Stage 3 / CBS compute
    it: UHF form of the ROHF determinant, alpha and beta occupied and virtual blocks
    canonicalized, UCCSD(T) with the lowest ``nfrozen`` orbitals of each spin frozen."""
    from pyscf import cc
    from .adaptive.stage3_execution import _ccsd_eris_within_memory
    umf = mf.to_uhf()
    umf.mo_energy, umf.mo_coeff = umf.canonicalize(umf.mo_coeff, umf.mo_occ)
    mycc = cc.CCSD(umf)
    mycc.frozen = nfrozen
    mycc.conv_tol, mycc.conv_tol_normt = policy.cc_convergence, SEMICANONICAL_CONV_TOL_NORMT
    from .adaptive.stage3_execution import CCSD_DIIS_SPACE
    mycc.diis_space = CCSD_DIIS_SPACE
    mycc.max_cycle, mycc.max_memory = policy.cc_max_iterations, policy.max_memory_mb
    eris = _ccsd_eris_within_memory(mycc)
    mycc.kernel(eris=eris)
    if not mycc.converged:
        raise RuntimeError("semicanonical UCCSD not converged")
    return float(mycc.e_corr), float(mycc.e_corr + mycc.ccsd_t(eris=eris))


def ccsd_t_point_memory_mb(spec: Any, basis: str, policy: PostCCSDTStagePolicy) -> float:
    """Memory of a CCSD(T) point: CCpy's estimate, and for an open shell the PySCF UCCSD
    integrals of the semicanonical baseline (out of core above ``policy.max_memory_mb``)."""
    need = cc_storage_mb(spec, basis, "CCSD(T)")
    if spec.spin_2s > 0:
        from .adaptive.stage3_execution import valence_frozen_core_orbitals
        from .uccsd_eris import blockwise_peak_mb
        mol = _mol(spec, basis, False, 1000)
        nao, nfz = mol.nao_nr(), valence_frozen_core_orbitals(mol)
        oa, ob = mol.nelec[0] - nfz, mol.nelec[1] - nfz
        peak = blockwise_peak_mb(nao, oa, ob, nao - nfz - oa, nao - nfz - ob)
        need = max(need, min(peak, float(policy.max_memory_mb)))
    return need


def cached_point(spec: Any, basis: str, method: str, policy: PostCCSDTStagePolicy, path: str) -> dict | None:
    out = Path(path)
    if out.is_file():
        cached = json.loads(out.read_text())
        if (cached.get("signature") == _point_signature(spec, basis, method, policy)
                and cached.get("status") == "COMPLETED"):
            return cached
    return None


def _run_point(spec: Any, basis: str, method: str, policy: PostCCSDTStagePolicy, path: str) -> dict:
    signature = _point_signature(spec, basis, method, policy)
    cached = cached_point(spec, basis, method, policy, path)
    if cached is not None:
        return cached
    from ccpy.drivers.driver import Driver
    from .adaptive.stage3_execution import valence_frozen_core_orbitals
    record: dict[str, Any] = {"signature": signature, "role": spec.role, "basis": basis, "method": method}
    try:
        symmetric = symmetric_ccp_applies(spec, method)
        blocked = symcc_applies(spec, method)
        mf = (reference_symmetric(spec, basis, policy) if (symmetric or blocked)
              else reference_c1(spec, basis, policy))
        record["scf_soscf_rescue"] = bool(getattr(mf, "openea_soscf_rescue", False))
        nfz = valence_frozen_core_orbitals(mf.mol)
        driver = Driver.from_pyscf(mf, nfrozen=nfz)
        driver.options["energy_convergence"] = policy.cc_convergence
        driver.options["amp_convergence"] = policy.cc_convergence
        driver.options["maximum_iterations"] = policy.cc_max_iterations
        out_of_core, diis_size = ((False, DIIS_SIZE) if (symmetric or blocked)
                                  else cc_mode(_amplitude_mb(spec, basis, method)))
        driver.options["diis_size"] = diis_size
        driver.options["diis_out_of_core"] = out_of_core
        record["diis_out_of_core"] = out_of_core
        record["diis_size"] = diis_size
        if out_of_core:
            import ccpy.drivers.solvers as solvers
            solvers.DIIS = cached_diis_class()
        t0 = time.perf_counter()
        log = _TeeStdout()
        with cc_scratch_dir() if out_of_core else nullcontext(), contextlib.redirect_stdout(log):
            if method == "CCSD(T)":
                driver.run_cc(method="ccsd")
                ccsd_corr = float(driver.correlation_energy)
                driver.run_ccp3(method="ccsd(t)")
                corr = ccsd_corr + float(driver.deltap3[0]["A"])
                if spec.spin_2s > 0:
                    sc_ccsd, sc_total = semicanonical_ccsd_t(mf, nfz, policy)
                    record.update(triples_baseline=TRIPLES_BASELINE, ccpy_ccsd_t_correlation_hartree=corr,
                                  ccsd_invariance_difference_hartree=sc_ccsd - ccsd_corr)
                    if abs(sc_ccsd - ccsd_corr) > CCSD_INVARIANCE_TOLERANCE_HARTREE:
                        raise RuntimeError(f"CCpy and PySCF CCSD differ by {sc_ccsd - ccsd_corr:.2e} Eh "
                                           "(correlation space or reference not the same)")
                    corr = sc_total
            elif blocked:
                from symcc.cc import blocked_hamiltonian, solve
                system = driver.system
                labels = [system.point_group_irrep_to_number[x] for x in system.orbital_symmetries]
                hb = blocked_hamiltonian(driver.hamiltonian, labels, labels, system.noccupied_alpha,
                                         system.noccupied_beta, "xor")
                driver.hamiltonian = None   # the dense blocks are not needed any more
                result = solve(hb, "ccsdtq", conv=policy.cc_convergence, max_iter=policy.cc_max_iterations,
                               diis_size=DIIS_SIZE)
                if not result["converged"]:
                    raise RuntimeError(f"symcc CCSDTQ: no convergence in {result['iterations']} iterations")
                print(f"   {CC_CONVERGED_MESSAGE} (symcc, {result['iterations']} iterations)", flush=True)
                corr = float(result["correlation"])
                record.update(quadruples_algorithm="SYMCC_SYMMETRY_BLOCKED", point_group=str(spec.point_group))
            elif symmetric:
                from .cc_pspace import run_symmetric_ccsdt
                system = driver.system
                labels = [system.point_group_irrep_to_number[x] for x in system.orbital_symmetries]
                counts = run_symmetric_ccsdt(driver, labels, labels, system.noccupied_alpha,
                                             system.noccupied_beta, "xor")
                corr = float(driver.correlation_energy)
                record.update(triples_algorithm="CCP_SYMMETRY_ALLOWED_TRIPLES", triples_count=counts,
                              point_group=str(spec.point_group))
            else:
                driver.run_cc(method=method.lower())
                corr = float(driver.correlation_energy)
        if CC_CONVERGED_MESSAGE not in log.text():
            raise RuntimeError(f"{method}: CCpy iterations ended without convergence "
                               f"({policy.cc_max_iterations} iterations)")
        record.update(status="COMPLETED", cc_wall_seconds=time.perf_counter() - t0, nfrozen=nfz,
                      reference_hartree=float(mf.e_tot), total_hartree=float(mf.e_tot) + corr)
    except Exception as exc:  # noqa: BLE001 - recorded, layer fails closed
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    _save_json(Path(path), record)
    return record


# CCpy's driver discards the solver's convergence flag; its message is the evidence.
CC_CONVERGED_MESSAGE = "CC calculation successfully converged"


class _TeeStdout(io.TextIOBase):
    """Passes CCpy's output on to the log and keeps a copy for the convergence check."""
    def __init__(self):
        self._target = sys.stdout
        self._parts: list[str] = []

    def write(self, text):
        self._parts.append(text)
        return self._target.write(text)

    def flush(self):
        self._target.flush()

    def text(self) -> str:
        return "".join(self._parts)


def _fno_increment(states: Mapping[str, Any], policy: PostCCSDTStagePolicy, output_dir: Path) -> dict:
    """Quadruples EA increment from FNO-CCSDTQ at two truncation levels (both states)."""
    from .fno_increment import FNOPolicy, ea_increment, run_fno_point, t4_amplitude_mb, truncation_levels
    from .adaptive.stage3_execution import valence_frozen_core_orbitals
    fno_policy, basis = FNOPolicy(), policy.quadruples_bases[-1]
    na = nb = nvir = 0
    for spec in states.values():
        mol = _mol(spec, basis, False, 1000)
        nfz = valence_frozen_core_orbitals(mol)
        na, nb = max(na, mol.nelec[0] - nfz), max(nb, mol.nelec[1] - nfz)
        nvir = max(nvir, mol.nao_nr() - mol.nelec[0])
    # neutral and anion of one level run together within the CCSDTQ memory limit
    levels = truncation_levels(na, nb, nvir, q_memory_limit_mb(policy) / 2, fno_policy)
    jobs, keys, reserve = [], [], []
    for level, keep in enumerate(levels):
        for role, spec in states.items():
            path = str(output_dir / "fno_ccsdtq" / f"{role}__{basis}__L{level}.json")
            print(f"[POST-CC] {spec.system} {role} {basis} FNO-CCSDTQ level {level} ({keep} virtual NOs)", flush=True)
            jobs.append((spec, basis, keep, fno_policy, policy, path))
            keys.append((role, level))
            reserve.append(fno_policy.amplitude_copies * t4_amplitude_mb(na, nb, keep, keep + na - nb))
    results = run_parallel(run_fno_point, jobs, reserve_mb=reserve, hold_reservations=True)
    points = dict(zip(keys, results))
    out = {"policy": asdict(fno_policy), "basis": basis, "virtuals_by_level": list(levels),
           "points": {f"{r}:L{lv}": {k: v for k, v in p.items() if k != "signature"} for (r, lv), p in points.items()}}
    failed = [f"{r} L{lv}: {p.get('error')}" for (r, lv), p in points.items() if p.get("status") != "COMPLETED"]
    if failed:
        return {**out, "status": "FAILED", "reason": "; ".join(failed)}
    from .electronic_ea_stage import HARTREE_TO_EV as h2ev
    return {**out, "status": "COMPLETED", **ea_increment(points, h2ev)}


def run_post_ccsd_t_stage(*, electronic_ea: Mapping[str, Any], requests: Sequence[Any],
                          point_records: Mapping[str, Mapping[str, Any]], refinement_dir: Path, output_dir: Path,
                          policy: PostCCSDTStagePolicy = PostCCSDTStagePolicy(), point_fn=None,
                          quadruples_basis: str = "aug-cc-pvdz") -> dict:
    output_dir = Path(output_dir)
    record: dict[str, Any] = {"stage": "POST_CCSD_T", "policy": asdict(policy), "is_production_ea": False,
                              "validation_metadata_used_in_computation": False}

    def done(**fields) -> dict:
        record.update(fields)
        _save_json(output_dir / "post_ccsd_t.json", record)
        return record

    if electronic_ea.get("status") != "ELECTRONIC_EA_EVALUATED":
        return done(status="NOT_ATTEMPTED", reason="single-basis electronic EA not evaluated")
    states = selected_state_specs(electronic_ea, requests, point_records, refinement_dir)
    if isinstance(states, str):
        return done(status="NOT_ATTEMPTED", reason=states)
    def need(basis, method):
        return max(cc_storage_mb(s, basis, method) for s in states.values())

    limit_mb, free_mb = q_memory_limit_mb(policy), scratch_free_mb()

    def affordable(basis, method):
        # A completed point is only read; otherwise RAM and the DIIS scratch must fit.
        if all(cached_point(s, basis, method, policy,
                            str(output_dir / "points" / f"{role}__{basis}__{METHOD_TOKEN[method]}.json"))
               for role, s in states.items()):
            return True
        return (need(basis, method) <= limit_mb
                and sum(cc_scratch_need_mb(s, basis, method) for s in states.values()) <= SCRATCH_SHARE * free_mb)

    q_candidates = quadruples_candidates(policy, quadruples_basis)
    record["quadruples_basis_choice"] = quadruples_basis
    estimates = {b: need(b, "CCSDTQ") for b in q_candidates}
    record["quadruples_storage_estimate_mb"] = estimates
    record["resource_limits"] = {"ram_limit_mb": limit_mb, "scratch_free_mb": free_mb,
                                 "scratch_share": SCRATCH_SHARE}
    q_basis = next((b for b in q_candidates if affordable(b, "CCSDTQ")), None)
    run_q = q_basis is not None
    record["quadruples_basis"] = q_basis
    t_bases = tuple(b for b in policy.triples_bases if affordable(b, "CCSDT"))
    record["triples_bases_affordable"] = list(t_bases)
    if not t_bases:
        return done(status="CLEARED_WITH_POST_CC_ALLOWANCE", quadruples_computed=False, triples_computed=False,
                    delta_post_ev=0.0, bound_ev=POST_CC_ALLOWANCE_EV, post_cc_allowance_ev=POST_CC_ALLOWANCE_EV,
                    reason=f"CCSDT exceeds {limit_mb} MB RAM or the scratch share in every triples basis; "
                           f"no correction, allowance {POST_CC_ALLOWANCE_EV} eV in the bound")
    tasks = [(basis, method) for basis in t_bases for method in ("CCSD(T)", "CCSDT")]
    if run_q:
        if q_basis not in t_bases:
            tasks.append((q_basis, "CCSD(T)"))
            tasks.append((q_basis, "CCSDT"))
        tasks.append((q_basis, "CCSDTQ"))
    fn = point_fn or _run_point
    jobs, reserves, groups = [], [], []
    for role, spec in states.items():
        for basis, method in tasks:
            path = str(output_dir / "points" / f"{role}__{basis}__{METHOD_TOKEN[method]}.json")
            # A completed point only reads its file: no memory reservation, so it does
            # not wait behind a running large point for the one-at-a-time slot.
            done_already = cached_point(spec, basis, method, policy, path) is not None
            reserves.append(0.0 if done_already else ccsd_t_point_memory_mb(spec, basis, policy)
                            if method == "CCSD(T)" else cc_storage_mb(spec, basis, method))
            groups.append(None if done_already else cc_limit_group(spec, basis, method))
            print(f"[POST-CC] {spec.system} {role} {basis} {method}{' (cached)' if done_already else ''}",
                  flush=True)
            jobs.append((spec, basis, method, policy, path))
    # CC points grow into their (exact) estimate over the first DIIS iterations: held (MEM_GATE_HOLD_S).
    results = run_parallel(fn, jobs, reserve_mb=reserves, limit_groups=groups, hold_reservations=True)
    energies = {(r["role"], r["basis"], r["method"]): r for r in results}
    record["points"] = [{k: v for k, v in r.items() if k != "signature"} for r in results]
    failed = [r for r in results if r.get("status") != "COMPLETED"]
    if failed:
        return done(status="EXECUTION_BLOCKED", reason="; ".join(f"{r['role']} {r['basis']} {r['method']}: "
                                                                   f"{r.get('error')}" for r in failed))

    def ea(basis: str, method: str) -> float:
        return (energies[("neutral", basis, method)]["total_hartree"]
                - energies[("anion", basis, method)]["total_hartree"]) * HARTREE_TO_EV

    delta_t = {b: ea(b, "CCSDT") - ea(b, "CCSD(T)") for b in t_bases}
    t_central = delta_t[t_bases[-1]]
    # Two bases: the last basis change; one basis only: 50 % of the increment.
    t_bound = (abs(delta_t[t_bases[-1]] - delta_t[t_bases[0]]) if len(t_bases) > 1
               else 0.5 * abs(t_central))
    record.update(delta_t_by_basis_ev=delta_t, delta_t_ev=t_central, delta_t_bound_ev=t_bound)
    if not run_q and FNO_BEYOND_TRIPLES:
        fno = _fno_increment(states, policy, output_dir)
        record["fno_ccsdtq"] = fno
        if fno["status"] == "COMPLETED":
            q_basis_used = q_candidates[-1]
            diffuse = q_diffuse_allowance_ev(q_basis_used)
            q_bound = fno["half_width_ev"] + policy.quadruples_relative_bound * abs(fno["delta_ev"]) + diffuse
            return done(status="CLEARED_FNO_CCSDTQ", quadruples_computed=False, beyond_triples_computed=True,
                        quadruples_basis=q_basis_used, delta_q_ev=fno["delta_ev"], delta_q_bound_ev=q_bound,
                        quadruples_diffuse_allowance_ev=diffuse,
                        delta_post_ev=t_central + fno["delta_ev"], bound_ev=t_bound + q_bound,
                        reason=f"CCSDTQ storage {estimates} MB exceeds {limit_mb} MB; beyond-CCSDT increment "
                               f"E(CCSDTQ) - E(CCSDT) in frozen-natural-orbital spaces ({q_basis_used})")
    if not run_q:
        allowance = max(Q_ALLOWANCE_MIN_EV, abs(t_central))
        return done(status="CLEARED_WITH_QUADRUPLES_ALLOWANCE", quadruples_computed=False,
                    delta_q_ev=None, quadruples_allowance_ev=allowance, post_cc_allowance_ev=allowance,
                    delta_post_ev=t_central, bound_ev=t_bound + allowance,
                    reason=f"CCSDTQ storage {estimates} MB exceeds {limit_mb} MB RAM or the scratch share; "
                           f"Delta_T applied, quadruples allowance {allowance:.4f} eV added to the bound")
    delta_q = ea(q_basis, "CCSDTQ") - ea(q_basis, "CCSDT")
    diffuse = q_diffuse_allowance_ev(q_basis)
    q_bound = policy.quadruples_relative_bound * abs(delta_q) + diffuse
    return done(status="CLEARED", quadruples_computed=True, delta_q_ev=delta_q, delta_q_bound_ev=q_bound,
                quadruples_diffuse_allowance_ev=diffuse,
                delta_post_ev=t_central + delta_q, bound_ev=t_bound + q_bound)
