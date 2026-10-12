"""First-order spin-orbit layer for diatomic states (Patch 47).

Experimental electron affinities connect the *lowest spin-orbit levels* of
anion and neutral; the CCSD(T)/CBS energies are spin-free.  For a state with
orbital angular momentum Lambda > 0 and spin S > 0 the first-order spin-orbit
coupling splits the level into Omega components and lowers the lowest one by

    Delta_SO = E(lowest SO level) - E(spin-free),  = -|A| Lambda S  for a regular multiplet.

Sigma states (Lambda = 0) and singlets have no first-order shift; their
second-order shift is not computed here (recorded, small for light atoms).

Method, per state (no molecule-specific input):

* full-valence CASSCF (ns, np of each atom; 1s for H/He; noble-gas cores
  inactive), the same rule for every main-group diatomic.  A minimal open-shell
  CAS was tried first and fails for polar pi* shells (ClO X2Pi: 196 cm-1 against
  320 cm-1 measured), because the ROHF-like pi* is too O-localized.  d-valence
  atoms, Delta/Phi states fail closed (``SOC_ACTIVE_SPACE_NOT_DERIVED``);
* OpenMolcas: ANO-RCC-VTZP, AMFI mean-field spin-orbit integrals, one CASSCF per
  Lambda component (C2v symmetries b1 and b2), RASSI spin-orbit state
  interaction between the two components;
* Delta_SO = lowest SO-RASSI energy - mean of the two spin-free energies.
  The two spin-free component energies must agree (degenerate by symmetry).

Calibration against measured splittings (not used in the computation): OH
133 / 139, NO 126 / 123, O2- 153 / ~160, ClO 244 / 320 cm-1.  The bound is a
pilot value covering ClO: 30 % of |Delta_SO| per state, combined linearly.

Delta_EA_SO = Delta_SO(neutral) - Delta_SO(anion)  (EA = E_neutral - E_anion).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from .electronic_ea_stage import HARTREE_TO_EV, _save_json

LAMBDA_OF = {"Sigma": 0, "Pi": 1, "Delta": 2, "Phi": 3}
# D2h irreps (molecular axis z, PySCF convention) collapsed onto C2v.
D2H_TO_C2V = {"Ag": "A1", "B1u": "A1", "B1g": "A2", "Au": "A2",
              "B3u": "B1", "B2g": "B1", "B2u": "B2", "B3g": "B2"}
MOLCAS_C2V_ORDER = ("A1", "B1", "B2", "A2")
DEFAULT_MOLCAS = Path(__file__).resolve().parents[2] / "external" / "openmolcas-install-v26.06-serial-ilp64"


@dataclass(frozen=True)
class SpinOrbitStagePolicy:
    """Explicit pilot policy, not a validated production threshold set."""
    basis: str = "ANO-RCC-VTZP"
    # Calibration (CASSCF/ANO-RCC-VTZP AMFI vs experiment): OH 133/139, NO 126/123,
    # O2- 153/~160, ClO 244/320 cm-1 (polar pi*, dynamic correlation missing).
    relative_bound: float = 0.30
    degeneracy_tolerance_hartree: float = 1.0e-5
    molcas_memory_mb: int = 4000


def parse_term_symbol(label: str) -> tuple[int, int] | None:
    """(2S+1, Lambda) from labels like '2Pi', '3Sigmag-', '2Pig'."""
    m = re.fullmatch(r"(\d+)(Sigma|Pi|Delta|Phi)[gu]?[+-]?", label or "")
    return None if m is None else (int(m.group(1)), LAMBDA_OF[m.group(2)])


def c2v_occupation(irrep_nelec: Mapping[str, Sequence[int]]) -> dict[str, tuple[int, int]]:
    out = {k: [0, 0] for k in MOLCAS_C2V_ORDER}
    for irrep, (na, nb) in irrep_nelec.items():
        target = D2H_TO_C2V.get(irrep, irrep)
        if target not in out:
            raise ValueError(f"irrep {irrep} has no C2v image")
        out[target][0] += int(na)
        out[target][1] += int(nb)
    return {k: (v[0], v[1]) for k, v in out.items()}


# Noble-gas cores (orbitals per C2v irrep, Molcas order a1 b1 b2 a2).
CORE_ORBITALS_BY_COUNT = {0: (0, 0, 0, 0), 1: (1, 0, 0, 0), 5: (3, 1, 1, 0), 9: (5, 2, 2, 0)}


def full_valence_space(atoms: Sequence[str], charge: int) -> dict | None:
    """Full-valence CAS for main-group atoms (ns, np; H/He: 1s), noble-gas cores inactive.

    Returns None for atoms with d valence or a core beyond [Ar] (fail closed).
    """
    from pyscf import gto
    from .core_valence_basis import core_orbital_count
    inactive, ras2, valence_electrons = [0, 0, 0, 0], [0, 0, 0, 0], 0
    for el in atoms:
        z = gto.charge(el)
        if z > 18 and el not in ("K", "Ca"):
            return None  # 3d elements and beyond: d valence not handled here
        ncore = core_orbital_count(el)
        core = CORE_ORBITALS_BY_COUNT.get(ncore)
        if core is None:
            return None
        inactive = [a + b for a, b in zip(inactive, core)]
        ras2 = [a + b for a, b in zip(ras2, (1, 0, 0, 0) if z <= 2 else (2, 1, 1, 0))]
        valence_electrons += z - 2 * ncore
    nact = valence_electrons - charge
    if not 0 < nact < 2 * sum(ras2):
        return None
    return {"inactive": tuple(inactive), "ras2": tuple(ras2), "active_electrons": nact,
            "rule": "FULL_VALENCE_NS_NP"}


def molcas_input(atoms: Sequence[str], r_angstrom: float, charge: int, spin_mult: int, space: Mapping,
                 policy: SpinOrbitStagePolicy) -> str:
    def rasscf(sym: int, title: str) -> str:
        return "\n".join([
            "&RASSCF", "Title", title, "Symmetry", str(sym), "Spin", str(spin_mult),
            "Charge", str(charge), "nActEl", f"{space['active_electrons']} 0 0",
            "Inactive", " ".join(map(str, space["inactive"])), "Ras2", " ".join(map(str, space["ras2"])),
            "CIRoot", "1 1 1", "Iterations", "200 100", "End of input", ""])
    return "\n".join([
        "&GATEWAY", "Coord", "2", "Angstrom",
        f"{atoms[0]}1  0.0 0.0 0.0", f"{atoms[1]}2  0.0 0.0 {float(r_angstrom):.8f}",
        "Group = X Y", f"Basis = {policy.basis}", "AMFI", "", "&SEWARD", "",
        rasscf(2, "Lambda component b1"),
        ">>> COPY $Project.JobIph JOB001", "",
        rasscf(3, "Lambda component b2"),
        ">>> COPY $Project.JobIph JOB002", "",
        "&RASSI", "SpinOrbit", "Nr of JobIph", "2 1 1", "1", "1", "End of input", ""])


def parse_rassi(text: str) -> tuple[list[float], list[float]]:
    sf = [float(x) for x in re.findall(r"::\s+RASSI State\s+\d+\s+Total energy:\s+(-?\d+\.\d+)", text)]
    so = [float(x) for x in re.findall(r"::\s+SO-RASSI State\s+\d+\s+Total energy:\s+(-?\d+\.\d+)", text)]
    return sf, so


def _run_molcas(workdir: Path, text: str, policy: SpinOrbitStagePolicy) -> str:
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "soc.out"
    stamp = workdir / "soc.input.sha256"
    digest = sha256(text.encode()).hexdigest()
    if out.is_file() and stamp.is_file() and stamp.read_text().strip() == digest and "Happy landing" in out.read_text():
        return out.read_text()
    (workdir / "soc.input").write_text(text)
    molcas = Path(os.environ.get("OPENEA_MOLCAS", DEFAULT_MOLCAS))
    env = dict(os.environ, MOLCAS=str(molcas), MOLCAS_WORKDIR=str(workdir / "scratch"),
               MOLCAS_MEM=str(policy.molcas_memory_mb), MOLCAS_PROJECT="soc")
    (workdir / "scratch").mkdir(exist_ok=True)
    with open(out, "w") as fh:
        subprocess.run(["nice", "-n", "19", str(molcas / "pymolcas"), "soc.input"], cwd=workdir, env=env,
                       stdout=fh, stderr=subprocess.STDOUT, check=False)
    stamp.write_text(digest)
    return out.read_text()


def state_spin_orbit_shift(*, atoms, r_angstrom, charge, symmetry_label, irrep_nelec, workdir: Path,
                           policy: SpinOrbitStagePolicy) -> dict:
    term = parse_term_symbol(symmetry_label)
    rec: dict[str, Any] = {"symmetry_label": symmetry_label, "r_angstrom": r_angstrom}
    if term is None:
        return {**rec, "status": "SOC_TERM_SYMBOL_UNKNOWN", "delta_so_ev": None}
    mult, lam = term
    if lam == 0 or mult == 1:
        return {**rec, "status": "FIRST_ORDER_ZERO", "delta_so_ev": 0.0, "bound_ev": 0.0,
                "second_order": "NOT_COMPUTED"}
    space = full_valence_space(atoms, charge) if lam == 1 else None
    if space is None:
        return {**rec, "status": "SOC_ACTIVE_SPACE_NOT_DERIVED", "delta_so_ev": None}
    rec["active_space"] = space
    text = _run_molcas(workdir, molcas_input(atoms, r_angstrom, charge, mult, space, policy), policy)
    sf, so = parse_rassi(text)
    if len(sf) != 2 or len(so) != 2 * mult:
        return {**rec, "status": "SOC_CALCULATION_FAILED", "delta_so_ev": None,
                "spin_free_hartree": sf, "spin_orbit_hartree": so}
    if abs(sf[0] - sf[1]) > policy.degeneracy_tolerance_hartree:
        return {**rec, "status": "SOC_COMPONENTS_NOT_DEGENERATE", "delta_so_ev": None,
                "spin_free_hartree": sf, "spin_orbit_hartree": so}
    shift = (min(so) - 0.5 * (sf[0] + sf[1])) * HARTREE_TO_EV
    splitting_cm = (max(so) - min(so)) * 219474.6313632
    return {**rec, "status": "CLEARED", "delta_so_ev": shift, "bound_ev": policy.relative_bound * abs(shift),
            "splitting_cm1": splitting_cm, "spin_free_hartree": sf, "spin_orbit_hartree": so,
            "method": f"CASSCF/{policy.basis} AMFI RASSI (OpenMolcas)"}


def run_spin_orbit_stage(*, electronic_ea: Mapping[str, Any], requests: Sequence[Any], output_dir: Path,
                         policy: SpinOrbitStagePolicy = SpinOrbitStagePolicy()) -> dict:
    output_dir = Path(output_dir)
    record: dict[str, Any] = {"stage": "SPIN_ORBIT_FIRST_ORDER", "policy": asdict(policy),
                              "is_production_ea": False, "validation_metadata_used_in_computation": False}

    def done(**fields) -> dict:
        record.update(fields)
        _save_json(output_dir / "spin_orbit.json", record)
        return record

    if electronic_ea.get("status") != "ELECTRONIC_EA_EVALUATED":
        return done(status="NOT_ATTEMPTED", reason="single-basis electronic EA not evaluated")
    template = {}
    for req in requests:
        template.setdefault(req.job_id, req)
    states = {}
    for role in ("neutral", "anion"):
        job_id = electronic_ea[f"selected_{role}_job"]
        cand = next(c for c in electronic_ea["candidates"][role] if c["job_id"] == job_id)
        req = template.get(job_id)
        if req is None:
            return done(status="NOT_ATTEMPTED", reason=f"{role}: no request template for {job_id}")
        print(f"[SOC] {req.system} {role} {req.symmetry_label}", flush=True)
        states[role] = state_spin_orbit_shift(
            atoms=req.atoms, r_angstrom=float(cand["r_central_angstrom"]), charge=req.charge,
            symmetry_label=req.symmetry_label, irrep_nelec=req.irrep_nelec, workdir=output_dir / role,
            policy=policy)
    record["states"] = states
    if any(s.get("delta_so_ev") is None for s in states.values()):
        return done(status="SOC_UNRESOLVED", delta_ea_so_ev=None, bound_ev=None)
    return done(status="CLEARED",
                delta_ea_so_ev=states["neutral"]["delta_so_ev"] - states["anion"]["delta_so_ev"],
                bound_ev=states["neutral"]["bound_ev"] + states["anion"]["bound_ev"])
