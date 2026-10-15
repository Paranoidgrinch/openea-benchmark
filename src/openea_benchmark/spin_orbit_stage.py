"""First-order spin-orbit layer for diatomic states (Patch 47).

Experimental electron affinities connect the *lowest spin-orbit levels* of
anion and neutral; the CCSD(T)/CBS energies are spin-free.  For a state with
orbital angular momentum Lambda > 0 and spin S > 0 the first-order spin-orbit
coupling splits the level into Omega components and lowers the lowest one by

    Delta_SO = E(lowest SO level) - E(spin-free),  = -|A| Lambda S  for a regular multiplet.

Sigma states (Lambda = 0) and singlets have no first-order shift; their
second-order shift is not computed here (recorded, small for light atoms).

Method, per state (no molecule-specific input):

* full-valence CASSCF (ns, np of each atom; 1s for H/He; 4s 3d for Sc-Zn;
  frozen cores inactive), the same rule for every diatomic up to Kr.  A minimal
  open-shell CAS was tried first and fails for polar pi* shells (ClO X2Pi: 196
  cm-1 against 320 cm-1 measured), because the ROHF-like pi* is too O-localized;
* OpenMolcas: ANO-RCC-VTZP, AMFI mean-field spin-orbit integrals.  The two
  Lambda components need not be the lowest roots (Sigma states share their
  symmetry).  Patch 55: in C2 (both components in one symmetry: a for even,
  b for odd Lambda) a first run state-averages the ``identify_roots`` lowest
  roots and finds the component pair by RASSI |<i|Lz|j>| = Lambda and
  degeneracy; a second RASSI couples that pair by spin-orbit;
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
    # Roots in the C2 symmetry for the <Lz> identification run, tried in turn.
    identify_roots: tuple[int, ...] = (6, 4, 2)
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


# Frozen cores (orbitals per C2v irrep, Molcas order a1 b1 b2 a2): noble gases, and
# [Ar]3d for Ga-Kr (d: a1 a1 b1 b2 a2).
CORE_ORBITALS_BY_COUNT = {0: (0, 0, 0, 0), 1: (1, 0, 0, 0), 5: (3, 1, 1, 0), 9: (5, 2, 2, 0),
                          14: (7, 3, 3, 1)}
# Valence shells per C2v irrep: ns np (main group), 1s (H, He), 4s 3d (Sc-Zn).
VALENCE_SP, VALENCE_S, VALENCE_SD = (2, 1, 1, 0), (1, 0, 0, 0), (3, 1, 1, 1)


def full_valence_space(atoms: Sequence[str], charge: int) -> dict | None:
    """Full-valence CAS, frozen cores inactive (rule in ``heavy_elements.full_valence_space``):
    main group ns np (H/He 1s), d block ns (n-1)d; None for open f shells."""
    from .heavy_elements import full_valence_space as rule
    return rule(atoms, charge)


def c2_space(space: Mapping) -> dict:
    """C2v (a1 b1 b2 a2) orbital counts merged to C2 (a = a1+a2, b = b1+b2)."""
    merge = lambda t: (t[0] + t[3], t[1] + t[2])
    return {**space, "inactive": merge(space["inactive"]), "ras2": merge(space["ras2"])}


def _molcas_head(atoms, r_angstrom, policy, angmom: bool) -> list[str]:
    # Group XY = C2 about the bond axis: both components of a Lambda > 0 state lie in
    # one symmetry (a: even Lambda, b: odd), so one state average treats them alike.
    return ["&GATEWAY", "Coord", "2", "Angstrom",
            f"{atoms[0]}1  0.0 0.0 0.0", f"{atoms[1]}2  0.0 0.0 {float(r_angstrom):.8f}",
            "Group = XY", f"Basis = {policy.basis}", "AMFI"] + (["AngMom = 0.0 0.0 0.0"] if angmom else []) \
        + ["", "&SEWARD", ""]


def identify_input(atoms, r_angstrom, charge, spin_mult, space, lam, nroots, policy) -> str:
    """Run 1: SA-CASSCF over the ``nroots`` lowest roots of the C2 symmetry of Lambda,
    RASSI <Lz> matrix between them."""
    sp = c2_space(space)
    lines = _molcas_head(atoms, r_angstrom, policy, angmom=True)
    lines += ["\n".join([
        "&RASSCF", "Symmetry", "2" if lam % 2 else "1", "Spin", str(spin_mult), "Charge", str(charge),
        "nActEl", f"{sp['active_electrons']} 0 0", "Inactive", " ".join(map(str, sp["inactive"])),
        "Ras2", " ".join(map(str, sp["ras2"])), "CIRoot", f"{nroots} {nroots} 1", "Iterations", "200 100",
        "End of input", ""]),
              ">>> COPY $Project.JobIph JOB001", "",
              "&RASSI", "Nr of JobIph", f"1 {nroots}", " ".join(map(str, range(1, nroots + 1))),
              "Prop", "1", "'AngMom' 3", "MEES", "End of input", ""]
    return "\n".join(lines)


def spin_orbit_rassi_input(roots: tuple[int, int]) -> str:
    """Run 2, in the scratch of run 1: RASSI spin-orbit between the two identified
    component roots of the state-averaged JobIph (state-specific CASSCF of an excited
    root can fall to another solution: TiO X3Delta 0.2 Eh too high)."""
    ia, ib = roots
    return "\n".join(["&RASSI", "SpinOrbit", "Nr of JobIph", "1 2", f"{ia} {ib}", "End of input", ""])


def parse_angmom_z(text: str, n: int):
    """The n x n RASSI matrix of ANGMOM component 3 (Lz, antihermitian real part)."""
    import numpy as np
    m = re.search(r"PROPERTY: ANGMOM\s+COMPONENT:\s+3\s*\n\s*ORIGIN:.*\n\s*\n\s*STATE[^\n]*\n\s*\n((?:\s*\d+(?:\s+-?\d\.\d+E[+-]\d+)+\s*\n)+)",
                  text)
    if m is None:
        return None
    rows = [list(map(float, line.split()[1:])) for line in m.group(1).strip().splitlines()]
    if len(rows) != n or any(len(r) != n for r in rows):
        return None  # wide matrices are printed in column blocks; not needed for n <= 8
    return np.array(rows)


def component_pair(sf: Sequence[float], lz, lam: int, nroots: int, *, lz_tolerance: float = 0.15,
                   degeneracy_hartree: float = 1.0e-5) -> tuple[int, int] | None:
    """Lowest pair of roots (1-based) with |<i|Lz|j>| = Lambda and equal energies.

    The pair must lie wholly inside the averaged roots: a degenerate partner cut off
    at the last root would leave one component unequally treated."""
    best = None
    for i in range(nroots):
        for j in range(i + 1, nroots):
            if abs(abs(lz[i, j]) - lam) > lz_tolerance or abs(sf[i] - sf[j]) > degeneracy_hartree:
                continue
            e = 0.5 * (sf[i] + sf[j])
            if best is None or e < best[0]:
                best = (e, i + 1, j + 1)
    return None if best is None else (best[1], best[2])


def parse_rassi(text: str) -> tuple[list[float], list[float]]:
    sf = [float(x) for x in re.findall(r"::\s+RASSI State\s+\d+\s+Total energy:\s+(-?\d+\.\d+)", text)]
    so = [float(x) for x in re.findall(r"::\s+SO-RASSI State\s+\d+\s+Total energy:\s+(-?\d+\.\d+)", text)]
    return sf, so


def _run_molcas(workdir: Path, text: str, policy: SpinOrbitStagePolicy, name: str = "soc") -> str:
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / f"{name}.out"
    stamp = workdir / f"{name}.input.sha256"
    digest = sha256(text.encode()).hexdigest()
    if out.is_file() and stamp.is_file() and stamp.read_text().strip() == digest and "Happy landing" in out.read_text():
        return out.read_text()
    (workdir / f"{name}.input").write_text(text)
    molcas = Path(os.environ.get("OPENEA_MOLCAS", DEFAULT_MOLCAS))
    env = dict(os.environ, MOLCAS=str(molcas), MOLCAS_WORKDIR=str(workdir / "scratch"),
               MOLCAS_MEM=str(policy.molcas_memory_mb), MOLCAS_PROJECT="soc")
    (workdir / "scratch").mkdir(exist_ok=True)
    with open(out, "w") as fh:
        subprocess.run(["nice", "-n", "19", str(molcas / "pymolcas"), f"{name}.input"], cwd=workdir, env=env,
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
    space = full_valence_space(atoms, charge)
    if space is None:
        return {**rec, "status": "SOC_ACTIVE_SPACE_NOT_DERIVED", "delta_so_ev": None}
    rec["active_space"] = space
    roots, ident_dir = None, None
    for nroots in policy.identify_roots:
        ident_dir = Path(workdir) / f"identify{nroots}"
        ident = _run_molcas(ident_dir, identify_input(atoms, r_angstrom, charge, mult, space, lam, nroots, policy),
                            policy)
        sf_id, _ = parse_rassi(ident)
        lz = parse_angmom_z(ident, nroots)
        if len(sf_id) != nroots or lz is None:
            continue  # e.g. fewer configurations than roots
        roots = component_pair(sf_id, lz, lam, nroots)
        rec["identification"] = {"nroots": nroots, "spin_free_hartree": sf_id,
                                 "lz": [[float(x) for x in row] for row in lz], "component_roots": roots}
        if roots is not None:
            break
    if roots is None:
        return {**rec, "status": "SOC_COMPONENTS_NOT_IDENTIFIED", "delta_so_ev": None}
    # The run-1 digest is part of the run-2 input, so a recomputed run 1 also reruns run 2.
    text = _run_molcas(ident_dir, spin_orbit_rassi_input(roots) + f"* run1 {sha256(ident.encode()).hexdigest()}\n",
                       policy, name="soc_so")
    sf, so = parse_rassi(text)
    if len(sf) != 2 or len(so) != 2 * mult:
        return {**rec, "status": "SOC_CALCULATION_FAILED", "delta_so_ev": None,
                "spin_free_hartree": sf, "spin_orbit_hartree": so}
    if abs(sf[0] - sf[1]) > policy.degeneracy_tolerance_hartree:
        return {**rec, "status": "SOC_COMPONENTS_NOT_DEGENERATE", "delta_so_ev": None,
                "spin_free_hartree": sf, "spin_orbit_hartree": so}
    shift = (min(so) - 0.5 * (sf[0] + sf[1])) * HARTREE_TO_EV
    splitting_cm = (max(so) - min(so)) * 219474.6313632
    # Both components come from one state average; any residual asymmetry is added.
    asym = 0.5 * abs(sf[0] - sf[1]) * HARTREE_TO_EV
    return {**rec, "status": "CLEARED", "delta_so_ev": shift,
            "bound_ev": policy.relative_bound * abs(shift) + asym, "component_asymmetry_ev": asym,
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
