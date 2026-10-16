"""Multistate (MR) layer for atomic fragments (Patch 61).

The atomic CCSD(T) energy (``fragment_execution``, configuration control of
Patch 60) is a single determinant per 2S: it is the energy of one configuration,
spin-free, and of one real-orbital component of its term.  For d/f atoms the
fragment ground state needs more:

* several configurations of different 2S lie within tenths of an eV (La 5d 6s2
  2D / 5d2 6s 4F, Ti 3d2 4s2 / 3d3 4s), and CCSD(T) of each determinant may
  order them wrongly;
* the measured ground level is the lowest spin-orbit level (La 2D3/2, I 2P3/2,
  Pb 3P0), up to an eV below the spin-free term for 6p atoms.

Per (element, charge) one OpenMolcas run computes, all-electron with ANO-RCC
(DKH2) and AMFI:

* SA-CASSCF per 2S over the active shells (every subshell whose occupation
  differs between the low configurations of ``state_candidates``, both charges),
  ``roots`` = largest (2L+1) of a Hund term with that 2S, capped;
* XMS-CASPT2 per 2S block (MS for one root);
* RASSI spin-orbit over all blocks with the XMS-CASPT2 energies (EJOB).

With the CCSD(T) anchor spin a (lowest CCSD(T) 2S) the fragment ground level is

  E0 = E_CC(a) + [E_SO,1 - E_PT2,min(a)]

so the bracket contains both the spin-free reordering (another 2S lower at
XMS-CASPT2) and the spin-orbit lowering.  Uncertainty: ``relative_bound`` *
|bracket| plus (1 - w) * ``mixing_gap_ev`` with w the weight of anchor-spin
states in the lowest SO level (the same terms as the molecular multistate layer,
``spin_orbit_multistate``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .electronic_ea_stage import HARTREE_TO_EV
from .spin_orbit_multistate import parse_so_weights
from .spin_orbit_stage import SpinOrbitStagePolicy, _run_molcas, parse_rassi
from .state_candidates import atomic_number, atomic_states, terms_of_configuration

_CAPACITY = {0: 2, 1: 6, 2: 10, 3: 14}


@dataclass(frozen=True)
class AtomicMultistatePolicy:
    basis: str = "ANO-RCC-VTZP"
    max_roots: int = 15
    relative_bound: float = 0.15
    mixing_gap_ev: float = 0.10
    molcas_memory_mb: int = 8000


def _parse_configuration(text: str) -> dict[tuple[int, int], int]:
    return {(int(t[0]), "spdf".index(t[1])): int(t[2:]) for t in text.split()}


def active_shells(element: str) -> list[tuple[int, int]]:
    """Subshells open in, or differently occupied between, the low configurations of both charges."""
    configs = [_parse_configuration(st.configuration)
               for q in (0, -1) for st in atomic_states(element, q)]
    shells = sorted({s for c in configs for s in c})
    out = []
    for s in shells:
        occ = {c.get(s, 0) for c in configs}
        if len(occ) > 1 or any(0 < k < _CAPACITY[s[1]] for k in occ):
            out.append(s)
    return out


def active_space(element: str, charge: int) -> dict:
    shells = active_shells(element)
    n_elec = atomic_number(element) - charge
    ground = _parse_configuration(next(st.configuration for st in atomic_states(element, charge)
                                       if st.configuration_kind == "GROUND_AUFBAU"))
    n_act = sum(ground.get(s, 0) for s in shells)
    return {"shells": [f"{n}{'spdf'[l]}" for n, l in shells], "active_electrons": n_act,
            "active_orbitals": sum(2 * l + 1 for _, l in shells), "inactive": (n_elec - n_act) // 2}


def _label_counts(label: str | None) -> dict[str, int] | None:
    """'s1d2' -> {'s': 1, 'd': 2} (configuration_label of the fragment); None for 'closed'/unknown."""
    import re
    if not label or label == "closed":
        return None
    return {l: int(k) for l, k in re.findall(r"([spdf])(\d+)", label)}


def roots_per_spin(element: str, charge: int, spins, max_roots: int,
                   labels: Mapping[int, str | None] | None = None) -> dict[int, int]:
    """Roots per 2S block: the components (2L+1) of the highest-L term with that 2S of the
    configuration the CCSD(T) determinant of that 2S occupies (its open-shell label); without
    a matching label the largest (2L+1) over all low configurations."""
    states = atomic_states(element, charge)
    out = {}
    for two_s in spins:
        label = (labels or {}).get(two_s)
        counts = _label_counts(label)
        pool = [st for st in states if st.term.two_s == two_s]
        if counts is not None:
            # open shells of the determinant itself (high spin: one electron per open orbital)
            occ = [(9, "spdf".index(l), k) for l, k in counts.items()]
            degeneracies = [2 * t.l + 1 for t in terms_of_configuration(occ) if t.two_s == two_s]
        elif label == "closed":
            # closed determinant: the configuration with the fewest components at this 2S
            per_config: dict[str, int] = {}
            for st in pool:
                per_config[st.configuration] = max(per_config.get(st.configuration, 0), 2 * st.term.l + 1)
            degeneracies = [min(per_config.values())] if per_config else []
        else:
            degeneracies = [2 * st.term.l + 1 for st in pool]
        out[two_s] = min(max_roots, max(degeneracies, default=1))
    return out


def _head(element: str, policy: "AtomicMultistatePolicy") -> list[str]:
    return ["&GATEWAY", "Coord", "1", "Angstrom", f"{element}1 0.0 0.0 0.0", "Group = NoSym",
            f"Basis = {policy.basis}", "AMFI", "", "&SEWARD", ""]


def block_input(element: str, charge: int, space: Mapping, two_s: int, n: int,
                policy: "AtomicMultistatePolicy", jobmix_path: Path) -> str:
    """One 2S block: SA-CASSCF over n roots, XMS-CASPT2 (MS for one root), JobMix saved."""
    return "\n".join(_head(element, policy) + [
        "&RASSCF", "Spin", str(two_s + 1), "Charge", str(charge),
        "nActEl", f"{space['active_electrons']} 0 0", "Inactive", str(space["inactive"]),
        "Ras2", str(space["active_orbitals"]), "CIRoot", f"{n} {n} 1", "Iterations", "200 100",
        "End of input", "&CASPT2", ("XMultistate" if n > 1 else "MultiState"),
        " ".join(map(str, [n] + list(range(1, n + 1)))), "End of input",
        f">>> COPY $Project.JobMix {jobmix_path}", ""])


def rassi_input(element: str, roots: Mapping[int, int], jobmix: Mapping[int, Path],
                policy: "AtomicMultistatePolicy") -> str:
    lines = _head(element, policy)
    lines += [f">>> COPY {jobmix[two_s]} JOB{k:03d}" for k, two_s in enumerate(roots, 1)] + [""]
    lines += ["&RASSI", "Nr of JobIph", f"{len(roots)} " + " ".join(str(n) for n in roots.values())]
    lines += [" ".join(map(str, range(1, n + 1))) for n in roots.values()]
    lines += ["SpinOrbit", "EJob", "End of input", ""]
    return "\n".join(lines)


def atomic_multistate(*, element: str, charge: int, cc_by_spin: Mapping[int, float], workdir: Path,
                      labels: Mapping[int, str | None] | None = None,
                      policy: AtomicMultistatePolicy = AtomicMultistatePolicy()) -> dict[str, Any]:
    """cc_by_spin: {2S: CCSD(T) energy} of the fragment scan (completed spins only);
    labels: {2S: configuration_label} of those determinants."""
    from concurrent.futures import ThreadPoolExecutor
    rec: dict[str, Any] = {"element": element, "charge": charge,
                           "method": f"SO-XMS-CASPT2/{policy.basis} DKH2 AMFI RASSI-EJOB (OpenMolcas)"}
    if not cc_by_spin:
        return {**rec, "status": "NO_CCSD_T_ANCHOR", "bracket_ev": None}
    anchor = min(cc_by_spin, key=cc_by_spin.get)
    space = active_space(element, charge)
    max_two_s = min(space["active_electrons"], 2 * space["active_orbitals"] - space["active_electrons"])
    spins = sorted({s for s in cc_by_spin if s <= max_two_s} | {s for s in (anchor - 2, anchor + 2)
                                                                   if 0 <= s <= max_two_s})
    roots = roots_per_spin(element, charge, spins, policy.max_roots, labels)
    rec.update(anchor_two_s=anchor, active_space=space, roots=roots)
    workdir = Path(workdir).resolve()
    so_policy = SpinOrbitStagePolicy(basis=policy.basis, molcas_memory_mb=policy.molcas_memory_mb)
    jobmix = {two_s: workdir / f"block_2S{two_s}" / "jobmix" for two_s in roots}

    def run_block(two_s):
        return _run_molcas(workdir / f"block_2S{two_s}",
                           block_input(element, charge, space, two_s, roots[two_s], policy, jobmix[two_s]),
                           so_policy, name="block")
    with ThreadPoolExecutor(max_workers=len(roots)) as ex:
        list(ex.map(run_block, roots))
    if not all(p.is_file() for p in jobmix.values()):
        return {**rec, "status": "MULTISTATE_CALCULATION_FAILED", "bracket_ev": None,
                "failed_blocks": [s for s, p in jobmix.items() if not p.is_file()]}
    text = _run_molcas(workdir / "rassi", rassi_input(element, roots, jobmix, policy), so_policy, name="rassi")
    sf, so = parse_rassi(text)
    if len(sf) != sum(roots.values()) or not so:
        return {**rec, "status": "MULTISTATE_CALCULATION_FAILED", "bracket_ev": None, "n_spin_free": len(sf)}
    blocks, offset = {}, 0
    for two_s, n in roots.items():
        blocks[two_s] = list(range(offset + 1, offset + n + 1))  # RASSI numbering
        offset += n
    lowest = {two_s: min(sf[i - 1] for i in idx) for two_s, idx in blocks.items()}
    e_anchor = lowest[anchor]
    bracket = (min(so) - e_anchor) * HARTREE_TO_EV
    weights = parse_so_weights(text)
    w = sum(wt for s, _, wt in weights.get(1, []) if s in blocks[anchor])
    gaps_mr = {two_s: (e - e_anchor) * HARTREE_TO_EV for two_s, e in lowest.items()}
    gaps_cc = {two_s: (e - cc_by_spin[anchor]) * HARTREE_TO_EV for two_s, e in cc_by_spin.items()}
    bound = policy.relative_bound * abs(bracket) + (1.0 - w) * policy.mixing_gap_ev
    return {**rec, "status": "CLEARED", "bracket_ev": bracket, "bound_ev": bound, "anchor_weight": w,
            "spin_free_gap_ev_mr": gaps_mr, "spin_free_gap_ev_cc": gaps_cc,
            "spin_free_lowering_ev": (min(lowest.values()) - e_anchor) * HARTREE_TO_EV,
            "spin_orbit_lowering_ev": (min(so) - min(sf)) * HARTREE_TO_EV,
            "lowest_so_composition": weights.get(1)}
