"""Multistate spin-orbit layer for heavy-element diatomics (heavy-element step 2).

The first-order layer (``spin_orbit_stage``) couples only the two Lambda components
of one state at CASSCF level.  Beyond Kr the spin-orbit coupling mixes different
Lambda-S states (second order, Sigma states included) and its size makes the
CASSCF spin-free gaps too crude.  Per selected state:

* spin-free manifold: full-valence SA-CASSCF in C2 (both symmetries) for the
  state's multiplicity and its neighbours 2S+1 +- 2 (spin-orbit couples dS <= 1),
  ``roots_per_block`` roots each; XMS-CASPT2 for every block (single root: MS);
* RASSI with EJOB: spin-orbit Hamiltonian over all blocks with the XMS-CASPT2
  energies on the diagonal (SO-XMS-CASPT2, AMFI, ANO-RCC-VTZP);
* reference = the selected Lambda-S state: lowest XMS root set of its block
  (degenerate pair for Lambda > 0, single root for Sigma);
* Delta_SO = lowest SO level - reference spin-free energy (both XMS-CASPT2).
  It is added to the spin-free CCSD(T)/CBS energy of the reference state.

Uncertainty per state: ``relative_bound`` * |Delta_SO| (atomic validation: SO-XMS-CASPT2
splittings of Br, I, Tl, Pb, Bi 5-15 % below experiment) plus (1 - w) * ``mixing_gap_ev``,
w the weight of the reference in the lowest SO level: transferring the CC energy of
the reference is exact only for w = 1; the admixed states enter with their PT2 gaps
(validated to ~0.1 eV for NEVPT2 term gaps, Patch 55b).  Below ``min_reference_weight``
the lowest level is no longer the reference state (Hund's case (c) with a different
parent) and the layer fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

from .electronic_ea_stage import HARTREE_TO_EV
from .spin_orbit_stage import (SpinOrbitStagePolicy, _molcas_head, _run_molcas, c2_space, full_valence_space,
                               parse_rassi, parse_term_symbol)


@dataclass(frozen=True)
class MultistateSpinOrbitPolicy:
    """Explicit pilot policy (validation: docs, heavy-element step 2)."""
    basis: str = "ANO-RCC-VTZP"
    roots_per_block: int = 6
    relative_bound: float = 0.15
    mixing_gap_ev: float = 0.10
    min_reference_weight: float = 0.5
    degeneracy_tolerance_hartree: float = 2.0e-5
    molcas_memory_mb: int = 8000


def applies(atoms: Sequence[str]) -> bool:
    """The multistate layer replaces the first-order one for molecules with an atom beyond Kr."""
    from pyscf import gto
    return any(gto.charge(el) > 36 for el in atoms)


def spin_blocks(mult: int, n_active_electrons: int, n_active_orbitals: int) -> list[int]:
    """Multiplicities coupled to ``mult`` by spin-orbit (dS <= 1) that the active space can hold."""
    max_two_s = min(n_active_electrons, 2 * n_active_orbitals - n_active_electrons)
    return [m for m in (mult - 2, mult, mult + 2) if m >= 1 and m - 1 <= max_two_s and (m - 1) % 2 == n_active_electrons % 2]


def multistate_input(atoms, r_angstrom, charge, space, blocks: Sequence[tuple[int, int, int]],
                     policy: MultistateSpinOrbitPolicy) -> str:
    """blocks: (multiplicity, C2 symmetry 1/2, nroots)."""
    sp = c2_space(space)
    lines = _molcas_head(atoms, r_angstrom, SpinOrbitStagePolicy(basis=policy.basis), angmom=False)
    for k, (mult, sym, n) in enumerate(blocks, 1):
        lines += ["&RASSCF", "Symmetry", str(sym), "Spin", str(mult), "Charge", str(charge),
                  "nActEl", f"{sp['active_electrons']} 0 0", "Inactive", " ".join(map(str, sp["inactive"])),
                  "Ras2", " ".join(map(str, sp["ras2"])), "CIRoot", f"{n} {n} 1", "Iterations", "200 100",
                  "End of input", "&CASPT2", ("XMultistate" if n > 1 else "MultiState"),
                  " ".join(map(str, [n] + list(range(1, n + 1)))), "End of input",
                  f">>> COPY $Project.JobMix JOB{k:03d}", ""]
    lines += ["&RASSI", "Nr of JobIph", f"{len(blocks)} " + " ".join(str(n) for _, _, n in blocks)]
    lines += [" ".join(map(str, range(1, n + 1))) for _, _, n in blocks]
    lines += ["SpinOrbit", "EJob", "End of input", ""]
    return "\n".join(lines)


def parse_so_weights(text: str) -> dict[int, list[tuple[int, float, float]]]:
    """{SO state: [(spin-free state, S, weight), ...]} from RASSI's five-largest-weights table."""
    m = re.search(r"Weights of the five most important spin-orbit-free states.*?-{20,}\n(.*?)\n\s*\n", text, re.S)
    out: dict[int, list[tuple[int, float, float]]] = {}
    if m is None:
        return out
    for line in m.group(1).splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        vals = parts[2:]
        out[int(parts[0])] = [(int(vals[i]), float(vals[i + 1]), float(vals[i + 2])) for i in range(0, len(vals) - 2, 3)]
    return out


def reference_roots(energies: Sequence[float], lam: int, tolerance: float) -> list[int] | None:
    """0-based roots of the reference inside its block: the lowest degenerate pair
    (Lambda > 0) or the lowest non-degenerate root (Sigma)."""
    order = sorted(range(len(energies)), key=lambda i: energies[i])
    i = 0
    while i < len(order):
        group = [order[i]]
        while i + len(group) < len(order) and abs(energies[order[i + len(group)]] - energies[order[i]]) <= tolerance:
            group.append(order[i + len(group)])
        if (lam > 0 and len(group) == 2) or (lam == 0 and len(group) == 1):
            return group
        i += len(group)
    return None


def state_multistate_shift(*, atoms, r_angstrom, charge, symmetry_label, workdir: Path,
                           policy: MultistateSpinOrbitPolicy = MultistateSpinOrbitPolicy()) -> dict:
    term = parse_term_symbol(symmetry_label)
    rec: dict[str, Any] = {"symmetry_label": symmetry_label, "r_angstrom": r_angstrom,
                           "method": f"SO-XMS-CASPT2/{policy.basis} AMFI RASSI-EJOB (OpenMolcas)"}
    if term is None:
        return {**rec, "status": "SOC_TERM_SYMBOL_UNKNOWN", "delta_so_ev": None}
    mult, lam = term
    space = full_valence_space(atoms, charge)
    if space is None:
        return {**rec, "status": "SOC_ACTIVE_SPACE_NOT_DERIVED", "delta_so_ev": None}
    norb = sum(space["ras2"])
    mults = spin_blocks(mult, space["active_electrons"], norb)
    ref_sym = 2 if lam % 2 else 1
    blocks = [(m, sym, policy.roots_per_block) for m in mults for sym in (ref_sym, 3 - ref_sym)]
    rec.update(active_space=space, blocks=blocks)
    text = _run_molcas(Path(workdir), multistate_input(atoms, r_angstrom, charge, space, blocks, policy),
                       SpinOrbitStagePolicy(basis=policy.basis, molcas_memory_mb=policy.molcas_memory_mb),
                       name="soc_ms")
    sf, so = parse_rassi(text)
    n_sf = sum(n for _, _, n in blocks)
    if len(sf) != n_sf or not so:
        return {**rec, "status": "SOC_CALCULATION_FAILED", "delta_so_ev": None, "n_spin_free": len(sf)}
    ref_block = blocks.index((mult, ref_sym, policy.roots_per_block))
    offset = sum(n for _, _, n in blocks[:ref_block])
    block_e = sf[offset:offset + policy.roots_per_block]
    roots = reference_roots(block_e, lam, policy.degeneracy_tolerance_hartree)
    if roots is None:
        return {**rec, "status": "SOC_REFERENCE_NOT_IDENTIFIED", "delta_so_ev": None, "spin_free_hartree": sf}
    ref_states = [offset + i + 1 for i in roots]  # RASSI numbering, 1-based
    e_ref = sum(sf[s - 1] for s in ref_states) / len(ref_states)
    lowest_sf = min(sf)
    weights = parse_so_weights(text)
    w_ref = sum(w for s, _, w in weights.get(1, []) if s in ref_states)
    shift = (min(so) - e_ref) * HARTREE_TO_EV
    rec.update(spin_free_hartree=sf, spin_orbit_hartree=so[:12], reference_states=ref_states,
               reference_weight=w_ref, lowest_spin_free_below_reference_ev=(e_ref - lowest_sf) * HARTREE_TO_EV,
               lowest_so_composition=weights.get(1))
    if (e_ref - lowest_sf) * HARTREE_TO_EV > 1e-3:
        return {**rec, "status": "SOC_REFERENCE_NOT_LOWEST_SPIN_FREE", "delta_so_ev": None}
    if w_ref < policy.min_reference_weight:
        return {**rec, "status": "SOC_LOWEST_LEVEL_HAS_OTHER_PARENT", "delta_so_ev": None}
    asym = 0.5 * (max(sf[s - 1] for s in ref_states) - min(sf[s - 1] for s in ref_states)) * HARTREE_TO_EV
    bound = policy.relative_bound * abs(shift) + (1.0 - w_ref) * policy.mixing_gap_ev + asym
    return {**rec, "status": "CLEARED", "delta_so_ev": shift, "bound_ev": bound, "component_asymmetry_ev": asym}
