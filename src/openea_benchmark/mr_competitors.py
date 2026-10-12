"""Multireference-only competitor states (Patch 51).

The symmetry scout lists Wigner-Witmer states that no single high-spin
determinant represents (1Delta, open-shell singlets, Sigma- from pi^2, Phi,
Gamma, ...).  Stage 3 cannot compute them, so until now every run carried them
as an explicit gap.  This layer closes the gap conservatively:

* every such state lies in a (charge, spin, C2v irrep) block: Sigma+ -> A1,
  Sigma- -> A2, Delta/Gamma -> A1 and A2 (the A2 component is used, A1 also
  holds Sigma+), Pi/Phi -> B1;
* per block, a two-root state-averaged full-valence CASSCF with SC-NEVPT2 per
  root (``mr_nuclear_curve`` active-space rule, def2-TZVPPD) on an R window
  around the selected state's equilibrium gives the lowest energy *of any state*
  in that block - a lower bound for the candidates in it;
* the selected state of the same charge is computed at the same level; in its
  own block the second root is the competitor bound;
* a block whose lower bound lies more than ``exclusion_margin_ev`` above the
  selected state excludes all its MR-only candidates (``EXCLUDED``); otherwise
  ``UNRESOLVED_MR_COMPETITOR`` keeps the ground-state assignment open.

Only lower bounds are used, so an exclusion is never optimistic; a block that
cannot be computed stays unresolved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .electronic_ea_stage import HARTREE_TO_EV, _save_json
from .mr_nuclear_curve import C2V_IRREPS, _c2v_counts
from .spin_orbit_stage import D2H_TO_C2V, full_valence_space

LAMBDA_IRREP = {"Sigma+": "A1", "Sigma-": "A2", "Pi": "B1", "Delta": "A2", "Phi": "B1", "Gamma": "A2"}


@dataclass(frozen=True)
class MRCompetitorPolicy:
    """Explicit pilot policy, not a validated production threshold set."""
    basis: str = "def2-TZVPPD"
    r_offsets_angstrom: tuple[float, ...] = (-0.15, -0.075, 0.0, 0.075, 0.15, 0.25)
    exclusion_margin_ev: float = 0.3
    # Blocks where CASSCF diverged and only CASCI on SCF orbitals exists: no
    # lower-bound property, so a wider margin is required.
    casci_only_exclusion_margin_ev: float = 1.0
    nroots: int = 2
    max_memory_mb: int = 16000


def candidate_block(label: str) -> tuple[int, int, str] | None:
    """'q=-1 2Deltag' -> (charge, 2S+1, C2v irrep)."""
    import re
    m = re.fullmatch(r"q=(-?\d+) (\d+)(Sigma|Pi|Delta|Phi|Gamma)[gu]?([+-]?)", label.strip())
    if m is None:
        return None
    lam = m.group(3) + (m.group(4) if m.group(3) == "Sigma" else "")
    irrep = LAMBDA_IRREP.get(lam)
    return None if irrep is None else (int(m.group(1)), int(m.group(2)), irrep)


def selected_block(label: str) -> tuple[int, str] | None:
    import re
    m = re.fullmatch(r"(\d+)(Sigma|Pi|Delta|Phi|Gamma)[gu]?([+-]?)", label)
    if m is None:
        return None
    lam = m.group(2) + (m.group(3) if m.group(2) == "Sigma" else "")
    irrep = LAMBDA_IRREP.get(lam)
    return None if irrep is None else (int(m.group(1)), irrep)


def block_energies(atoms, charge, mult, irrep, r, policy: MRCompetitorPolicy, cache: Path) -> dict:
    """Two-root SA-CASSCF/SC-NEVPT2 energies of one (charge, spin, irrep) block at one R."""
    if cache.is_file():
        return json.loads(cache.read_text())
    from pyscf import gto, mcscf, mrpt, scf
    space = full_valence_space(atoms, charge)
    if space is None:
        return {"status": "MR_ACTIVE_SPACE_NOT_DERIVED"}
    spin_2s = mult - 1
    record: dict[str, Any] = {"charge": charge, "mult": mult, "irrep": irrep, "r": r}
    try:
        mol = gto.M(atom=[(atoms[0], (0, 0, 0)), (atoms[1], (0, 0, float(r)))], basis=policy.basis, charge=charge,
                    spin=spin_2s, unit="Angstrom", symmetry="C2v", verbose=0, max_memory=policy.max_memory_mb)
        mf = (scf.ROHF(mol) if spin_2s else scf.RHF(mol)).run()
        ncas, nelecas = sum(space["ras2"]), space["active_electrons"]
        mo = mcscf.sort_mo_by_irrep(mcscf.CASSCF(mf, ncas, nelecas), mf.mo_coeff,
                                    {k: v for k, v in zip(C2V_IRREPS, space["ras2"]) if v},
                                    {k: v for k, v in zip(C2V_IRREPS, space["inactive"]) if v})
        for n_avg in (policy.nroots, policy.nroots + 1, 1):
            # A stalled two-root average is retried with three roots (a near-degenerate
            # third state often causes the stall), then with the lowest root alone.
            # A block may hold fewer states than requested in the active space.
            mc = mcscf.CASSCF(mf, ncas, nelecas)
            mc.fcisolver.wfnsym = irrep
            mc.fix_spin_(ss=0.5 * spin_2s * (0.5 * spin_2s + 1))
            if n_avg > 1:
                mc = mc.state_average_([1.0 / n_avg] * n_avg)
            mc.max_cycle_macro = 200
            try:
                mc.kernel(mo)
            except ValueError:  # fewer states in the block than averaged roots
                continue
            if mc.converged:
                break
        else:
            mc = None  # orbital optimization diverges: CASCI on the SCF orbitals (weaker evidence)
        record["averaged_roots"] = None if mc is None else n_avg
        record["casci_only"] = mc is None
        ci = mcscf.CASCI(mf, ncas, nelecas)
        ci.fcisolver.wfnsym = irrep
        ci.fcisolver.nroots = policy.nroots
        ci.fix_spin_(ss=0.5 * spin_2s * (0.5 * spin_2s + 1))
        e_casci = np.atleast_1d(ci.kernel(mo if mc is None else mc.mo_coeff)[0])
        totals = [float(e_casci[i]) + float(mrpt.NEVPT(ci, root=i).kernel()) for i in range(len(e_casci))]
        record.update(status="COMPLETED", energies_hartree=sorted(totals))
    except Exception as exc:  # noqa: BLE001 - recorded, the block stays unresolved
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    _save_json(cache, record)
    return record


def _block_task(atoms, charge, mult, irrep, r, policy, cache):
    return block_energies(atoms, charge, mult, irrep, r, policy, Path(cache))


def run_mr_competitor_stage(*, electronic_ea: Mapping[str, Any], requests: Sequence[Any],
                            mr_only_labels: Sequence[str], output_dir: Path,
                            policy: MRCompetitorPolicy = MRCompetitorPolicy(), block_fn=None) -> dict:
    from .point_pool import run_parallel
    output_dir = Path(output_dir)
    record: dict[str, Any] = {"stage": "MR_ONLY_COMPETITORS", "policy": asdict(policy),
                              "validation_metadata_used_in_computation": False}

    def done(**fields) -> dict:
        record.update(fields)
        _save_json(output_dir / "mr_competitors.json", record)
        return record

    if electronic_ea.get("status") != "ELECTRONIC_EA_EVALUATED":
        return done(status="NOT_ATTEMPTED", reason="single-basis electronic EA not evaluated")
    template = {r.job_id: r for r in requests}
    selected = {}
    for role in ("neutral", "anion"):
        job = electronic_ea[f"selected_{role}_job"]
        cand = next(c for c in electronic_ea["candidates"][role] if c["job_id"] == job)
        req = template.get(job)
        block = None if req is None else selected_block(req.symmetry_label)
        if block is None:
            return done(status="NOT_ATTEMPTED", reason=f"{role}: selected state block unknown")
        selected[req.charge] = {"role": role, "job": job, "label": req.symmetry_label, "mult": block[0],
                                "irrep": block[1], "r": float(cand["r_central_angstrom"]), "atoms": tuple(req.atoms)}
    blocks: dict[tuple[int, int, str], list[str]] = {}
    unknown = []
    for label in mr_only_labels:
        key = candidate_block(label)
        if key is None or key[0] not in selected:
            unknown.append(label)
            continue
        blocks.setdefault(key, []).append(label)
    keys = set(blocks) | {(q, s["mult"], s["irrep"]) for q, s in selected.items()}
    fn = block_fn or _block_task
    jobs, index = [], []
    for (q, mult, irrep) in sorted(keys):
        sel = selected[q]
        for dr in policy.r_offsets_angstrom:
            r = round(sel["r"] + dr, 6)
            jobs.append((sel["atoms"], q, mult, irrep, r, policy,
                         str(output_dir / "blocks" / f"q{q}_m{mult}_{irrep}_r{r:.4f}.json")))
            index.append((q, mult, irrep))
    for job in jobs:
        print(f"[MR] {electronic_ea.get('system')} q={job[1]} 2S+1={job[2]} {job[3]} R={job[4]:.3f}", flush=True)
    results = run_parallel(fn, jobs)
    per_block: dict[tuple, list] = {}
    for key, res in zip(index, results):
        per_block.setdefault(key, []).append(res)

    def lowest(key, root):
        """Lowest energy of the given root over the R window; None if any point lacks it.

        A block holding only one state in the active space has no second root:
        then nothing else in that block can compete (``inf``)."""
        vals = []
        for r in per_block[key]:
            if r.get("status") != "COMPLETED":
                return None
            energies = r["energies_hartree"]
            vals.append(energies[root] if root < len(energies) else float("inf"))
        return min(vals) if vals else None

    verdicts, open_blocks = [], []
    for q, sel in selected.items():
        e_sel = lowest((q, sel["mult"], sel["irrep"]), 0)
        sel["mr_energy_hartree"] = e_sel
        for key, labels in sorted(blocks.items()):
            if key[0] != q:
                continue
            same = key == (q, sel["mult"], sel["irrep"])
            e_bound = lowest(key, 1 if same else 0)
            casci_only = any(r.get("casci_only") for r in per_block[key])
            threshold = policy.casci_only_exclusion_margin_ev if casci_only else policy.exclusion_margin_ev
            if e_sel is None or e_bound is None:
                status, margin = "UNRESOLVED_NOT_COMPUTED", None
            else:
                margin = (e_bound - e_sel) * HARTREE_TO_EV
                status = "EXCLUDED" if margin > threshold else "UNRESOLVED_MR_COMPETITOR"
            verdicts.append({"charge": q, "mult": key[1], "irrep_c2v": key[2], "candidates": labels,
                             "lower_bound_margin_ev": margin, "status": status, "casci_only": casci_only,
                             "bound_root": "second" if same else "lowest"})
            if status != "EXCLUDED":
                open_blocks.append(verdicts[-1])
    record.update(selected={str(q): s for q, s in selected.items()}, blocks=verdicts,
                  unclassified_labels=unknown)
    if unknown:
        return done(status="UNRESOLVED_UNCLASSIFIED_LABELS")
    return done(status="CLEARED" if not open_blocks else "UNRESOLVED_MR_COMPETITOR")
