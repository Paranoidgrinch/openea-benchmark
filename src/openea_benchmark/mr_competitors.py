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

Lambda resolution (Patch 53): a C2v block mixes Lambda values (B1 holds Pi and
Phi, A2 holds Sigma- and Delta), so its lowest root can be a state Stage 3
already computes (C2 a3Piu below the 3Phi candidates).  Each root's <Lz^2> over
the active space assigns Lambda = round(sqrt(<Lz^2>)); a candidate's bound is
the lowest root of its Lambda (or of unassigned Lambda).  If none of the
``classify_roots`` lowest CASCI roots carries it, the candidate lies above all
of them and the highest computed root is its bound.  For like atoms the g/u
parity of each root (product of the D2h orbital parities over the CI vector)
is matched as well (C2 3Pig/3Piu share the C2v B1 block).

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

# Version of the per-root Lambda / g-u classification; older block caches are recomputed.
ROOT_CLASSIFICATION = 3  # 3: high-spin ROHF start orbitals tried as well
LAMBDA_OF = {"Sigma": 0, "Pi": 1, "Delta": 2, "Phi": 3, "Gamma": 4}
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
    # CASCI roots classified by Lambda (0 = no classification, old behaviour).
    classify_roots: int = 6
    lambda_tolerance: float = 0.25
    max_memory_mb: int = 16000


def candidate_lambda(label: str) -> int | None:
    import re
    m = re.search(r"(Sigma|Pi|Delta|Phi|Gamma)", label)
    return None if m is None else LAMBDA_OF[m.group(1)]


def candidate_parity(label: str) -> int | None:
    """+1 for g, -1 for u, None for unlike atoms."""
    import re
    m = re.search(r"(Sigma|Pi|Delta|Phi|Gamma)([gu])", label)
    return None if m is None else (1 if m.group(2) == "g" else -1)


def active_orbital_parities(atoms, r, basis, mo_active) -> list[int] | None:
    """Inversion parity of each active orbital from its D2h label; None if not pure."""
    from pyscf import gto, symm
    if atoms[0] != atoms[1]:
        return None
    mol = gto.M(atom=[(atoms[0], (0, 0, 0)), (atoms[1], (0, 0, float(r)))], basis=basis, unit="Angstrom",
                symmetry="D2h", verbose=0)
    try:
        labels = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_active)
    except ValueError:
        return None
    return [1 if str(x).endswith("g") else -1 for x in labels]


def root_parity(civec, ncas, nelecas, orbital_parity, tolerance: float = 0.1) -> int | None:
    """<inversion> of a CI root (closed inactive shells are gerade)."""
    from pyscf.fci import cistring
    na, nb = nelecas if isinstance(nelecas, (tuple, list)) else ((nelecas + 1) // 2, nelecas // 2)
    par = np.asarray(orbital_parity)

    def string_parity(n):
        strings = cistring.make_strings(range(ncas), n)
        return np.array([np.prod(par[[i for i in range(ncas) if s >> i & 1]]) if n else 1.0 for s in strings])
    c = np.asarray(civec).reshape(len(string_parity(na)), len(string_parity(nb)))
    val = float(np.einsum("ab,a,b->", c * c, string_parity(na), string_parity(nb)) / np.sum(c * c))
    return 1 if val > 1 - tolerance else (-1 if val < -1 + tolerance else None)


def active_lz2(mol, mo_coeff, ncore, ncas, fcisolver, civec, nelecas) -> float:
    """<Lz^2> of one CI root over the active orbitals (molecule on the z axis).

    The closed inactive shells carry no net Lz; core-active couplings of Lz are
    neglected (they vanish for orbitals of pure Lambda)."""
    with mol.with_common_orig((0, 0, 0)):
        d = mol.intor("int1e_cg_irxp", comp=3)[2]  # matrix of the real antisymmetric operator x d/dy - y d/dx
    c = mo_coeff[:, ncore:ncore + ncas]
    d = c.T @ d @ c
    dm1, dm2 = fcisolver.make_rdm12(civec, ncas, nelecas)
    val = float(np.einsum("pq,rs,pqrs", d, d, dm2) + np.einsum("pq,qs,ps", d, d, dm1))
    return abs(val)  # Lz = -i D: <Lz^2> = -<D^2>; sign convention of the integral not relied on


def root_lambda(lz2: float, tolerance: float, irrep: str | None = None) -> int | None:
    """Lambda of a root, None if <Lz^2> is not near a square allowed in its C2v block.

    A1/A2 hold even Lambda, B1/B2 odd: a Pi/Phi mixture with <Lz^2> near 4 in B1
    is not a Delta state and must stay unassigned (it may be either)."""
    x = float(np.sqrt(max(lz2, 0.0)))
    lam = int(round(x))
    if abs(x - lam) > tolerance:
        return None
    if irrep is not None and lam % 2 != (0 if irrep in ("A1", "A2") else 1):
        return None
    return lam


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
        cached = json.loads(cache.read_text())
        if (not policy.classify_roots or cached.get("status") != "COMPLETED"
                or cached.get("root_classification") == ROOT_CLASSIFICATION):
            return cached
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
        ras2 = {k: v for k, v in zip(C2V_IRREPS, space["ras2"]) if v}
        inactive = {k: v for k, v in zip(C2V_IRREPS, space["inactive"]) if v}

        def optimize(mo):
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
                    return mc, n_avg
            return None, None

        def casci(orbitals):
            for n_ci in sorted({max(policy.nroots, policy.classify_roots), policy.nroots}, reverse=True):
                ci = mcscf.CASCI(mf, ncas, nelecas)
                ci.fcisolver.wfnsym = irrep
                ci.fcisolver.nroots = n_ci
                ci.fix_spin_(ss=0.5 * spin_2s * (0.5 * spin_2s + 1))
                try:
                    return ci, np.atleast_1d(ci.kernel(orbitals)[0])
                except (ValueError, IndexError):  # fewer states in the block than requested roots
                    continue
            raise RuntimeError("CASCI found no root in the block")

        # Start orbitals (ORBITAL_GUESS 2): the block's own SCF, and the ROHF two
        # spins higher.  A closed-shell or low-spin SCF can split degenerate pi
        # orbitals (NH 1Delta: 0.29 eV too high); the high-spin ROHF occupies
        # them equally.  The CASSCF solution with the lowest variational CASCI
        # ground root is kept, so the lower-bound property only improves.
        guesses = [("block_scf", mf.mo_coeff)]
        if spin_2s + 2 <= min(nelecas, 2 * ncas - nelecas):  # room for two more unpaired active electrons
            try:
                hs = gto.M(atom=mol.atom, basis=policy.basis, charge=charge, spin=spin_2s + 2, unit="Angstrom",
                           symmetry="C2v", verbose=0, max_memory=policy.max_memory_mb)
                mf_hs = scf.ROHF(hs).run()
                if mf_hs.converged:
                    guesses.append(("high_spin_rohf", mf_hs.mo_coeff))
            except (RuntimeError, ValueError):
                pass
        best, tried = None, []
        for name, coeff in guesses:
            try:
                mo = mcscf.sort_mo_by_irrep(mcscf.CASSCF(mf, ncas, nelecas), coeff, ras2, inactive)
                mc, n_avg = optimize(mo)
                orbitals = mo if mc is None else mc.mo_coeff
                ci, e_casci = casci(orbitals)
            except Exception:  # noqa: BLE001 - this start fails; the other may not
                if name == guesses[-1][0] and best is None:
                    raise
                continue
            v0 = np.array(ci.ci[0] if len(e_casci) > 1 else ci.ci)
            lz2_0 = active_lz2(mol, np.array(ci.mo_coeff), ci.ncore, ncas, ci.fcisolver, v0, ci.nelecas)
            pure = root_lambda(lz2_0, 0.005, irrep) is not None
            tried.append({"guess": name, "casscf_converged": mc is not None, "e_casci_hartree": float(e_casci[0]),
                          "lz2_lowest_root": lz2_0, "pure_lambda": pure})
            # Converged CASSCF beats CASCI on SCF orbitals; then orbitals of pure Lambda
            # (eigenstates of a linear molecule; split pi orbitals are an artefact);
            # then the lower ground root.
            rank = (mc is None, not pure, float(e_casci[0]))
            if best is None or rank < best[0]:
                best = (rank, name, mc, n_avg, ci, e_casci)
        _, guess_name, mc, n_avg, ci, e_casci = best
        record["orbital_guess"] = guess_name
        record["orbital_guesses_tried"] = tried
        record["averaged_roots"] = n_avg
        record["casci_only"] = mc is None
        # Classify every root before NEVPT2: it rotates the active orbitals and CI
        # vectors of the CASCI object to a semicanonical, reordered basis.
        civecs = [np.array(v) for v in (ci.ci if len(e_casci) > 1 else [ci.ci])]
        mo_ci = np.array(ci.mo_coeff)
        orb_par = active_orbital_parities(atoms, r, policy.basis, mo_ci[:, ci.ncore:ci.ncore + ncas])
        roots = []
        for i in range(len(e_casci)):
            lz2 = active_lz2(mol, mo_ci, ci.ncore, ncas, ci.fcisolver, civecs[i], ci.nelecas)
            roots.append({"casci_index": i, "lz2": lz2, "lambda": root_lambda(lz2, policy.lambda_tolerance, irrep),
                          "parity": None if orb_par is None else root_parity(civecs[i], ncas, ci.nelecas, orb_par)})
        for i, root in enumerate(roots):
            root["e_hartree"] = float(e_casci[i]) + float(mrpt.NEVPT(ci, root=i).kernel())
        roots.sort(key=lambda x: x["e_hartree"])
        record.update(status="COMPLETED", energies_hartree=[x["e_hartree"] for x in roots], roots=roots,
                      root_classification=ROOT_CLASSIFICATION)
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

    def lambda_bound(key, lam, parity, skip_selected):
        """Lowest energy over the R window of a root that can be a Lambda=lam (g/u) state."""
        vals = []
        for r in per_block[key]:
            if r.get("status") != "COMPLETED" or "roots" not in r:
                return None
            roots = list(r["roots"])
            if skip_selected and roots:
                roots = roots[1:]  # the selected state itself
            match = [x["e_hartree"] for x in roots if (x["lambda"] is None or x["lambda"] == lam)
                     and (parity is None or x.get("parity") is None or x["parity"] == parity)]
            if match:
                vals.append(min(match))
            elif roots:  # above every computed root of this block
                vals.append(max(x["e_hartree"] for x in roots))
            else:
                vals.append(float("inf"))
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
            kinds = {(candidate_lambda(label), candidate_parity(label)) for label in labels}
            lambda_resolved = False
            if policy.classify_roots and all(lam is not None for lam, _ in kinds):
                bounds = [lambda_bound(key, lam, par, same) for lam, par in kinds]
                if all(b is not None for b in bounds):
                    e_bound, lambda_resolved = min(bounds), True
            casci_only = any(r.get("casci_only") for r in per_block[key])
            threshold = policy.casci_only_exclusion_margin_ev if casci_only else policy.exclusion_margin_ev
            if e_sel is None or e_bound is None:
                status, margin = "UNRESOLVED_NOT_COMPUTED", None
            else:
                margin = (e_bound - e_sel) * HARTREE_TO_EV
                status = "EXCLUDED" if margin > threshold else "UNRESOLVED_MR_COMPETITOR"
            verdicts.append({"charge": q, "mult": key[1], "irrep_c2v": key[2], "candidates": labels,
                             "lower_bound_margin_ev": margin, "status": status, "casci_only": casci_only,
                             "bound_root": "second" if same else "lowest", "lambda_resolved": lambda_resolved})
            if status != "EXCLUDED":
                open_blocks.append(verdicts[-1])
    record.update(selected={str(q): s for q, s in selected.items()}, blocks=verdicts,
                  unclassified_labels=unknown)
    if unknown:
        return done(status="UNRESOLVED_UNCLASSIFIED_LABELS")
    return done(status="CLEARED" if not open_blocks else "UNRESOLVED_MR_COMPETITOR")
