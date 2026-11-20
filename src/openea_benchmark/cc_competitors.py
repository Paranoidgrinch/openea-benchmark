"""Coupled-cluster route for MR-only competitor states.

Stage 3 (ROHF -> CCSD(T)) cannot represent states that need several determinants
(open-shell singlets such as 1Pi or 1Delta, high-Lambda states such as 5Phi or 6Gamma
in real orbitals).  The MR layer bounds them with full-valence SA-CASSCF/SC-NEVPT2,
which fails for transition-metal oxides: in verify_v1 CuO- intruder roots (NEVPT2
correction off by -0.44 Eh) became the "lowest 1Pi", the curves jumped by 3-7 eV
between neighbouring points, and the term gap came out at -3 eV.  EOM-CCSD from the
1Sigma+ ground state puts that 1Pi at +0.91 eV (91 % single excitation).

This layer computes, for every (charge, 2S+1, C2v irrep) block holding an MR-only
candidate, the lowest states of that block at each R of the MR layer's window
(``mr_competitors.run_mr_competitor_stage`` does the window, the edge extensions and
the exclusion test; only the block energies differ):

* reference determinants are symmetric ROHF determinants of the run's own Stage-3
  candidates (lowest candidate of that charge and spin; the selected state for its
  own block), else the aufbau determinant of that spin;
* the reference's CCSD energy is a root when its irrep is the block's;
* EOM-EE-CCSD roots of the block's irrep from the Ms = S reference (spin-adapted
  singlets from a closed-shell reference), and for S <= 1/2 also spin-flip EOM-CCSD
  (Ms = S + 1 -> S): open-shell singlets/doublets such as 1Delta.  Spin-keeping roots
  of an open-shell reference and spin-flip roots include states of higher spin of
  the same irrep; they only lower the block's bound (never hide a competitor);
* all energies at the CCSD level (selected state too), so the margin compares one
  method; a root with a single-excitation weight below ``min_singles_weight`` is
  doubly excited, where EOM-CCSD lies too high: it is lowered by
  ``low_singles_allowance_ev``.

Margin: block energies in validation (scripts/openea_cc_competitor_validation.py: NH, SO,
S2 singlet blocks by spin flip, 1Delta lowest) lie 0.06-0.09 eV above the measured T0;
the 1Sigma+ roots 0.11-0.13 eV (2026-11-20).  The exclusion margin is 0.3 eV.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .electronic_ea_stage import HARTREE_TO_EV, _save_json
from .mr_competitors import MRCompetitorPolicy, run_mr_competitor_stage, selected_block

C2V_IRREPS = ("A1", "A2", "B1", "B2")


@dataclass(frozen=True)
class CCCompetitorPolicy:
    """Explicit pilot policy (validation in the module docstring)."""
    basis: str = "def2-TZVPPD"
    nroots: int = 4
    exclusion_margin_ev: float = 0.3
    min_singles_weight: float = 0.8
    low_singles_allowance_ev: float = 0.7
    max_memory_mb: int = 16000
    scf_conv_tol: float = 1.0e-9
    max_spin_flip_s: float = 0.5
    # A spin-flip root this close to the reference (excitation ~0) is the reference's own
    # Ms = S component (NH: 1e-6 eV), a state of the higher-spin block.
    spin_partner_ev: float = 0.05

    def window_policy(self) -> MRCompetitorPolicy:
        """Window, edge extensions and exclusion test of the MR layer, with this margin."""
        return MRCompetitorPolicy(basis=self.basis, exclusion_margin_ev=self.exclusion_margin_ev,
                                  casci_only_exclusion_margin_ev=self.exclusion_margin_ev,
                                  max_memory_mb=self.max_memory_mb)


def _molecule(atoms, r, charge, two_s, basis, max_memory_mb):
    from pyscf import gto
    from .heavy_elements import mole_basis_kwargs
    return gto.M(atom=[(atoms[0], (0.0, 0.0, 0.0)), (atoms[1], (0.0, 0.0, float(r)))],
                 **mole_basis_kwargs(atoms, basis), charge=charge, spin=two_s, symmetry="C2v",
                 verbose=0, max_memory=max_memory_mb, dump_input=False)


def _irrep_id(mol, name: str) -> int:
    return int(mol.irrep_id[list(mol.irrep_name).index(name)])


def _irrep_name(mol, ident: int) -> str:
    return str(mol.irrep_name[list(mol.irrep_id).index(int(ident))])


def reference_scf(atoms, r, charge, two_s, irrep_nelec, policy: CCCompetitorPolicy):
    """Symmetric ROHF (RHF for a closed shell) in the given irrep occupation, SOSCF rescue."""
    from pyscf import scf
    from .heavy_elements import apply_hamiltonian
    mol = _molecule(atoms, r, charge, two_s, policy.basis, policy.max_memory_mb)
    mf = apply_hamiltonian(scf.RHF(mol) if two_s == 0 else scf.ROHF(mol))
    if irrep_nelec:
        mf.irrep_nelec = ({k: int(v[0]) + int(v[1]) for k, v in irrep_nelec.items()} if two_s == 0
                          else {k: (int(v[0]), int(v[1])) for k, v in irrep_nelec.items()})
    mf.conv_tol, mf.max_cycle = policy.scf_conv_tol, 200
    mf.kernel()
    if not mf.converged:
        soscf = mf.newton()
        soscf.conv_tol, soscf.max_cycle = policy.scf_conv_tol, 200
        soscf.kernel(mf.mo_coeff, mf.mo_occ)
        if soscf.converged:
            mf.mo_coeff, mf.mo_occ, mf.mo_energy, mf.e_tot, mf.converged = (
                soscf.mo_coeff, soscf.mo_occ, soscf.mo_energy, soscf.e_tot, True)
    if not mf.converged:
        raise RuntimeError(f"reference SCF not converged (2S={two_s}, {irrep_nelec})")
    return mf


def _occupied_first(mf):
    """UHF copy of ``mf`` with occupied orbitals first in each spin (energy order kept)."""
    from pyscf import scf
    umf = scf.addons.convert_to_uhf(mf)
    coeff, occ, energy = [], [], []
    for s in range(2):
        order = np.argsort(-np.asarray(umf.mo_occ[s]), kind="stable")
        coeff.append(umf.mo_coeff[s][:, order])
        occ.append(np.asarray(umf.mo_occ[s])[order])
        energy.append(np.asarray(umf.mo_energy[s])[order])
    umf.mo_coeff, umf.mo_occ, umf.mo_energy = np.array(coeff), np.array(occ), np.array(energy)
    return umf


def _orbsym(mol, coeff) -> np.ndarray:
    from pyscf import symm
    return np.asarray(symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, coeff), dtype=int)


def _singles_weight(r1_parts, vector) -> float:
    norm = float(np.dot(vector, vector))
    return 0.0 if norm == 0.0 else float(sum(np.sum(np.asarray(p) ** 2) for p in r1_parts)) / norm


def _guess_vectors(size, entries, nroots):
    """Unit vectors at the given (address, orbital-energy gap) entries, lowest gaps first."""
    out = []
    for addr, _ in sorted(entries, key=lambda x: x[1])[:nroots]:
        g = np.zeros(size)
        g[addr] = 1.0
        out.append(g)
    return out


def block_roots(atoms, charge, mult, irrep, r, policy: CCCompetitorPolicy,
                keep_reference: Mapping[str, Any] | None,
                flip_reference: Mapping[str, Any] | None) -> list[dict]:
    """CCSD / EOM-CCSD states of one (charge, 2S+1, irrep) block at R (energies in Eh).

    ``keep_reference`` / ``flip_reference``: {"two_s", "irrep_nelec"} of the Ms = S and
    Ms = S + 1 reference determinants (irrep_nelec None: aufbau)."""
    from pyscf import cc
    from pyscf.cc import eom_rccsd, eom_uccsd
    from .adaptive.stage3_execution import valence_frozen_core_orbitals
    two_s = mult - 1
    roots: list[dict] = []
    for kind, ref in (("keep", keep_reference), ("flip", flip_reference)):
        if ref is None:
            continue
        mf = reference_scf(atoms, r, charge, ref["two_s"], ref.get("irrep_nelec"), policy)
        mol = mf.mol
        nfz = valence_frozen_core_orbitals(mol)
        target = _irrep_id(mol, irrep)
        if kind == "keep" and ref["two_s"] == 0:
            from .post_ccsd_t_stage import occupation_ordered
            mf = occupation_ordered(mf)
            mycc = cc.RCCSD(mf)
            mycc.frozen = nfz
            mycc.kernel()
            if not mycc.converged:
                raise RuntimeError("closed-shell CCSD not converged")
            sym = _orbsym(mol, mf.mo_coeff)[nfz:]
            nocc = mycc.nocc
            nvir = sym.size - nocc
            if target == 0:  # totally symmetric: the reference itself is a state of this block
                roots.append({"e_hartree": float(mycc.e_tot), "method": "CCSD", "singles_weight": 1.0})
            eom = eom_rccsd.EOMEESinglet(mycc)
            gaps = [(i * nvir + a, mf.mo_energy[nfz + nocc + a] - mf.mo_energy[nfz + i])
                    for i in range(nocc) for a in range(nvir) if sym[i] ^ sym[nocc + a] == target]
            if not gaps:
                continue
            e, v = eom.kernel(nroots=min(policy.nroots, len(gaps)),
                              guess=_guess_vectors(eom.vector_size(), gaps, policy.nroots))
            for ei, vi in zip(np.atleast_1d(e), v if isinstance(v, list) else list(np.atleast_2d(v))):
                r1, _ = eom.vector_to_amplitudes(vi)
                i, a = np.unravel_index(np.argmax(abs(r1)), r1.shape)
                if sym[i] ^ sym[nocc + a] != target:
                    continue  # drifted out of the block
                roots.append({"e_hartree": float(mycc.e_tot + ei), "method": "EOM-EE-CCSD singlet",
                              "singles_weight": _singles_weight([r1], vi)})
            continue
        umf = _occupied_first(mf)
        mycc = cc.UCCSD(umf)
        mycc.frozen = nfz
        mycc.kernel()
        if not mycc.converged:
            raise RuntimeError(f"UCCSD not converged (2S={ref['two_s']})")
        syma, symb = _orbsym(mol, umf.mo_coeff[0])[nfz:], _orbsym(mol, umf.mo_coeff[1])[nfz:]
        ea, eb = umf.mo_energy[0][nfz:], umf.mo_energy[1][nfz:]
        nocca, noccb = mycc.nocc
        nvira, nvirb = syma.size - nocca, symb.size - noccb
        ref_irrep = 0
        for x in list(syma[:nocca]) + list(symb[:noccb]):
            ref_irrep ^= int(x)
        if kind == "keep":
            if ref_irrep == target:
                roots.append({"e_hartree": float(mycc.e_tot), "method": "CCSD", "singles_weight": 1.0})
            eom = eom_uccsd.EOMEESpinKeep(mycc)
            size = eom.vector_size()
            sizea = nocca * nvira + nocca * (nocca - 1) // 2 * nvira * (nvira - 1) // 2
            gaps = [(i * nvira + a, ea[nocca + a] - ea[i]) for i in range(nocca) for a in range(nvira)
                    if ref_irrep ^ syma[i] ^ syma[nocca + a] == target]
            gaps += [(sizea + i * nvirb + a, eb[noccb + a] - eb[i]) for i in range(noccb) for a in range(nvirb)
                     if ref_irrep ^ symb[i] ^ symb[noccb + a] == target]
            method = "EOM-EE-CCSD spin-keeping"
        else:
            eom = eom_uccsd.EOMEESpinFlip(mycc)
            size = eom.vector_size()
            # r1ab block first: alpha occupied -> beta virtual (Ms - 1)
            gaps = [(i * nvirb + a, eb[noccb + a] - ea[i]) for i in range(nocca) for a in range(nvirb)
                    if ref_irrep ^ syma[i] ^ symb[noccb + a] == target]
            method = "EOM-SF-CCSD"
        if not gaps:
            continue
        e, v = eom.kernel(nroots=min(policy.nroots, len(gaps)), guess=_guess_vectors(size, gaps, policy.nroots))
        for ei, vi in zip(np.atleast_1d(e), v if isinstance(v, list) else list(np.atleast_2d(v))):
            if kind == "flip" and ref_irrep == target and abs(ei) * HARTREE_TO_EV < policy.spin_partner_ev:
                continue  # the Ms = S component of the Ms = S + 1 reference itself: another block
            r1, _ = eom.vector_to_amplitudes(vi)
            roots.append({"e_hartree": float(mycc.e_tot + ei), "method": method,
                          "singles_weight": _singles_weight(list(r1), vi)})
    for root in roots:
        root["lambda"], root["parity"] = None, None
        if root["singles_weight"] < policy.min_singles_weight:
            root["e_raw_hartree"] = root["e_hartree"]
            root["e_hartree"] -= policy.low_singles_allowance_ev / HARTREE_TO_EV
            root["low_singles_weight"] = True
    _ = two_s
    return sorted(roots, key=lambda x: x["e_hartree"])


def cc_block_task(references: Mapping[str, Any], cc_policy: CCCompetitorPolicy, atoms, charge, mult, irrep, r,
                  policy, cache) -> dict:
    """Block record in the MR layer's format (``energies_hartree``, ``roots``), cached per R."""
    import json
    cache = Path(cache)
    if cache.is_file():
        cached = json.loads(cache.read_text())
        if cached.get("policy") == asdict(cc_policy) and cached.get("status") in ("COMPLETED", "FAILED"):
            return cached
    record: dict[str, Any] = {"charge": charge, "mult": mult, "irrep": irrep, "r": float(r),
                              "policy": asdict(cc_policy), "method": "CCSD/EOM-CCSD"}
    key = f"{charge}:{mult}:{irrep}"
    selected = references.get("selected", {}).get(key)
    two_s = mult - 1
    try:
        if selected is not None:
            # The selected state's own block: its determinant, spin-keeping EOM only (a
            # spin-flip partner of the selected state would pose as a second root).
            keep, flip = {"two_s": two_s, "irrep_nelec": selected}, None
        else:
            keep = {"two_s": two_s, "irrep_nelec": references.get("lowest", {}).get(f"{charge}:{two_s}")}
            flip = (None if two_s / 2.0 > cc_policy.max_spin_flip_s else
                    {"two_s": two_s + 2, "irrep_nelec": references.get("lowest", {}).get(f"{charge}:{two_s + 2}")})
        roots = block_roots(tuple(atoms), charge, mult, irrep, r, cc_policy, keep, flip)
        if not roots:
            raise RuntimeError("no CC state in this block")
        if selected is not None:
            # Root 0 is the selected state's own CCSD energy; an EOM root below it is a
            # lower state of the same block and must show up as a competitor (second
            # root), never replace the selected energy.
            own = [x for x in roots if x["method"] == "CCSD"]
            if not own:
                raise RuntimeError("selected state's CCSD reference missing in its block")
            roots = own[:1] + [x for x in roots if x is not own[0]]
        record.update(status="COMPLETED", roots=roots, energies_hartree=[x["e_hartree"] for x in roots],
                      casci_only=False)
    except Exception as exc:  # noqa: BLE001 - recorded, the block stays unresolved
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    _save_json(cache, record)
    return record


def stage_references(electronic_ea: Mapping[str, Any], requests: Sequence[Any]) -> dict:
    """Irrep occupations of the selected states (per block) and of the lowest Stage-3
    candidate per (charge, 2S)."""
    template = {r.job_id: r for r in requests}
    out: dict[str, dict] = {"selected": {}, "lowest": {}}
    for role in ("neutral", "anion"):
        req = template.get(electronic_ea.get(f"selected_{role}_job"))
        if req is None or req.irrep_nelec is None:
            continue
        block = selected_block(req.symmetry_label or "")
        if block is not None:
            out["selected"][f"{req.charge}:{block[0]}:{block[1]}"] = {k: list(v) for k, v in req.irrep_nelec.items()}
    best: dict[str, tuple[float, dict]] = {}
    for role in ("neutral", "anion"):
        for cand in (electronic_ea.get("candidates") or {}).get(role, ()):
            req = template.get(cand.get("job_id"))
            energy = cand.get("energy_central_hartree")
            if req is None or req.irrep_nelec is None or energy is None:
                continue
            k = f"{req.charge}:{req.spin_2s}"
            if k not in best or energy < best[k][0]:
                best[k] = (float(energy), {x: list(v) for x, v in req.irrep_nelec.items()})
    out["lowest"] = {k: v for k, (_, v) in best.items()}
    return out


def run_cc_competitor_stage(*, electronic_ea: Mapping[str, Any], requests: Sequence[Any],
                            mr_only_labels: Sequence[str], output_dir: Path,
                            policy: CCCompetitorPolicy = CCCompetitorPolicy(),
                            candidate_r_angstrom: Mapping[str, float] | None = None, block_fn=None) -> dict:
    """The MR layer's window and exclusion test on CCSD / EOM-CCSD block energies."""
    output_dir = Path(output_dir)
    references = stage_references(electronic_ea, requests) if electronic_ea.get("candidates") else {}
    fn = block_fn or partial(cc_block_task, references, policy)
    record = run_mr_competitor_stage(electronic_ea=electronic_ea, requests=requests, mr_only_labels=mr_only_labels,
                                     output_dir=output_dir, policy=policy.window_policy(), block_fn=fn,
                                     candidate_r_angstrom=candidate_r_angstrom)
    record = dict(record, stage="CC_COMPETITORS", cc_policy=asdict(policy), references=references)
    _save_json(output_dir / "cc_competitors.json", record)
    return record


def merge_competitor_records(cc: Mapping[str, Any] | None, mr: Mapping[str, Any] | None) -> dict | None:
    """One competitor record from the CC and the MR layer: a block counts as excluded when
    either layer excludes it (each test is one-sided: it can only fail to exclude)."""
    if cc is None or mr is None:
        return None if cc is None and mr is None else dict(cc or mr)
    if "blocks" not in mr or "blocks" not in cc:  # a layer that did not get to the blocks
        return dict(mr if "blocks" in mr or "blocks" not in cc else cc)
    key = lambda b: (b["charge"], b["mult"], b["irrep_c2v"])  # noqa: E731
    by_cc = {key(b): b for b in cc.get("blocks", ())}
    blocks = []
    for b in mr.get("blocks", ()):
        c = by_cc.get(key(b))
        if c is not None and c["status"] == "EXCLUDED" and b["status"] != "EXCLUDED":
            blocks.append({**c, "source": "CC", "mr_status": b["status"]})
        else:
            blocks.append({**b, "source": "MR", "cc_status": None if c is None else c["status"]})
    seen = {key(b) for b in blocks}
    blocks += [{**b, "source": "CC"} for k, b in by_cc.items() if k not in seen]
    unknown = sorted(set(mr.get("unclassified_labels") or ()))  # same label parser in both layers
    status = ("UNRESOLVED_UNCLASSIFIED_LABELS" if unknown else
              "CLEARED" if all(b["status"] == "EXCLUDED" for b in blocks) else "UNRESOLVED_MR_COMPETITOR")
    return {**dict(mr), "stage": "MR_ONLY_COMPETITORS", "status": status, "blocks": blocks,
            "unclassified_labels": unknown, "cc_status": cc.get("status"), "mr_status": mr.get("status")}
