#!/usr/bin/env python3
"""Build src/openea_benchmark/data/f_in_core_pp.json from the Stuttgart/Cologne library (Route B).

    build_f_in_core_pp.py COLOGNE_DIR OUT.json

COLOGNE_DIR holds the Molpro-format pages of downloads/fetch_an_pp.sh, fetch_an_basis.sh and
fetch_ln_pp.sh: ``{an,ln}/{El}_ECP{n}MWB.molpro.html`` and ``{an,ln}_basis/{El}_{set}.molpro.html``.
Only the f-in-core pseudopotentials are taken: ECP(46+n_f)MWB for Ce–Yb, ECP(78+n_f)MWB for
Ac–Lr (n_f = 4f/5f electrons in the core).

Molpro conventions (library "Read me first"): ``ECP,El,ncore,lmax,lmax';`` is followed by
V(lmax) (local) and V(l), l = 0..lmax-1, each as ``k; n,a,A; ...`` for terms
A r^(n-2) exp(-a r^2); this is PySCF's layout with l = -1 for the local part.  The comment line
carries the core-polarization potential (CPP: alpha, delta, ncut).  A basis block
``s,El,a1,...;`` is followed by ``C,i.j,c_i,...,c_j;`` contractions; primitives in no
contraction stay uncontracted.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

L_LETTERS = "spdfghi"
LANTHANIDES = "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb".split()


def page_text(path: Path) -> str:
    text = path.read_text(encoding="latin-1")
    text = re.sub(r"<[^>]*>", "\n", text).replace("&nbsp;", " ")
    return text


def number(s: str) -> float:
    return float(s.strip().replace("D", "E").replace("d", "e"))


def parse_ecp(text: str, element: str) -> dict:
    head = re.search(rf"^ECP,{element},(\d+),(\d+),(\d+);", text, re.M)
    if head is None:
        raise ValueError("no ECP card")
    ncore, lmax = int(head.group(1)), int(head.group(2))
    comment = text[:head.start()].strip().splitlines()[-1]
    body = text[head.end():].split("!")[0]
    tokens = [t.strip() for t in body.replace("\n", " ").split(";") if t.strip()]
    channels, pos = [], 0
    for l in [-1] + list(range(lmax)):
        k = int(tokens[pos]); pos += 1
        by_power: list[list] = [[] for _ in range(7)]
        for _ in range(k):
            n, a, coef = tokens[pos].split(","); pos += 1
            by_power[int(n)].append([number(a), number(coef)])
        channels.append([l, by_power])
    q = re.search(r"Q=\s*([\d.]+)", comment)
    cpp = re.search(r"alpha=\s*([\d.]+);\s*delta=\s*([\d.]+);\s*ncut=\s*(\d+)", comment)
    ref = re.search(r"Ref\s+(\d+)", comment)
    refs = dict(re.findall(r"^!\s*\[(\d+)\]\s*(.+)$", text, re.M))
    return {"core_electrons": ncore, "valence_charge": float(q.group(1)) if q else None,
            "cpp": {"alpha": float(cpp.group(1)), "delta": float(cpp.group(2)), "ncut": int(cpp.group(3))}
            if cpp else None,
            "comment": comment.lstrip("! ").strip(),
            "reference": refs.get(ref.group(1)) if ref else None,
            "pyscf_ecp": [ncore, channels]}


def parse_basis(text: str, element: str) -> list:
    """PySCF shells [[l, [a, c1, c2, ...], ...], ...] of one Molpro basis page."""
    shells = []
    lines = [ln.strip() for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        m = re.match(rf"^([{L_LETTERS}]),{element},(.*);$", lines[i])
        if not m:
            i += 1
            continue
        l = L_LETTERS.index(m.group(1))
        exps = [number(x) for x in m.group(2).split(",") if x.strip()]
        contractions = []
        i += 1
        while i < len(lines) and lines[i].startswith("C,"):
            fields = lines[i].rstrip(";").split(",")
            lo, hi = (int(x) for x in fields[1].split("."))
            coefs = [number(x) for x in fields[2:]]
            if len(coefs) != hi - lo + 1:
                raise ValueError(f"contraction length mismatch in {lines[i]!r}")
            contractions.append((lo - 1, hi - 1, coefs))
            i += 1
        covered = set()
        for lo, hi, coefs in contractions:
            shells.append([l] + [[exps[j], coefs[j - lo]] for j in range(lo, hi + 1)])
            covered.update(range(lo, hi + 1))
        shells += [[l, [exps[j], 1.0]] for j in range(len(exps)) if j not in covered]
    return shells


def main(src: Path, out: Path) -> None:
    data: dict = {"source": "Stuttgart/Cologne energy-consistent pseudopotentials "
                            "(www.tc.uni-koeln.de/PP), f-in-core MWB sets, Molpro format",
                  "elements": {}}
    for family, base in (("an", 78), ("ln", 46)):
        for page in sorted((src / family).glob("*_ECP*MWB.molpro.html")):
            element, name = page.name.split(".")[0].split("_", 1)
            if (element in LANTHANIDES) != (family == "ln"):
                continue
            text = page_text(page)
            ecp = parse_ecp(text, element)
            ecp["f_in_core"] = ecp["core_electrons"] - base
            ecp["basis_sets"] = {}
            for bpage in sorted((src / f"{family}_basis").glob(f"{element}_{name}*.molpro.html")):
                bname = bpage.name.split(".")[0].split("_", 1)[1]
                if bname.endswith("GUESS"):
                    continue  # starting orbitals, not a basis
                shells = parse_basis(page_text(bpage), element)
                if shells:
                    ecp["basis_sets"][bname] = shells
            data["elements"].setdefault(element, {})[name] = ecp
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=0) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
