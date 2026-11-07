#!/usr/bin/env python3
"""Build src/openea_benchmark/data/nist_f_shell_levels.json from NIST ASD level tables (Route B).

Input: tab-separated ASD energy-level files ``{El}_{I,II}.tsv`` (downloads/fetch_fshell_data.sh).
Only MEASURED levels are kept (ASD level help, levelshelp.html, version 5.12):
  [..]  semi-empirical (interpolation/extrapolation)  -> rejected
  (..)  ab-initio calculation / not evaluated experimental data -> rejected
  ?     level may not be real                         -> rejected
  +x    unknown offset to the main level system       -> rejected
  a     autoionization broadened; o undocumented      -> rejected
A configuration label with "?" (questionable assignment) is rejected as well, as is an f shell
written without occupation whose parent term does not identify it (e.g. "4f.(8S*)").
An f shell missing from the label is empty, except for Yb, No, Lr (closed f14 shell omitted).

Energies are put on one scale relative to the neutral ground level: Ln II levels are shifted
by the neutral ionization energy, used only when that limit itself is measured.
For every element, f count and charge the lowest measured level of each valence configuration
is stored with configuration, term, J, uncertainty and ASD reference code.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

LANTHANIDES = "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb".split()
ACTINIDES = "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split()
CLOSED_F_OMITTED = {"Yb", "No", "Lr"}
CM1_TO_EV = 1.0 / 8065.543937


def f_count(conf: str, shell: str, element: str) -> int | None:
    if "?" in conf:
        return None
    m = re.search(shell + r"(\d*)", conf)
    if not m:
        return 14 if element in CLOSED_F_OMITTED else 0
    if m.group(1):
        return int(m.group(1))
    parent = re.match(r"\.?\s*\(([^)]*)\)", conf[m.end():])
    if parent and parent.group(1).strip() != "2F*":
        return None
    return 1


def valence_configuration(conf: str, shell: str) -> str:
    """Configuration without the f shell and without parent terms, e.g. '5d.6s'."""
    stripped = re.sub(r"\([^)]*\)|<[^>]*>", "", conf)
    parts = [p.strip() for p in stripped.split(".") if p.strip()]
    return ".".join(p for p in parts if not p.startswith(shell)) or "-"


def clean(value: str) -> str:
    return value.strip().strip('"').strip()


def read_spectrum(path: Path, element: str, shell: str):
    levels, limit = [], None
    for row in csv.DictReader(open(path), delimiter="\t"):
        r = {k: clean(v or "") for k, v in row.items() if k}
        flag = r.get("Prefix", "") + r.get("Suffix", "")
        try:
            value = float(r.get("Level (cm-1)", ""))
        except ValueError:
            continue
        if r.get("Term") == "Limit":
            limit = {"value_cm1": value, "measured": not flag, "flag": flag or None,
                     "reference": r.get("Reference") or None}
            continue
        if flag:
            continue
        k = f_count(r.get("Configuration", ""), shell, element)
        if k is None or not r.get("Configuration"):
            continue
        levels.append({"f_count": k, "valence": valence_configuration(r["Configuration"], shell),
                       "level_cm1": value, "configuration": r["Configuration"], "term": r.get("Term", ""),
                       "J": r.get("J", ""), "uncertainty_cm1": r.get("Uncertainty (cm-1)") or None,
                       "reference": r.get("Reference") or None})
    return levels, limit


def main(src: Path, out: Path) -> None:
    data = {"source": "NIST Atomic Spectra Database, energy levels (ASD v5.12); measured levels only",
            "rules": __doc__.split("Only MEASURED")[1].split("Energies are")[0].strip(),
            "elements": {}}
    for element in LANTHANIDES + ACTINIDES:
        shell = "4f" if element in LANTHANIDES else "5f"
        neutral, ie = read_spectrum(src / f"{element}_I.tsv", element, shell)
        cation, _ = read_spectrum(src / f"{element}_II.tsv", element, shell)
        entries = [dict(lv, charge=0, energy_cm1=lv["level_cm1"]) for lv in neutral]
        if ie and ie["measured"]:
            entries += [dict(lv, charge=1, energy_cm1=lv["level_cm1"] + ie["value_cm1"]) for lv in cation]
        lowest: dict[tuple, dict] = {}
        for e in entries:
            key = (e["f_count"], e["charge"], e["valence"])
            if key not in lowest or e["energy_cm1"] < lowest[key]["energy_cm1"]:
                lowest[key] = e
        by_f: dict[str, list] = {}
        for (k, _, _), e in sorted(lowest.items(), key=lambda kv: (kv[0][0], kv[1]["energy_cm1"])):
            by_f.setdefault(str(k), []).append(dict(e, energy_ev=e["energy_cm1"] * CM1_TO_EV))
        data["elements"][element] = {"f_shell": shell, "neutral_ionization_limit": ie,
                                     "levels_by_f_count": by_f}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
