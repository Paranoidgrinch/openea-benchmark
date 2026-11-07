#!/usr/bin/env python3
"""Build src/openea_benchmark/data/f_shell_levels.json: measured atomic levels per f count (Route B).

    build_f_shell_levels.py NIST_DIR LAC_DIR OUT.json

Sources (downloads/fetch_fshell_data.sh, downloads/fetch_lac_tables.sh):

1. NIST ASD energy-level tables ``{El}_{I,II}.tsv`` (Ce–Yb, Ac–Lr).  Only measured levels are
   kept (ASD level help, levelshelp.html, version 5.12):
     [..]  semi-empirical (interpolation/extrapolation)  -> rejected
     (..)  ab-initio calculation / not evaluated experimental data -> rejected
     ?     level may not be real                         -> rejected
     +x    unknown offset to the main level system       -> rejected
     a     autoionization broadened; o undocumented      -> rejected
2. Blaise & Wyart, "Selected Constants: Energy Levels and Atomic Spectra of Actinides"
   (Laboratoire Aimé Cotton; archived copy of lac.universite-paris-saclay.fr/Data/Database),
   energy-level tables of the first and second spectra (Tab-energy/*/*-tables/Xx{1,2}{e,o}*.html,
   Ac–Es).  These list levels observed in spectra with their discovery reference; Table I of
   the compilation, which also holds predicted configuration onsets (given with ±), is not used.
   A row is kept only if its energy is a plain number (no ?, typos or shifted columns) and its
   configuration label identifies the f count.  Laser-spectroscopy lists without
   configurations are not used.

Common rules: a configuration label that does not parse into subshells (e.g. "5f 9 6d 7S")
or carries "?" is rejected, as is an f shell written without
occupation whose parent term does not identify it (e.g. "4f.(8S*)").  An f shell missing
from the label is empty, except for Yb, No, Lr (closed f14 shell omitted).

Energies are put on one scale relative to the neutral ground level: cation levels are shifted
by the neutral ionization energy, used only when the NIST limit itself is measured.
For every element, f count and charge the lowest measured level of each valence configuration
is stored with configuration, term, J, uncertainty, source and reference.
"""
from __future__ import annotations

import csv
from html.parser import HTMLParser
import json
import re
import sys
from pathlib import Path

LANTHANIDES = "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb".split()
ACTINIDES = "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split()
LAC_DIRS = {"Ac": "Actinium", "Th": "Thorium", "Pa": "Protactinium", "U": "Uranium", "Np": "Neptunium",
            "Pu": "Plutonium", "Am": "Americium", "Cm": "Curium", "Bk": "Berkelium", "Cf": "Californium",
            "Es": "Einsteinium"}
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
    if parent and parent.group(1).strip().rstrip("*") != "2F":
        return None
    return 1


def valence_configuration(conf: str, shell: str) -> str:
    """Configuration without the f shell and without parent terms, e.g. '5d.6s'."""
    stripped = re.sub(r"\([^)]*\)|<[^>]*>", "", conf)
    parts = [p.strip() for p in stripped.split(".") if p.strip()]
    return ".".join(p for p in parts if not p.startswith(shell)) or "-"


def clean(value: str) -> str:
    return value.strip().strip('"').strip()


def level_entry(element: str, shell: str, conf: str, term: str, j: str, value: float,
                uncertainty: str | None, reference: str | None, source: str) -> dict | None:
    # every subshell token must read like '5f3' or '7s' once parent terms are removed
    tokens = re.sub(r"\([^)]*\)|<[^>]*>|\s", "", conf).split(".")
    if not conf or not all(re.fullmatch(r"\d+[spdfg]\d*", t) for t in tokens if t):
        return None
    k = f_count(conf, shell, element)
    if k is None:
        return None
    return {"f_count": k, "valence": valence_configuration(conf, shell), "level_cm1": value,
            "configuration": conf, "term": term, "J": j, "uncertainty_cm1": uncertainty,
            "reference": reference, "source": source}


def read_nist(path: Path, element: str, shell: str):
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
        e = level_entry(element, shell, r.get("Configuration", ""), r.get("Term", ""), r.get("J", ""), value,
                        r.get("Uncertainty (cm-1)") or None, r.get("Reference") or None, "NIST ASD")
        if e:
            levels.append(e)
    return levels, limit


class _Rows(HTMLParser):
    """Table rows of an HTML page as lists of cell texts."""

    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def read_lac(path: Path, element: str, spectrum: str) -> list[dict]:
    """Levels of one Blaise-Wyart table.  Column layout per data row: the configuration is the
    first cell that looks like one; term, J and energy follow it; the two cells before it are
    the discovery and identification references (when present)."""
    parser = _Rows()
    parser.feed(path.read_text(encoding="latin-1").replace("&nbsp;", " "))
    code = re.compile(rf"^{element}{spectrum}[eo]")
    levels = []
    for row in parser.rows:
        if not row or not code.match(row[0]):
            continue
        c = next((i for i, cell in enumerate(row[1:], 1) if re.match(r"^\d[spdf]", cell)), None)
        if c is None or c + 3 >= len(row):
            continue
        energy = row[c + 3].replace(" ", "")
        if not re.fullmatch(r"\d+(\.\d+)?", energy) or not re.fullmatch(r"\d+(/ ?2)?", row[c + 2]):
            continue
        refs = [r for r in row[1:c] if r]
        e = level_entry(element, "5f", row[c], row[c + 1], row[c + 2].replace(" ", ""), float(energy), None,
                        " / ".join(refs) or None, f"Blaise-Wyart {element} {spectrum} ({path.name})")
        if e:
            levels.append(e)
    return levels


def main(nist: Path, lac: Path, out: Path) -> None:
    data = {"source": "measured atomic levels: NIST ASD (v5.12) and Blaise & Wyart actinide tables (LAC)",
            "rules": __doc__.split("Sources")[1].split("Energies are")[0].strip(),
            "elements": {}}
    for element in LANTHANIDES + ACTINIDES:
        shell = "4f" if element in LANTHANIDES else "5f"
        neutral, ie = read_nist(nist / f"{element}_I.tsv", element, shell)
        cation, _ = read_nist(nist / f"{element}_II.tsv", element, shell)
        if element in LAC_DIRS:
            tables = lac / "Tab-energy" / LAC_DIRS[element] / f"{element}-tables"
            for spectrum, digit, target in (("I", "1", neutral), ("II", "2", cation)):
                for f in sorted(tables.glob(f"{element}{digit}[eo]*.html")):
                    if "laser" not in f.name:
                        target += read_lac(f, element, spectrum)
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
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
