"""State inventory for finished (or running) runs, from the files they already wrote.

usage: python scripts/openea_state_inventory.py RUN_DIR [MOLECULE ...] [--write]

RUN_DIR holds one directory per molecule (``<MOL>/benchmark_auto``).  Prints the
table of every state found; ``--write`` also stores ``state_inventory.json`` next
to ``auto_state.json``.  Reads only; no computation.  MR stage files are used only
if the run's ``auto_state.json`` refers to them (older runs leave stale files).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from openea_benchmark.state_inventory import build_state_inventory, format_state_table


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def inventory_for(acq: Path) -> dict | None:
    electronic = _load(acq / "electronic_ea" / "electronic_ea.json")
    state = _load(acq / "auto_state.json") or {}
    gap = _load(acq / "mr_term_gap" / "mr_term_gap.json") if state.get("mr_term_gap") else None
    comp = _load(acq / "mr_competitors" / "mr_competitors.json") if state.get("mr_competitors") else None
    return build_state_inventory(electronic, final_verdict=state.get("final_verdict"), mr_term_gap=gap,
                                 mr_competitors=comp,
                                 mr_only_labels=state.get("mr_only_candidates_not_computed") or ())


def main(argv: list[str]) -> int:
    write = "--write" in argv
    args = [a for a in argv if a != "--write"]
    if not args:
        print(__doc__)
        return 2
    root, names = Path(args[0]), args[1:]
    dirs = [root / n for n in names] if names else sorted(p for p in root.iterdir() if (p / "benchmark_auto").is_dir())
    for d in dirs:
        acq = d / "benchmark_auto"
        inv = inventory_for(acq)
        outcome = (_load(acq / "auto_state.json") or {}).get("outcome")
        print(f"=== {d.name}  (final: {outcome})")
        print(format_state_table(inv))
        if write and inv is not None:
            (acq / "state_inventory.json").write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
