"""Scout validation: is the literature ground state among the Stage-3 candidates?

Usage: .venv/bin/python scripts/scout_validation_report.py RESULTS_DIR [--json OUT]
       e.g.  scripts/scout_validation_report.py ../openea_results/scout_v1

The state selection is recomputed with the current code from each molecule's stored
symmetry_scout.json (+ open_pec_extension.json), so cached scouts are judged by the
current rules.  The literature table (validation/ground_state_terms_v1.json) is read
here only; the workflow itself never sees it.

Per charge:
  SEED          literature term is the lowest bracketed candidate (provisional seed)
  *_BLOCKED     (suffix) found, but an open component at or below the seed blocks the EA
  CANDIDATE     literature term is a bracketed candidate, not the seed (Stage 3 decides)
  SAFE_BLOCKED  term missing from the candidates, but an open component at or below the
                seed blocks the EA (UNRESOLVED instead of a wrong number)
  MR_GAP        term has no single-determinant block in the frontier window
  NOT_SCANNED   a block with the term exists but no bracketed minimum and nothing blocks
  MISSED        term missing and nothing blocks: the workflow would select a wrong state
  UNBOUND_EXPECTED  no bound anion in the literature
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openea_benchmark.open_pec_extension import apply_open_pec_extension  # noqa: E402
from openea_benchmark.state_selection import build_state_selection_report  # noqa: E402
from openea_benchmark.symmetry_scout import to_legacy_summary  # noqa: E402

HARTREE_TO_EV = 27.211386245988
BLOCK_MARGIN_EV = 1.0  # ElectronicEAStagePolicy.competitor_exclusion_margin_ev


def _label(component_id: str) -> str:
    return component_id.split("__")[1]


def evaluate(system_dir: Path, expected: dict) -> dict:
    scout_path = system_dir / "symmetry_scout" / "symmetry_scout.json"
    if not scout_path.exists():
        return {"status": "NO_SCOUT"}
    record = json.loads(scout_path.read_text())
    ext_path = system_dir / "open_pec_extension.json"
    extension = json.loads(ext_path.read_text()) if ext_path.exists() else None
    summary = to_legacy_summary(record, atomic=(extension or {}).get("atomic"))
    out: dict = {}
    if not summary["sectors"]:
        for label, q in (("neutral", "0"), ("anion", "-1")):
            out[label] = {"verdict": "MR_GAP" if expected.get(label) else "UNBOUND_EXPECTED",
                          "expected": expected.get(label), "seed": None, "candidates": []}
        return out
    if extension:
        summary = apply_open_pec_extension(summary, extension)
    selection = build_state_selection_report(summary)
    labels = {sc["component_id"]: sc["symmetry_block"]["label"]
              for sec in summary["sectors"] for sc in sec["manifold_pec_scouts"]}
    _label = lambda cid: labels.get(cid, cid.split("__")[1])  # noqa: E731  (relabelled blocks)
    for label, q in (("neutral", "0"), ("anion", "-1")):
        group = selection["charge_groups"][label]
        want = expected.get(label)
        seed = group.get("provisional_high_level_seed")
        cands = [_label(c["component_id"]) for c in group["bracketed_candidates"]]
        ref = seed["energy_hartree"] if seed else None
        blockers = [{"component": u["component_id"], "status": u["minimum_status"],
                     "gap_ev": (u["observed_lowest_energy_hartree"] - ref) * HARTREE_TO_EV}
                    for u in group["unresolved_components"]
                    if ref is not None and u.get("observed_lowest_energy_hartree") is not None
                    and (u["observed_lowest_energy_hartree"] - ref) * HARTREE_TO_EV < BLOCK_MARGIN_EV]
        plan = record["charges"].get(q, {}).get("plan", {})
        planned = {b["label"] for b in plan.get("blocks", ())}
        if want is None:
            verdict = "UNBOUND_EXPECTED"
        elif seed and _label(seed["component_id"]) == want:
            verdict = "SEED"
        elif want in cands:
            verdict = "CANDIDATE"
        elif blockers:
            verdict = "SAFE_BLOCKED"
        elif want not in planned:
            verdict = "MR_GAP"
        else:
            verdict = "NOT_SCANNED" if not cands else "MISSED"
        if verdict in ("SEED", "CANDIDATE") and blockers:
            verdict += "_BLOCKED"  # right term found, but an open competitor still blocks the EA
        out[label] = {"verdict": verdict, "expected": want,
                      "seed": _label(seed["component_id"]) if seed else None,
                      "candidates": cands, "blockers": blockers,
                      "expected_blocks": {bid: blk["status"] for bid, blk in
                                          record["charges"].get(q, {}).get("blocks", {}).items()
                                          if blk["label"] == want}}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--table", type=Path, default=ROOT / "validation" / "ground_state_terms_v1.json")
    args = parser.parse_args()
    table = json.loads(args.table.read_text())["systems"]
    report, tally = {}, {}
    print(f"{'mol':5} {'conf':6} {'neutral (exp/seed)':32} {'anion (exp/seed)':32}")
    for system, expected in table.items():
        system_dir = args.results / system
        res = evaluate(system_dir, expected) if system_dir.exists() else {"status": "NOT_RUN"}
        report[system] = res
        if "status" in res:
            print(f"{system:5} {expected['confidence']:6} {res['status']}")
            continue
        cells = []
        for label in ("neutral", "anion"):
            r = res[label]
            tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
            cells.append(f"{r['verdict']:16} {str(r['expected']):9}/{str(r['seed']):9}")
        print(f"{system:5} {expected['confidence']:6} {cells[0]:36} {cells[1]:36}")
    print("tally:", dict(sorted(tally.items())))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
