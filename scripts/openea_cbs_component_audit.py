#!/usr/bin/env python3
"""Fast, non-destructive audit before component-resolved CBS work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from openea_benchmark.attachment.cbs_component_audit import (
    audit_cbs_components,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument(
        "--output",
        default="cbs_component_audit.json",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if args.run_id:
        run_dirs = tuple(run_root / item for item in args.run_id)
    else:
        run_dirs = tuple(
            path for path in sorted(run_root.iterdir())
            if path.is_dir() and path.name != "latest"
        ) if run_root.exists() else ()

    print(
        "[OPENEA][AUDIT] current=SCAN_STAGE_CHECKPOINTS "
        "| completed=- "
        "| next=EXTRACT_HF_CCSD_TRIPLES -> "
        "ASSESS_QZ_5Z_COMPONENT_AVAILABILITY -> PLAN_CBS",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[OPENEA][AUDIT] scanning="
        + (",".join(str(x) for x in run_dirs) if run_dirs else "<none>"),
        file=sys.stderr,
        flush=True,
    )

    audit = audit_cbs_components(run_dirs=run_dirs)
    payload = audit.to_dict()

    output = Path(args.output)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "[OPENEA][AUDIT] current=AUDIT_COMPLETE "
        "| completed=SCAN_STAGE_CHECKPOINTS,EXTRACT_COMPONENTS "
        "| next=" + " -> ".join(audit.next_actions),
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[OPENEA][AUDIT] report={output}",
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
