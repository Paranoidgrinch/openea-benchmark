#!/usr/bin/env python3
"""Resolve corrected frozen-core OH CBS without new QC calculations."""
from __future__ import annotations
import argparse, json
from pathlib import Path

from openea_benchmark.attachment.frozen_core_cbs_resolution import (
    resolve_frozen_core_cbs,
)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-root", default="runs")
    p.add_argument("--frozen-core-run-id",
                   default="oh_frozen_core_cbs_qz5z_daug5_20261005")
    p.add_argument("--reference-run-id",
                   default="oh_diffuse_x5_resume_20261003")
    p.add_argument("--output", default="oh_frozen_core_cbs.json")
    p.add_argument("--model-spread-target-ev", type=float, default=0.005)
    args=p.parse_args()

    root=Path(args.run_root)
    fc=json.loads((root/args.frozen_core_run_id/"result.json").read_text())
    ref=json.loads((root/args.reference_run_id/"result.json").read_text())

    if fc.get("status")!="READY":
        raise RuntimeError(f"frozen-core evidence not READY: {fc.get('status')}")

    indirect=(ref.get("final_assessment") or {}).get("residual_estimate_ev")

    print("[OPENEA][FC-CBS] current=LOAD_FROZEN_CORE_COMPONENTS "
          "| completed=- | next=SCF_CBS -> CCSD_CBS -> TRIPLES_CBS -> "
          "DIRECT_DAUG_CORRECTION -> UNCERTAINTY", flush=True)

    result=resolve_frozen_core_cbs(
        points=list(fc["points"]),
        indirect_diffuse_residual_ev=indirect,
        model_spread_target_ev=args.model_spread_target_ev,
    )
    payload=result.to_dict()
    payload.update({
        "frozen_core_run_id":args.frozen_core_run_id,
        "reference_run_id":args.reference_run_id,
        "correlation_space":"FROZEN_CORE_PYSCF_AUTO_CHEMCORE",
        "diffuse_residual_evidence_tier":
            "INDIRECTLY_ESTIMATED_FROM_PRIOR_ALL_ELECTRON_DIFFUSE_AXIS"
            if indirect is not None else "UNKNOWN",
        "scientific_scope":[
            "Valence frozen-core CCSD(T) CBS baseline.",
            "Direct frozen-core d-aug-minus-aug 5Z correction.",
            "Prior all-electron diffuse residual is used only as indirect uncertainty evidence.",
            "No core-valence, scalar-relativistic, SOC, post-CCSD(T), or nuclear-motion correction included.",
            "Not a production adiabatic electron affinity.",
        ],
    })
    Path(args.output).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(
        f"[OPENEA][FC-CBS][{result.status}] current=FROZEN_CORE_CBS_RESOLVED "
        f"| EA={result.ea_cbs_plus_diffuse_ev:.12f} eV "
        f"| half_width={result.conservative_half_width_ev:.12f} eV "
        "| next="+" -> ".join(result.next_actions),
        flush=True,
    )
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
