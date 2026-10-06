#!/usr/bin/env python3
"""Resolve OH component-resolved CBS evidence without new quantum chemistry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

from openea_benchmark.attachment.component_resolved_cbs import (
    HARTREE_TO_EV,
    resolve_cbs,
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _find_reference_total(reference_geometry: dict) -> float:
    path = Path(reference_geometry["source_loop_path"])
    rid = str(reference_geometry["request_id"])
    with path.open("rb") as handle:
        loop = pickle.load(handle)
    matches = [
        result for result in tuple(getattr(loop, "results", ()) or ())
        if str(getattr(result, "request_id", "")) == rid
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{path}: expected exactly one reference result for {rid}, "
            f"found {len(matches)}"
        )
    value = getattr(matches[0], "ccsd_t_total_hartree", None)
    if value is None:
        raise RuntimeError(f"{path}: reference CCSD(T) total missing")
    return float(value)


def _find_aug5_full_pec(reference_result: dict) -> float | None:
    for point in reference_result.get("evaluated_points", []):
        if int(point.get("augmentation_level", -1)) == 1:
            return float(point["ea_central_ev"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component-run-id",
        default="oh_cbs_components_qz5z_20261005",
    )
    parser.add_argument(
        "--reference-run-id",
        default="oh_diffuse_x5_resume_20261003",
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument(
        "--output",
        default="oh_component_resolved_cbs.json",
    )
    parser.add_argument(
        "--model-spread-target-ev",
        type=float,
        default=0.005,
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    component_path = run_root / args.component_run_id / "result.json"
    reference_path = run_root / args.reference_run_id / "result.json"

    print(
        "[OPENEA][CBS] current=LOAD_COMPONENT_EVIDENCE "
        "| completed=- "
        "| next=EXTRAPOLATE_SCF -> EXTRAPOLATE_CCSD -> "
        "EXTRAPOLATE_TRIPLES -> APPLY_DIFFUSE_CORRECTION -> "
        "ASSESS_MODEL_SENSITIVITY",
        file=sys.stderr,
        flush=True,
    )

    component = _load_json(component_path)
    if component.get("status") != "READY":
        raise RuntimeError(
            f"component evidence not READY: {component.get('status')}"
        )
    reference = _load_json(reference_path)

    geoms = {
        item["role"]: item
        for item in component["reference_geometries"]
    }
    neutral_daug = _find_reference_total(geoms["neutral"])
    anion_daug = _find_reference_total(geoms["anion"])
    daug_fixed_ea_ev = (neutral_daug - anion_daug) * HARTREE_TO_EV

    final_assessment = reference.get("final_assessment") or {}
    diffuse_residual = final_assessment.get("residual_estimate_ev")
    full_aug5 = _find_aug5_full_pec(reference)

    print(
        "[OPENEA][CBS] current=COMPONENT_EXTRAPOLATION "
        "| completed=LOAD_COMPONENT_EVIDENCE "
        "| next=PRIMARY_COMPONENT_MODEL -> SENSITIVITY_MODEL -> "
        "COMBINED_CORRELATION_SANITY",
        file=sys.stderr,
        flush=True,
    )

    result = resolve_cbs(
        points=list(component["points"]),
        daug_fixed_ea_ev=daug_fixed_ea_ev,
        full_pec_aug5_ea_ev=full_aug5,
        diffuse_residual_estimate_ev=diffuse_residual,
        model_spread_target_ev=args.model_spread_target_ev,
    )
    payload = result.to_dict()
    payload.update(
        {
            "component_run_id": args.component_run_id,
            "reference_run_id": args.reference_run_id,
            "daug_fixed_reference_ea_ev": daug_fixed_ea_ev,
            "model_spread_target_ev": args.model_spread_target_ev,
            "scientific_scope": [
                "Legacy Stage-3 all-electron CCSD(T) / valence-basis CBS diagnostic only.",
                "Do not add a core-valence correction to this legacy result.",
                "QZ/5Z cardinal extrapolation uses fixed d-aug-5Z reference geometries.",
                "SCF, CCSD correlation, and (T) are extrapolated separately.",
                "d-aug minus aug at 5Z is retained as a separate diffuse correction.",
                "No core-valence, scalar-relativistic, SOC, post-CCSD(T), ZPE, or nuclear-motion correction is included.",
                "This is not a production adiabatic electron affinity.",
            ],
        }
    )

    output = Path(args.output)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"[OPENEA][CBS][{result.status.value}] "
        "current=COMPONENT_RESOLVED_CBS_COMPLETE "
        "| completed=SCF_CBS,CCSD_CBS,TRIPLES_CBS,"
        "DIFFUSE_CORRECTION,MODEL_SENSITIVITY "
        "| next=" + " -> ".join(result.next_actions),
        file=sys.stderr,
        flush=True,
    )
    print(
        "[OPENEA][CBS][DETAIL] "
        + json.dumps(
            {
                "ea_cbs_aug_primary_ev":
                    result.primary_model.ea_cbs_aug_ev,
                "diffuse_correction_ev":
                    result.diffuse_correction_ev,
                "ea_cbs_plus_diffuse_primary_ev":
                    result.ea_cbs_plus_diffuse_primary_ev,
                "cbs_model_sensitivity_bound_ev":
                    result.cbs_model_sensitivity_bound_ev,
                "conservative_intermediate_half_width_ev":
                    result.conservative_intermediate_half_width_ev,
                "output": str(output),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
