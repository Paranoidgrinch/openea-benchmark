"""Run the automated electronic-EA stage on an existing benchmark-auto acquisition.

Usage (lowest priority, from openea-benchmark/):
    nice -n 19 python scripts/openea_electronic_ea_from_acquisition.py \
        ../openea_diverse_wave0_scout/MgH/benchmark_auto_v2 --atoms Mg H \
        --stability-following-steps 4 --rhf-external-instability RECORD_AND_CONTINUE

The Stage-3 execution settings must match the acquisition that produced the
points (stability policy); the refinement checkpoints go next to the existing ones.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openea_benchmark.adaptive.stage3_execution import Stage3ExecutionRequest, Stage3ExecutionSettings
from openea_benchmark.electronic_ea_stage import run_electronic_ea_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("acquisition_dir", type=Path)
    parser.add_argument("--atoms", nargs=2, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=12000)
    parser.add_argument("--stability-following-steps", type=int, default=4)
    parser.add_argument("--rhf-external-instability", default="RECORD_AND_CONTINUE")
    parser.add_argument("--output-name", default="electronic_ea")
    parser.add_argument("--cbs", action="store_true", help="also run the frozen-core CBS layer")
    parser.add_argument("--cbs-memory-mb", type=int, default=48000)
    args = parser.parse_args()

    from pyscf import lib
    lib.num_threads(args.threads)

    acq = args.acquisition_dir
    # Only points of the current Stage-3 plan (and its reference-branch jobs):
    # an acquisition directory may hold records of jobs a later plan excluded.
    plan_jobs = {job["job_id"] for job in json.loads((acq / "stage3_plan.json").read_text())["jobs"]}
    reviews = json.loads((acq / "stage3_pec_reviews.json").read_text())
    requests, records = [], {}
    for path in sorted((acq / "points").glob("*.json")):
        data = json.loads(path.read_text())
        job = data["request"]["job_id"]
        if job not in plan_jobs and job not in reviews:
            continue
        if str((reviews.get(job) or {}).get("status", "")).startswith("SUPERSEDED"):
            continue
        req = dict(data["request"])
        req["atoms"] = tuple(req["atoms"])
        req["methods"] = tuple(req["methods"])
        requests.append(Stage3ExecutionRequest(**req))
        records[data["request_id"]] = data["result"]
    settings = Stage3ExecutionSettings(
        max_memory_mb=args.memory_mb, artifact_dir=str(acq / "checkpoints"),
        stability_following_max_steps=args.stability_following_steps,
        rhf_external_instability_policy=args.rhf_external_instability,
    )
    result = run_electronic_ea_stage(
        system=requests[0].system, atoms=tuple(args.atoms), requests=requests, point_records=records,
        pec_reviews=reviews, stage3_settings=settings, output_dir=acq / args.output_name,
    )
    summary = {k: result.get(k) for k in ("system", "status", "verdict", "sign_at_this_level", "ea_lower_ev", "ea_upper_ev", "ea_central_ev",
                                          "selected_neutral_job", "selected_anion_job", "reason", "basis")}
    summary["competition"] = {k: v["status"] for k, v in result["high_level_competition"].items()}
    summary["candidates"] = {lab: [(c["job_id"].split("__manifold")[0], c["loop_status"], c["equilibrium_status"])
                                   for c in items] for lab, items in result["candidates"].items()}
    if args.cbs:
        from openea_benchmark.cbs_stage import CBSStagePolicy, run_cbs_stage
        cbs = run_cbs_stage(electronic_ea=result, requests=requests, point_records=records,
                            refinement_dir=acq / args.output_name / "refinement_points", stage3_settings=settings,
                            output_dir=acq / (args.output_name + "_cbs"),
                            policy=CBSStagePolicy(max_memory_mb=args.cbs_memory_mb))
        summary["cbs"] = {k: cbs.get(k) for k in ("status", "reason", "ea_cbs_ev", "ea_cbs_lower_ev",
                                                  "ea_cbs_upper_ev", "fixed_basis_ea_ev")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
