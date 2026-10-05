#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, pickle
from datetime import datetime, timezone
from pathlib import Path

from pyscf import gto, scf
from openea_benchmark.adaptive.run_feedback import ProgressReporter
from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus, Stage3ExecutionSettings, run_stage3_point,
)
from openea_benchmark.attachment.cbs_component_evidence import (
    load_reference_geometries, make_cbs_single_point_request,
    point_from_stage3_result, summarize_component_evidence,
)

BASIS_SPECS = (("aug-cc-pvqz", 4), ("aug-cc-pv5z", 5))


def default_run_id():
    return os.environ.get("OPENEA_RUN_ID") or (
        "oh_cbs_components_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )


def source_checkpoint(path: Path, *, role: str, basis: str, r: float, max_memory_mb: int):
    if path.is_file():
        return
    charge, spin = ((0, 1) if role == "neutral" else (-1, 0))
    mol = gto.M(
        atom=f"O 0 0 0; H 0 0 {r}", basis=basis, charge=charge, spin=spin,
        unit="Angstrom", symmetry=False, verbose=0, max_memory=max_memory_mb,
    )
    mf = scf.ROHF(mol) if spin else scf.RHF(mol)
    mf.chkfile = str(path)
    mf.conv_tol = 1e-9
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        mf2 = mf.newton()
        mf2.chkfile = str(path)
        mf2.conv_tol = 1e-9
        mf2.max_cycle = 100
        mf2.kernel(mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
        if not mf2.converged:
            raise RuntimeError(f"source SCF did not converge: {role} {basis}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-run-id", default="oh_diffuse_x5_resume_20261003")
    p.add_argument("--reference-basis", default="d-aug-cc-pv5z")
    p.add_argument("--run-root", default=os.environ.get("OPENEA_RUN_ROOT", "runs"))
    p.add_argument("--run-id", default=default_run_id())
    p.add_argument("--max-memory-mb", type=int, default=12000)
    args = p.parse_args()

    run_root = Path(args.run_root)
    run_dir = run_root / args.run_id
    ref_dir = run_root / args.reference_run_id
    result_dir = run_dir / "component_points"
    source_dir = run_dir / "source_checkpoints"
    result_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    reporter = ProgressReporter(
        run_id=args.run_id, workflow="OH_CBS_COMPONENT_EVIDENCE",
        status_file=run_dir / "status.json",
    )
    reporter.update(
        current_step="LOAD_REFERENCE_GEOMETRIES", completed_steps=[],
        next_steps=["QZ_NEUTRAL", "QZ_ANION", "5Z_NEUTRAL", "5Z_ANION", "ASSESS_COMPONENT_EVIDENCE"],
        details={"reference_run": args.reference_run_id, "reference_basis": args.reference_basis},
    )
    neutral, anion = load_reference_geometries(
        run_dir=ref_dir, reference_basis=args.reference_basis
    )
    geoms = {"neutral": neutral, "anion": anion}

    settings = Stage3ExecutionSettings(
        scf_conv_tol=1e-9, scf_conv_tol_grad=1e-6,
        cc_conv_tol=1e-8, max_memory_mb=args.max_memory_mb, verbose=0,
    )

    completed = ["REFERENCE_GEOMETRIES_LOADED"]
    points = []
    for basis, cardinal in BASIS_SPECS:
        for role in ("neutral", "anion"):
            label = f"{role}:{basis}"
            pkl = result_dir / f"{role}__{basis}.pkl"
            js = result_dir / f"{role}__{basis}.json"
            result = None
            if pkl.is_file():
                with pkl.open("rb") as h:
                    candidate = pickle.load(h)
                if (
                    getattr(candidate, "status", None) is PointExecutionStatus.COMPLETED
                    and getattr(candidate, "basis", None) == basis
                    and abs(float(candidate.r_angstrom) - geoms[role].r_angstrom) <= 1e-10
                ):
                    result = candidate
                    reporter.update(
                        current_step=f"REUSE_{label}",
                        completed_steps=completed + [label],
                        next_steps=["CONTINUE_COMPONENT_SERIES"],
                        details={"source": "COMPONENT_CHECKPOINT"},
                    )
                else:
                    pkl.unlink(missing_ok=True)
                    js.unlink(missing_ok=True)

            if result is None:
                reporter.update(
                    current_step=f"CALCULATE_{label}", completed_steps=completed,
                    next_steps=["SAVE_COMPONENT_CHECKPOINT", "CONTINUE_COMPONENT_SERIES"],
                    details={"basis": basis, "role": role, "r_angstrom": geoms[role].r_angstrom},
                )
                src = source_dir / f"{role}__{basis}.chk"
                source_checkpoint(
                    src, role=role, basis=basis, r=geoms[role].r_angstrom,
                    max_memory_mb=args.max_memory_mb,
                )
                req = make_cbs_single_point_request(
                    role=role, basis=basis, cardinal_number=cardinal,
                    r_angstrom=geoms[role].r_angstrom,
                    source_checkpoint_path=src,
                )
                result = run_stage3_point(req, settings=settings)
                js.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
                if result.status is not PointExecutionStatus.COMPLETED:
                    reporter.blocked(
                        current_step=f"BLOCKED_{label}", completed_steps=completed,
                        next_steps=["INSPECT_SINGLE_POINT_FAILURE", "RERUN_SAME_RUN_ID"],
                        details={
                            "status": result.status.value,
                            "error_type": result.error_type,
                            "error_message": result.error_message,
                            "result_json": str(js),
                        },
                    )
                    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
                    raise SystemExit(2)
                with pkl.open("wb") as h:
                    pickle.dump(result, h, protocol=pickle.HIGHEST_PROTOCOL)

            point = point_from_stage3_result(result, role=role, cardinal_number=cardinal)
            points.append(point)
            if label not in completed:
                completed.append(label)
            reporter.update(
                current_step=f"COMPLETED_{label}", completed_steps=completed,
                next_steps=["CONTINUE_COMPONENT_SERIES"],
                details={
                    "scf_energy_hartree": point.scf_energy_hartree,
                    "ccsd_correlation_hartree": point.ccsd_correlation_hartree,
                    "triples_correction_hartree": point.triples_correction_hartree,
                    "ccsd_t_total_hartree": point.ccsd_t_total_hartree,
                },
            )

    evidence = summarize_component_evidence(
        reference_basis=args.reference_basis,
        reference_geometries=(neutral, anion),
        points=points,
        required_bases=tuple(x[0] for x in BASIS_SPECS),
    )
    output = evidence.to_dict()
    output.update({
        "run_id": args.run_id,
        "reference_run_id": args.reference_run_id,
        "scientific_caveats": [
            "All cardinal points use fixed d-aug-cc-pV5Z reference geometries.",
            "No QZ/5Z PEC optimization is repeated.",
            "No atomic fragment thresholds are recomputed.",
            "No CBS extrapolation is performed here.",
        ],
    })
    out = run_dir / "result.json"
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    if evidence.status.value == "READY":
        reporter.completed(
            current_step="CBS_COMPONENT_EVIDENCE_READY",
            completed_steps=completed + ["COMPONENT_EVIDENCE_ASSESSED"],
            next_steps=[
                "COMPONENT_RESOLVED_CBS_EXTRAPOLATION",
                "APPLY_SEPARATE_DIFFUSE_CORRECTION",
                "PROPAGATE_UNCERTAINTY",
            ],
            details={"result_file": str(out)},
        )
    else:
        reporter.blocked(
            current_step="CBS_COMPONENT_EVIDENCE_INCOMPLETE",
            completed_steps=completed,
            next_steps=list(evidence.next_actions),
            details={"missing_points": list(evidence.missing_points)},
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
