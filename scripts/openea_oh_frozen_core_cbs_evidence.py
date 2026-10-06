#!/usr/bin/env python3
"""Recompute only the frozen-core component evidence needed for OH CBS.

The earlier Stage-3 component run used PySCF's default all-electron CCSD(T).
This runner intentionally computes a clean valence frozen-core baseline.

Only six fixed-geometry molecular single points are run:
neutral/anion x aug-QZ, aug-5Z, d-aug-5Z.
No PECs and no fragments are recomputed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import pickle

from pyscf import cc, gto, scf
from pyscf.data import elements

from openea_benchmark.adaptive.run_feedback import ProgressReporter
from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionSettings,
    run_stage3_point,
)
from openea_benchmark.attachment.cbs_component_evidence import (
    load_reference_geometries,
    make_cbs_single_point_request,
    point_from_stage3_result,
)
from openea_benchmark.attachment.frozen_core_cbs_evidence import (
    summarize_frozen_core_points,
)

BASIS_SPECS = (
    ("aug-cc-pvqz", 4),
    ("aug-cc-pv5z", 5),
    ("d-aug-cc-pv5z", 5),
)


def default_run_id():
    return os.environ.get("OPENEA_RUN_ID") or (
        "oh_frozen_core_cbs_" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )


def source_checkpoint(path, role, basis, r, max_memory_mb):
    if path.is_file():
        return
    charge = 0 if role == "neutral" else -1
    spin = 1 if role == "neutral" else 0
    mol = gto.M(
        atom=f"O 0 0 0; H 0 0 {r}",
        basis=basis,
        charge=charge,
        spin=spin,
        unit="Angstrom",
        symmetry=False,
        verbose=0,
        max_memory=max_memory_mb,
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
            raise RuntimeError(f"source SCF failed: {role} {basis}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-run-id", default="oh_diffuse_x5_resume_20261003")
    p.add_argument("--reference-basis", default="d-aug-cc-pv5z")
    p.add_argument("--run-root", default="runs")
    p.add_argument("--run-id", default=default_run_id())
    p.add_argument("--max-memory-mb", type=int, default=12000)
    args = p.parse_args()

    if importlib.util.find_spec("basis_set_exchange") is None:
        raise SystemExit("basis-set-exchange is required for d-aug-cc-pV5Z")

    run_root = Path(args.run_root)
    run_dir = run_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    point_dir = run_dir / "component_points"
    source_dir = run_dir / "source_checkpoints"
    hf_dir = run_dir / "hf_artifacts"
    point_dir.mkdir(exist_ok=True)
    source_dir.mkdir(exist_ok=True)
    hf_dir.mkdir(exist_ok=True)

    reporter = ProgressReporter(
        run_id=args.run_id,
        workflow="OH_FROZEN_CORE_CBS_EVIDENCE",
        status_file=run_dir / "status.json",
    )

    neutral_geom, anion_geom = load_reference_geometries(
        run_dir=run_root / args.reference_run_id,
        reference_basis=args.reference_basis,
    )
    geoms = {"neutral": neutral_geom, "anion": anion_geom}

    # Explicitly verify the automatic PySCF core rule for this OH validation.
    probe = gto.M(
        atom=f"O 0 0 0; H 0 0 {neutral_geom.r_angstrom}",
        basis="aug-cc-pvqz",
        charge=0,
        spin=1,
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )
    frozen_spatial = int(elements.chemcore(probe))
    if frozen_spatial != 1:
        reporter.blocked(
            current_step="FROZEN_CORE_PREFLIGHT_BLOCKED",
            completed_steps=[],
            next_steps=["INSPECT_PYSCF_CHEMCORE_POLICY"],
            details={"chemcore_spatial_orbitals": frozen_spatial},
        )
        raise SystemExit(
            f"OH validation expected one frozen spatial core orbital, got {frozen_spatial}"
        )

    settings = Stage3ExecutionSettings(
        scf_conv_tol=1e-9,
        scf_conv_tol_grad=1e-6,
        cc_conv_tol=1e-8,
        max_memory_mb=args.max_memory_mb,
        verbose=0,
        frozen_core=True,
        artifact_dir=str(hf_dir),
    )

    completed = ["REFERENCE_GEOMETRIES_LOADED", "FROZEN_CORE_POLICY_VERIFIED"]
    points = []

    for basis, cardinal in BASIS_SPECS:
        for role in ("neutral", "anion"):
            label = f"{role}:{basis}"
            pkl_path = point_dir / f"{role}__{basis}.pkl"
            json_path = point_dir / f"{role}__{basis}.json"

            if pkl_path.is_file():
                with pkl_path.open("rb") as fh:
                    result = pickle.load(fh)
                if (
                    getattr(result, "status", None) is PointExecutionStatus.COMPLETED
                    and getattr(result, "basis", None) == basis
                    and abs(float(result.r_angstrom) - geoms[role].r_angstrom) < 1e-10
                ):
                    point = point_from_stage3_result(
                        result, role=role, cardinal_number=cardinal
                    )
                    points.append(point)
                    completed.append(label)
                    reporter.update(
                        current_step=f"REUSE_{label}",
                        completed_steps=completed,
                        next_steps=["CONTINUE_FROZEN_CORE_SERIES"],
                        details={"source": "CHECKPOINT"},
                    )
                    continue
                pkl_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)

            reporter.update(
                current_step=f"CALCULATE_{label}",
                completed_steps=completed,
                next_steps=["SAVE_POINT", "CONTINUE_FROZEN_CORE_SERIES"],
                details={
                    "basis": basis,
                    "role": role,
                    "r_angstrom": geoms[role].r_angstrom,
                    "correlation_space": "FROZEN_CORE_PYSCF_AUTO_CHEMCORE",
                    "frozen_spatial_core_orbitals": frozen_spatial,
                },
            )

            source = source_dir / f"{role}__{basis}.chk"
            source_checkpoint(
                source, role, basis, geoms[role].r_angstrom, args.max_memory_mb
            )
            request = make_cbs_single_point_request(
                role=role,
                basis=basis,
                cardinal_number=cardinal,
                r_angstrom=geoms[role].r_angstrom,
                source_checkpoint_path=source,
            )
            result = run_stage3_point(request, settings=settings)
            json_path.write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
            )
            if result.status is not PointExecutionStatus.COMPLETED:
                reporter.blocked(
                    current_step=f"BLOCKED_{label}",
                    completed_steps=completed,
                    next_steps=["INSPECT_FAILURE", "RERUN_SAME_RUN_ID"],
                    details={
                        "status": result.status.value,
                        "error_type": result.error_type,
                        "error_message": result.error_message,
                        "result_json": str(json_path),
                    },
                )
                raise SystemExit(2)

            with pkl_path.open("wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)

            point = point_from_stage3_result(
                result, role=role, cardinal_number=cardinal
            )
            points.append(point)
            completed.append(label)

    point_dicts = [p.to_dict() for p in points]
    summary = summarize_frozen_core_points(point_dicts)
    payload = summary.to_dict()
    payload.update({
        "run_id": args.run_id,
        "reference_run_id": args.reference_run_id,
        "reference_basis": args.reference_basis,
        "frozen_spatial_core_orbitals": frozen_spatial,
        "frozen_core_rule": "PySCF CCSD.set_frozen() -> elements.chemcore",
        "points": point_dicts,
        "reference_geometries": [
            neutral_geom.to_dict(), anion_geom.to_dict()
        ],
        "scientific_scope": [
            "Valence frozen-core CCSD(T) component evidence.",
            "Fixed d-aug-cc-pV5Z reference geometries.",
            "No PEC or atomic-fragment recalculation.",
            "No core-valence correction is included here.",
            "No CBS extrapolation is performed here.",
        ],
    })
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if summary.status == "READY":
        reporter.completed(
            current_step="FROZEN_CORE_CBS_EVIDENCE_READY",
            completed_steps=completed + ["EVIDENCE_ASSESSED"],
            next_steps=[
                "REBUILD_FROZEN_CORE_COMPONENT_RESOLVED_CBS",
                "COMPUTE_CORE_VALENCE_CORRECTION_SEPARATELY",
            ],
            details={"result_file": str(result_path)},
        )
    else:
        reporter.blocked(
            current_step="FROZEN_CORE_CBS_EVIDENCE_PARTIAL",
            completed_steps=completed,
            next_steps=list(summary.next_actions),
            details={"missing_points": list(summary.missing_points)},
        )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
