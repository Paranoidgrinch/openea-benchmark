#!/usr/bin/env python3
"""Adaptive scalar-relativistic OH correction.

For each cardinal number:
  O : aug-cc-pCVXZ-DK
  H : aug-cc-pVXZ-DK

At the fixed validated neutral/anion geometries, run all-electron CCSD(T):
  NR neutral, NR anion, SFX2C1E neutral, SFX2C1E anion

Then:
  Delta_SR(X) = EA_SFX2C1E(X) - EA_NR(X)

SOC is explicitly NOT included.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle

from pyscf import gto, scf

from openea_benchmark.adaptive.run_feedback import ProgressReporter
from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionSettings,
    run_stage3_point,
)
from openea_benchmark.attachment.cbs_component_evidence import (
    load_reference_geometries,
    make_cbs_single_point_request,
)
from openea_benchmark.attachment.scalar_relativity import (
    ScalarRelativityPoint,
    assess_scalar_relativity,
)

HARTREE_TO_EV = 27.211386245988

BASIS_BY_X = {
    3: {
        "label": "O:aug-cc-pCVTZ-DK|H:aug-cc-pVTZ-DK",
        "by_element": {
            "O": "aug-cc-pcvtz-dk",
            "H": "aug-cc-pvtz-dk",
        },
    },
    4: {
        "label": "O:aug-cc-pCVQZ-DK|H:aug-cc-pVQZ-DK",
        "by_element": {
            "O": "aug-cc-pcvqz-dk",
            "H": "aug-cc-pvqz-dk",
        },
    },
    5: {
        "label": "O:aug-cc-pCV5Z-DK|H:aug-cc-pV5Z-DK",
        "by_element": {
            "O": "aug-cc-pcv5z-dk",
            "H": "aug-cc-pv5z-dk",
        },
    },
}

def default_run_id():
    return os.environ.get("OPENEA_RUN_ID") or (
        "oh_scalar_rel_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

def make_source(path, role, basis_by_element, r, max_memory):
    if path.is_file():
        return
    charge = 0 if role == "neutral" else -1
    spin = 1 if role == "neutral" else 0
    mol = gto.M(
        atom=f"O 0 0 0; H 0 0 {r}",
        basis=dict(basis_by_element),
        charge=charge,
        spin=spin,
        unit="Angstrom",
        symmetry=False,
        verbose=0,
        max_memory=max_memory,
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
            raise RuntimeError(f"source SCF failed: {role}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-run-id", default="oh_diffuse_x5_resume_20261003")
    p.add_argument("--reference-basis", default="d-aug-cc-pv5z")
    p.add_argument("--run-root", default="runs")
    p.add_argument("--run-id", default=default_run_id())
    p.add_argument("--target-change-ev", type=float, default=0.0005)
    p.add_argument("--max-cardinal", type=int, default=5)
    p.add_argument("--max-memory-mb", type=int, default=96000)
    args = p.parse_args()

    run_root = Path(args.run_root)
    run_dir = run_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    points_dir = run_dir / "points"
    src_dir = run_dir / "source_checkpoints"
    artifacts = run_dir / "hf_artifacts"
    points_dir.mkdir(exist_ok=True)
    src_dir.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)

    reporter = ProgressReporter(
        run_id=args.run_id,
        workflow="OH_SCALAR_RELATIVITY",
        status_file=run_dir / "status.json",
    )

    neutral_geom, anion_geom = load_reference_geometries(
        run_dir=run_root / args.reference_run_id,
        reference_basis=args.reference_basis,
    )
    geoms = {"neutral": neutral_geom, "anion": anion_geom}
    # Validate the full requested relativistically recontracted basis series
# before starting any
    # expensive CCSD(T) work.  This deliberately fails fast if the local
    # PySCF/Basis Set Exchange installation lacks a required recontraction.
    requested_x = tuple(range(3, args.max_cardinal + 1))
    for x in requested_x:
        spec = BASIS_BY_X[x]
        for atom, basis_name in spec["by_element"].items():
            try:
                gto.basis.load(basis_name, atom)
            except Exception as exc:
                raise RuntimeError(
                    f"relativistic DK-basis preflight failed: X={x} atom={atom} "
                    f"basis={basis_name}. No QC calculation was started."
                ) from exc

    completed = ["REFERENCE_GEOMETRIES_LOADED", "RELATIVISTIC_BASIS_PREFLIGHT_VERIFIED"]
    sr_points = []

    for x in requested_x:
        spec = BASIS_BY_X[x]
        basis_label = spec["label"]
        basis_map = dict(spec["by_element"])

        totals = {}
        safe = basis_label.replace(":", "_").replace("|", "__").replace("/", "_")

        for role in ("neutral", "anion"):
            source = src_dir / f"{role}__X{x}.chk"
            make_source(
                source, role, basis_map, geoms[role].r_angstrom,
                args.max_memory_mb,
            )
            for ham in ("NR", "SFX2C1E"):
                label = f"X{x}:{role}:{ham}"
                pkl = points_dir / f"{safe}__{role}__{ham}.pkl"
                js = points_dir / f"{safe}__{role}__{ham}.json"

                result = None
                if pkl.is_file():
                    with pkl.open("rb") as fh:
                        candidate = pickle.load(fh)
                    if (
                        getattr(candidate, "status", None)
                        is PointExecutionStatus.COMPLETED
                        and candidate.basis == basis_label
                        and abs(
                            float(candidate.r_angstrom) - geoms[role].r_angstrom
                        ) < 1e-10
                    ):
                        result = candidate
                        reporter.update(
                            current_step=f"REUSE_{label}",
                            completed_steps=completed + [label],
                            next_steps=["CONTINUE_SCALAR_RELATIVITY_SERIES"],
                            details={
                                "basis": basis_label,
                                "basis_by_element": basis_map,
                                "role": role,
                                "hamiltonian": ham,
                            },
                        )

                if result is None:
                    reporter.update(
                        current_step=f"CALCULATE_{label}",
                        completed_steps=completed,
                        next_steps=["SAVE_POINT", "CONTINUE_SCALAR_RELATIVITY_SERIES"],
                        details={
                            "basis": basis_label,
                            "basis_by_element": basis_map,
                            "role": role,
                            "hamiltonian": ham,
                            "correlation_space": "ALL_ELECTRON",
                        },
                    )
                    request = make_cbs_single_point_request(
                        role=role,
                        basis=basis_label,
                        cardinal_number=x,
                        r_angstrom=geoms[role].r_angstrom,
                        source_checkpoint_path=source,
                        basis_by_element=basis_map,
                    )
                    settings = Stage3ExecutionSettings(
                        scf_conv_tol=1e-9,
                        scf_conv_tol_grad=1e-6,
                        cc_conv_tol=1e-8,
                        max_memory_mb=args.max_memory_mb,
                        frozen_core=False,
                        scalar_relativistic=(
                            "NONE" if ham == "NR" else "SFX2C1E"
                        ),
                        verbose=0,
                        artifact_dir=str(artifacts / f"X{x}_{role}_{ham}"),
                    )
                    result = run_stage3_point(request, settings=settings)
                    payload = result.to_dict()
                    payload["hamiltonian"] = ham
                    payload["basis_by_element"] = basis_map
                    js.write_text(
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
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
                            },
                        )
                        raise SystemExit(2)
                    with pkl.open("wb") as fh:
                        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)

                totals[(role, ham)] = float(result.ccsd_t_total_hartree)
                if label not in completed:
                    completed.append(label)

        ea_nr = (
            totals[("neutral", "NR")] - totals[("anion", "NR")]
        ) * HARTREE_TO_EV
        ea_x2c = (
            totals[("neutral", "SFX2C1E")]
            - totals[("anion", "SFX2C1E")]
        ) * HARTREE_TO_EV

        sr_points.append(
            ScalarRelativityPoint(
                cardinal=x,
                basis=basis_label,
                ea_nr_ev=ea_nr,
                ea_sfx2c1e_ev=ea_x2c,
                delta_sr_ev=ea_x2c - ea_nr,
            )
        )
        assessment = assess_scalar_relativity(
            sr_points,
            target_change_ev=args.target_change_ev,
            max_cardinal=args.max_cardinal,
        )
        reporter.update(
            current_step=f"ASSESS_SCALAR_RELATIVITY_X{x}",
            completed_steps=completed,
            next_steps=[assessment.action],
            details=assessment.to_dict(),
        )
        if assessment.status == "CLEARED":
            break
        if assessment.status == "UNRESOLVED":
            break

    payload = assessment.to_dict()
    payload.update({
        "run_id": args.run_id,
        "reference_run_id": args.reference_run_id,
        "basis_family": "O:aug-cc-pCVXZ-DK | H:aug-cc-pVXZ-DK",
        "definition": "Delta_SR = EA_SFX2C1E - EA_NR in the same DK-recontracted basis",
        "correlation_space": "ALL_ELECTRON",
        "includes_soc": False,
        "relativistic_scope": "SPIN_FREE_ONE_ELECTRON_X2C",
        "includes_two_electron_relativistic_terms": False,
        "method_note": (
            "PySCF spin-free one-electron X2C. SOC and two-electron "
            "relativistic terms are outside this layer and must be "
            "assessed separately when relevant."
        ),
        "next_after_clear": [
            "ADD_DELTA_SR_TO_CBS_PLUS_CORE_VALENCE_BASELINE",
            "ASSESS_TWO_ELECTRON_RELATIVISTIC_RELEVANCE",
            "ASSESS_POST_CCSD_T",
            "ASSESS_SOC",
        ],
        "is_production_ea": False,
    })
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if assessment.status == "CLEARED":
        reporter.completed(
            current_step="SCALAR_ONE_BODY_RELATIVITY_CLEARED",
            completed_steps=completed + ["SCALAR_RELATIVITY_CONVERGENCE_CLEARED"],
            next_steps=payload["next_after_clear"],
            details={
                "central_correction_ev": assessment.central_correction_ev,
                "convergence_bound_ev": assessment.convergence_bound_ev,
                "result_file": str(result_path),
            },
        )
    else:
        reporter.blocked(
            current_step="SCALAR_RELATIVITY_NOT_CLEARED",
            completed_steps=completed,
            next_steps=[assessment.action],
            details={"result_file": str(result_path)},
        )

    print(json.dumps(payload, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
