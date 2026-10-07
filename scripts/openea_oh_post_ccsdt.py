#!/usr/bin/env python3
"""OH post-CCSD(T) evidence using CCpy.

The calculation is performed at the fixed high-level reference geometries
already established by OpenEA.

Frozen-core valence correction:
    Delta_T3(X) = EA_CCSDT(X)  - EA_CCSD(T)(X)
    Delta_T4(X) = EA_CCSDTQ(X) - EA_CCSDT(X)

Basis sequence:
    aug-cc-pVDZ (X=2)
    aug-cc-pVTZ (X=3)

CCSDTQ is always evaluated at DZ.  It is evaluated at TZ only when the
direct DZ quadruples correction is larger than the configured small-term
threshold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path

from pyscf import gto, scf
from ccpy.drivers.driver import Driver

from openea_benchmark.adaptive.run_feedback import ProgressReporter
from openea_benchmark.attachment.cbs_component_evidence import (
    load_reference_geometries,
)
from openea_benchmark.attachment.post_ccsd_t import (
    PostCCPoint,
    assess_post_ccsd_t,
)

HARTREE_TO_EV = 27.211386245988
BASIS_BY_X = {2: "aug-cc-pvdz", 3: "aug-cc-pvtz"}


def default_run_id() -> str:
    return os.environ.get("OPENEA_RUN_ID") or (
        "oh_post_ccsdt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )


def ccpy_version() -> str:
    for name in ("coupled-cluster-py", "ccpy"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "UNKNOWN"


def make_mean_field(*, role: str, basis: str, r_angstrom: float, max_memory_mb: int):
    charge = 0 if role == "neutral" else -1
    spin = 1 if role == "neutral" else 0
    mol = gto.M(
        atom=f"O 0 0 0; H 0 0 {r_angstrom}",
        basis=basis,
        charge=charge,
        spin=spin,
        unit="Angstrom",
        symmetry=False,
        cart=False,
        verbose=0,
        max_memory=max_memory_mb,
    )
    mf = scf.ROHF(mol) if spin else scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.max_cycle = 150
    mf.kernel()
    if not mf.converged:
        mf2 = mf.newton()
        mf2.conv_tol = 1.0e-10
        mf2.max_cycle = 150
        mf2.kernel(mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
        mf = mf2
    if not mf.converged:
        raise RuntimeError(f"SCF failed for {role} {basis}")
    return mf


def configure_driver(mf):
    # OH contains one chemically frozen O(1s) spatial orbital.
    driver = Driver.from_pyscf(mf, nfrozen=1)
    driver.options["energy_convergence"] = 1.0e-08
    driver.options["amp_convergence"] = 1.0e-08
    driver.options["maximum_iterations"] = 100
    return driver


def extract_ccsd_t_correction(driver) -> tuple[float, dict]:
    raw = getattr(driver, "deltap3", None)
    if raw is None or len(raw) == 0:
        raise RuntimeError("CCpy CCSD(T) did not populate driver.deltap3")
    item = raw[0]
    if not isinstance(item, dict):
        raise RuntimeError(f"Unexpected CCpy deltap3 structure: {type(item)!r}")

    finite = {
        str(k): float(v)
        for k, v in item.items()
        if v is not None and math.isfinite(float(v))
    }
    if not finite:
        raise RuntimeError(f"No finite CCSD(T) correction in deltap3: {item!r}")

    # CCpy's conventional CCSD(T) implementation is exposed through
    # run_ccp3(method="ccsd(t)").  Current versions store the correction in
    # the A slot.  The fallback only accepts an unambiguous single value.
    if "A" in finite:
        value = finite["A"]
    elif len(set(round(v, 14) for v in finite.values())) == 1:
        value = next(iter(finite.values()))
    elif len(finite) == 1:
        value = next(iter(finite.values()))
    else:
        raise RuntimeError(
            "Ambiguous CCpy CCSD(T) deltap3 payload; refusing to guess: "
            f"{finite}"
        )
    return value, finite


def run_method(*, role, basis, cardinal, r_angstrom, method, point_dir, max_memory_mb):
    method_token = {
        "CCSD(T)": "CCSD_pT",
        "CCSDT": "CCSDT",
        "CCSDTQ": "CCSDTQ",
    }.get(method)
    if method_token is None:
        raise ValueError(f"Unsupported post-CC method for checkpoint naming: {method}")
    path = point_dir / f"X{cardinal}__{role}__{method_token}.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("status") == "COMPLETED"
            and data.get("role") == role
            and data.get("basis") == basis
            and data.get("method") == method
            and abs(float(data.get("r_angstrom")) - r_angstrom) <= 1.0e-10
        ):
            return data

    mf = make_mean_field(
        role=role,
        basis=basis,
        r_angstrom=r_angstrom,
        max_memory_mb=max_memory_mb,
    )
    driver = configure_driver(mf)

    payload = {
        "status": "RUNNING",
        "role": role,
        "basis": basis,
        "cardinal": cardinal,
        "r_angstrom": r_angstrom,
        "method": method,
        "reference_energy_hartree": float(mf.e_tot),
        "nfrozen_spatial_orbitals": 1,
        "ccpy_version": ccpy_version(),
    }

    if method == "CCSD(T)":
        driver.run_cc(method="ccsd")
        ccsd_corr = float(driver.correlation_energy)
        driver.run_ccp3(method="ccsd(t)")
        triples, raw = extract_ccsd_t_correction(driver)
        corr = ccsd_corr + triples
        payload.update({
            "ccsd_correlation_hartree": ccsd_corr,
            "perturbative_triples_hartree": triples,
            "ccpy_deltap3": raw,
        })
    elif method == "CCSDT":
        driver.run_cc(method="ccsdt")
        corr = float(driver.correlation_energy)
    elif method == "CCSDTQ":
        driver.run_cc(method="ccsdtq")
        corr = float(driver.correlation_energy)
    else:
        raise ValueError(method)

    total = float(mf.e_tot) + corr
    payload.update({
        "status": "COMPLETED",
        "correlation_energy_hartree": corr,
        "total_energy_hartree": total,
    })
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def electron_affinity(neutral: dict, anion: dict) -> float:
    return (
        float(neutral["total_energy_hartree"])
        - float(anion["total_energy_hartree"])
    ) * HARTREE_TO_EV


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-run-id", default="oh_diffuse_x5_resume_20261003")
    p.add_argument("--reference-basis", default="d-aug-cc-pv5z")
    p.add_argument("--run-root", default="runs")
    p.add_argument("--run-id", default=default_run_id())
    p.add_argument("--triples-target-ev", type=float, default=0.001)
    p.add_argument("--quadruples-target-ev", type=float, default=0.001)
    p.add_argument("--max-memory-mb", type=int, default=96000)
    args = p.parse_args()

    run_root = Path(args.run_root)
    run_dir = run_root / args.run_id
    point_dir = run_dir / "points"
    run_dir.mkdir(parents=True, exist_ok=True)
    point_dir.mkdir(exist_ok=True)

    reporter = ProgressReporter(
        run_id=args.run_id,
        workflow="OH_POST_CCSDT",
        status_file=run_dir / "status.json",
    )

    neutral_geom, anion_geom = load_reference_geometries(
        run_dir=run_root / args.reference_run_id,
        reference_basis=args.reference_basis,
    )
    geoms = {"neutral": neutral_geom.r_angstrom, "anion": anion_geom.r_angstrom}

    # Fail before expensive work if the required package/bases are unavailable.
    version = ccpy_version()
    if version == "UNKNOWN":
        raise RuntimeError("CCpy/coupled-cluster-py is not installed")
    for basis in BASIS_BY_X.values():
        for atom in ("O", "H"):
            gto.basis.load(basis, atom)

    completed = ["REFERENCE_GEOMETRIES_LOADED", "CCPY_PREFLIGHT_VERIFIED"]
    points: list[PostCCPoint] = []

    # First obtain T3-(T) at DZ and TZ.  At DZ also obtain full connected T4.
    for x in (2, 3):
        basis = BASIS_BY_X[x]
        energies = {}
        methods = ("CCSD(T)", "CCSDT", "CCSDTQ") if x == 2 else ("CCSD(T)", "CCSDT")
        for method in methods:
            for role in ("neutral", "anion"):
                label = f"X{x}:{role}:{method}"
                reporter.update(
                    current_step=f"CALCULATE_{label}",
                    completed_steps=completed,
                    next_steps=["SAVE_METHOD_RESULT", "CONTINUE_POST_CC_SERIES"],
                    details={
                        "basis": basis,
                        "role": role,
                        "method": method,
                        "correlation_space": "FROZEN_CORE",
                        "nfrozen_spatial_orbitals": 1,
                    },
                )
                result = run_method(
                    role=role,
                    basis=basis,
                    cardinal=x,
                    r_angstrom=geoms[role],
                    method=method,
                    point_dir=point_dir,
                    max_memory_mb=args.max_memory_mb,
                )
                energies[(role, method)] = result
                completed.append(label)

        ea_t = electron_affinity(
            energies[("neutral", "CCSD(T)")],
            energies[("anion", "CCSD(T)")],
        )
        ea_tfull = electron_affinity(
            energies[("neutral", "CCSDT")],
            energies[("anion", "CCSDT")],
        )
        if x == 2:
            ea_q = electron_affinity(
                energies[("neutral", "CCSDTQ")],
                energies[("anion", "CCSDTQ")],
            )
            delta_t4 = ea_q - ea_tfull
        else:
            ea_q = None
            delta_t4 = None

        points.append(PostCCPoint(
            cardinal=x,
            basis=basis,
            ea_ccsd_t_ev=ea_t,
            ea_ccsdt_ev=ea_tfull,
            delta_t3_ev=ea_tfull - ea_t,
            ea_ccsdtq_ev=ea_q,
            delta_t4_ev=delta_t4,
        ))

    assessment = assess_post_ccsd_t(
        points,
        triples_target_ev=args.triples_target_ev,
        quadruples_target_ev=args.quadruples_target_ev,
        max_quadruples_cardinal=3,
    )

    # Only escalate full CCSDTQ to aug-TZ if the directly computed DZ T4
    # correction is too large to be conservatively bounded by its own magnitude.
    if assessment.action == "COMPUTE_T4_X3":
        basis = BASIS_BY_X[3]
        energies = {}
        for method in ("CCSDT", "CCSDTQ"):
            for role in ("neutral", "anion"):
                label = f"X3:{role}:{method}"
                reporter.update(
                    current_step=f"CALCULATE_{label}",
                    completed_steps=completed,
                    next_steps=["SAVE_METHOD_RESULT", "REASSESS_POST_CC"],
                    details={
                        "basis": basis,
                        "role": role,
                        "method": method,
                        "correlation_space": "FROZEN_CORE",
                    },
                )
                result = run_method(
                    role=role,
                    basis=basis,
                    cardinal=3,
                    r_angstrom=geoms[role],
                    method=method,
                    point_dir=point_dir,
                    max_memory_mb=args.max_memory_mb,
                )
                energies[(role, method)] = result
                if label not in completed:
                    completed.append(label)

        ea_tfull = electron_affinity(
            energies[("neutral", "CCSDT")],
            energies[("anion", "CCSDT")],
        )
        ea_q = electron_affinity(
            energies[("neutral", "CCSDTQ")],
            energies[("anion", "CCSDTQ")],
        )
        points = [
            p if p.cardinal != 3 else PostCCPoint(
                cardinal=3,
                basis=p.basis,
                ea_ccsd_t_ev=p.ea_ccsd_t_ev,
                ea_ccsdt_ev=p.ea_ccsdt_ev,
                delta_t3_ev=p.delta_t3_ev,
                ea_ccsdtq_ev=ea_q,
                delta_t4_ev=ea_q - ea_tfull,
            )
            for p in points
        ]
        assessment = assess_post_ccsd_t(
            points,
            triples_target_ev=args.triples_target_ev,
            quadruples_target_ev=args.quadruples_target_ev,
            max_quadruples_cardinal=3,
        )

    payload = assessment.to_dict()
    payload.update({
        "run_id": args.run_id,
        "reference_run_id": args.reference_run_id,
        "ccpy_version": version,
        "basis_family": "aug-cc-pVXZ",
        "correlation_space": "FROZEN_CORE",
        "nfrozen_spatial_orbitals": 1,
        "definition": {
            "delta_t3": "EA_CCSDT - EA_CCSD(T)",
            "delta_t4": "EA_CCSDTQ - EA_CCSDT",
        },
        "scope": [
            "Valence post-CCSD(T) correction at fixed reference geometries.",
            "Nonrelativistic Hamiltonian; scalar relativity and SOC are separate.",
            "Diffuse aug-cc-pVXZ bases are used for both neutral and anion.",
            "No uncomputed higher-order term is assigned zero.",
            "No post-CC basis extrapolation is assumed in v1.",
        ],
        "next_after_clear": [
            "ADD_POST_CCSDT_CORRECTION_TO_ELECTRONIC_BASELINE",
            "ASSESS_SOC",
            "SOLVE_NUCLEAR_MOTION",
        ],
        "is_production_ea": False,
    })

    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    reporter.update(
        current_step="ASSESS_POST_CCSDT",
        completed_steps=completed,
        next_steps=[assessment.action],
        details=assessment.to_dict(),
    )

    if assessment.status == "CLEARED":
        reporter.completed(
            current_step="POST_CCSDT_CLEARED",
            completed_steps=completed + ["POST_CCSDT_ASSESSED"],
            next_steps=payload["next_after_clear"],
            details={
                "central_correction_ev": assessment.central_correction_ev,
                "combined_bound_ev": assessment.combined_bound_ev,
                "result_file": str(result_path),
            },
        )
    else:
        reporter.blocked(
            current_step="POST_CCSDT_NOT_CLEARED",
            completed_steps=completed,
            next_steps=[assessment.action],
            details={"result_file": str(result_path)},
        )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
