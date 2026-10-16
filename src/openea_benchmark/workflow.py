"""OpenEA entry point: two element symbols in, electron affinity (or UNBOUND) out.

    python -m openea_benchmark.workflow --atoms O H --output runs/OH

The whole route (symmetry scout, fragment asymptotes, Stage-3 CCSD(T) curves,
electronic EA, CBS, core-valence, scalar relativity, spin-orbit, post-CCSD(T),
multireference competitors, zero-point vibration) is chosen by the workflow
itself; no reference data enter the calculation.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .branch_continuity import BranchThresholds
from .state_identity import IdentityThresholds


# State-identity and branch-continuity thresholds of the Stage-3 / ZPE curve checks.
WEEKEND_IDENTITY_THRESHOLDS = IdentityThresholds(
    same_energy_mev=0.5,
    same_delta_s2=1.0e-5,
    same_total_spectrum_max=1.0e-5,
    same_spin_spectrum_max=1.0e-5,
    same_total_density_rel_fro=1.0e-5,
    same_spin_density_rel_fro=1.0e-5,

    distinct_energy_mev=10.0,
    distinct_delta_s2=0.10,
    distinct_total_spectrum_max=0.10,
    distinct_spin_spectrum_max=0.10,
    distinct_total_density_rel_fro=0.10,
    distinct_spin_density_rel_fro=0.10,
)


WEEKEND_BRANCH_THRESHOLDS = BranchThresholds(
    continuous_occ_min=0.95,
    discontinuous_occ_min=0.50,
    continuous_delta_s2=0.02,
    discontinuous_delta_s2=0.20,
    max_step_angstrom=0.10,
)


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    return value


def _jsonable(
    value: Any,
    *,
    field_name: str | None = None,
) -> Any:
    """
    Conservative serialization for provenance.

    Large AO objects and PySCF Mole objects are deliberately not serialized.
    """
    value = _enum_value(
        value
    )

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        result = {
            "type": "ndarray",
            "shape": list(
                value.shape
            ),
            "dtype": str(
                value.dtype
            ),
        }

        if value.size:
            if np.issubdtype(
                value.dtype,
                np.number,
            ):
                result[
                    "min"
                ] = float(
                    np.min(
                        value
                    )
                )

                result[
                    "max"
                ] = float(
                    np.max(
                        value
                    )
                )

        return result

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _jsonable(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            tuple,
            list,
            set,
        ),
    ):
        return [
            _jsonable(
                item
            )
            for item in value
        ]

    if is_dataclass(
        value
    ):
        output = {}

        for item in fields(
            value
        ):
            name = item.name

            if name == "mol":
                output[name] = (
                    "<PySCF Mole omitted>"
                )

                continue

            output[name] = _jsonable(
                getattr(
                    value,
                    name,
                ),
                field_name=name,
            )

        return output

    #
    # Avoid huge / unstable repr() output for PySCF-like objects.
    #
    module = getattr(
        type(value),
        "__module__",
        "",
    )

    if module.startswith(
        "pyscf"
    ):
        return (
            f"<{type(value).__name__} omitted>"
        )

    return repr(
        value
    )


def _write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            _jsonable(
                value
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OpenEA: electron affinity of a diatomic from its two atoms"
        )
    )

    parser.add_argument(
        "--atoms", nargs=2, metavar=("A", "B"), required=True, help="Two element symbols",
    )

    parser.add_argument(
        "--mode", choices=("symmetry-auto",), default="symmetry-auto",
        help="only route (kept for existing launch scripts)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--grid-level",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-stage3-points", type=int, default=100,
        help="Bounded per-invocation Stage-3 CCSD(T) point budget",
    )
    parser.add_argument(
        "--stage3-memory-mb", type=int, default=12000,
    )
    parser.add_argument(
        "--no-provisional-stage3", action="store_true",
        help="Execute Stage-3 only when adaptive release gates are cleared",
    )
    parser.add_argument(
        "--auto-dry-run", action="store_true",
        help="Plan/select automatically without executing Stage-3 points",
    )
    parser.add_argument(
        "--stability-following-steps", type=int, default=4,
        help="Max D02 internal-instability following steps per Stage-3 HF (0 = fail on instability)",
    )
    parser.add_argument(
        "--rhf-external-instability", choices=("FAIL", "RECORD_AND_CONTINUE"),
        default="RECORD_AND_CONTINUE",
        help="RHF->UHF instability: fail the point, or keep RHF and record UHF lowering/<S^2> for D03",
    )
    parser.add_argument(
        "--no-electronic-ea", action="store_true",
        help="Stop after Stage-3 acquisition (skip refinement, equilibrium, fragments and electronic EA)",
    )
    parser.add_argument(
        "--no-cbs", action="store_true",
        help="Skip the frozen-core aug-cc-pV{Q,5}Z (+d-aug) CBS layer after the electronic EA",
    )
    parser.add_argument(
        "--no-zpe", action="store_true",
        help="Skip the zero-point (v=0) nuclear-motion layer and the adiabatic-EA assembly",
    )
    parser.add_argument(
        "--no-mr-competitors", action="store_true",
        help="Skip the multireference check of MR-only Wigner-Witmer states (SA-CASSCF/NEVPT2)",
    )
    parser.add_argument(
        "--no-post-cc", action="store_true",
        help="Skip the post-CCSD(T) layer (CCSDT and CCSDTQ increments, CCpy)",
    )
    parser.add_argument(
        "--no-sr", action="store_true",
        help="Skip the scalar-relativity (all-electron SFX2C1E minus NR CCSD(T)) layer",
    )
    parser.add_argument(
        "--no-soc", action="store_true",
        help="Skip the first-order spin-orbit layer (OpenMolcas CASSCF/AMFI)",
    )
    parser.add_argument(
        "--no-cv", action="store_true",
        help="Skip the core-valence (all-electron minus frozen-core CCSD(T)) layer",
    )
    parser.add_argument(
        "--cbs-memory-mb", type=int, default=48000,
        help="PySCF memory for the CBS-layer CCSD(T) points",
    )
    parser.add_argument(
        "--point-workers", type=int, default=1,
        help="concurrent independent calculations (scout blocks, Stage-3/CBS points), each with --threads",
    )
    parser.add_argument(
        "--core-slots", type=int, default=0,
        help="machine-wide limit of concurrent calculations shared by all runs using --core-slots-dir",
    )
    parser.add_argument(
        "--core-slots-dir", type=Path, default=None,
        help="lock directory of the shared core-slot budget",
    )
    parser.add_argument(
        "--mem-limit-frac", type=float, default=0.0,
        help="start a calculation only while machine memory in use plus reservations stays below "
             "this fraction of RAM (needs --core-slots-dir; 0 disables)",
    )
    parser.add_argument(
        "--mem-reserve-mb", type=int, default=None,
        help="memory reserved per calculation by the memory gate (default: --cbs-memory-mb)",
    )
    parser.add_argument(
        "--no-open-pec-extension", action="store_true",
        help="Skip the D09 EXTEND_OPEN_PEC continuation scans and DFT fragment asymptotes",
    )
    parser.add_argument(
        "--acquisition-dir", default="benchmark_auto",
        help="Per-system subdirectory  records (use a new one when settings change)",
    )

    return parser


def _run_symmetry_auto(args) -> int:
    """Two element symbols -> symmetry scout -> asymptotes -> Stage-3 -> electronic EA."""
    from pyscf import lib as pyscf_lib

    from .benchmark_auto import AcquisitionSettings, run_benchmark_acquisition
    from .open_pec_extension import run_open_pec_extension
    from .symmetry_scout import manifest_for_two_atoms, run_symmetry_scout, to_legacy_summary

    if args.output is None:
        raise SystemExit("--output DIR is required")
    atom_a, atom_b = args.atoms
    from .heavy_elements import element_route
    unsupported = {el: element_route(el)["reason"] for el in (atom_a, atom_b) if element_route(el)["route"] is None}
    if unsupported:  # fail closed before any computation
        verdict = {"system": f"{atom_a}{atom_b}", "final_verdict": "ELEMENT_NOT_SUPPORTED", "elements": unsupported}
        _write_json(args.output / f"{atom_a}{atom_b}" / "final_verdict.json", verdict)
        print("[SYMMETRY AUTO] " + json.dumps(verdict), flush=True)
        return 2
    from .heavy_elements import set_model_hamiltonian
    print(f"[SYMMETRY AUTO] model Hamiltonian {set_model_hamiltonian((atom_a, atom_b))}", flush=True)
    from .compute_threads import set_compute_threads
    from .point_pool import configure_point_pool
    set_compute_threads(args.threads)
    configure_point_pool(workers=args.point_workers, threads=args.threads,
                         slot_dir=args.core_slots_dir, n_slots=args.core_slots,
                         mem_limit_frac=args.mem_limit_frac,
                         mem_reserve_mb=(args.cbs_memory_mb if args.mem_reserve_mb is None
                                         else args.mem_reserve_mb))
    system_dir = args.output / f"{atom_a}{atom_b}"
    print(f"[SYMMETRY AUTO] symmetry scout {atom_a}{atom_b}", flush=True)
    record = run_symmetry_scout(atom_a, atom_b, output_dir=system_dir / "symmetry_scout")
    preliminary = to_legacy_summary(record)
    extension = None
    if not args.no_open_pec_extension:
        print(f"[SYMMETRY AUTO] fragment asymptotes / open PECs {atom_a}{atom_b}", flush=True)
        from .point_pool import core_slot
        with core_slot():
            extension = run_open_pec_extension(summary=preliminary, output_dir=system_dir,
                                               grid_level=args.grid_level)
    summary = to_legacy_summary(record, atomic=(extension or {}).get("atomic"))
    manifest = manifest_for_two_atoms(summary)
    _write_json(system_dir / "summary.json", summary)
    _write_json(system_dir / "manifest.json", manifest)
    state = run_benchmark_acquisition(
        summary=summary, manifest=manifest, open_pec_extension=extension,
        output_dir=system_dir / args.acquisition_dir,
        settings=AcquisitionSettings(
            max_points=args.max_stage3_points, threads=args.threads, memory_mb=args.stage3_memory_mb,
            enable_provisional_points=not args.no_provisional_stage3, compute_points=not args.auto_dry_run,
            stability_following_max_steps=args.stability_following_steps,
            rhf_external_instability_policy=args.rhf_external_instability,
            assemble_electronic_ea=not args.no_electronic_ea,
            cbs_stage=not args.no_cbs and not args.no_electronic_ea,
            cbs_memory_mb=args.cbs_memory_mb,
            zpe_stage=not args.no_zpe and not args.no_electronic_ea,
            cv_stage=not args.no_cv and not args.no_electronic_ea,
            soc_stage=not args.no_soc and not args.no_electronic_ea,
            sr_stage=not args.no_sr and not args.no_electronic_ea,
            post_cc_stage=not args.no_post_cc and not args.no_electronic_ea,
            mr_competitor_stage=not args.no_mr_competitors and not args.no_electronic_ea,
        ),
    )
    print("[SYMMETRY AUTO] " + json.dumps({
        "system": summary["system"], "stage3_release": state["stage3_release"],
        "completed": state["completed_point_count"], "failed": state["failed_point_count"],
        "mr_only_candidates": len(summary["mr_only_candidates"]),
        "electronic_ea": state.get("electronic_ea"),
        "cbs": state.get("cbs"),
        "core_valence": state.get("core_valence"),
        "spin_orbit": state.get("spin_orbit"),
        "scalar_relativity": state.get("scalar_relativity"),
        "post_ccsd_t": state.get("post_ccsd_t"),
        "mr_competitors": state.get("mr_competitors"),
        "final_verdict": state.get("final_verdict"),
        "adiabatic_ea": state.get("adiabatic_ea"),
    }, default=str), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return _run_symmetry_auto(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
