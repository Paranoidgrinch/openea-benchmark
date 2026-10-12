"""Automated core-valence layer (Patch 46).

    Delta_CV = EA_AE(X) - EA_FC(X)

at the def2-TZVPPD equilibrium of the selected neutral and anion states,
all-electron (AE) and frozen-core (FC) CCSD(T) in the same core-valence basis.
FC freezes the noble-gas core (``valence_frozen_core_orbitals``), the same
space the frozen-core CBS layer left uncorrelated, so EA_CBS(FC) + Delta_CV is
the all-electron valence+core value.

Execution reuses ``adaptive.core_valence_runner`` (matched AE/FC points and
the ``assess_core_valence`` convergence policy: X=T,Q first, X=5 when the T->Q
change exceeds the target; the highest cardinal is the central value, the last
change is the bound; no extrapolation).  New here:

* the per-element basis rule (``core_valence_basis``), so no caller has to
  supply a basis policy;
* the state's symmetry block (point group, irrep occupation) and the
  seed-continuous reference branch are carried into every CV point, as in the
  CBS layer, so the open-shell reference stays the validated state;
* the initial T/Q points run concurrently on the point pool and are cached.

Fails closed: an element without a core-valence basis (e.g. K, Ca in the local
libraries) blocks the layer (``POLICY_BLOCKED_NO_CV_BASIS``); a failed point
blocks it (``EXECUTION_BLOCKED``); no convergence at the highest available
cardinal gives ``CORE_VALENCE_UNRESOLVED``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .adaptive.core_valence_runner import (
    CoreValenceBasisSpec, CoreValenceCorrelationSpace, CoreValenceStateSpec, _request_for, _settings_for,
    run_adaptive_core_valence_series,
)
from .adaptive.stage3_execution import (
    SEED_CONTINUOUS_BRANCH, Stage3ExecutionRequest, Stage3ExecutionSettings, Stage3PointResult,
)
from .cbs_stage import _job_point_records
from .core_valence_basis import cv_element_series
from .electronic_ea_stage import _save_json, cached_point_runner
from .point_pool import run_parallel

BASIS_RULE_ID = "OPENEA_CV_BASIS_RULE_V1"


@dataclass(frozen=True)
class CoreValenceStagePolicy:
    """Explicit pilot policy, not a validated production threshold set."""
    initial_cardinals: tuple[int, ...] = (3, 4)
    maximum_cardinal: int = 5
    target_change_ev: float = 0.002
    max_contraction_ratio: float = 0.8
    max_memory_mb: int = 48000


def basis_specs_for(atoms: Sequence[str], policy: Any, *, element_series=cv_element_series,
                    rule_id: str = BASIS_RULE_ID, make_spec=None
                    ) -> tuple[tuple[Any, ...], dict] | None:
    """Basis specs for every cardinal up to the highest one all elements share."""
    make_spec = make_spec or (lambda **kw: CoreValenceBasisSpec(**kw))
    elements = list(dict.fromkeys(atoms))
    cardinals = tuple(range(policy.initial_cardinals[0], policy.maximum_cardinal + 1))
    while len(cardinals) >= len(policy.initial_cardinals):
        series = {el: element_series(el, cardinals) for el in elements}
        if all(series.values()):
            break
        cardinals = cardinals[:-1]
    else:
        return None
    if cardinals[: len(policy.initial_cardinals)] != policy.initial_cardinals:
        return None
    family = ",".join(f"{el}:" + re.sub(r"V[DTQ56]Z", "VXZ", series[el][cardinals[0]][1]) for el in elements)
    specs, provenance = [], {}
    for x in cardinals:
        mapping = {el: series[el][x][0] for el in elements}
        prov = {el: series[el][x][1] for el in elements}
        provenance[x] = prov
        specs.append(make_spec(cardinal=x, family_id=f"{rule_id}:{family}",
                               label=f"X{x}[" + ";".join(f"{el}={p}" for el, p in prov.items()) + "]",
                               basis_by_element=mapping))
    return tuple(specs), provenance


def _state_spec(role: str, job_id: str, r: float, template: Stage3ExecutionRequest,
                seed: Mapping[str, Any], spec_cls: type = CoreValenceStateSpec) -> Any:
    return spec_cls(
        role=role, system=template.system, atoms=tuple(template.atoms), charge=template.charge,
        spin_2s=template.spin_2s, state_id=job_id, r_angstrom=float(r),
        source_root_id=str(seed.get("request_id") or job_id),
        source_checkpoint_path=str(seed["high_level_checkpoint_path"]),
        state_identity_validated=True, point_group=template.point_group,
        irrep_nelec=None if template.irrep_nelec is None else dict(template.irrep_nelec),
        reference_branch=SEED_CONTINUOUS_BRANCH)


def selected_state_specs(electronic_ea: Mapping[str, Any], requests: Sequence[Stage3ExecutionRequest],
                         point_records: Mapping[str, Mapping[str, Any]], refinement_dir: Path,
                         spec_cls: type = CoreValenceStateSpec) -> dict | str:
    """Selected neutral/anion at their TZ equilibrium, seeded from the nearest
    high-level checkpoint; a reason string when a seed is missing."""
    template: dict[str, Stage3ExecutionRequest] = {}
    for req in requests:
        template.setdefault(req.job_id, req)
    states = {}
    for role in ("neutral", "anion"):
        job_id = electronic_ea[f"selected_{role}_job"]
        cand = next(c for c in electronic_ea["candidates"][role] if c["job_id"] == job_id)
        recs = _job_point_records(job_id, point_records, Path(refinement_dir))
        if job_id not in template or not recs:
            return f"{role}: no seed reference for {job_id}"
        r = float(cand["r_central_angstrom"])
        seed = min(recs, key=lambda rec: abs(float(rec["r_angstrom"]) - r))
        states[role] = _state_spec(role, job_id, r, template[job_id], seed, spec_cls)
    return states


def run_core_valence_stage(*, electronic_ea: Mapping[str, Any], requests: Sequence[Stage3ExecutionRequest],
                           point_records: Mapping[str, Mapping[str, Any]], refinement_dir: Path,
                           stage3_settings: Stage3ExecutionSettings, output_dir: Path,
                           policy: CoreValenceStagePolicy = CoreValenceStagePolicy(),
                           point_backend: Callable[..., Stage3PointResult] | None = None) -> dict:
    output_dir = Path(output_dir)
    record: dict[str, Any] = {"stage": "CORE_VALENCE", "policy": asdict(policy), "is_production_ea": False,
                              "basis_rule": BASIS_RULE_ID, "validation_metadata_used_in_computation": False}

    def done(**fields) -> dict:
        record.update(fields)
        _save_json(output_dir / "core_valence.json", record)
        return record

    if electronic_ea.get("status") != "ELECTRONIC_EA_EVALUATED":
        return done(status="NOT_ATTEMPTED", reason="single-basis electronic EA not evaluated")
    states = selected_state_specs(electronic_ea, requests, point_records, refinement_dir)
    if isinstance(states, str):
        return done(status="NOT_ATTEMPTED", reason=states)
    record["states"] = {role: {k: v for k, v in asdict(s).items() if k != "irrep_nelec"}
                        for role, s in states.items()}

    atoms = states["neutral"].atoms
    built = basis_specs_for(atoms, policy)
    if built is None:
        return done(status="POLICY_BLOCKED_NO_CV_BASIS",
                    reason=f"no core-valence basis series {policy.initial_cardinals} for all of {list(atoms)}")
    specs, provenance = built
    record["basis_provenance"] = {f"X{x}": p for x, p in provenance.items()}

    settings = replace(stage3_settings, scalar_relativistic="NONE", max_memory_mb=policy.max_memory_mb,
                       artifact_dir=str(output_dir / "checkpoints"))
    signature = sha256(json.dumps({"basis_rule": BASIS_RULE_ID}, sort_keys=True).encode()).hexdigest()
    runner = cached_point_runner(output_dir / "points", signature, point_backend)

    # The initial cardinals' eight points are independent: run them on the pool;
    # the series below then reads them from the cache.
    initial = [s for s in specs if s.cardinal in policy.initial_cardinals]
    jobs = [(runner, _request_for(state, spec, space), _settings_for(settings, space))
            for spec in initial for state in states.values() for space in CoreValenceCorrelationSpace]
    for _, req, _ in jobs:
        print(f"[CV] {req.system} {req.job_id}", flush=True)
    run_parallel(_run_point, [(r, q, c) for r, q, c in jobs])

    result = run_adaptive_core_valence_series(
        neutral=states["neutral"], anion=states["anion"], basis_specs=specs,
        initial_cardinals=policy.initial_cardinals, maximum_cardinal=max(s.cardinal for s in specs),
        target_change_ev=policy.target_change_ev, max_contraction_ratio=policy.max_contraction_ratio,
        execution_settings=settings, point_runner=runner)
    assessment = result.assessment
    points = [] if assessment is None else [p.to_dict() for p in assessment.points]
    return done(status=result.status.value,
                delta_cv_ev=None if assessment is None else assessment.central_correction_ev,
                delta_cv_bound_ev=None if assessment is None else assessment.convergence_bound_ev,
                highest_cardinal=None if assessment is None else assessment.highest_cardinal,
                action=None if assessment is None else assessment.action,
                points=points, execution_error=result.execution_error_message,
                point_status={f"X{ev.cardinal}:{c.role}:{c.correlation_space.value}": c.result.status.value
                              for ev in result.evidence for c in ev.calculations})


def _run_point(runner, request, settings):
    return runner(request, settings)
