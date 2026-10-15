"""Automated scalar-relativity layer (Patch 48).

    Delta_SR = EA_SFX2C1E(X) - EA_NR(X)

all-electron CCSD(T) at the TZ equilibrium of the selected states; the two
members of a pair differ only in the one-electron Hamiltonian (spin-free X2C).
Execution reuses ``adaptive.scalar_relativity_runner`` (X=T,Q first, X=5 when
the change exceeds 0.5 meV; highest X central, last change the bound).

Basis rule (per element, ``core_valence_basis.sr_element_series``): SFX2C1E in
the Douglas-Kroll recontracted aug-cc-pVXZ-DK, NR in aug-cc-pVXZ (same primitives,
NR contraction); else the cc-pVXZ-DK / cc-pVXZ pair with the diffuse primitives
of aug-cc-pVXZ.  A DK contraction used for the NR member gave NR energies rising
with X (MgH) and a 0.02 eV spurious 5Z change (ClO).  States carry their symmetry block and the
seed-continuous reference branch, as in the CBS and CV layers.

The result is the one-electron spin-free X2C correction only; spin-orbit is a
separate layer and two-electron relativistic terms are not included.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .adaptive.scalar_relativity_runner import (
    ScalarRelativityBasisSpec, ScalarRelativityHamiltonian, ScalarRelativityStateSpec, _request_for, _settings_for,
    run_adaptive_scalar_relativity_series,
)
from .adaptive.stage3_execution import Stage3ExecutionRequest, Stage3ExecutionSettings, Stage3PointResult
from .core_valence_basis import sr_element_series
from .core_valence_stage import _run_point, basis_specs_for, selected_state_specs
from .electronic_ea_stage import _save_json, cached_point_runner
from .point_pool import run_parallel

BASIS_RULE_ID = "OPENEA_SR_BASIS_RULE_V2"  # V2: NR member in the nonrelativistic partner family


@dataclass(frozen=True)
class ScalarRelativityStagePolicy:
    """Explicit pilot policy, not a validated production threshold set."""
    initial_cardinals: tuple[int, ...] = (3, 4)
    maximum_cardinal: int = 5
    target_change_ev: float = 0.0005
    max_memory_mb: int = 48000


def _sr_spec(*, cardinal, family_id, label, basis_by_element):
    """Split the (DK, NR) element pairs into the two members' basis mappings."""
    return ScalarRelativityBasisSpec(
        cardinal=cardinal, family_id=family_id, label=label, scalar_relativistically_recontracted=True,
        basis_by_element={el: pair[0] for el, pair in basis_by_element.items()},
        nonrelativistic_basis_by_element={el: pair[1] for el, pair in basis_by_element.items()})


def run_scalar_relativity_stage(*, electronic_ea: Mapping[str, Any], requests: Sequence[Stage3ExecutionRequest],
                                point_records: Mapping[str, Mapping[str, Any]], refinement_dir: Path,
                                stage3_settings: Stage3ExecutionSettings, output_dir: Path,
                                policy: ScalarRelativityStagePolicy = ScalarRelativityStagePolicy(),
                                point_backend: Callable[..., Stage3PointResult] | None = None) -> dict:
    output_dir = Path(output_dir)
    record: dict[str, Any] = {"stage": "SCALAR_RELATIVITY_SFX2C1E", "policy": asdict(policy),
                              "is_production_ea": False, "basis_rule": BASIS_RULE_ID,
                              "includes_two_electron_relativistic_terms": False,
                              "validation_metadata_used_in_computation": False}

    def done(**fields) -> dict:
        record.update(fields)
        _save_json(output_dir / "scalar_relativity.json", record)
        return record

    if electronic_ea.get("status") != "ELECTRONIC_EA_EVALUATED":
        return done(status="NOT_ATTEMPTED", reason="single-basis electronic EA not evaluated")
    states = selected_state_specs(electronic_ea, requests, point_records, refinement_dir,
                                  spec_cls=ScalarRelativityStateSpec)
    if isinstance(states, str):
        return done(status="NOT_ATTEMPTED", reason=states)
    record["states"] = {role: {k: v for k, v in asdict(s).items() if k != "irrep_nelec"}
                        for role, s in states.items()}
    from .heavy_elements import requires_x2c
    if requires_x2c(states["neutral"].atoms):  # every stage already runs sfX2C (heavy_elements)
        return done(status="SCALAR_RELATIVITY_CLEARED", reason="INCLUDED_IN_MODEL_HAMILTONIAN",
                    delta_sr_ev=0.0, delta_sr_bound_ev=0.0)
    built = basis_specs_for(states["neutral"].atoms, policy, element_series=sr_element_series,
                            rule_id=BASIS_RULE_ID,
                            make_spec=_sr_spec)
    if built is None:
        return done(status="POLICY_BLOCKED_NO_SR_BASIS",
                    reason=f"no DK-recontracted basis series {policy.initial_cardinals} for all atoms")
    specs, provenance = built
    record["basis_provenance"] = {f"X{x}": p for x, p in provenance.items()}

    settings = replace(stage3_settings, frozen_core=False, scalar_relativistic="NONE",
                       max_memory_mb=policy.max_memory_mb, artifact_dir=str(output_dir / "checkpoints"))
    signature = sha256(json.dumps({"basis_rule": BASIS_RULE_ID}, sort_keys=True).encode()).hexdigest()
    runner = cached_point_runner(output_dir / "points", signature, point_backend)
    initial = [s for s in specs if s.cardinal in policy.initial_cardinals]
    jobs = [(runner, _request_for(state, spec, ham), _settings_for(settings, ham))
            for spec in initial for state in states.values() for ham in ScalarRelativityHamiltonian]
    for _, req, _ in jobs:
        print(f"[SR] {req.system} {req.job_id}", flush=True)
    run_parallel(_run_point, jobs)

    result = run_adaptive_scalar_relativity_series(
        neutral=states["neutral"], anion=states["anion"], basis_specs=specs,
        initial_cardinals=policy.initial_cardinals, maximum_cardinal=max(s.cardinal for s in specs),
        target_change_ev=policy.target_change_ev, execution_settings=settings, point_runner=runner,
        basis_validator=lambda element, basis: None)  # rule-built shells, loaded above
    assessment = result.assessment
    return done(status=result.status.value,
                delta_sr_ev=None if assessment is None else assessment.central_correction_ev,
                delta_sr_bound_ev=None if assessment is None else assessment.convergence_bound_ev,
                highest_cardinal=None if assessment is None else getattr(assessment, "highest_cardinal", None),
                action=None if assessment is None else getattr(assessment, "action", None),
                execution_error=result.execution_error_message)
