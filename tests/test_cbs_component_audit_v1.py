from types import SimpleNamespace

from openea_benchmark.attachment.cbs_component_audit import (
    ComponentAvailability,
    audit_cbs_components,
    extract_loop_component_evidence,
    summarize_basis_components,
)


def enumlike(value):
    return SimpleNamespace(value=value)


def result(request_id, basis="aug-cc-pv5z", complete=True):
    return SimpleNamespace(
        request_id=request_id,
        basis=basis,
        r_angstrom=0.97,
        status=enumlike("COMPLETED" if complete else "SCF_NOT_CONVERGED"),
        scf_energy_hartree=-75.0 if complete else -74.9,
        ccsd_correlation_hartree=-0.2 if complete else None,
        ccsd_total_hartree=-75.2 if complete else None,
        triples_correction_hartree=-0.01 if complete else None,
        ccsd_t_total_hartree=-75.21 if complete else None,
    )


def loop(role_basis="aug-cc-pv5z", complete=True):
    rid = "req-min"
    pec_point = SimpleNamespace(
        canonical_request_id=rid,
        status=enumlike("ACCEPTED"),
    )
    candidate = SimpleNamespace(canonical_request_id=rid)
    pec = SimpleNamespace(
        points=(pec_point,),
        minimum_scout=SimpleNamespace(candidates=(candidate,)),
    )
    return SimpleNamespace(
        status=enumlike("CONVERGED"),
        results=(result(rid, role_basis, complete),),
        final_pec=pec,
    )


def test_complete_component_point_is_recognized(tmp_path):
    evidence = extract_loop_component_evidence(
        loop(),
        source_path=tmp_path / "neutral_loop.pkl",
        role="neutral",
    )
    assert evidence.points[0].component_availability is ComponentAvailability.COMPLETE
    assert evidence.minimum_candidate_request_ids == ("req-min",)


def test_missing_triples_is_partial(tmp_path):
    obj = loop()
    p = obj.results[0]
    p.triples_correction_hartree = None
    evidence = extract_loop_component_evidence(
        obj,
        source_path=tmp_path / "neutral_loop.pkl",
        role="neutral",
    )
    assert evidence.points[0].component_availability is ComponentAvailability.PARTIAL


def test_basis_summary_requires_neutral_and_anion_minimum_components(tmp_path):
    neutral = extract_loop_component_evidence(
        loop(),
        source_path=tmp_path / "neutral_loop.pkl",
        role="neutral",
    )
    anion = extract_loop_component_evidence(
        loop(),
        source_path=tmp_path / "anion_loop.pkl",
        role="anion",
    )
    summaries = summarize_basis_components((neutral, anion))
    assert summaries[0].ready_for_component_extraction is True


def test_daug_does_not_substitute_for_aug_cardinal_series(tmp_path):
    run = tmp_path / "run"
    stage = run / "basis_stage_checkpoints" / "d-aug-cc-pv5z"
    stage.mkdir(parents=True)
    import pickle
    for role in ("neutral", "anion"):
        with (stage / f"{role}_loop.pkl").open("wb") as handle:
            pickle.dump(loop("d-aug-cc-pv5z"), handle)

    audit = audit_cbs_components(run_dirs=(run,))
    assert audit.diffuse_reference_available is True
    assert audit.missing_required_bases == (
        "aug-cc-pvqz",
        "aug-cc-pv5z",
    )
    assert "DO_NOT_EXTRAPOLATE_CBS_YET" in audit.next_actions


def test_required_qz_5z_ready_enables_cbs_action(tmp_path):
    import pickle
    run = tmp_path / "run"
    for basis in ("aug-cc-pvqz", "aug-cc-pv5z"):
        stage = run / "basis_stage_checkpoints" / basis
        stage.mkdir(parents=True)
        for role in ("neutral", "anion"):
            with (stage / f"{role}_loop.pkl").open("wb") as handle:
                pickle.dump(loop(basis), handle)

    audit = audit_cbs_components(run_dirs=(run,))
    assert audit.missing_required_bases == ()
    assert "BUILD_COMPONENT_RESOLVED_CBS_EXTRAPOLATION" in audit.next_actions


def test_audit_never_claims_production_ea(tmp_path):
    audit = audit_cbs_components(run_dirs=(tmp_path,))
    assert audit.is_production_ea is False
