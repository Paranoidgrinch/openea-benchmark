import json
from dataclasses import replace
import pytest

from openea_benchmark.adaptive.attachment_eom_runner import (
    DiffuseShellSelector, G2EOMAuthorization, G2EOMBasis, G2EOMNeutralState,
    G2EOMSettings, G2EOMStabilization, G2EOMStatus, G2EOMRoot,
    G2EOMRawResult, evidence_points_from_review, run_g2_eom_diagnostics,
    scaled_diffuse_basis,
)
from openea_benchmark.adaptive.attachment_continuum import (
    AttachmentContinuumSettings, EOMEAAssessmentStatus, assess_eom_ea_diffuse_series,
)
from openea_benchmark.adaptive.model import Review, ReviewStatus


def reference(tmp_path):
    cp = tmp_path / "n.chk"
    if not cp.exists():
        cp.write_text("neutral checkpoint")
    return G2EOMNeutralState("XY", ("X", "Y"), 0, 0, "N", 1.2, "NROOT",
                              str(cp), True, True, "RHF")


def bases():
    return tuple(G2EOMBasis(i, f"aug{i}", "FAMILY", {"X": f"x{i}", "Y": f"y{i}"})
                 for i in range(3))


def auth(keys=("aug:0", "aug:1", "aug:2"), approved=True):
    return G2EOMAuthorization(approved, "Scientifically motivated D08 review",
                              ("G2-REVIEW",), keys)


def loader(element, basis):
    # S (uncontracted) and P (uncontracted) shells with different exponents.
    number = int(basis[-1]) if basis[-1:].isdigit() else 0
    return [[0, [0.08 + 0.01 * number, 1.0]], [1, [0.04 + 0.005 * number, 1.0]]]


def fake_backend(calls):
    def run(req, settings, material):
        calls.append((req.key, material))
        levels = {0: -0.10, 1: -0.11, 2: -0.114}
        omega = levels[req.basis.augmentation_level]
        if req.scale_factor is not None:
            omega += (req.scale_factor - 1.0) * 0.001
        return G2EOMRawResult(req.key,
                              (G2EOMRoot(0, omega, -omega * 27.211386245988),
                               G2EOMRoot(1, omega + 0.2, -(omega + 0.2) * 27.211386245988),
                               G2EOMRoot(2, omega + 0.4, -(omega + 0.4) * 27.211386245988)),
                              req.state.scf_reference, "FAKE", True, True, True)
    return run


def run(tmp_path, **kwargs):
    calls = []
    opts = dict(authorization=auth(), neutral=reference(tmp_path), basis_specs=bases(),
                backend=fake_backend(calls), basis_loader=loader,
                checkpoint_dir=tmp_path / "points", backend_id="FAKE-BACKEND-v1")
    opts.update(kwargs)
    return run_g2_eom_diagnostics(**opts), calls


def test_runs_three_root_spectra_with_no_implicit_boundness(tmp_path):
    result, calls = run(tmp_path)
    assert result.status is G2EOMStatus.COMPLETE_ROOT_REVIEW_REQUIRED
    assert len(result.subpoints) == len(calls) == 3
    assert result.method_role == "DIAGNOSTIC"
    assert not result.is_production_ea
    assert all(not x.checkpoint_reused for x in result.subpoints)
    assert [r.attachment_ea_ev for r in result.subpoints[0].result.roots][:1] == pytest.approx([2.7211386245988])


def test_restart_reads_identical_checkpoints_without_compute(tmp_path):
    result, calls = run(tmp_path)
    assert len(calls) == 3
    new, newcalls = run(tmp_path)
    assert len(newcalls) == 0
    assert all(x.checkpoint_reused for x in new.subpoints)
    assert result.subpoints[0].evidence_id == new.subpoints[0].evidence_id


def test_checkpoint_rejects_source_content_change(tmp_path):
    old, _ = run(tmp_path)
    st = reference(tmp_path)
    from pathlib import Path
    Path(st.source_checkpoint_path).write_text("changed source")
    with pytest.raises(ValueError, match="Incompatible G2 EOM checkpoint"):
        run(tmp_path)


def test_checkpoint_rejects_method_settings_change(tmp_path):
    run(tmp_path)
    with pytest.raises(ValueError, match="Incompatible G2 EOM checkpoint"):
        run(tmp_path, settings=G2EOMSettings(nroots=4))


def test_no_compute_without_approval(tmp_path):
    result, calls = run(tmp_path, authorization=auth(approved=False))
    assert result.status is G2EOMStatus.POLICY_BLOCKED
    assert calls == []


def test_unapproved_point_can_only_be_reused(tmp_path):
    result, _ = run(tmp_path)
    restored, calls = run(tmp_path, authorization=auth(keys=("aug:2",)))
    assert restored.status is G2EOMStatus.COMPLETE_ROOT_REVIEW_REQUIRED
    assert len(calls) == 0
    (tmp_path / "points" / "aug_0.json").unlink()
    blocked, calls = run(tmp_path, authorization=auth(keys=("aug:2",)))
    assert blocked.status is G2EOMStatus.POLICY_BLOCKED
    assert not calls


def test_requires_all_basis_labels_and_atom_mapping_before_compute(tmp_path):
    broken = (G2EOMBasis(0, "0", "FAMILY", {"X": "onlyX"}),)
    with pytest.raises(ValueError, match="all molecular elements"):
        run(tmp_path, basis_specs=broken)


def test_augmentation_level_unique_and_ordered(tmp_path):
    with pytest.raises(ValueError, match="unique and sorted"):
        run(tmp_path, basis_specs=bases()[::-1])


def test_bad_stabilization_shell_blocks_before_computation(tmp_path):
    scan = (G2EOMStabilization(2, 0.8, (DiffuseShellSelector("X", 20),)),)
    with pytest.raises(ValueError, match="out of range"):
        run(tmp_path, stabilization_specs=scan)
    assert not list((tmp_path / "points").glob("*.json")) if (tmp_path / "points").exists() else True


def test_diffuse_exponent_scaling_only_selected_shell():
    generated = scaled_diffuse_basis({"X": "x", "Y": "y"},
                        (DiffuseShellSelector("X", 0),), 0.5, loader=loader)
    assert generated["X"][0][1][0] == pytest.approx(0.04)
    assert generated["X"][1][1][0] == pytest.approx(0.04)
    assert generated["Y"][0][1][0] == pytest.approx(0.08)


def test_reject_scaling_contracted_shell():
    def contracted(a, b):
        return [[0, [0.08, 0.5], [0.03, 0.5]]]
    with pytest.raises(ValueError, match="uncontracted"):
        scaled_diffuse_basis({"X": "x"}, (DiffuseShellSelector("X", 0),), 0.5,
                             loader=contracted)


def test_stabilization_points_require_explicit_authorization(tmp_path):
    scan = (G2EOMStabilization(2, 0.8, (DiffuseShellSelector("X", 0),)),)
    blocked, calls = run(tmp_path, stabilization_specs=scan)
    assert blocked.status is G2EOMStatus.POLICY_BLOCKED
    assert len(calls) == 3  # explicitly authorized augmentation jobs only
    series, calls = run(tmp_path, stabilization_specs=scan,
                        authorization=auth(keys=("scale:2:0.8",)))
    assert series.status is G2EOMStatus.COMPLETE_ROOT_REVIEW_REQUIRED
    assert len(calls) == 1
    assert series.subpoints[-1].request.key == "scale:2:0.8"


def test_independent_identity_review_required_before_g2_assessment(tmp_path):
    series, _ = run(tmp_path)
    pending = {p.request.key: (0, Review(ReviewStatus.UNRESOLVED, (), "unreviewed"))
               for p in series.subpoints}
    eom, stab = evidence_points_from_review(series, selections=pending)
    assert stab == ()
    assert assess_eom_ea_diffuse_series(eom, settings=AttachmentContinuumSettings()).status is EOMEAAssessmentStatus.UNRESOLVED
    assert all(p.state_identity.status is ReviewStatus.UNRESOLVED for p in eom)


def test_review_does_not_silently_approve_self_citing_root(tmp_path):
    series, _ = run(tmp_path)
    p = series.subpoints[0]
    selections = {s.request.key: (0, Review(ReviewStatus.CLEARED, (s.evidence_id,), "I declare myself cleared"))
                  for s in series.subpoints}
    with pytest.raises(ValueError, match="separate reviewed evidence"):
        evidence_points_from_review(series, selections=selections)


def test_root_selection_from_review_can_convert_verified_evidence(tmp_path):
    series, _ = run(tmp_path)
    selections = {s.request.key: (0, Review(ReviewStatus.CLEARED, (f"root-overlap:{s.request.key}",), "identity reviewed"))
                  for s in series.subpoints}
    eom, stab = evidence_points_from_review(series, selections=selections)
    assert len(eom) == 3 and not stab
    assert all(p.state_identity.status is ReviewStatus.CLEARED for p in eom)


def test_missing_root_index_rejected(tmp_path):
    series, _ = run(tmp_path)
    selections = {s.request.key: (99, Review(ReviewStatus.UNRESOLVED, (), "pending"))
                  for s in series.subpoints}
    with pytest.raises(ValueError, match="does not exist"):
        evidence_points_from_review(series, selections=selections)


def test_reference_gate_blocks_unvalidated_spin_or_reference(tmp_path):
    state = reference(tmp_path)
    with pytest.raises(ValueError, match="neutral state and SR"):
        replace(state, reference_character_validated=False)
    with pytest.raises(ValueError, match="RHF for singlet"):
        replace(state, scf_reference="UHF")
    with pytest.raises(ValueError, match="RHF for singlet"):
        replace(state, spin_2s=1)


def test_input_basis_policy_must_reject_ecp():
    with pytest.raises(ValueError, match="ECP"):
        replace(bases()[0], electron_model="ECP")


def test_no_automatic_one_particle_weight_is_invented(tmp_path):
    result, _ = run(tmp_path)
    review = {s.request.key: (0, Review(ReviewStatus.UNRESOLVED, (), "pending"))
              for s in result.subpoints}
    eom, _ = evidence_points_from_review(result, selections=review)
    assert all(p.one_particle_weight is None for p in eom)


def test_checkpoint_payload_carries_request_signature(tmp_path):
    series, _ = run(tmp_path)
    raw = json.loads((tmp_path / "points" / "aug_0.json").read_text())
    assert raw["schema"] == "G2_EOM_RAW_V1"
    assert raw["signature"] in series.subpoints[0].evidence_id
    assert raw["result"]["roots"][0]["omega_hartree"] == pytest.approx(-0.1)


def test_duplicate_physical_basis_across_augmentation_rejected(tmp_path):
    def identical(a, b):
        return [[0, [0.08, 1.0]]]
    calls = []
    with pytest.raises(ValueError, match="identical orbital bases"):
        run(tmp_path, basis_loader=identical, backend=fake_backend(calls))
    assert calls == []


def test_stabilization_must_keep_one_fixed_augmentation_baseline(tmp_path):
    selections = (G2EOMStabilization(1, 0.8, (DiffuseShellSelector("X", 0),)),
                  G2EOMStabilization(2, 1.2, (DiffuseShellSelector("X", 0),)))
    with pytest.raises(ValueError, match="one fixed"):
        run(tmp_path, stabilization_specs=selections)


def test_failed_backend_returns_blocked_status_not_g2_result(tmp_path):
    def fails(request, settings, basis):
        raise RuntimeError("synthetic SCF failed")
    result, _ = run(tmp_path, backend=fails)
    assert result.status is G2EOMStatus.EXECUTION_BLOCKED
    assert "synthetic SCF failed" in result.notes[0]
    assert not (tmp_path / "points" / "aug_0.json").exists()


def test_change_backend_version_invalidates_checkpoint(tmp_path):
    result, calls = run(tmp_path)
    with pytest.raises(ValueError, match="Incompatible G2 EOM checkpoint"):
        run(tmp_path, backend_id="FAKE-BACKEND-v2")


def test_fake_backend_without_version_tag_is_rejected(tmp_path):
    state = reference(tmp_path)
    with pytest.raises(ValueError, match="requires explicit backend_id"):
        run_g2_eom_diagnostics(authorization=auth(), neutral=state,
                               basis_specs=bases(), backend=fake_backend([]),
                               basis_loader=loader)
