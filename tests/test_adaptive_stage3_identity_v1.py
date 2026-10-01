from types import SimpleNamespace

import numpy as np

from openea_benchmark.branch_continuity import BranchComparison, BranchRelation, BranchThresholds
from openea_benchmark.checkpoint_fingerprint import CheckpointAuditSettings
from openea_benchmark.root_record import SCFRootRecord, SCFRunStatus
from openea_benchmark.state_identity import (
    IdentityThresholds,
    StateComparison,
    StateFingerprint,
    StateRelation,
)
from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus,
    Stage3ExecutionRequest,
    Stage3PointResult,
)
from openea_benchmark.adaptive.stage3_identity import (
    HighLevelCheckpointAudit,
    _CheckpointEvidence,
    _principal_values,
    _spin_resolved_hf_density,
    review_high_level_geometry_continuity,
    review_high_level_initializations,
)
from openea_benchmark.adaptive.stage3_pec import IdentityReviewStatus, StateIdentityReview


IDENTITY = IdentityThresholds(
    same_energy_mev=0.5,
    same_delta_s2=1.0e-5,
    same_total_spectrum_max=1.0e-5,
    same_spin_spectrum_max=1.0e-5,
    same_total_density_rel_fro=1.0e-5,
    same_spin_density_rel_fro=1.0e-5,
    distinct_energy_mev=10.0,
    distinct_delta_s2=0.10,
    distinct_total_spectrum_max=1.0e-2,
    distinct_spin_spectrum_max=1.0e-2,
    distinct_total_density_rel_fro=1.0e-2,
    distinct_spin_density_rel_fro=1.0e-2,
)

BRANCH = BranchThresholds(
    continuous_occ_min=0.95,
    discontinuous_occ_min=0.50,
    continuous_delta_s2=0.02,
    discontinuous_delta_s2=0.20,
    max_step_angstrom=0.10,
)


def request(r=1.0, init=0, root="root_A"):
    return Stage3ExecutionRequest(
        request_id=f"job__init{init:02d}_{root}__r{r:.3f}",
        job_id="job",
        system="OH",
        atoms=("O", "H"),
        charge=0,
        spin_2s=1,
        component_id="component_A",
        r_angstrom=r,
        basis="sto-3g",
        methods=("CCSD", "CCSD(T)"),
        requested_reference="ROHF",
        scf_reference="ROHF",
        source_link_status="MULTIPLE_DFT_INITIALIZATIONS",
        source_root_id=root,
        source_checkpoint_path=f"/tmp/{root}.chk",
        source_origin_guess="minao",
        grid_index=0,
        initialization_index=init,
        dft_center_r_angstrom=1.0,
        dft_center_energy_hartree=-75.0,
        requires_independent_state_identity_validation=True,
    )


def result(req, *, checkpoint="/tmp/hl.chk"):
    return Stage3PointResult(
        request_id=req.request_id,
        job_id=req.job_id,
        status=PointExecutionStatus.COMPLETED,
        system=req.system,
        charge=req.charge,
        spin_2s=req.spin_2s,
        component_id=req.component_id,
        r_angstrom=req.r_angstrom,
        basis=req.basis,
        source_root_id=req.source_root_id,
        source_checkpoint_path=req.source_checkpoint_path,
        scf_reference=req.scf_reference,
        cc_reference="SEMICANONICAL_UHF_FROM_ROHF",
        pyscf_version="2.14.0",
        scf_converged=True,
        scf_energy_hartree=-74.0,
        s2=0.75,
        multiplicity=2.0,
        internal_stable=True,
        external_stable=None,
        external_stability_available=False,
        semicanonicalization="UHF_OCCUPIED_VIRTUAL_CANONICALIZATION",
        cc_class="UCCSD",
        ccsd_converged=True,
        ccsd_correlation_hartree=-0.1,
        ccsd_total_hartree=-74.1,
        triples_correction_hartree=-0.001,
        ccsd_t_total_hartree=-74.101,
        t1_diagnostic=None,
        t1_diagnostic_definition=None,
        error_type=None,
        error_message=None,
        high_level_checkpoint_path=checkpoint,
    )


def root(req):
    return SCFRootRecord(
        root_id=req.request_id,
        molecule=req.system,
        atom_a=req.atoms[0],
        atom_b=req.atoms[1],
        charge=req.charge,
        spin_2s=req.spin_2s,
        r_angstrom=req.r_angstrom,
        functional="HF_STAGE3",
        basis=req.basis,
        origin_guess=req.source_root_id,
        scf_path="stage3_high_level_hf",
        reference=req.scf_reference,
        status=SCFRunStatus.CANONICALIZED,
        energy_hartree=-74.0,
        internal_stable=True,
        s2=0.75,
        observed_multiplicity=2.0,
        checkpoint_path="/tmp/hl.chk",
    )


def fingerprint():
    return StateFingerprint(
        total_spectrum=(2.0, 1.0),
        spin_spectrum=(1.0, 0.0),
        total_density_orth=((1.0, 0.0), (0.0, 1.0)),
        spin_density_orth=((1.0, 0.0), (0.0, 0.0)),
        total_trace=3.0,
        spin_trace=1.0,
    )


def audit(req):
    return HighLevelCheckpointAudit(
        request_id=req.request_id,
        checkpoint_path="/tmp/hl.chk",
        reference=req.scf_reference,
        checkpoint_energy_hartree=-74.0,
        result_energy_hartree=-74.0,
        energy_delta_hartree=0.0,
        checkpoint_r_angstrom=req.r_angstrom,
        request_r_angstrom=req.r_angstrom,
        geometry_delta_angstrom=0.0,
        electron_count=9.0,
        expected_electron_count=9,
        spin_population=1.0,
        expected_spin_2s=1,
        overlap_min_eigenvalue=0.5,
        overlap_max_eigenvalue=1.5,
    )


def evidence(req):
    return _CheckpointEvidence(
        request=req,
        result=result(req),
        root=root(req),
        fingerprint=fingerprint(),
        mol=SimpleNamespace(),
        alpha_occ_coeff=np.eye(2),
        beta_occ_coeff=np.eye(2),
        audit=audit(req),
    )


def state_comparison(a, b, relation):
    return StateComparison(
        root_a=a.request_id,
        root_b=b.request_id,
        delta_energy_mev=0.0,
        delta_s2=0.0,
        total_spectrum_max=0.0,
        total_spectrum_l2=0.0,
        spin_spectrum_max=0.0,
        spin_spectrum_l2=0.0,
        total_density_rel_fro=0.0,
        spin_density_rel_fro=0.0,
        relation=relation,
    )


def branch_comparison(a, b, relation):
    return BranchComparison(
        root_a=a.request_id,
        root_b=b.request_id,
        r_a_angstrom=a.r_angstrom,
        r_b_angstrom=b.r_angstrom,
        delta_r_angstrom=abs(b.r_angstrom - a.r_angstrom),
        delta_energy_mev=0.0,
        delta_s2=0.0,
        alpha_singular_values=(0.99,),
        beta_singular_values=(0.99,),
        alpha_occ_overlap_min=0.99,
        alpha_occ_overlap_mean=0.99,
        beta_occ_overlap_min=0.99,
        beta_occ_overlap_mean=0.99,
        relation=relation,
    )


def test_rohf_density_split_uses_double_and_single_occupations():
    coeff = np.eye(3)
    alpha, beta = _spin_resolved_hf_density(coeff, np.array([2.0, 1.0, 0.0]))
    assert np.allclose(np.diag(alpha), [1.0, 1.0, 0.0])
    assert np.allclose(np.diag(beta), [1.0, 0.0, 0.0])


def test_rhf_density_split_gives_equal_alpha_beta():
    coeff = np.eye(3)
    alpha, beta = _spin_resolved_hf_density(coeff, np.array([2.0, 2.0, 0.0]))
    assert np.allclose(alpha, beta)


def test_bad_hf_occupation_is_rejected():
    try:
        _spin_resolved_hf_density(np.eye(2), np.array([2.5, 0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid occupation should fail")


def test_principal_values_are_subspace_not_energy_metrics():
    vals = _principal_values(np.eye(2), np.eye(2), np.eye(2))
    assert vals == (1.0, 1.0)


def test_initialization_same_state_clears(monkeypatch):
    a, b = request(1.0, 0, "A"), request(1.0, 1, "B")
    lookup = {a.request_id: evidence(a), b.request_id: evidence(b)}
    monkeypatch.setattr(
        "openea_benchmark.adaptive.stage3_identity._load_high_level_checkpoint",
        lambda req, res, audit_settings: lookup[req.request_id],
    )
    monkeypatch.setattr(
        "openea_benchmark.adaptive.stage3_identity.compare_states",
        lambda ar, af, br, bf, thresholds: state_comparison(a, b, StateRelation.SAME_STATE),
    )
    review, comparisons, audits, _ = review_high_level_initializations(
        requests=[a, b],
        results=[result(a), result(b)],
        identity_thresholds=IDENTITY,
        audit_settings=CheckpointAuditSettings(),
    )
    assert review.status is IdentityReviewStatus.CLEARED
    assert len(comparisons) == 1
    assert len(audits) == 2


def test_initialization_distinct_state_fails_closed(monkeypatch):
    a, b = request(1.0, 0, "A"), request(1.0, 1, "B")
    lookup = {a.request_id: evidence(a), b.request_id: evidence(b)}
    monkeypatch.setattr(
        "openea_benchmark.adaptive.stage3_identity._load_high_level_checkpoint",
        lambda req, res, audit_settings: lookup[req.request_id],
    )
    monkeypatch.setattr(
        "openea_benchmark.adaptive.stage3_identity.compare_states",
        lambda ar, af, br, bf, thresholds: state_comparison(a, b, StateRelation.DISTINCT_STATE),
    )
    review, _, _, _ = review_high_level_initializations(
        requests=[a, b], results=[result(a), result(b)], identity_thresholds=IDENTITY
    )
    assert review.status is IdentityReviewStatus.UNRESOLVED
    assert "DISTINCT_STATE" in review.rationale


def test_initialization_ambiguous_fails_closed(monkeypatch):
    a, b = request(1.0, 0, "A"), request(1.0, 1, "B")
    lookup = {a.request_id: evidence(a), b.request_id: evidence(b)}
    monkeypatch.setattr(
        "openea_benchmark.adaptive.stage3_identity._load_high_level_checkpoint",
        lambda req, res, audit_settings: lookup[req.request_id],
    )
    monkeypatch.setattr(
        "openea_benchmark.adaptive.stage3_identity.compare_states",
        lambda ar, af, br, bf, thresholds: state_comparison(a, b, StateRelation.AMBIGUOUS),
    )
    review, _, _, _ = review_high_level_initializations(
        requests=[a, b], results=[result(a), result(b)], identity_thresholds=IDENTITY
    )
    assert review.status is IdentityReviewStatus.UNRESOLVED


def test_missing_checkpoint_audit_fails_closed(monkeypatch):
    a = request()
    monkeypatch.setattr(
        "openea_benchmark.adaptive.stage3_identity._load_high_level_checkpoint",
        lambda req, res, audit_settings: (_ for _ in ()).throw(ValueError("missing")),
    )
    review, _, _, _ = review_high_level_initializations(
        requests=[a], results=[result(a)], identity_thresholds=IDENTITY
    )
    assert review.status is IdentityReviewStatus.UNRESOLVED
    assert "audit failed" in review.rationale


def test_continuity_waits_for_initialization_clearance():
    a, b = request(0.95), request(1.00)
    pending = StateIdentityReview(
        status=IdentityReviewStatus.UNRESOLVED,
        covered_request_ids=(),
        evidence_ids=(),
        rationale="not resolved",
    )
    review, comps = review_high_level_geometry_continuity(
        requests=[a, b], evidence={}, initialization_review=pending, branch_thresholds=BRANCH
    )
    assert review.status is IdentityReviewStatus.UNRESOLVED
    assert comps == ()


def test_continuous_adjacent_points_clear(monkeypatch):
    a, b, c = request(0.95), request(1.00), request(1.05)
    ev = {x.request_id: evidence(x) for x in (a, b, c)}
    init = StateIdentityReview(
        status=IdentityReviewStatus.CLEARED,
        covered_request_ids=tuple(x.request_id for x in (a, b, c)),
        evidence_ids=("init:ok",),
        rationale="same-state initializations cleared",
    )
    monkeypatch.setattr(
        "openea_benchmark.adaptive.stage3_identity._compare_high_level_geometry",
        lambda left, right, thresholds: branch_comparison(
            left.request, right.request, BranchRelation.CONTINUOUS
        ),
    )
    review, comps = review_high_level_geometry_continuity(
        requests=[a, b, c], evidence=ev, initialization_review=init, branch_thresholds=BRANCH
    )
    assert review.status is IdentityReviewStatus.CLEARED
    assert len(comps) == 2


def test_ambiguous_continuity_fails_closed(monkeypatch):
    a, b = request(0.95), request(1.00)
    ev = {x.request_id: evidence(x) for x in (a, b)}
    init = StateIdentityReview(
        status=IdentityReviewStatus.CLEARED,
        covered_request_ids=(a.request_id, b.request_id),
        evidence_ids=("init:ok",),
        rationale="cleared",
    )
    monkeypatch.setattr(
        "openea_benchmark.adaptive.stage3_identity._compare_high_level_geometry",
        lambda left, right, thresholds: branch_comparison(
            left.request, right.request, BranchRelation.AMBIGUOUS
        ),
    )
    review, _ = review_high_level_geometry_continuity(
        requests=[a, b], evidence=ev, initialization_review=init, branch_thresholds=BRANCH
    )
    assert review.status is IdentityReviewStatus.UNRESOLVED
    assert "AMBIGUOUS" in review.rationale
