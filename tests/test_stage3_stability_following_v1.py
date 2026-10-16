"""D02 stability following, quantified RHF->UHF instability and partial PEC reviews."""
from pathlib import Path

import pytest

from openea_benchmark.adaptive.stage3_execution import (
    PointExecutionStatus, RHF_EXTERNAL_INSTABILITY_WARNING, Stage3ExecutionRequest,
    Stage3ExecutionSettings, Stage3PointResult, _empty_result, run_stage3_point,
)
from openea_benchmark.benchmark_auto import (
    AcquisitionSettings, _reference_warning_summary, run_benchmark_acquisition,
)
from openea_benchmark.workflow import build_parser

from test_benchmark_auto_v1 import fixture


def test_settings_reject_invalid_stability_policy():
    with pytest.raises(ValueError):
        Stage3ExecutionSettings(stability_following_max_steps=-1)
    with pytest.raises(ValueError):
        Stage3ExecutionSettings(rhf_external_instability_policy="SWITCH_TO_UHF")
    with pytest.raises(ValueError):
        AcquisitionSettings(stability_following_max_steps=-1)


def test_legacy_defaults_keep_fail_closed_behaviour():
    settings = Stage3ExecutionSettings()
    assert settings.stability_following_max_steps == 0
    assert settings.rhf_external_instability_policy == "FAIL"


def test_cli_enables_following_and_separate_acquisition_dir():
    args = build_parser().parse_args(["--atoms", "H", "F"])
    assert args.stability_following_steps == 4
    assert args.rhf_external_instability == "RECORD_AND_CONTINUE"
    assert args.acquisition_dir == "benchmark_auto"


def _request(**over):
    data = dict(
        request_id="r", job_id="j", system="HX", atoms=("H", "H"), charge=0, spin_2s=0,
        component_id="c", r_angstrom=1.0, basis="sto-3g", methods=("CCSD", "CCSD(T)"),
        requested_reference="ROHF", scf_reference="RHF",
        source_link_status="SINGLE_DFT_INITIALIZATION", source_root_id="root",
        source_checkpoint_path="x.chk", source_origin_guess=None, grid_index=0,
        initialization_index=0, dft_center_r_angstrom=1.0, dft_center_energy_hartree=-1.0,
        requires_independent_state_identity_validation=True,
    )
    data.update(over)
    return Stage3ExecutionRequest(**data)


def test_result_warning_round_trip_from_json_list():
    result = _empty_result(_request(), PointExecutionStatus.COMPLETED,
                           reference_warnings=[RHF_EXTERNAL_INSTABILITY_WARNING])
    data = result.to_dict()
    assert data["reference_warnings"] == [RHF_EXTERNAL_INSTABILITY_WARNING]
    again = Stage3PointResult(**{**data, "status": PointExecutionStatus(data["status"])})
    assert again.reference_warnings == (RHF_EXTERNAL_INSTABILITY_WARNING,)


def test_warning_summary_reports_magnitudes_without_classification():
    reqs = [_request(request_id="a"), _request(request_id="b", grid_index=1)]
    records = {
        "a": {"reference_warnings": ["RHF_EXTERNAL_INSTABILITY_RECORDED"],
              "rhf_external_uhf_lowering_hartree": 0.003, "rhf_external_uhf_s2": 0.46},
        "b": {"reference_warnings": ["INTERNAL_INSTABILITY_FOLLOWED", "RHF_EXTERNAL_INSTABILITY_RECORDED"],
              "stability_following_lowering_hartree": 0.0167,
              "rhf_external_uhf_lowering_hartree": 0.05, "rhf_external_uhf_s2": 1.03},
    }
    summary = _reference_warning_summary(reqs, records)["j"]
    assert summary["points_with_rhf_external_instability"] == 2
    assert summary["points_with_followed_instability"] == 1
    assert summary["max_uhf_lowering_hartree"] == pytest.approx(0.05)
    assert summary["max_uhf_s2"] == pytest.approx(1.03)


def test_single_failed_point_no_longer_blocks_review_of_the_rest(tmp_path):
    summary, manifest = fixture(tmp_path, open_highspin=False)
    failing = []

    def runner(req, settings):
        if req.grid_index == 0 and req.charge == 0:
            failing.append(req.request_id)
            return _empty_result(req, PointExecutionStatus.SCF_UNSTABLE)
        return _empty_result(req, PointExecutionStatus.COMPLETED, scf_converged=True,
                             scf_energy_hartree=-1.0)

    state = run_benchmark_acquisition(
        summary=summary, manifest=manifest, output_dir=tmp_path / "out",
        settings=AcquisitionSettings(), point_runner=runner,
    )
    neutral_job = next(k for k in state["stage3_pec_reviews"] if "q+0" in k)
    import json
    reviews = json.loads((tmp_path / "out" / "stage3_pec_reviews.json").read_text())
    entry = reviews[neutral_job]
    assert entry["status"] != "WAITING_FOR_COMPLETE_POINTS"
    assert [x["request_id"] for x in entry["excluded_failed_points"]] == failing
    # The stand-in checkpoint cannot pass the HF audit, so the partial review
    # stays unresolved; what matters is that it ran and listed the exclusion.
    assert state["stage3_identity_summary"][neutral_job]["complete_point_coverage"] is False
    assert state["stage3_identity_summary"][neutral_job]["excluded_failed_point_count"] == 1
    assert not state["ea_computed"]


@pytest.fixture(scope="module")
def stretched_h2_checkpoint(tmp_path_factory):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    path = tmp_path_factory.mktemp("h2") / "rhf.chk"
    mol = gto.M(atom="H 0 0 0; H 0 0 2.2", basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.chkfile = str(path)
    mf.kernel()
    return str(path)


def _h2_request(chk):
    return _request(r_angstrom=2.2, source_checkpoint_path=chk)


def test_pyscf_rhf_external_instability_fails_under_legacy_policy(stretched_h2_checkpoint):
    result = run_stage3_point(_h2_request(stretched_h2_checkpoint),
                              settings=Stage3ExecutionSettings(verbose=0))
    assert result.status is PointExecutionStatus.SCF_UNSTABLE
    assert result.external_stable is False


def test_pyscf_rhf_external_instability_recorded_and_quantified(stretched_h2_checkpoint):
    result = run_stage3_point(
        _h2_request(stretched_h2_checkpoint),
        settings=Stage3ExecutionSettings(
            verbose=0, stability_following_max_steps=4,
            rhf_external_instability_policy="RECORD_AND_CONTINUE"),
    )
    assert result.status is PointExecutionStatus.COMPLETED, result.error_message
    assert result.cc_reference == "RHF"
    assert result.external_stable is False
    assert RHF_EXTERNAL_INSTABILITY_WARNING in result.reference_warnings
    assert result.rhf_external_uhf_lowering_hartree > 1.0e-3
    assert 0.2 < result.rhf_external_uhf_s2 < 1.2
    assert result.ccsd_t_total_hartree is not None
