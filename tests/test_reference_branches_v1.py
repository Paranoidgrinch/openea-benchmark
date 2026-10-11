"""Patch 38: split a job mixing two ROHF reference branches (wave-0 AlO X2Sigma+)."""
from pathlib import Path

import pytest

from openea_benchmark.adaptive.stage3_execution import Stage3ExecutionRequest
from openea_benchmark.adaptive.stage3_naming import checkpoint_artifact_basename
from openea_benchmark.electronic_ea_stage import _competition
from openea_benchmark.reference_branches import (
    branch_parent, branches_identical, build_branch_requests, mixed_reference_jobs,
)

H = 1.0 / 27.211386245988
RS = (1.545, 1.57, 1.595, 1.62, 1.645)


def request(i, r, job="AlO__q+0__2Sigma+"):
    return Stage3ExecutionRequest(
        request_id=f"{job}__g{i:03d}", job_id=job, system="AlO", atoms=("Al", "O"), charge=0, spin_2s=1,
        component_id="c", r_angstrom=r, basis="def2-tzvppd", methods=("CCSD", "CCSD(T)"),
        requested_reference="ROHF", scf_reference="ROHF", source_link_status="SINGLE_DFT_INITIALIZATION",
        source_root_id="root", source_checkpoint_path="/scout.chk", source_origin_guess="test", grid_index=i,
        initialization_index=0, dft_center_r_angstrom=1.595, dft_center_energy_hartree=-317.0,
        requires_independent_state_identity_validation=True)


def records(reqs, followed):
    return {r.request_id: {"status": "COMPLETED", "r_angstrom": r.r_angstrom,
                           "stability_following_steps": 1 if r.r_angstrom in followed else 0,
                           "high_level_checkpoint_path": f"/hl/{r.request_id}.chk"} for r in reqs}


def test_mixed_jobs_detected_only_when_following_is_partial():
    reqs = [request(i, r) for i, r in enumerate(RS)]
    assert mixed_reference_jobs(reqs, records(reqs, {1.62, 1.645})) == ["AlO__q+0__2Sigma+"]
    assert mixed_reference_jobs(reqs, records(reqs, set())) == []
    assert mixed_reference_jobs(reqs, records(reqs, set(RS))) == []


def test_branch_requests_seed_and_neighbour_continuation(tmp_path):
    reqs = [request(i, r) for i, r in enumerate(RS)]
    recs = records(reqs, {1.62, 1.645})
    out = build_branch_requests(reqs, recs, tmp_path)
    seed = [r for r in out if r.reference_branch == "SEED_CONTINUOUS"]
    fol = [r for r in out if r.reference_branch == "STABILITY_FOLLOWED"]
    assert len(seed) == len(fol) == 5
    assert all(r.source_checkpoint_path == "/scout.chk" and r.job_id.endswith("__refS") for r in seed)
    # anchors first (own high-level reference), then outward from the nearest anchor
    assert [r.r_angstrom for r in fol] == [1.62, 1.645, 1.595, 1.57, 1.545]
    assert fol[0].source_checkpoint_path == f"/hl/{reqs[3].request_id}.chk"
    by_r = {r.r_angstrom: r for r in fol}
    expected = str(Path(tmp_path) / checkpoint_artifact_basename(by_r[1.62].request_id))
    assert by_r[1.595].source_checkpoint_path == expected
    assert branch_parent(fol[0].job_id) == "AlO__q+0__2Sigma+"
    assert branch_parent("AlO__q+0__2Sigma+") is None


def test_identical_branches_are_merged():
    a = [{"status": "COMPLETED", "r_angstrom": r, "ccsd_t_total_hartree": -1.0 - r} for r in RS]
    b = [dict(x, ccsd_t_total_hartree=x["ccsd_t_total_hartree"] + 1e-8) for x in a]
    c = [dict(x, ccsd_t_total_hartree=x["ccsd_t_total_hartree"] + 1e-3) for x in a]
    assert branches_identical(a, b) and not branches_identical(a, c)


def test_unknown_branch_rejected():
    with pytest.raises(ValueError):
        Stage3ExecutionRequest(**{**request(0, 1.6).to_dict(), "reference_branch": "OTHER"})


def entry(job, center, branch_of=None, status="RESOLVED"):
    return {"job_id": job, "equilibrium_status": status, "energy_central_hartree": center,
            "energy_lower_hartree": center - 1e-5, "energy_upper_hartree": center + 1e-5,
            "lowest_sampled_hartree": center, "branch_of": branch_of}


def test_reference_branches_of_one_state_do_not_compete():
    res = _competition("neutral", [entry("p__refS", -1.0, "p"), entry("p__refF", -1.0 + 0.05 * H, "p"),
                                   entry("q", -1.0 + 0.6 * H)], 1.0)
    assert res["status"] == "CLEARED" and res["selected_job_id"] == "p__refS"
    assert abs(res["reference_sensitivity_ev"] - 0.05) < 1e-9
    # a genuine other state inside the interval still blocks
    blocked = _competition("neutral", [entry("p__refS", -1.0, "p"), entry("q", -1.0 + 1e-6)], 1.0)
    assert blocked["status"] == "UNRESOLVED"
