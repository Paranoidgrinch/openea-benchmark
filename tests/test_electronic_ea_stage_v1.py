"""Automated electronic-EA stage: competition, fragments, channels, wiring."""
import pytest

from openea_benchmark.attachment.fragment_execution import FragmentExecutionStatus, _empty_result as _fragment_result
from openea_benchmark.electronic_ea_stage import (
    ElectronicEAStagePolicy, _competition, anion_dissociation_channels, scan_atomic_fragments,
)
from openea_benchmark.adaptive.stage3_execution import Stage3ExecutionSettings
from openea_benchmark.workflow import build_parser

H = 1.0 / 27.211386245988


def entry(job, status="RESOLVED", center=None, lo=None, hi=None, sampled=None):
    return {"job_id": job, "equilibrium_status": status, "energy_central_hartree": center,
            "energy_lower_hartree": lo, "energy_upper_hartree": hi, "lowest_sampled_hartree": sampled}


def test_competition_clears_only_with_separated_intervals():
    ok = _competition("neutral", [entry("a", center=-1.0, lo=-1.0001, hi=-0.9999),
                                  entry("b", center=-0.9, lo=-0.9001, hi=-0.8999)], 1.0)
    assert ok["status"] == "CLEARED" and ok["selected_job_id"] == "a"
    overlap = _competition("neutral", [entry("a", center=-1.0, lo=-1.0001, hi=-0.9999),
                                       entry("b", center=-0.99995, lo=-1.00005, hi=-0.99985)], 1.0)
    assert overlap["status"] == "UNRESOLVED" and overlap["selected_job_id"] is None


def test_unresolved_competitor_needs_explicit_margin():
    near = _competition("anion", [entry("a", center=-1.0, lo=-1.0001, hi=-0.9999),
                                  entry("x", status="NOT_ELIGIBLE_REVIEW_NOT_CLEARED", sampled=-0.9999 + 0.5 * H)], 1.0)
    assert near["status"] == "UNRESOLVED"
    far = _competition("anion", [entry("a", center=-1.0, lo=-1.0001, hi=-0.9999),
                                 entry("x", status="NOT_ELIGIBLE_REVIEW_NOT_CLEARED", sampled=-0.9999 + 3.0 * H)], 1.0)
    assert far["status"] == "CLEARED"
    assert far["excluded_by_margin"][0]["job_id"] == "x"
    missing = _competition("anion", [entry("a", center=-1.0, lo=-1.0001, hi=-0.9999),
                                     entry("y", status="NOT_ELIGIBLE_REVIEW_NOT_CLEARED", sampled=None)], 1.0)
    assert missing["status"] == "UNRESOLVED"


def test_no_resolved_candidate_is_unresolved():
    assert _competition("neutral", [entry("a", status="BRIDGE_NOT_READY")], 1.0)["status"] == "UNRESOLVED"


def _fake_fragment(energies):
    def runner(request, settings):
        e = energies.get((request.element, request.charge, request.spin_2s))
        status = FragmentExecutionStatus.COMPLETED if e is not None else FragmentExecutionStatus.SCF_UNSTABLE
        return _fragment_result(request, status, ccsd_t_total_hartree=e)
    return runner


def test_atomic_scan_picks_lowest_spin_and_builds_channels(tmp_path):
    energies = {("O", 0, 0): -74.9, ("O", 0, 2): -75.0, ("O", 0, 4): -74.7,
                ("O", -1, 1): -75.05, ("O", -1, 3): -74.8,
                ("H", 0, 1): -0.5, ("H", -1, 0): -0.52}
    table = scan_atomic_fragments(atoms=("O", "H"), basis="sto-3g", settings=Stage3ExecutionSettings(),
                                  policy=ElectronicEAStagePolicy(), cache_dir=tmp_path,
                                  runner=_fake_fragment(energies))
    assert table["O+0"]["lowest"]["spin_2s"] == 2
    assert table["O-1"]["lowest"]["spin_2s"] == 1
    assert table["O+0"]["all_spins_completed"] is False  # O 2S=6 not in fake table
    channels = {c.channel_id: c.asymptotic_energy_hartree for c in anion_dissociation_channels(("O", "H"), table)}
    assert channels["O-1+H+0"] == pytest.approx(-75.55)
    assert channels["O+0+H-1"] == pytest.approx(-75.52)
    # cached: a second scan never calls the runner
    again = scan_atomic_fragments(atoms=("O", "H"), basis="sto-3g", settings=Stage3ExecutionSettings(),
                                  policy=ElectronicEAStagePolicy(), cache_dir=tmp_path,
                                  runner=lambda *a, **k: pytest.fail("recomputed"))
    assert again["O+0"]["lowest"] == table["O+0"]["lowest"]


def test_cli_runs_electronic_ea_by_default():
    args = build_parser().parse_args(["m.yaml", "--mode", "benchmark-auto"])
    assert args.no_electronic_ea is False
