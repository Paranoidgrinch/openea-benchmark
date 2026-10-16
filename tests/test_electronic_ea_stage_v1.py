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
    args = build_parser().parse_args(["--atoms", "H", "F"])
    assert args.no_electronic_ea is False


def test_single_basis_interval_reports_sign_not_verdict():
    from openea_benchmark.electronic_ea_stage import single_basis_ea_interval
    n = entry("n", center=-1.0, lo=-1.0 - 1e-6, hi=-1.0 + 1e-6)
    a = entry("a", center=-1.0 + 0.007 * H, lo=-1.0 + 0.007 * H - 1e-6, hi=-1.0 + 0.007 * H + 1e-6)
    lower, upper, central, sign = single_basis_ea_interval(n, a)
    assert central == pytest.approx(-0.007, abs=1e-6)
    assert sign == "NEGATIVE"   # a near-threshold negative value is only a sign at this level


def test_reference_collapse_evidence_flags_large_t2_or_triples():
    from openea_benchmark.electronic_ea_stage import reference_collapse_evidence
    pol = ElectronicEAStagePolicy()
    healthy = {"max_abs_t2_amplitude": 0.34, "triples_correction_hartree": -0.04, "ccsd_correlation_hartree": -0.4}
    assert reference_collapse_evidence(healthy, pol) is None
    c2 = {"max_abs_t2_amplitude": 1.77, "triples_correction_hartree": -0.26, "ccsd_correlation_hartree": -0.6}
    assert reference_collapse_evidence(c2, pol)["reasons"] == ["MAX_T2_AMPLITUDE", "TRIPLES_FRACTION"]
    assert reference_collapse_evidence({"max_abs_t2_amplitude": None}, pol) is None


def test_collapsed_candidate_excluded_only_by_reference_gap():
    low = {**entry("x", center=-1.0, lo=-1.0001, hi=-0.9999), "scf_lowest_hartree": -0.6}
    far = {**entry("c", status="SINGLE_REFERENCE_COLLAPSED"), "scf_lowest_hartree": -0.6 + 6.0 * H}
    res = _competition("neutral", [low, far], 1.0, 3.0)
    assert res["status"] == "CLEARED" and res["multireference_only_excluded"][0]["job_id"] == "c"
    near = {**far, "scf_lowest_hartree": -0.6 + 1.0 * H}
    res = _competition("neutral", [low, near], 1.0, 3.0)
    assert res["status"] == "UNRESOLVED" and res["blocking"][0]["status"] == "MULTIREFERENCE_METHOD_REQUIRED"
    assert _competition("neutral", [low, {**far, "scf_lowest_hartree": None}], 1.0, 3.0)["status"] == "UNRESOLVED"


def test_coupled_cluster_failed_candidate_separated_by_reference_gap():
    # TiO- 6Pi: CCSD not converged at any point, reference determinant 4.6 eV above.
    low = {**entry("x", center=-1.0, lo=-1.0001, hi=-0.9999), "scf_lowest_hartree": -0.6}
    failed = {**entry("f", status="NOT_ELIGIBLE_REVIEW_NOT_CLEARED"), "scf_only_evidence": True,
              "scf_lowest_hartree": -0.6 + 4.6 * H}
    res = _competition("anion", [low, failed], 1.0, 3.0)
    assert res["status"] == "CLEARED"
    assert res["multireference_only_excluded"][0]["evidence"] == "SCF_ONLY_COUPLED_CLUSTER_NOT_CONVERGED"
    near = {**failed, "scf_lowest_hartree": -0.6 + 2.0 * H}
    assert _competition("anion", [low, near], 1.0, 3.0)["status"] == "UNRESOLVED"
    # Without converged references the state still blocks.
    assert _competition("anion", [low, entry("g", status="NOT_ELIGIBLE_REVIEW_NOT_CLEARED")], 1.0, 3.0)["status"] == "UNRESOLVED"


def test_atomic_fragments_run_inside_a_core_slot(tmp_path, monkeypatch):
    # Every atomic calculation holds a machine-wide core slot (core and memory budget);
    # before, fragment CCSD(T) and the scout's reference DFT ran outside it.
    from contextlib import contextmanager
    import openea_benchmark.point_pool as pool
    held = {"now": 0, "calls": []}

    @contextmanager
    def slot(reserve_mb=None):
        held["now"] += 1
        try:
            yield
        finally:
            held["now"] -= 1
    monkeypatch.setattr(pool, "core_slot", slot)
    fake = _fake_fragment({("H", 0, 1): -0.5, ("H", -1, 0): -0.52})

    def runner(request, settings):
        held["calls"].append(held["now"])
        return fake(request, settings)
    scan_atomic_fragments(atoms=("H", "H"), basis="sto-3g", settings=Stage3ExecutionSettings(),
                          policy=ElectronicEAStagePolicy(), cache_dir=tmp_path, runner=runner)
    assert held["calls"] and all(n == 1 for n in held["calls"])


def test_symmetry_scout_reference_dft_runs_inside_a_core_slot(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import openea_benchmark.symmetry_scout as scout
    held = {"now": 0, "plans": []}

    @contextmanager
    def slot(reserve_mb=None):
        held["now"] += 1
        try:
            yield
        finally:
            held["now"] -= 1

    def plan(*args, **kwargs):
        held["plans"].append(held["now"])
        return {"blocks": [], "mr_only_or_outside_window": [], "candidate_labels": []}
    monkeypatch.setattr(scout, "core_slot", slot)
    monkeypatch.setattr(scout, "plan_blocks", plan)
    scout.run_symmetry_scout("H", "H", output_dir=tmp_path)
    assert held["plans"] == [1, 1]  # one plan per charge, each inside a slot
