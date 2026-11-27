"""Route B (lanthanide monoxides, axial determinants) with a fake point function."""
import json

import pytest

from openea_benchmark import ln_oxide_route as rb
from openea_benchmark.axial.ln_oxide import LnConfig

H = 1.0 / 27.211386245988


def fake_energy(spec):
    """Neutral: f8 6s ML=3 lowest; anion: f8 6s2 ML=3 lowest; harmonic wells at 1.80 / 1.85 A."""
    cfg = LnConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec.config.items()})
    anion = cfg.nf + cfg.ns == 65 - 54 - 1
    r0 = 1.85 if anion else 1.80
    base = -100.0 - (0.04 if anion else 0.0)        # EA ~ 1.09 eV before corrections
    base += 0.004 * (3 - cfg.m_l) + (0.2 if cfg.nf == 9 else 0.0)
    return base + 0.3 * (spec.r - r0) ** 2


def fake_point(spec):
    rec = {"spec": {**spec.__dict__, "methods": list(spec.methods)}, "seconds": 0.0}
    if spec.so:
        return {**rec, "status": "COMPLETED", "delta_so_hartree": -0.002, "delta_so_hf_hartree": -0.0019,
                "omega_twice": 1}
    cfg = LnConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec.config.items()})
    if cfg.nf == 9 and FAIL_F9:
        return {**rec, "status": "CC_NOT_CONVERGED", "e_scf": fake_energy(spec) + 0.5, "scf_converged": True,
                "correlation": {}, "f_purity": {}, "scf_model": "frozen_f"}
    e = fake_energy(spec)
    corr = {"ccsd(t)": -0.5 - (0.01 if spec.basis == "Q" else 0.0)}
    if "ccsdt" in spec.methods:
        corr["ccsdt"] = corr["ccsd(t)"] - (0.001 if cfg.ns == 2 else 0.0005)
    if spec.cv:
        corr = {"val_ccsd(t)": -0.5, "cv_ccsd(t)": -0.6 - (0.0003 if cfg.ns == 2 else 0.0)}
    return {**rec, "status": "COMPLETED", "e_scf": e + 0.5, "scf_converged": True, "correlation": corr,
            "f_purity": {"a3": 1.0}, "scf_model": "free"}


FAIL_F9 = False


def test_route_selection():
    assert rb.route_b_element("Tb", "O") == "Tb" and rb.route_b_element("O", "Pr") == "Pr"
    assert rb.route_b_element("La", "O") is None and rb.route_b_element("Lu", "O") is None
    assert rb.route_b_unavailable("Pr", "F").startswith("ROUTE_B_NOT_AVAILABLE")
    assert rb.route_b_unavailable("U", "O").startswith("ROUTE_B_NOT_AVAILABLE")
    assert rb.route_b_unavailable("Th", "O") is None and rb.route_b_unavailable("O", "H") is None


def test_quartic_minimum_of_a_harmonic_well():
    pts = {round(1.76 + 0.02 * i, 4): 0.3 * (1.76 + 0.02 * i - 1.80) ** 2 for i in range(5)}
    mn = rb.quartic_minimum(pts, (158.925343, 15.994915))
    assert mn["r_e"] == pytest.approx(1.80, abs=1e-6) and mn["omega_cm1"] > 0


def test_route_b_end_to_end_with_fake_points(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__(__name__), "FAIL_F9", False)
    rec = rb.run_route_b(element="Tb", output_dir=tmp_path / "rb", point_fn=fake_point)
    assert rec["status"] == "CLEARED"
    n, a = rec["charges"]["neutral"], rec["charges"]["anion"]
    assert n["selection"]["selected"]["label"].startswith("4f8") and "ML=3" in n["selection"]["selected"]["label"]
    assert n["selection"]["minimum"]["r_e"] == pytest.approx(1.80, abs=1e-4)
    assert a["selection"]["minimum"]["r_e"] == pytest.approx(1.85, abs=1e-4)
    t = rec["terms"]
    assert t["ea_cbs_ev"] == pytest.approx(0.04 / H, abs=1e-6)          # same correlation, SCF offset
    assert t["delta_t_ev"] == pytest.approx((-0.0005 + 0.001) / H, abs=1e-9)
    assert t["delta_cv_ev"] == pytest.approx(0.0003 / H, abs=1e-9)
    assert t["delta_so_ev"] == pytest.approx(0.0, abs=1e-12)
    assert rec["ea0_lower_ev"] < rec["ea0_ev"] < rec["ea0_upper_ev"]
    v = rb.final_verdict(rec)
    assert v["outcome"] == "EA" and v["ea_ev"] == rec["ea0_ev"]
    saved = json.loads((tmp_path / "rb" / "route_b.json").read_text())
    assert saved["status"] == "CLEARED"
    # points are cached: a rerun computes nothing
    calls = []
    rb.run_route_b(element="Tb", output_dir=tmp_path / "rb", point_fn=lambda s: calls.append(s) or fake_point(s))
    assert calls == []


def test_unconverged_configuration_far_above_does_not_block(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__(__name__), "FAIL_F9", True)
    rec = rb.run_route_b(element="Tb", output_dir=tmp_path / "rb", point_fn=fake_point)
    # f9 fails CC, its SCF lies 0.2 Eh = 5.4 eV above the lowest SCF: excluded, not blocking
    assert rec["status"] == "CLEARED"


def test_unconverged_configuration_close_in_scf_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(__import__(__name__), "FAIL_F9", True)
    monkeypatch.setattr(rb, "UNCONVERGED_SCF_MARGIN_EV", 10.0)
    rec = rb.run_route_b(element="Tb", output_dir=tmp_path / "rb", point_fn=fake_point)
    assert rec["status"] == "UNRESOLVED" and "screen" in rec["reason"]
    assert rb.final_verdict(rec)["outcome"] == "UNRESOLVED"


def test_workflow_fails_closed_for_open_f_pairs_outside_the_model(tmp_path):
    from openea_benchmark.workflow import main
    assert main(["--atoms", "Pr", "F", "--output", str(tmp_path), "--quadruples-basis", "cc-pvdz"]) == 0
    verdict = json.loads((tmp_path / "PrF" / "final_verdict.json").read_text())
    assert verdict["final_verdict"]["outcome"] == "UNRESOLVED"
    assert verdict["final_verdict"]["blocking"][0].startswith("ROUTE_B_NOT_AVAILABLE")


def test_workflow_runs_route_b_for_a_lanthanide_monoxide(tmp_path, monkeypatch):
    from openea_benchmark import workflow
    seen = {}

    def fake_run(*, element, output_dir):
        seen["element"] = element
        return {"status": "CLEARED", "ea0_ev": 1.2, "ea0_lower_ev": 1.1, "ea0_upper_ev": 1.3,
                "competitor_widening_ev": {}}
    monkeypatch.setattr(rb, "run_route_b", fake_run)
    assert workflow.main(["--atoms", "Tb", "O", "--output", str(tmp_path), "--quadruples-basis", "cc-pvdz"]) == 0
    verdict = json.loads((tmp_path / "TbO" / "final_verdict.json").read_text())
    assert seen["element"] == "Tb" and verdict["route"] == "B_AXIAL_LN_OXIDE"
    assert verdict["outcome"] == "EA" and verdict["final_verdict"]["ea_ev"] == 1.2
