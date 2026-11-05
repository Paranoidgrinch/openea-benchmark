"""CBS geometry relaxation: the low-cardinal minimum is located around the single-basis r_e."""
import math
from dataclasses import replace

from openea_benchmark.adaptive.stage3_execution import PointExecutionStatus, _empty_result
from openea_benchmark.cbs_stage import (
    GEOMETRY_MAX_PROBES, _relaxation_chain, geometry_relaxation_correction, relaxation_from_points,
)
from test_reference_branches_v1 import request as base_request


def morse(de, a, re):
    return lambda r: de * (1.0 - math.exp(-a * (r - re))) ** 2 - de


def runner_for(curve, fail_beyond=None):
    def run(req, settings):
        if fail_beyond is not None and req.r_angstrom > fail_beyond:
            return _empty_result(req, PointExecutionStatus.SCF_NOT_CONVERGED, reference_warnings=())
        return _empty_result(req, PointExecutionStatus.COMPLETED, ccsd_t_total_hartree=curve(req.r_angstrom),
                             reference_warnings=())
    return run


def chain(curve, r0, **kw):
    req = replace(base_request(0, r0, job="N"), request_id="N__CBS__aug-cc-pvqz")
    return _relaxation_chain(runner_for(curve, **kw), req, r0, curve(r0), None)


def test_covalent_shift_bracketed_with_two_probes():
    curve = morse(0.2, 2.0, 1.000)  # single-basis r_e 1.005, low-cardinal r_e 1.000
    res = chain(curve, 1.005)
    assert res["status"] == "BRACKETED" and len(res["probes"]) == 2
    assert abs(res["r_min"] - 1.0) < 2e-3
    exact = curve(1.0) - curve(1.005)
    assert res["relaxation_hartree"] <= 0.0
    # anharmonic curve: the exact value lies inside the reported half-width
    assert abs(res["relaxation_hartree"] - exact) <= abs(res["relaxation_hartree"]) + res["model_uncertainty_hartree"]
    assert abs(res["relaxation_hartree"] - exact) < 0.2 * abs(exact)


def test_van_der_waals_shift_followed_outward():
    curve = morse(0.0019, 1.0, 3.98)  # Mg2-like: well 1.9 mEh, single-basis r_e 4.40
    res = chain(curve, 4.40)
    assert res["status"] == "BRACKETED" and len(res["probes"]) <= GEOMETRY_MAX_PROBES
    assert abs(res["r_min"] - 3.98) < 0.1
    exact = curve(3.98) - curve(4.40)
    assert abs(res["relaxation_hartree"] - exact) <= res["model_uncertainty_hartree"] + 0.1 * abs(exact)


def test_flat_curve_gives_no_correction():
    res = chain(lambda r: -1.0, 1.2)
    assert res["status"] == "FLAT" and res["relaxation_hartree"] == 0.0


def test_failed_probe_keeps_one_sided_bound():
    curve = morse(0.0019, 1.0, 4.6)
    res = chain(curve, 4.40, fail_beyond=4.43)
    assert res["status"] == "PROBE_FAILED"
    assert res["relaxation_hartree"] <= 0.0 and res["model_uncertainty_hartree"] == abs(res["relaxation_hartree"])


def test_correction_sign_and_half_width():
    h = 1.0 / 27.211386245988
    relax = {"neutral": {"relaxation_hartree": -0.0002 * h, "model_uncertainty_hartree": 0.0},
             "anion": {"relaxation_hartree": -0.0001 * h, "model_uncertainty_hartree": 0.00005 * h}}
    corr, half = geometry_relaxation_correction(relax)
    assert abs(corr - (-0.0001)) < 1e-12  # neutral relaxes more: EA decreases
    assert abs(half - 0.00035) < 1e-12


def test_next_probe_direction():
    pts = {1.0: -1.0, 0.98: -0.99, 1.02: -1.01}  # still falling to larger r
    v = relaxation_from_points(pts, 1.0)
    assert v["status"] == "NEXT" and v["next_r"] > 1.02
