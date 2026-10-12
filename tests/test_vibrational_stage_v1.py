"""Patch 42: automated ZPE layer wiring and adiabatic-EA assembly."""
from types import SimpleNamespace

import pytest

from openea_benchmark.vibrational_stage import (
    ZPEStagePolicy, assemble_adiabatic_ea, cross_model_identity_evidence, main_isotopologue_masses,
)


def cbs(**over):
    base = {"status": "CLEARED", "binding_at_cbs": "BOUND", "ea_cbs_ev": 1.80, "ea_cbs_lower_ev": 1.79,
            "ea_cbs_upper_ev": 1.81, "binding_reason": "CBS_INTERVAL_POSITIVE"}
    return {**base, **over}


def nuclear(**over):
    base = {"status": "CLEARED", "delta_zpe_ev": 0.03, "delta_zpe_half_width_ev": 0.004,
            "anion_vibrational_binding": "BOUND"}
    return {**base, **over}


def test_adiabatic_ea_adds_delta_zpe_and_widens_interval():
    rec = assemble_adiabatic_ea(cbs=cbs(), nuclear=nuclear())
    assert rec["status"] == "CLEARED" and rec["binding"] == "BOUND"
    assert rec["ea0_ev"] == pytest.approx(1.83)
    assert rec["ea0_lower_ev"] == pytest.approx(1.79 + 0.03 - 0.004)
    assert rec["ea0_upper_ev"] == pytest.approx(1.81 + 0.03 + 0.004)
    assert rec["verdict"].startswith("NOT_ISSUED")


def test_unbound_cbs_anion_is_not_zpe_corrected():
    rec = assemble_adiabatic_ea(cbs=cbs(binding_at_cbs="UNBOUND", ea_cbs_ev=None), nuclear=nuclear())
    assert rec["status"] == "ANION_UNBOUND" and rec["ea0_ev"] is None


def test_missing_zpe_bound_fails_closed():
    rec = assemble_adiabatic_ea(cbs=cbs(), nuclear=nuclear(status="MODEL_CROSS_MODEL_IDENTITY_REVIEW_REQUIRED",
                                                           delta_zpe_half_width_ev=None))
    assert rec["status"] == "NEED_MORE_EVIDENCE_ZPE" and rec["ea0_ev"] is None


def test_zpe_can_move_a_near_threshold_ea_across_zero():
    rec = assemble_adiabatic_ea(cbs=cbs(ea_cbs_ev=0.01, ea_cbs_lower_ev=0.009, ea_cbs_upper_ev=0.011),
                                nuclear=nuclear(delta_zpe_ev=-0.02, delta_zpe_half_width_ev=0.002))
    assert rec["binding"] == "UNBOUND" and rec["status"] == "NEED_MORE_EVIDENCE_BINDING_UNBOUND"


def test_vibrationally_unbound_anion_is_not_bound():
    rec = assemble_adiabatic_ea(cbs=cbs(), nuclear=nuclear(anion_vibrational_binding="UNRESOLVED"))
    assert rec["binding"] == "UNDETERMINED"


def test_main_isotopologue_masses():
    m = main_isotopologue_masses(("Mg", "H"))
    assert m.mass_a_u == pytest.approx(23.985042) and m.mass_b_u == pytest.approx(1.007825)


def context(role, energies, following=0):
    points = [SimpleNamespace(r_angstrom=r, energy_hartree=e) for r, e in energies]
    return SimpleNamespace(role=role, pec=SimpleNamespace(points=points),
                           results=[SimpleNamespace(stability_following_steps=following)])


def morse(r, re=1.0, d=0.2, a=2.0, shift=0.0):
    import math
    return d * (1 - math.exp(-a * (r - re))) ** 2 + shift


GRID = [0.85 + 0.05 * i for i in range(7)]


def test_cross_model_identity_accepts_smooth_basis_shift():
    prim = context("neutral", [(r, morse(r)) for r in GRID])
    comp = context("neutral", [(r, morse(r, re=1.005, shift=-0.05) + 1e-4 * r) for r in GRID])
    ids, checks = cross_model_identity_evidence(prim, comp, ZPEStagePolicy())
    assert len(ids) == 3 and "failed" not in checks


def test_cross_model_identity_rejects_kink_and_following():
    prim = context("anion", [(r, morse(r)) for r in GRID])
    kinked = [(r, morse(r, shift=-0.05) + (0.01 if r > 1.0 else 0.0)) for r in GRID]
    ids, checks = cross_model_identity_evidence(prim, context("anion", kinked), ZPEStagePolicy())
    assert ids == () and "not smooth" in checks["failed"]
    ids, checks = cross_model_identity_evidence(
        prim, context("anion", [(r, morse(r, shift=-0.05)) for r in GRID], following=1), ZPEStagePolicy())
    assert ids == () and "instability" in checks["failed"]
