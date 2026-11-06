"""Route A: FNO-DMRG beyond-CCSDT increment (pure parts; the real chain is validated on PO)."""
from openea_benchmark.fno_dmrg_increment import FNODMRGPolicy, ea_increment, truncation_levels


def test_truncation_levels_respect_the_active_space_cap():
    p = FNODMRGPolicy(max_active_orbitals=40, second_level_fraction=0.6)
    assert truncation_levels(9, 38, p) == (31, 19)  # explicit cap 40
    assert truncation_levels(6, 20, p) == (20, 12)  # whole virtual space fits
    large, small = truncation_levels(38, 50, p)
    assert large == p.min_virtuals and small < large


def test_ea_increment_and_half_width():
    h2ev = 27.211386245988
    pt = lambda inc, d: {"increment_hartree": inc, "dmrg": {"extrapolation_distance_hartree": d}}
    pts = {("neutral", 0): pt(-1.30e-3, 1e-6), ("anion", 0): pt(-1.00e-3, 2e-6),
           ("neutral", 1): pt(-1.20e-3, 0.0), ("anion", 1): pt(-0.95e-3, 0.0)}
    res = ea_increment(pts, h2ev)
    assert abs(res["delta_ev"] - (-0.30e-3 * h2ev)) < 1e-12
    assert abs(res["level_change_ev"] - 0.05e-3 * h2ev) < 1e-12
    assert abs(res["half_width_ev"] - (0.05e-3 + 3e-6) * h2ev) < 1e-12
