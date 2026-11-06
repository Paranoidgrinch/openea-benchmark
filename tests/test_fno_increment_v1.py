"""Route A: FNO-CCSDTQ beyond-CCSDT increment (pure parts; the real chain is validated on PO)."""
from openea_benchmark.fno_increment import FNOPolicy, ea_increment, t4_amplitude_mb, truncation_levels


def test_truncation_levels_follow_the_memory_limit():
    p = FNOPolicy(second_level_fraction=0.6)
    large, small = truncation_levels(7, 6, 20, 100000.0, p)
    assert p.amplitude_copies * t4_amplitude_mb(7, 6, large, large + 1) <= 100000.0
    assert large == 20 or p.amplitude_copies * t4_amplitude_mb(7, 6, large + 1, large + 2) > 100000.0
    assert small == round(0.6 * large)
    assert truncation_levels(7, 6, 20, 1e12, p)[0] == 20  # whole virtual space fits
    large, small = truncation_levels(40, 40, 200, 1.0, p)
    assert large == p.min_virtuals and small < large


def test_ea_increment_and_half_width():
    h2ev = 27.211386245988
    pt = lambda inc: {"increment_hartree": inc}
    pts = {("neutral", 0): pt(-1.30e-3), ("anion", 0): pt(-1.00e-3),
           ("neutral", 1): pt(-1.20e-3), ("anion", 1): pt(-0.95e-3)}
    res = ea_increment(pts, h2ev)
    assert abs(res["delta_ev"] - (-0.30e-3 * h2ev)) < 1e-12
    assert abs(res["level_change_ev"] - 0.05e-3 * h2ev) < 1e-12
    assert abs(res["half_width_ev"] - 0.05e-3 * h2ev) < 1e-12
