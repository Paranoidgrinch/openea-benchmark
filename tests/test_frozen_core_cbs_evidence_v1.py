from openea_benchmark.attachment.frozen_core_cbs_evidence import (
    summarize_frozen_core_points,
)

def p(role, basis, reusable=True):
    return {"role": role, "basis": basis, "reusable": reusable}

def test_all_six_points_are_required():
    points = [
        p("neutral", "aug-cc-pvqz"), p("anion", "aug-cc-pvqz"),
        p("neutral", "aug-cc-pv5z"), p("anion", "aug-cc-pv5z"),
        p("neutral", "d-aug-cc-pv5z"), p("anion", "d-aug-cc-pv5z"),
    ]
    result = summarize_frozen_core_points(points)
    assert result.status == "READY"
    assert not result.missing_points
    assert result.correlation_space == "FROZEN_CORE_PYSCF_AUTO_CHEMCORE"

def test_missing_daug_blocks_rebuild():
    points = [
        p("neutral", "aug-cc-pvqz"), p("anion", "aug-cc-pvqz"),
        p("neutral", "aug-cc-pv5z"), p("anion", "aug-cc-pv5z"),
    ]
    result = summarize_frozen_core_points(points)
    assert result.status == "PARTIAL"
    assert "neutral:d-aug-cc-pv5z" in result.missing_points
    assert "DO_NOT_REBUILD_CBS_YET" in result.next_actions
