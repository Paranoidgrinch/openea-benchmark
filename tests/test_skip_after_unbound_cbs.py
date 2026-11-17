from openea_benchmark.benchmark_auto import anion_unbound_at_cbs, mr_labels_to_run
from openea_benchmark.final_resolver import resolve_final_verdict


def test_unbound_cbs_skips_layers_and_keeps_only_anion_mr_candidates():
    labels = ["q=0 1Deltag", "q=0 3Sigmau-", "q=-1 2Pig"]
    assert anion_unbound_at_cbs({"status": "ANION_UNBOUND_AT_CBS"})
    assert not anion_unbound_at_cbs({"status": "CLEARED"}) and not anion_unbound_at_cbs(None)
    assert mr_labels_to_run(labels, True) == ["q=-1 2Pig"]
    assert mr_labels_to_run(labels, False) == labels


def test_unbound_verdict_needs_no_skipped_layer():
    ea = {"status": "ELECTRONIC_EA_EVALUATED", "high_level_competition": {}}
    cbs = {"status": "ANION_UNBOUND_AT_CBS", "binding_reason": "CBS_INTERVAL_NEGATIVE"}
    verdict = resolve_final_verdict(electronic_ea=ea, cbs=cbs, adiabatic=None, mr_competitors=None,
                                    mr_only_labels=["q=0 1Deltag"], mr_term_gap=None, no_bound_state=())
    assert verdict["outcome"] == "UNBOUND"
    open_anion = resolve_final_verdict(electronic_ea=ea, cbs=cbs, adiabatic=None, mr_competitors=None,
                                       mr_only_labels=["q=-1 2Pig"], mr_term_gap=None, no_bound_state=())
    assert open_anion["outcome"] == "UNRESOLVED"
