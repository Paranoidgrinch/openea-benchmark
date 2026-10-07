import json
from pathlib import Path


def test_postcc_checkpoint_tokens_are_collision_free():
    tokens = {
        "CCSD(T)": "CCSD_pT",
        "CCSDT": "CCSDT",
        "CCSDTQ": "CCSDTQ",
    }
    assert len(set(tokens.values())) == len(tokens)
    assert tokens["CCSD(T)"] != tokens["CCSDT"]


def test_w4_source_result_schema_contains_required_aggregate_evidence(tmp_path):
    result = {
        "status": "UNRESOLVED",
        "points": [
            {
                "cardinal": 2,
                "ea_ccsd_t_ev": 1.62,
                "ea_ccsdt_ev": 1.61,
                "delta_t3_ev": -0.01,
                "ea_ccsdtq_ev": 1.63,
                "delta_t4_ev": 0.02,
            },
            {
                "cardinal": 3,
                "ea_ccsd_t_ev": 1.73,
                "ea_ccsdt_ev": 1.72,
                "delta_t3_ev": -0.01,
                "ea_ccsdtq_ev": None,
                "delta_t4_ev": None,
            },
        ],
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(result), encoding="utf-8")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    by_x = {int(x["cardinal"]): x for x in loaded["points"]}
    assert by_x[2]["delta_t4_ev"] == 0.02
    assert by_x[3]["delta_t3_ev"] == -0.01
