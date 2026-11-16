# Smoke tests for the brief evaluation harness on the 5 hand-written sample briefs.
# Run: python3 -m pytest tests/test_faithfulness.py -q

import json
import os
import tempfile

from src.llm.agent_eval import RunLogger, load_logs
from src.llm.faithfulness import (
    auditor_feedback,
    number_supported,
    retrieval_metrics,
    score_brief,
    score_run,
)

FX = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures/sample_briefs.json")))
SNAP = {s["client_id"]: s for s in FX["snapshots"]}
EV = FX["evidence"]


def _score(i):
    b = FX["briefs"][i]
    return score_brief(b, SNAP[b["client_id"]], EV)


def test_good_brief_passes():
    r = _score(0)
    assert r["all_pass"], auditor_feedback(r)
    assert r["faithfulness_mean"] >= 3.5


def test_fabricated_citation():
    r = _score(1)
    assert not r["all_pass"]
    assert set(r["details"]["citations"]["invalid"]) == {"TYP-11", "SOP-v2025-9.9"}


def test_wrong_numbers():
    r = _score(2)
    bad = {u["number"] for u in r["details"]["numbers"]["unsupported"]}
    assert "45%" in bad and "0.75" in bad
    assert "-21.5" not in bad and "18%" not in bad


def test_expired_rule_version():
    r = _score(3)
    assert r["rule_applicability_rate"] == 0.0
    assert r["citation_valid_rate"] == 1.0 and r["number_accuracy_rate"] == 1.0


def test_weak_support_and_schema():
    r = _score(4)
    assert not r["schema_valid"]
    assert r["details"]["faithfulness"]["scores"][0] <= 2


def test_number_rounding_and_percent():
    assert number_supported("87%", 87, [0.8734])
    assert number_supported("0.87", 0.87, [0.8734])
    assert not number_supported("0.88", 0.88, [0.8734])
    assert number_supported("1,200", 1200, [1200])


def test_retrieval_metrics():
    m = retrieval_metrics(
        {"q1": ["a", "b", "c"], "q2": ["x", "y", "z", "w", "g"]}, {"q1": {"b"}, "q2": {"g"}}
    )
    assert m["recall@3"] == 0.5 and m["recall@5"] == 1.0
    assert abs(m["mrr"] - (0.5 + 0.2) / 2) < 1e-9


def test_score_run_and_logger():
    items = [{"brief": b, "snapshot": SNAP[b["client_id"]], "evidence": EV} for b in FX["briefs"]]
    agg, res = score_run(items)
    assert agg["n_briefs"] == 5 and agg["all_pass_share"] == 0.2
    with tempfile.TemporaryDirectory() as d:
        with RunLogger("single", "train_Client_101", "M10", log_dir=d) as log:
            log.tool_call("get_risk_score", {"client_id": "train_Client_101"}, {"score": 0.87})
            log.llm_call({"input_tokens": 1200, "output_tokens": 300})
            log.set_brief(FX["briefs"][0])
        L = load_logs("single", d)
        assert L[0]["n_tool_calls"] == 1 and L[0]["input_tokens"] == 1200


def test_comma_separated_citations_are_split():
    b = json.loads(json.dumps(FX["briefs"][0]))
    ids = b["evidence"][0]["citations"]
    b["evidence"][0]["citations"] = [", ".join(ids)]
    b["onsite_checks"][0] = b["onsite_checks"][0].replace("]", ", SNAPSHOT]", 1)
    r = score_brief(b, SNAP[b["client_id"]], EV)
    assert r["citation_valid_rate"] == 1.0, r["details"]["citations"]
    assert r["number_accuracy_rate"] == _score(0)["number_accuracy_rate"]
