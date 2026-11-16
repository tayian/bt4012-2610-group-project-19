# Unit tests for src/llm/agent.py: each tool alone, the tool budget, and both agent systems
# end to end with a scripted fake Gemini client (no network, no model download).

import json
import os
from types import SimpleNamespace

from src.llm import agent, briefs
from src.llm.agent_eval import load_logs
from src.llm.retrieval import HybridRetriever

FX = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures/sample_briefs.json")))
CID = "train_Client_100109"


class StubRetriever:
    def __init__(self):
        self.calls = []
        self.id_to_chunk = {c["id"]: c for c in FX["evidence"]}

    def retrieve(self, query, filters=None, top_k=5, method="hybrid"):
        self.calls.append({"query": query, "filters": filters})
        return [dict(c) for c in FX["evidence"][:top_k]]


def _usage():
    return SimpleNamespace(prompt_token_count=50, candidates_token_count=10, thoughts_token_count=0)


def _call(name, **args):
    return SimpleNamespace(name=name, args=args, id=None)


def _resp(text="", calls=None):
    from google.genai import types

    parts = [types.Part(text=text or "ok")]
    return SimpleNamespace(
        text=text,
        function_calls=calls or None,
        candidates=[SimpleNamespace(content=types.Content(role="model", parts=parts))],
        usage_metadata=_usage(),
    )


class ScriptedModels:
    # Tool-loop turns come from `turns`; JSON calls are answered by system prompt.
    def __init__(self, turns, briefs_out, plan=None):
        self.turns = list(turns)
        self.briefs_out = list(briefs_out)
        self.plan = plan or {"questions": ["Why is the score high?", "Which SOP applies?"]}
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(config)
        system = (
            config.get("system_instruction")
            if isinstance(config, dict)
            else config.system_instruction
        )
        if not isinstance(config, dict) and config.tools:
            return self.turns.pop(0) if self.turns else _resp("DONE")
        if system == agent.PLANNER_PROMPT:
            return _resp(json.dumps(self.plan))
        out = self.briefs_out.pop(0) if len(self.briefs_out) > 1 else self.briefs_out[0]
        return _resp(json.dumps(out))


def _good_brief(cid=CID, chunk_id=None):
    risk = agent.get_risk_score(cid)
    cites = ["SNAPSHOT"] + ([chunk_id] if chunk_id else [])
    return {
        "client_id": cid,
        "risk_score": risk["risk_score"],
        "risk_rank_pct": risk["risk_rank_pct"],
        "risk_tier": risk["risk_tier"],
        "evidence": [
            {"point": "The model ranks this client for inspection.", "citations": cites},
            {"point": "The client has several invoices on file.", "citations": ["SNAPSHOT"]},
            {"point": "The client profile is recorded.", "citations": ["SNAPSHOT"]},
        ],
        "onsite_checks": ["Check the meter seals."],
        "applicable_rules": [],
        "similar_cases": [],
        "uncertainty": "Evidence is limited to the snapshot.",
    }


def always5(point, cited_texts):
    return 5


# Each tool alone


def test_get_client_profile():
    out = agent.get_client_profile(CID)
    for k in ("client_catg", "region", "tenure_years", "n_counters", "n_invoices", "counter_type"):
        assert k in out
    assert out["client_id"] == CID and out["split"] == "train"


def test_get_risk_score():
    out = agent.get_risk_score(CID)
    assert 0 <= out["risk_score"] <= 1 and 0 <= out["risk_rank_pct"] <= 100
    assert out["risk_tier"] == briefs.risk_tier(out["risk_rank_pct"])
    assert out["model_version"].startswith("M10_v4")
    assert "not a verified probability" in out["note"]


def test_explain_shap():
    out = agent.explain_shap(CID)
    assert [d["rank"] for d in out["drivers"]] == [1, 2, 3, 4, 5]
    for d in out["drivers"]:
        assert d["direction"] == ("raises risk" if d["shap_value"] > 0 else "lowers risk")


def test_benford_check():
    out = agent.benford_check(CID)
    assert out["n_positive_readings"] >= 0
    assert out["benford_flag"] in {
        "close conformity",
        "acceptable",
        "marginal",
        "nonconformity",
        "insufficient readings",
    }
    assert isinstance(out["round_reading_flag"], bool)


def test_retrieve_evidence_returns_ids_and_metadata():
    chunks = [
        {**c, "jurisdiction": "TN", "counter_type": "both", "valid_from": "2005-01-01"}
        for c in FX["evidence"]
    ]
    for c in chunks:
        c.pop("valid_to", None)
    retr = HybridRetriever(chunks=chunks)
    retr._dense_rank = lambda q, filtered, n: ([c["id"] for c in filtered[:n]], {})
    out = agent.retrieve_evidence("counter_statue 5 meter faulty", {"doc": "codebook"}, retr)
    assert out["chunks"] and len(out["chunks"]) <= 5
    for c in out["chunks"]:
        assert c["id"] and c["doc"] == "codebook" and "text" in c and "version" in c


def test_plot_history_writes_png(tmp_path):
    out = agent.plot_history(CID, out_dir=tmp_path)
    assert out["path"].endswith(f"{CID}.png") and os.path.getsize(out["path"]) > 1000
    assert out["n_invoices"] > 0 and out["n_drops"] >= 0
    assert out["first_year"] == int(out["first_invoice"][:4])
    assert out["drop_years"] == sorted({int(d[:4]) for d in out["drop_dates"]})


def test_drop_years_count_as_snapshot_numbers():
    # Old logs lack drop_years; scoring derives them so a correct year is not flagged.
    hist = {"path": "x.png", "drop_dates": ["2009-05-19 (ELEC)"], "first_invoice": "2006-01-20"}
    snap = agent.scoring_snapshot(CID, {"plot_history": hist})
    brief = _good_brief()
    brief["evidence"][0]["point"] = "Consumption dropped sharply in 2009."
    from src.llm.faithfulness import score_brief

    assert score_brief(brief, snap, [])["number_accuracy_rate"] == 1.0


def test_tool_schemas_match_functions():
    names = [t["name"] for t in agent.TOOL_SCHEMAS]
    assert names == list(agent.TOOL_FUNCS) and len(names) == 6
    for t in agent.TOOL_SCHEMAS:
        assert t["parameters"]["type"] == "object" and t["parameters"]["required"]
    decls = agent.gemini_tools()[0].function_declarations
    assert [d.name for d in decls] == names


# The toolbox and the loop


def test_toolbox_binds_client_and_filters():
    retr = StubRetriever()
    box = agent.ToolBox(CID, retriever=retr)
    out = box.call("get_client_profile", {"client_id": "train_Client_0"})
    assert out["client_id"] == CID
    box.call(
        "retrieve_evidence",
        {"query": "bypass", "filters": {"doc": "sop", "alert_date": "2020-01-01", "x": 1}},
    )
    f = retr.calls[0]["filters"]
    assert f["doc"] == "sop" and f["exclude_client_id"] == CID
    assert f["alert_date"] == box.snapshot["alert_date"] and "x" not in f
    assert set(box.chunks) == {c["id"] for c in FX["evidence"][:5]}
    assert "error" in box.call("no_such_tool", {})
    snap = box.agent_snapshot()
    assert set(snap["tool_outputs"]) == {"get_client_profile"}


def test_tool_loop_stops_at_budget(tmp_path):
    many = [_resp(calls=[_call("get_risk_score", client_id=CID)] * 3) for _ in range(6)]
    fake = SimpleNamespace(models=ScriptedModels(many, [_good_brief()]))
    with agent.RunLogger("single", CID, "m", str(tmp_path)) as log:
        box = agent.ToolBox(CID, log, StubRetriever())
        agent.tool_loop(fake, "m", "sys", [agent._user("go")], box, log, role="agent")
    assert box.n_calls == agent.MAX_TOOL_CALLS
    assert len(log.rec["tool_calls"]) == agent.MAX_TOOL_CALLS


# Both systems end to end


def test_run_single_logs_brief(tmp_path):
    turns = [
        _resp(
            calls=[
                _call("get_risk_score", client_id=CID),
                _call("retrieve_evidence", query="meter bypass"),
            ]
        ),
        _resp("DONE"),
    ]
    brief = _good_brief(chunk_id=FX["evidence"][0]["id"])
    fake = SimpleNamespace(models=ScriptedModels(turns, [brief]))
    rec = agent.run_single(CID, fake, "m", log_dir=str(tmp_path), retriever=StubRetriever())
    assert rec["brief"]["client_id"] == CID and rec["n_tool_calls"] == 2
    assert rec["evidence_ids"][0] == FX["evidence"][0]["id"]
    assert rec["llm_model"] == "m" and rec["model_version"].startswith("M10_v4")
    assert (tmp_path / "single" / f"{CID}.json").is_file()
    assert agent._done("single", CID, str(tmp_path), "m")
    assert not agent._done("single", CID, str(tmp_path), "other-model")


def test_run_multi_revises_once_then_passes(tmp_path):
    turns = [_resp(calls=[_call("explain_shap", client_id=CID)]), _resp("notes")]
    bad = {**_good_brief(), "risk_score": 0.12345}
    fake = SimpleNamespace(models=ScriptedModels(turns, [bad, _good_brief()]))
    rec = agent.run_multi(CID, fake, always5, "m", log_dir=str(tmp_path), retriever=StubRetriever())
    assert rec["plan"] == ["Why is the score high?", "Which SOP applies?"]
    assert rec["auditor_verdict"] == "pass_after_revision"
    assert any("risk_score" in m for m in rec["auditor_feedback"])
    assert len(rec["audits"]) == 2 and rec["brief"]["risk_score"] != 0.12345
    # Planner reads profile and risk score; the gatherer made one call.
    assert [t["tool"] for t in rec["tool_calls"]] == [
        "get_client_profile",
        "get_risk_score",
        "explain_shap",
    ]
    roles = [c["role"] for c in rec["llm_calls"]]
    assert roles[0] == "planner" and roles[-1] == "writer_revision"


def test_run_multi_flags_human_review_and_compare(tmp_path):
    turns = [_resp("nothing found")]
    bad = {**_good_brief(), "risk_score": 0.12345}
    fake = SimpleNamespace(models=ScriptedModels(turns, [bad]))
    rec = agent.run_multi(CID, fake, always5, "m", log_dir=str(tmp_path), retriever=StubRetriever())
    assert rec["auditor_verdict"] == "needs_human_review"
    assert len(load_logs("multi", str(tmp_path))) == 1


def test_compare_table(tmp_path, monkeypatch):
    fake = SimpleNamespace(models=ScriptedModels([_resp("DONE")], [_good_brief()]))
    agent.run_single(CID, fake, "m", log_dir=str(tmp_path), retriever=StubRetriever())
    fake = SimpleNamespace(models=ScriptedModels([_resp("notes")], [_good_brief()]))
    agent.run_multi(CID, fake, always5, "m", log_dir=str(tmp_path), retriever=StubRetriever())
    monkeypatch.setattr(agent, "get_retriever", lambda: StubRetriever())
    t = agent.compare(["single", "multi"], always5, "always5", log_dir=str(tmp_path))
    t = t.set_index("system")
    assert t.loc["multi", "auditor_pass_rate"] == 1.0
    assert t.loc["single", "n_clients"] == 1 and t.loc["multi", "n_clients"] == 1
    assert t.loc["single", "citation_validity"] == 1.0
    assert t.loc["single", "faithfulness_mean"] == 5.0
    for col in (
        "tool_calls_mean",
        "tokens_mean",
        "seconds_mean",
        "schema_valid",
        "number_accuracy",
    ):
        assert col in t.columns
