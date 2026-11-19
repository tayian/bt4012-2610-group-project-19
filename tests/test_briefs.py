# Tests for brief writing and the Gemini judge; the LLM and retriever are stubs.

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.llm import briefs
from src.llm.faithfulness import make_gemini_judge, score_brief
from src.llm.snapshot import SNAPSHOT_CONFIG, get_client_snapshot

FX = json.loads((Path(__file__).parent / "fixtures/sample_briefs.json").read_text())
CID = "train_Client_100109"


def _usage():
    return SimpleNamespace(
        prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5
    )


class FakeModels:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        if isinstance(text, Exception):
            raise text
        return SimpleNamespace(text=text, usage_metadata=_usage())


def fake_client(*texts):
    return SimpleNamespace(models=FakeModels(texts))


class StubRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.id_to_chunk = {c["id"]: c for c in chunks}
        self.filters = []

    def retrieve(self, query, filters=None, top_k=5, method="hybrid"):
        self.filters.append(filters)
        return [dict(c) for c in self.chunks[:top_k]]


def _brief_for(snap, citations=("SNAPSHOT",)):
    return {
        "client_id": snap["client_id"],
        "risk_score": snap["risk_score"],
        "risk_rank_pct": snap["risk_rank_pct"],
        "risk_tier": briefs.risk_tier(snap["risk_rank_pct"]),
        "evidence": [
            {
                "point": f"The client has {snap['profile']['n_invoices']} invoices.",
                "citations": list(citations),
            },
            {"point": "The model score places the client in its tier.", "citations": ["SNAPSHOT"]},
            {"point": "The client has meter readings on file.", "citations": ["SNAPSHOT"]},
        ],
        "onsite_checks": ["Check the meter seals."],
        "applicable_rules": [],
        "similar_cases": [],
        "uncertainty": "No retrieved evidence.",
    }


def test_risk_tier_by_rank_percentile():
    assert briefs.risk_tier(99.5) == "high"
    assert briefs.risk_tier(99.0) == "high"
    assert briefs.risk_tier(97.0) == "medium"
    assert briefs.risk_tier(95.0) == "medium"
    assert briefs.risk_tier(50.0) == "low"
    assert briefs.risk_tier(None) == "low"


@pytest.mark.needs_data
def test_brief_snapshot_and_prompt_order():
    snap = briefs.brief_snapshot(get_client_snapshot(CID))
    assert snap["risk_tier"] == briefs.risk_tier(snap["risk_rank_pct"])
    assert "not a verified probability" in snap["risk_note"]
    ev = FX["evidence"][:3]
    text = briefs.format_evidence(ev)
    # The top-ranked chunk sits last, nearest the task instruction.
    assert text.rfind(ev[0]["id"]) > text.rfind(ev[2]["id"])
    prompt = briefs.build_brief_prompt(snap, ev)
    assert prompt.index("SNAPSHOT:") < prompt.index("EVIDENCE:") < prompt.index("Write the")
    assert "<untrusted_data>" in prompt
    assert "no evidence retrieved" in briefs.format_evidence([])


def test_top_clients_are_highest_oof_scores():
    ids = briefs.top_clients(5)
    oof = pd.read_parquet(SNAPSHOT_CONFIG["oof_path"]).set_index("client_id")["score"]
    assert len(ids) == 5 and len(set(ids)) == 5
    assert oof.loc[ids].min() >= oof.drop(ids).max()
    assert str(SNAPSHOT_CONFIG["oof_path"]).endswith("M10_v4.parquet")


@pytest.mark.needs_data
def test_write_brief_uses_schema_and_parses():
    snap = briefs.brief_snapshot(get_client_snapshot(CID))
    client = fake_client(json.dumps(_brief_for(snap)))
    brief, raw, usage = briefs.write_brief(snap, [], client, model="m")
    cfg = client.models.calls[0]["config"]
    assert cfg["system_instruction"] == briefs.BRIEF_WRITER_SYSTEM_PROMPT
    assert cfg["response_schema"] is briefs.InspectorBrief
    assert cfg["temperature"] == 0.0
    assert brief["client_id"] == CID and usage == {"input_tokens": 100, "output_tokens": 25}
    res = score_brief(brief, snap, [])
    assert res["schema_valid"] and res["citation_valid_rate"] == 1.0


@pytest.mark.needs_data
def test_generate_run_caches_and_filters(tmp_path):
    snap = briefs.brief_snapshot(get_client_snapshot(CID))
    good = json.dumps(_brief_for(snap))
    retr = StubRetriever(FX["evidence"])
    queries = json.dumps(["meter bypass", "counter_statue 5", "estimated readings"])

    client = fake_client(queries, good)
    recs = briefs.generate_run(
        "filtered_hybrid",
        [CID],
        client=client,
        model="m",
        out_dir=tmp_path,
        retriever=retr,
        sleep=0,
    )
    assert recs[0]["brief"]["client_id"] == CID
    assert len(recs[0]["evidence_ids"]) == briefs.EVIDENCE_K
    assert all(f["exclude_client_id"] == CID for f in retr.filters)
    assert (tmp_path / "filtered_hybrid" / f"{CID}.json").is_file()
    assert (tmp_path / "queries" / f"{CID}.json").is_file()

    # The unfiltered run reuses the cached queries and passes no client filters.
    retr.filters.clear()
    client2 = fake_client(good)
    briefs.generate_run(
        "hybrid", [CID], client=client2, model="m", out_dir=tmp_path, retriever=retr, sleep=0
    )
    assert len(client2.models.calls) == 1 and retr.filters == [None, None, None]

    # A rerun is served from the cache without any LLM call.
    client3 = fake_client(RuntimeError("must not be called"))
    again = briefs.generate_run(
        "filtered_hybrid", [CID], client=client3, model="m", out_dir=tmp_path, retriever=retr
    )
    assert again[0]["brief"] == recs[0]["brief"] and client3.models.calls == []


@pytest.mark.needs_data
def test_generate_run_records_llm_failure(tmp_path):
    client = fake_client(ValueError("400 bad request"))
    recs = briefs.generate_run(
        "no_evidence", [CID], client=client, model="m", out_dir=tmp_path, sleep=0
    )
    assert recs[0]["brief"] is None and "bad request" in recs[0]["error"]
    rows = briefs.score_records(recs, {}, judge=None)
    assert rows[0]["llm_error"] and not rows[0]["schema_valid"]


def test_generate_with_retry_backs_off_on_429(monkeypatch):
    monkeypatch.setattr(briefs.time, "sleep", lambda s: None)
    client = fake_client(RuntimeError("429 RESOURCE_EXHAUSTED"), "ok")
    resp = briefs.generate_with_retry(client, "m", "hi", {})
    assert resp.text == "ok" and len(client.models.calls) == 2
    with pytest.raises(ValueError):
        briefs.generate_with_retry(fake_client(ValueError("400 invalid")), "m", "hi", {})


def test_summarise_runs_table():
    rows = [
        {
            "run": run,
            "llm_error": False,
            "schema_valid": True,
            "citation_valid_rate": 1.0,
            "number_accuracy_rate": 0.5,
            "rule_applicability_rate": 1.0,
            "faithfulness_mean": 4.0,
            "all_pass": False,
            "risk_tier_ok": True,
            "id_fields_ok": True,
            "own_case_in_evidence": run == "hybrid",
            "expired_sop_in_evidence": False,
        }
        for run in briefs.RUNS
    ]
    t = briefs.summarise_runs(rows, "lexical").set_index("run")
    assert list(t.index) == list(briefs.RUNS)
    assert t.loc["hybrid", "own_case_card_retrieved_pct"] == 100.0
    assert t.loc["no_evidence", "number_accuracy_pct"] == 50.0


def test_gemini_judge_batches_caches_and_falls_back(tmp_path):
    snap = FX["snapshots"][0]
    brief = FX["briefs"][0]
    client = fake_client(json.dumps([5, 4, 3, 2]))
    judge = make_gemini_judge(model="m", client=client, cache_path=tmp_path / "judge.json")
    judge.prime(brief, snap, FX["evidence"])
    assert len(client.models.calls) == 1
    assert "independent fact-checker" in client.models.calls[0]["contents"]
    res = score_brief(brief, snap, FX["evidence"], judge=judge)
    assert res["details"]["faithfulness"]["scores"] == [5, 4, 3, 2]
    assert len(client.models.calls) == 1

    # A new judge reads the disk cache; an API failure falls back to the lexical judge.
    broken = fake_client(ValueError("400 invalid"))
    judge2 = make_gemini_judge(model="m", client=broken, cache_path=tmp_path / "judge.json")
    assert score_brief(brief, snap, FX["evidence"], judge=judge2)["faithfulness_mean"] == 3.5
    score = judge2("An unseen claim.", ["some source"])
    assert 1 <= score <= 5 and judge2.n_fallback == 1


def test_cache_model_picks_the_saved_model(tmp_path):
    for i, model in enumerate(["a", "b", "b"]):
        (tmp_path / f"{i}.json").write_text(json.dumps({"model": model, "brief": {}}))
    (tmp_path / "3.json").write_text(json.dumps({"model": "c", "brief": None}))
    paths = [tmp_path / f"{i}.json" for i in range(5)]
    assert briefs.cache_model(paths) == "b"
    assert briefs.cache_model([tmp_path / "missing.json"]) is None


def test_cached_brief_from_an_old_scoring_model_is_not_reused():
    now = SNAPSHOT_CONFIG["model_version"]
    assert briefs._current({"model": "m", "model_version": now}, "m")
    assert not briefs._current({"model": "m", "model_version": "M10_v1"}, "m")
    assert not briefs._current({"model": "m"}, "m")
    assert not briefs._current({"model": "other", "model_version": now}, "m")
