# Agent-evaluation harness, used to compare the single agent and the multi-agent workflow on
# the same 30 clients.
#
# Logging (wrap each agent run):
#
#   from src.llm.agent_eval import RunLogger
#   with RunLogger(system="multi", client_id=cid, model_version="M10@model-freeze") as log:
#       log.tool_call("get_risk_score", {"client_id": cid}, result)
#       log.llm_call(response.usage)          # anthropic usage object or dict
#       log.evidence(["SOP-v2025-2.1", "CASE-train_Client_9"])
#       log.set_brief(brief_dict)
#       log.set_verdict("pass")               # Auditor verdict, multi-agent only
#   -> results/agent_logs/<system>/<client_id>.json
#
# Comparison table (report T6):
#
#   python3 -m src.llm.agent_eval --systems single multi
#   -> results/tables/agent_comparison.csv
#      rows = systems; columns = citation validity, number accuracy, rule applicability,
#      faithfulness, all-pass share, tool calls, tokens, seconds, needs_human_review share

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from statistics import mean, median

LOG_DIR = "results/agent_logs"


class RunLogger:
    def __init__(self, system: str, client_id: str, model_version: str = "", log_dir=LOG_DIR):
        self.path = os.path.join(log_dir, system, f"{client_id}.json")
        self.rec = {
            "system": system,
            "client_id": client_id,
            "model_version": model_version,
            "tool_calls": [],
            "llm_calls": [],
            "evidence_ids": [],
            "brief": None,
            "auditor_verdict": None,
            "error": None,
        }

    def __enter__(self):
        self._t = time.perf_counter()
        self.rec["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        return self

    def tool_call(self, name, args, result=None, seconds=None):
        self.rec["tool_calls"].append(
            {"tool": name, "args": args, "seconds": seconds, "result_preview": str(result)[:300]}
        )

    def llm_call(self, usage, role: str = ""):
        if not isinstance(usage, dict):
            usage = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
            }
        self.rec["llm_calls"].append({"role": role, **usage})

    def evidence(self, ids):
        self.rec["evidence_ids"] += [i for i in ids if i not in self.rec["evidence_ids"]]

    def set_brief(self, brief):
        self.rec["brief"] = brief

    def set_verdict(self, verdict):
        self.rec["auditor_verdict"] = verdict

    def __exit__(self, et, ev, tb):
        self.rec["seconds"] = time.perf_counter() - self._t
        if ev is not None:
            self.rec["error"] = repr(ev)
        self.rec["n_tool_calls"] = len(self.rec["tool_calls"])
        self.rec["input_tokens"] = sum(c.get("input_tokens", 0) for c in self.rec["llm_calls"])
        self.rec["output_tokens"] = sum(c.get("output_tokens", 0) for c in self.rec["llm_calls"])
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(self.rec, open(self.path, "w"), indent=2, default=str)
        return False


def load_logs(system, log_dir=LOG_DIR):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(log_dir, system, "*.json")))]


def summarise(system, snapshot_fn, evidence_fn, judge=None, log_dir=LOG_DIR) -> dict:
    # snapshot_fn(client_id) -> snapshot dict; evidence_fn(log) -> retrieved chunks (normally
    # the chunks whose IDs are in log['evidence_ids']).
    from src.llm.faithfulness import score_run

    logs = load_logs(system, log_dir)
    items = [
        {
            "brief": log["brief"],
            "snapshot": snapshot_fn(log["client_id"]),
            "evidence": evidence_fn(log),
        }
        for log in logs
        if log.get("brief")
    ]
    agg, _ = score_run(items, judge=judge) if items else ({}, [])
    tok = [log["input_tokens"] + log["output_tokens"] for log in logs]
    return {
        "system": system,
        "n_runs": len(logs),
        "n_briefs": len(items),
        **{k: v for k, v in agg.items() if k != "n_briefs"},
        "tool_calls_mean": mean(log["n_tool_calls"] for log in logs) if logs else None,
        "tokens_mean": mean(tok) if tok else None,
        "seconds_median": median(log["seconds"] for log in logs) if logs else None,
        "needs_human_review_share": (
            mean(log.get("auditor_verdict") == "needs_human_review" for log in logs)
            if logs
            else None
        ),
        "errors": sum(bool(log.get("error")) for log in logs),
    }


def _default_loaders():
    # Uses the retrieval module (src.llm.retrieval, not in the repo yet) when it exists;
    # falls back to the sample fixtures.
    try:
        from src.llm.retrieval import get_client_snapshot, load_chunks

        chunks = {c["id"]: c for c in load_chunks()}
        return get_client_snapshot, lambda log: [
            chunks[i] for i in log["evidence_ids"] if i in chunks
        ]
    except Exception:
        fx = json.load(open("tests/fixtures/sample_briefs.json"))
        snaps = {s["client_id"]: s for s in fx["snapshots"]}
        chunks = {c["id"]: c for c in fx["evidence"]}
        return (
            lambda cid: snaps[cid],
            lambda log: [chunks[i] for i in log["evidence_ids"] if i in chunks],
        )


if __name__ == "__main__":
    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=["single", "multi"])
    p.add_argument("--llm-judge", action="store_true")
    p.add_argument("--out", default="results/tables/agent_comparison.csv")
    a = p.parse_args()
    judge = None
    if a.llm_judge:
        from src.llm.faithfulness import make_anthropic_judge

        judge = make_anthropic_judge()
    snap_fn, ev_fn = _default_loaders()
    rows = [summarise(s, snap_fn, ev_fn, judge) for s in a.systems]
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.round(3).to_string(index=False))
