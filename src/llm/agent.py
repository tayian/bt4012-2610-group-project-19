# Investigation agents: a single function-calling agent and a multi-agent workflow (planner,
# evidence gatherer, brief writer, auditor) that write the same Inspector Brief as briefs.py.
# Run with `python3 -m src.llm.agent`; a client whose log already holds a brief is skipped.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.llm.agent_eval import LOG_DIR, RunLogger, load_logs
from src.llm.briefs import (
    BRIEF_WRITER_SYSTEM_PROMPT,
    DEFAULT_MODEL,
    N_CLIENTS,
    DailyQuotaExceeded,
    InspectorBrief,
    brief_config,
    brief_snapshot,
    cache_model,
    generate_with_retry,
    get_gemini_client,
    top_clients,
    try_parse_brief,
    usage_dict,
    write_brief,
)
from src.llm.faithfulness import (
    auditor_feedback,
    lexical_judge,
    make_gemini_judge,
    score_brief,
    score_run,
)
from src.llm.query_rewriter import low_thinking
from src.llm.retrieval import client_filters, get_retriever
from src.llm.snapshot import SNAPSHOT_CONFIG, get_client_snapshot

MAX_TOOL_CALLS = 8
MAX_TURNS = 12
PLOT_DIR = Path(LOG_DIR) / "plots"
COMPARISON_PATH = Path("results/tables/agent_comparison.csv")
JUDGE_CACHE = Path(LOG_DIR) / "judge_cache.json"

# Nigrini's first-digit MAD bands (also used by SOP-v2025-5.1: nonconformity above 0.015).
BENFORD_BANDS = ((0.006, "close conformity"), (0.012, "acceptable"), (0.015, "marginal"))
BENFORD_MIN_READINGS = 20
ROUND_SHARE_FLAG = 0.40

# Filters the agent may add to retrieve_evidence. The client's own filters (alert date, meter
# type, own case card excluded) are always applied on top and cannot be overridden.
AGENT_FILTER_KEYS = ("doc", "field", "value", "section")


# Each tool takes plain arguments and returns a JSON-serialisable dict.
def get_client_profile(client_id: str) -> dict[str, Any]:
    snap = get_client_snapshot(client_id)
    return {"client_id": snap["client_id"], "split": snap["split"], **snap["profile"]}


def get_risk_score(client_id: str) -> dict[str, Any]:
    snap = brief_snapshot(get_client_snapshot(client_id))
    return {
        "client_id": snap["client_id"],
        "risk_score": snap["risk_score"],
        "risk_rank_pct": snap["risk_rank_pct"],
        "risk_tier": snap["risk_tier"],
        "model_version": snap["model_version"],
        "note": snap["risk_note"],
    }


def explain_shap(client_id: str) -> dict[str, Any]:
    snap = get_client_snapshot(client_id)
    feats = snap["key_features"]
    drivers = []
    for s in snap["shap_top5"]:
        drivers.append(
            {
                "rank": s["rank"],
                "feature": s["feature"],
                "value": s["value"],
                "train_pct": (feats.get(s["feature"]) or {}).get("train_pct"),
                "shap_value": s["shap_value"],
                "direction": "raises risk" if s["shap_value"] > 0 else "lowers risk",
            }
        )
    return {
        "client_id": snap["client_id"],
        "source": "SHAP of the tuned LightGBM component of the stacked model",
        "drivers": drivers,
    }


def _client_invoices(client_id: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(
        SNAPSHOT_CONFIG["interim_invoice_clean_path"],
        columns=["client_id", *columns],
        filters=[("client_id", "==", str(client_id))],
    )


def benford_check(client_id: str) -> dict[str, Any]:
    snap = get_client_snapshot(client_id)
    inv = _client_invoices(client_id, ["cons"])
    n_pos = int((inv["cons"] > 0).sum())
    mad, rshare = snap["benford_mad"], snap["round_reading_share"]
    if mad is None or n_pos < BENFORD_MIN_READINGS:
        flag = "insufficient readings"
    else:
        flag = next((label for cut, label in BENFORD_BANDS if mad <= cut), "nonconformity")
    rpct = (snap["key_features"].get("round_reading_share") or {}).get("train_pct")
    return {
        "client_id": snap["client_id"],
        "benford_mad": mad,
        "round_reading_share": rshare,
        "round_reading_share_train_pct": rpct,
        "n_positive_readings": n_pos,
        "benford_flag": flag,
        "round_reading_flag": bool(rshare is not None and rshare > ROUND_SHARE_FLAG),
        "caveat": (
            "Benford deviation is a screening signal only; in this data benford_mad is on "
            "average lower for fraud clients, and round_reading_share carries no signal."
        ),
    }


def retrieve_evidence(
    query: str,
    filters: dict[str, Any] | None = None,
    retriever: Any | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    retriever = retriever or get_retriever()
    hits = retriever.retrieve(query=query, filters=filters, top_k=top_k, method="hybrid")
    keep = ("id", "doc", "section", "version", "valid_from", "valid_to", "counter_type", "text")
    return {"query": query, "chunks": [{k: h.get(k) for k in keep} for h in hits]}


def with_years(history: dict[str, Any]) -> dict[str, Any]:
    # Years as plain numbers: the number check ignores full dates, so a brief saying "drops in
    # 2009" would otherwise count as an unsupported number.
    years = sorted({int(d[:4]) for d in history.get("drop_dates", [])})
    first, last = history.get("first_invoice"), history.get("last_invoice")
    return {
        **history,
        "drop_years": years,
        "first_year": int(first[:4]) if first else None,
        "last_year": int(last[:4]) if last else None,
    }


def plot_history(client_id: str, out_dir: Path | str = PLOT_DIR) -> dict[str, Any]:
    # Consumption per month (cons_pm) per meter type; a drop is a reading below half the
    # median of the same meter type's previous three readings.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inv = _client_invoices(client_id, ["invoice_date", "counter_type", "cons_pm"])
    inv = inv.dropna(subset=["invoice_date", "cons_pm"])
    series = (
        inv.groupby(["counter_type", "invoice_date"], as_index=False)["cons_pm"]
        .sum()
        .sort_values(["counter_type", "invoice_date"])
    )
    out = Path(out_dir) / f"{client_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    drops = []
    for ctype, g in series.groupby("counter_type"):
        vals = g["cons_pm"].to_numpy(dtype=float)
        dates = g["invoice_date"].to_numpy()
        ref = pd.Series(vals).shift(1).rolling(3, min_periods=3).median().to_numpy()
        is_drop = np.where(np.isfinite(ref) & (ref > 0), vals < 0.5 * ref, False)
        ax.plot(dates, vals, marker=".", lw=1, label=str(ctype))
        ax.scatter(dates[is_drop], vals[is_drop], color="red", zorder=3, s=30)
        drops += [(str(pd.Timestamp(d).date()), str(ctype)) for d in dates[is_drop]]
    ax.set_title(f"{client_id}: consumption per month (red = drop below half the recent median)")
    ax.set_ylabel("cons_pm")
    if len(series):
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    drops.sort()
    inv_dates = inv["invoice_date"]
    return with_years(
        {
            "client_id": str(client_id),
            "path": str(out),
            "n_invoices": int(len(inv)),
            "first_invoice": str(pd.Timestamp(inv_dates.min()).date()) if len(inv) else None,
            "last_invoice": str(pd.Timestamp(inv_dates.max()).date()) if len(inv) else None,
            "n_drops": len(drops),
            "drop_dates": [f"{d} ({c})" for d, c in drops[-10:]],
        }
    )


_CLIENT_ARG = {
    "type": "object",
    "properties": {"client_id": {"type": "string", "description": "e.g. train_Client_42"}},
    "required": ["client_id"],
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_client_profile",
        "description": "Client category, region, district, tenure in years, meter types, "
        "number of counters and invoice count.",
        "parameters": _CLIENT_ARG,
    },
    {
        "name": "get_risk_score",
        "description": "The fraud model's score for the client, its rank percentile among "
        "training clients, the risk tier and the model version.",
        "parameters": _CLIENT_ARG,
    },
    {
        "name": "explain_shap",
        "description": "The 5 features that moved the client's score most (SHAP), with the "
        "feature value and its training percentile.",
        "parameters": _CLIENT_ARG,
    },
    {
        "name": "benford_check",
        "description": "Benford first-digit MAD, share of round meter readings, number of "
        "positive readings and an interpretation flag.",
        "parameters": _CLIENT_ARG,
    },
    {
        "name": "retrieve_evidence",
        "description": "Search the playbook (TYP-*), code book (CODE-*), inspection SOP "
        "(SOP-*) and precedent case cards (CASE-*). Returns the top 5 chunks with IDs and "
        "metadata. Use plain policy-manual words and exact codes, e.g. 'counter_statue 5'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "object",
                    "description": "Optional narrowing filters.",
                    "properties": {
                        "doc": {"type": "string", "enum": ["playbook", "codebook", "sop", "case"]},
                        "field": {
                            "type": "string",
                            "description": "code book field, e.g. counter_statue",
                        },
                        "value": {"type": "string", "description": "code value, e.g. 5"},
                    },
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "plot_history",
        "description": "Draw the client's monthly consumption with drops marked; returns the "
        "PNG path, invoice period and the dates of marked drops.",
        "parameters": _CLIENT_ARG,
    },
]

TOOL_FUNCS = {
    "get_client_profile": get_client_profile,
    "get_risk_score": get_risk_score,
    "explain_shap": explain_shap,
    "benford_check": benford_check,
    "retrieve_evidence": retrieve_evidence,
    "plot_history": plot_history,
}


class ToolBox:
    # Tools bound to one client: every call is about this client (a different client_id is
    # replaced), retrieval always carries the client's filters, and outputs are kept for the
    # brief writer and the checks.

    def __init__(self, client_id: str, logger: RunLogger | None = None, retriever: Any = None):
        self.client_id = str(client_id)
        self.snapshot = get_client_snapshot(self.client_id)
        self.logger = logger
        self.retriever = retriever
        self.n_calls = 0
        self.outputs: dict[str, Any] = {}
        self.chunks: dict[str, dict[str, Any]] = {}

    def call(self, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
        args = dict(args or {})
        t0 = time.perf_counter()
        self.n_calls += 1
        try:
            if name not in TOOL_FUNCS:
                result = {"error": f"unknown tool {name}"}
            elif name == "retrieve_evidence":
                extra = {
                    k: v for k, v in (args.get("filters") or {}).items() if k in AGENT_FILTER_KEYS
                }
                filters = client_filters(self.snapshot, **extra)
                result = retrieve_evidence(
                    str(args.get("query", "")), filters, retriever=self.retriever
                )
                for c in result["chunks"]:
                    self.chunks.setdefault(c["id"], c)
            else:
                result = TOOL_FUNCS[name](self.client_id)
                self.outputs[name] = result
        except Exception as exc:
            result = {"error": repr(exc)[:300]}
        if self.logger is not None:
            self.logger.tool_call(name, args, result, seconds=round(time.perf_counter() - t0, 3))
            if name == "retrieve_evidence" and "chunks" in result:
                self.logger.evidence([c["id"] for c in result["chunks"]])
        return result

    def evidence(self) -> list[dict[str, Any]]:
        return list(self.chunks.values())

    def agent_snapshot(self) -> dict[str, Any]:
        # What the brief writer may cite as SNAPSHOT: the risk header plus the tool outputs.
        snap = brief_snapshot(self.snapshot)
        head = ("client_id", "alert_date", "risk_score", "risk_rank_pct", "risk_tier")
        return {
            **{k: snap[k] for k in head},
            "model_version": snap["model_version"],
            "risk_note": snap["risk_note"],
            "tool_outputs": self.outputs,
        }


def scoring_snapshot(client_id: str, tool_outputs: dict[str, Any]) -> dict[str, Any]:
    # Checks run against the full client snapshot plus what the tools returned (tool outputs
    # are deterministic facts the agent was told to cite as SNAPSHOT).
    snap = brief_snapshot(get_client_snapshot(client_id))
    outputs = dict(tool_outputs)
    if "path" in outputs.get("plot_history", {}):
        outputs["plot_history"] = with_years(outputs["plot_history"])
    snap["tool_outputs"] = outputs
    return snap


def _types():
    from google.genai import types

    return types


def gemini_tools() -> list[Any]:
    types = _types()
    decls = [
        types.FunctionDeclaration(
            name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]
        )
        for t in TOOL_SCHEMAS
    ]
    return [types.Tool(function_declarations=decls)]


def tool_loop(
    client: Any,
    model: str,
    system_prompt: str,
    contents: list[Any],
    box: ToolBox,
    logger: RunLogger,
    role: str,
    max_calls: int = MAX_TOOL_CALLS,
) -> str:
    # Function-calling loop. Stops when the model answers in text or the tool budget is spent;
    # calls beyond the budget are answered with an error instead of being run.
    types = _types()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=gemini_tools(),
        temperature=0.0,
        **low_thinking(model),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    used = 0
    text = ""
    for _ in range(MAX_TURNS):
        resp = generate_with_retry(client, model, contents, config)
        logger.llm_call(usage_dict(resp), role=role)
        calls = resp.function_calls or []
        if not resp.candidates:
            break
        contents.append(resp.candidates[0].content)
        if not calls:
            text = resp.text or ""
            break
        parts = []
        for fc in calls:
            if used < max_calls:
                used += 1
                result = box.call(fc.name, dict(fc.args or {}))
            else:
                result = {"error": f"tool budget of {max_calls} calls is spent; stop calling tools"}
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=getattr(fc, "id", None), name=fc.name, response={"result": result}
                    )
                )
            )
        contents.append(types.Content(role="user", parts=parts))
        if used >= max_calls:
            break
    return text


def _user(text: str) -> Any:
    types = _types()
    return types.Content(role="user", parts=[types.Part(text=text)])


TOOL_GUIDE = (
    "Tools: get_risk_score, get_client_profile, explain_shap, benford_check, plot_history "
    "(facts about the client; cite them as [SNAPSHOT]) and retrieve_evidence (playbook TYP-*, "
    "code book CODE-*, SOP clauses SOP-*, precedent cases CASE-*; cite the chunk IDs it "
    f"returns). You may make at most {MAX_TOOL_CALLS} tool calls in total. Query "
    "retrieve_evidence once per suspicious pattern, in plain policy-manual words, with exact "
    "codes when a meter status or remark code matters. Make independent calls together in "
    "one turn."
)

SINGLE_AGENT_PROMPT = (
    BRIEF_WRITER_SYSTEM_PROMPT
    + "\n\nYou are an investigation agent. First gather facts and evidence with tools, then "
    "write the brief. " + TOOL_GUIDE + " When you have enough, stop calling tools and reply DONE."
)

PLANNER_PROMPT = (
    "You plan the investigation of one client flagged by a utility-fraud model. You see the "
    "client's profile and model risk score. You do not decide guilt. Write 2 to 4 short "
    "questions an evidence gatherer should answer with tools (model drivers, reading "
    "irregularities, consumption history, applicable playbook typologies, code meanings, SOP "
    'clauses, precedent cases). Return JSON {"questions": [...]}.'
)

GATHERER_PROMPT = (
    "You gather evidence for a utility-fraud investigation plan. You do not write the brief "
    "and you do not decide guilt. "
    + TOOL_GUIDE
    + " When done, reply with short notes: one line per question naming the IDs (chunk IDs "
    "or SNAPSHOT) that answer it, or saying that no evidence was found."
)


class Plan(BaseModel):
    questions: list[str]


def final_brief_from_history(
    client: Any, model: str, contents: list[Any], client_id: str, logger: RunLogger
) -> dict[str, Any] | None:
    contents.append(
        _user(
            f"Now write the Inspector Brief for client {client_id} as JSON. Copy risk_score, "
            "risk_rank_pct and risk_tier from get_risk_score. Cite only [SNAPSHOT] and chunk "
            "IDs that retrieve_evidence returned."
        )
    )
    config = brief_config(model, InspectorBrief)
    config["system_instruction"] = SINGLE_AGENT_PROMPT
    resp = generate_with_retry(client, model, contents, config)
    logger.llm_call(usage_dict(resp), role="writer")
    return try_parse_brief(resp.text or "")


def run_single(
    client_id: str,
    client: Any,
    model: str = DEFAULT_MODEL,
    log_dir: str = LOG_DIR,
    retriever: Any = None,
) -> dict[str, Any]:
    snap = get_client_snapshot(client_id)
    with RunLogger("single", client_id, f"{snap['model_version']}|{model}", log_dir) as log:
        box = ToolBox(client_id, log, retriever)
        contents = [
            _user(
                f"Investigate client {client_id} (alert date {snap['alert_date']}) and prepare "
                "an Inspector Brief."
            )
        ]
        tool_loop(client, model, SINGLE_AGENT_PROMPT, contents, box, log, role="agent")
        brief = final_brief_from_history(client, model, contents, client_id, log)
        log.rec["llm_model"] = model
        log.rec["tool_outputs"] = box.outputs
        log.set_brief(brief)
    return log.rec


def audit(
    brief: dict[str, Any] | None,
    snap: dict[str, Any],
    evidence: list[dict[str, Any]],
    judge: Any,
) -> tuple[bool, list[str], dict[str, Any]]:
    # The brief checks plus the LLM judge. Pass = every automatic check at 100% and no evidence
    # point judged 2 or below.
    if brief is None:
        return False, ["The brief is not valid JSON; return JSON matching the schema."], {}
    if hasattr(judge, "prime"):
        judge.prime(brief, snap, evidence)
    res = score_brief(brief, snap, evidence, judge=judge)
    weak = any(s <= 2 for s in res["details"]["faithfulness"]["scores"])
    ok = bool(res["all_pass"]) and not weak
    summary = {k: v for k, v in res.items() if k != "details"}
    summary["faithfulness_scores"] = res["details"]["faithfulness"]["scores"]
    return ok, auditor_feedback(res), summary


def run_multi(
    client_id: str,
    client: Any,
    judge: Any,
    model: str = DEFAULT_MODEL,
    log_dir: str = LOG_DIR,
    retriever: Any = None,
) -> dict[str, Any]:
    snap = get_client_snapshot(client_id)
    with RunLogger("multi", client_id, f"{snap['model_version']}|{model}", log_dir) as log:
        log.rec["llm_model"] = model
        box = ToolBox(client_id, log, retriever)

        # 1. Planner reads the profile and the risk score.
        profile = box.call("get_client_profile", {"client_id": client_id})
        risk = box.call("get_risk_score", {"client_id": client_id})
        plan_cfg = {
            "system_instruction": PLANNER_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": Plan,
            "temperature": 0.0,
            **low_thinking(model),
        }
        resp = generate_with_retry(
            client,
            model,
            "PROFILE:\n" + json.dumps(profile) + "\n\nRISK:\n" + json.dumps(risk),
            plan_cfg,
        )
        log.llm_call(usage_dict(resp), role="planner")
        try:
            questions = Plan.model_validate_json(resp.text or "").questions[:4]
        except Exception:
            questions = ["Which model drivers and reading irregularities explain the score?"]
        log.rec["plan"] = questions

        # 2. Evidence Gatherer answers the questions with tools (its own budget of 8 calls).
        gather_msg = (
            f"Client {client_id}, alert date {snap['alert_date']}.\nPLAN:\n"
            + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
            + "\nThe profile and risk score are already known:\n"
            + json.dumps({"profile": profile, "risk": risk})
        )
        notes = tool_loop(
            client, model, GATHERER_PROMPT, [_user(gather_msg)], box, log, role="gatherer"
        )
        log.rec["gatherer_notes"] = notes
        log.rec["tool_outputs"] = box.outputs

        # 3. Brief Writer uses the gathered facts and chunks only.
        writer_snap = box.agent_snapshot()
        evidence = box.evidence()
        context = "INVESTIGATION PLAN:\n" + "\n".join(questions)
        brief, _, usage = write_brief(writer_snap, evidence, client, model, extra=context)
        log.llm_call(usage, role="writer")

        # 4. Auditor: checks plus LLM judge; one revision round, else human review.
        snap_full = scoring_snapshot(client_id, box.outputs)
        ok, feedback, summary = audit(brief, snap_full, evidence, judge)
        audits = [summary]
        verdict = "pass"
        if not ok:
            revise = (
                context + "\n\nAUDITOR FEEDBACK on your previous brief (fix every item, change "
                "nothing else):\n- "
                + "\n- ".join(feedback or ["Unspecified failure; recheck every citation."])
                + "\n\nPREVIOUS BRIEF:\n"
                + json.dumps(brief, ensure_ascii=False)
            )
            brief2, _, usage = write_brief(writer_snap, evidence, client, model, extra=revise)
            log.llm_call(usage, role="writer_revision")
            ok, feedback2, summary2 = audit(brief2, snap_full, evidence, judge)
            audits.append(summary2)
            log.rec["auditor_feedback"] = feedback
            if brief2 is not None:
                brief = brief2
            verdict = "pass_after_revision" if ok else "needs_human_review"
            if not ok:
                log.rec["auditor_feedback_final"] = feedback2
        log.rec["audits"] = audits
        log.set_brief(brief)
        log.set_verdict(verdict)
    return log.rec


def _finished(rec: dict[str, Any], model: str | None) -> bool:
    # A log with a brief, no error, the same LLM (any if model is None) and the current model.
    return (
        rec.get("brief") is not None
        and not rec.get("error")
        and (model is None or rec.get("llm_model") == model)
        and str(rec.get("model_version", "")).startswith(f"{SNAPSHOT_CONFIG['model_version']}|")
    )


def _done(system: str, client_id: str, log_dir: str, model: str) -> bool:
    path = Path(log_dir) / system / f"{client_id}.json"
    return path.is_file() and _finished(json.loads(path.read_text(encoding="utf-8")), model)


def compare(
    systems: list[str],
    judge: Any,
    judge_name: str,
    log_dir: str = LOG_DIR,
    model: str | None = None,
) -> pd.DataFrame:
    # One row per system, over the clients every system has a finished run for (so a partly
    # finished system is never compared on a different set of clients).
    chunks = dict(get_retriever().id_to_chunk)
    logs = {
        system: {
            log["client_id"]: log for log in load_logs(system, log_dir) if _finished(log, model)
        }
        for system in systems
    }
    common = sorted(set.intersection(*(set(v) for v in logs.values()))) if logs else []

    rows = []
    for system in systems:
        recs = [logs[system][cid] for cid in common]
        items = [
            {
                "brief": r["brief"],
                "snapshot": scoring_snapshot(r["client_id"], r.get("tool_outputs") or {}),
                "evidence": [chunks[i] for i in r["evidence_ids"] if i in chunks],
            }
            for r in recs
        ]
        if hasattr(judge, "prime"):
            for it in items:
                judge.prime(it["brief"], it["snapshot"], it["evidence"])
        agg, _ = score_run(items, judge=judge) if items else ({}, [])
        verdicts = [r.get("auditor_verdict") for r in recs]
        multi = system == "multi" and bool(recs)
        tokens = [r["input_tokens"] + r["output_tokens"] for r in recs]
        rows.append(
            {
                "system": system,
                "n_clients": len(recs),
                **agg,
                "tool_calls_mean": float(np.mean([r["n_tool_calls"] for r in recs]))
                if recs
                else None,
                "tool_calls_max": max((r["n_tool_calls"] for r in recs), default=None),
                "llm_calls_mean": float(np.mean([len(r["llm_calls"]) for r in recs]))
                if recs
                else None,
                "tokens_mean": float(np.mean(tokens)) if recs else None,
                "seconds_mean": float(np.mean([r["seconds"] for r in recs])) if recs else None,
                "seconds_median": float(np.median([r["seconds"] for r in recs])) if recs else None,
                "auditor_pass_rate": (
                    float(np.mean([v in ("pass", "pass_after_revision") for v in verdicts]))
                    if multi
                    else None
                ),
                "auditor_first_pass_rate": (
                    float(np.mean([v == "pass" for v in verdicts])) if multi else None
                ),
                "needs_human_review_share": (
                    float(np.mean([v == "needs_human_review" for v in verdicts])) if multi else None
                ),
                "judge": judge_name,
                "llm_model": ",".join(sorted({str(r.get("llm_model")) for r in recs})),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Single agent vs multi-agent Inspector Briefs.")
    p.add_argument("--systems", nargs="+", default=["single", "multi"], choices=["single", "multi"])
    p.add_argument("--n-clients", type=int, default=N_CLIENTS)
    p.add_argument(
        "--model", default=None, help="default: the logged runs' model, else GEMINI_MODEL"
    )
    p.add_argument("--force", action="store_true", help="rerun clients that already have a log")
    p.add_argument("--lexical-judge", action="store_true", help="score offline")
    p.add_argument("--out", default=str(COMPARISON_PATH))
    a = p.parse_args()
    ids = top_clients(a.n_clients)
    logged = [Path(LOG_DIR) / s / f"{cid}.json" for s in a.systems for cid in ids]
    a.model = a.model or cache_model(logged) or DEFAULT_MODEL
    print(f"Agent model: {a.model}")

    if a.lexical_judge:
        judge, judge_name = lexical_judge, "lexical"
    else:
        judge = make_gemini_judge(model=a.model, cache_path=JUDGE_CACHE)
        judge_name = f"gemini:{a.model}"

    client = None
    retriever = get_retriever()
    stop = False
    for system in a.systems:
        for i, cid in enumerate(ids, start=1):
            if stop or (not a.force and _done(system, cid, LOG_DIR, a.model)):
                continue
            client = client or get_gemini_client()
            t0 = time.perf_counter()
            try:
                if system == "single":
                    rec = run_single(cid, client, a.model, retriever=retriever)
                else:
                    rec = run_multi(cid, client, judge, a.model, retriever=retriever)
                status = rec.get("auditor_verdict") or ("ok" if rec.get("brief") else "no brief")
            except DailyQuotaExceeded as exc:
                # Drop the half-finished log so a later rerun redoes this client.
                (Path(LOG_DIR) / system / f"{cid}.json").unlink(missing_ok=True)
                print(f"Stopped: {exc}. Rerun later to finish; finished runs are kept.")
                stop = True
                continue
            except Exception as exc:
                status = f"error {exc!r}"[:200]
            print(f"{system} {i}/{len(ids)} {cid}: {status} ({time.perf_counter() - t0:.0f}s)")

    try:
        table = compare(a.systems, judge, judge_name, model=a.model)
    except DailyQuotaExceeded as exc:
        raise SystemExit(f"Stopped while scoring: {exc}. Rerun later, or use --lexical-judge.")
    if getattr(judge, "n_fallback", 0):
        table["judge"] = f"{judge_name} ({judge.n_fallback} points fell back to lexical)"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
