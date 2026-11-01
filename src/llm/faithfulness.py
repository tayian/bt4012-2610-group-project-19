# Evaluation harness for Inspector Briefs (section 14.6).
#
# Contract (4.4):  score_brief(brief, snapshot, evidence) -> dict
#
#   brief     dict following the 14.5 schema
#   snapshot  dict from the retrieval module's get_client_snapshot(client_id); may be nested.
#             Optional key "alert_date" (YYYY-MM-DD) used for rule applicability; defaults
#             to today.
#   evidence  list of retrieved chunks, each a dict with at least "id" and "text"; SOP chunks
#             also carry "version", "valid_from", "valid_to".
#
# Checks (pass rule = 100% unless stated)
#   1 citation_exists     every cited ID is in the evidence set or is "SNAPSHOT"
#   2 number_accuracy     every number in the brief text appears in SNAPSHOT or in a chunk
#                         the same point cites (rounding and % allowed)
#   3 rule_applicability  every cited SOP clause is valid on the alert date
#   4 faithfulness        judge scores each evidence point 1-5 against its cited text; mean
#                         reported. Pass a real LLM judge via `judge=`; the default is an
#                         offline lexical judge so the harness runs with no API key (it is a
#                         proxy, not the final number).
#   5 retrieval           retrieval_metrics(ranked, gold): Recall@3, Recall@5, MRR
#
# Also: schema_valid, and risk_score / risk_rank_pct must equal the snapshot.

from __future__ import annotations

import datetime as dt
import json
import math
import re
from typing import Callable, Iterable

SCHEMA_KEYS = {
    "client_id": str,
    "risk_score": (int, float),
    "risk_rank_pct": (int, float),
    "risk_tier": str,
    "evidence": list,
    "onsite_checks": list,
    "applicable_rules": list,
    "similar_cases": list,
    "uncertainty": str,
}
TIERS = {"high", "medium", "low"}

# IDs that contain digits and must not be read as numbers
ID_RE = re.compile(
    r"\b(?:SOP-v\d{4}-[\d.]+|TYP-\d+|CODE-[\w.-]+|CASE-[\w.-]+|"
    r"(?:train|test)_Client_\d+)\b"
)
CITE_RE = re.compile(r"\[([^\]]+)\]")
NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")
SOP_RE = re.compile(r"^SOP-(v\d{4})-")


# Utilities
def _flatten_numbers(obj) -> list[float]:
    # All numeric values in a nested snapshot (numbers in strings included).
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            # digits inside field names (slope_last3y -> 3) are definitions the
            # brief may restate in words ("last 3 years")
            out += [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(k))]
            out += _flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _flatten_numbers(v)
    elif isinstance(obj, str):
        out += [n for n, _ in _numbers_in(obj)]
    return out


def _numbers_in(text: str) -> list[tuple[float, str]]:
    # (value, raw token) for each number in text, ignoring citation IDs and dates.
    text = CITE_RE.sub(" ", text)
    text = ID_RE.sub(" ", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)
    out = []
    for tok in NUM_RE.findall(text):
        raw = tok
        t = tok.rstrip("%").replace(",", "")
        try:
            out.append((float(t), raw))
        except ValueError:
            pass
    return out


def _decimals(raw: str) -> int:
    t = raw.rstrip("%").replace(",", "")
    return len(t.split(".")[1]) if "." in t else 0


def number_supported(raw: str, value: float, sources: Iterable[float]) -> bool:
    # True if `value` equals a source number after rounding to the brief's precision, allowing
    # a fraction written as a percentage (0.873 -> 87.3%).
    d = _decimals(raw)
    tol = 0.5 * 10 ** (-d) + 1e-9
    for s in sources:
        cands = [s]
        if raw.endswith("%") or 1 < value <= 100:
            cands.append(s * 100)
        for c in cands:
            if abs(c - value) <= tol:
                return True
    return False


def _brief_text_fields(brief) -> list[tuple[str, str, list[str]]]:
    # (location, text, cited_ids) for every free-text field in the brief.
    out = []
    for i, e in enumerate(brief.get("evidence", [])):
        out.append((f"evidence[{i}]", str(e.get("point", "")), list(e.get("citations", []))))
    for i, t in enumerate(brief.get("onsite_checks", [])):
        out.append((f"onsite_checks[{i}]", str(t), CITE_RE.findall(str(t))))
    for i, r in enumerate(brief.get("applicable_rules", [])):
        out.append((f"applicable_rules[{i}]", str(r.get("action", "")), [r.get("rule_id", "")]))
    out.append(
        (
            "uncertainty",
            str(brief.get("uncertainty", "")),
            CITE_RE.findall(str(brief.get("uncertainty", ""))),
        )
    )
    return out


def all_citations(brief) -> list[str]:
    ids = []
    for _, text, cites in _brief_text_fields(brief):
        ids += [c.strip() for c in cites if c]
        ids += [c.strip() for c in CITE_RE.findall(text)]
    ids += [c for c in brief.get("similar_cases", [])]
    return [i for i in ids if i]


# Checks
def check_schema(brief) -> dict:
    errors = []
    for k, typ in SCHEMA_KEYS.items():
        if k not in brief:
            errors.append(f"missing key {k}")
        elif not isinstance(brief[k], typ):
            errors.append(f"{k} has type {type(brief[k]).__name__}")
    if brief.get("risk_tier") not in TIERS:
        errors.append(f"risk_tier {brief.get('risk_tier')!r} not in {sorted(TIERS)}")
    for i, e in enumerate(brief.get("evidence", []) or []):
        if not isinstance(e, dict) or not e.get("citations"):
            errors.append(f"evidence[{i}] has no citations")
    n = len(brief.get("evidence", []) or [])
    if not 3 <= n <= 5:
        errors.append(f"{n} evidence points (spec: 3 to 5)")
    return {"pass": not errors, "errors": errors}


def check_citations(brief, evidence) -> dict:
    allowed = {c["id"] for c in evidence} | {"SNAPSHOT"}
    cites = all_citations(brief)
    bad = [c for c in cites if c not in allowed]
    rate = 1.0 if not cites else 1 - len(bad) / len(cites)
    return {"pass": not bad, "rate": rate, "n_citations": len(cites), "invalid": bad}


def check_numbers(brief, snapshot, evidence) -> dict:
    chunk_nums = {c["id"]: _flatten_numbers(c.get("text", "")) for c in evidence}
    snap_nums = _flatten_numbers(snapshot)
    bad, n = [], 0
    for loc, text, cites in _brief_text_fields(brief):
        cites = set(cites) | set(CITE_RE.findall(text))
        sources = list(snap_nums)
        for c in cites:
            sources += chunk_nums.get(c, [])
        for value, raw in _numbers_in(text):
            n += 1
            if not number_supported(raw, value, sources):
                bad.append({"where": loc, "number": raw})
    # header numbers must equal the snapshot exactly (rounded to 4 dp)
    for key in ("risk_score", "risk_rank_pct"):
        if key in brief and key in snapshot:
            n += 1
            if round(float(brief[key]), 4) != round(float(snapshot[key]), 4):
                bad.append({"where": key, "number": str(brief[key]), "expected": snapshot[key]})
    rate = 1.0 if n == 0 else 1 - len(bad) / n
    return {"pass": not bad, "rate": rate, "n_numbers": n, "unsupported": bad}


def check_rules(brief, evidence, alert_date=None) -> dict:
    alert = dt.date.fromisoformat(str(alert_date)) if alert_date else dt.date.today()
    meta = {c["id"]: c for c in evidence}
    sop_ids = {c for c in all_citations(brief) if c.startswith("SOP-")}
    sop_ids |= {
        r.get("rule_id", "")
        for r in brief.get("applicable_rules", [])
        if str(r.get("rule_id", "")).startswith("SOP-")
    }
    bad = []
    for sid in sorted(sop_ids):
        c = meta.get(sid)
        if c is None:
            bad.append({"rule": sid, "reason": "not in retrieved evidence"})
            continue
        vf = dt.date.fromisoformat(str(c.get("valid_from", "1900-01-01")))
        vt = dt.date.fromisoformat(str(c.get("valid_to") or "2999-12-31"))
        if not vf <= alert <= vt:
            bad.append({"rule": sid, "reason": f"valid {vf}..{vt}, alert {alert}"})
        m = SOP_RE.match(sid)
        if m and c.get("version") and m.group(1) != c["version"]:
            bad.append({"rule": sid, "reason": f"ID version {m.group(1)} != chunk {c['version']}"})
    rate = 1.0 if not sop_ids else 1 - len({b["rule"] for b in bad}) / len(sop_ids)
    return {"pass": not bad, "rate": rate, "n_rules": len(sop_ids), "violations": bad}


_STOP = set(
    "the a an of to in on for and or is are was were be by with at as from that this "
    "it its than over under per client clients has have had which".split()
)


def lexical_judge(point: str, cited_texts: list[str]) -> int:
    # Offline proxy judge, 1-5: share of the point's content words (and numbers) found in the
    # cited text. Replace with an LLM judge for reported results.
    words = [w for w in re.findall(r"[a-z_]+|\d+(?:\.\d+)?", point.lower()) if w not in _STOP]
    if not words or not cited_texts:
        return 1
    src = " ".join(cited_texts).lower()
    hit = sum(w in src for w in words) / len(words)
    return 1 + int(round(4 * hit))


JUDGE_PROMPT = """You check one claim from a utility-fraud inspection brief against its cited sources.
Score 1-5: 5 = every part of the claim is directly stated in the sources;
3 = partly supported or needs an inference; 1 = unsupported or contradicted.
Reply with the single digit only.

CLAIM: {point}

SOURCES:
{sources}"""


def make_anthropic_judge(model="claude-haiku-5-5", client=None) -> Callable[[str, list[str]], int]:
    # LLM judge (different prompt from the brief writer, as 14.6 requires).
    import anthropic

    client = client or anthropic.Anthropic()

    def judge(point, cited_texts):
        msg = client.messages.create(
            model=model,
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        point=point, sources="\n---\n".join(cited_texts) or "(none)"
                    ),
                }
            ],
        )
        m = re.search(r"[1-5]", msg.content[0].text)
        return int(m.group()) if m else 1

    return judge


def check_faithfulness(brief, snapshot, evidence, judge=None) -> dict:
    judge = judge or lexical_judge
    texts = {c["id"]: c.get("text", "") for c in evidence}
    texts["SNAPSHOT"] = json.dumps(snapshot, default=str)
    scores = []
    for e in brief.get("evidence", []):
        cited = [texts[c] for c in e.get("citations", []) if c in texts]
        scores.append(judge(e.get("point", ""), cited))
    mean = sum(scores) / len(scores) if scores else float("nan")
    return {"mean": mean, "scores": scores, "judge": getattr(judge, "__name__", "custom")}


def score_brief(
    brief: dict, snapshot: dict, evidence: list[dict], judge=None, alert_date=None
) -> dict:
    # Run all brief-level checks. Returns a flat-friendly dict.
    if isinstance(brief, str):
        brief = json.loads(brief)
    alert_date = alert_date or snapshot.get("alert_date")
    schema = check_schema(brief)
    cit = check_citations(brief, evidence)
    num = check_numbers(brief, snapshot, evidence)
    rule = check_rules(brief, evidence, alert_date)
    faith = check_faithfulness(brief, snapshot, evidence, judge)
    return {
        "client_id": brief.get("client_id"),
        "schema_valid": schema["pass"],
        "citation_valid_rate": cit["rate"],
        "number_accuracy_rate": num["rate"],
        "rule_applicability_rate": rule["rate"],
        "faithfulness_mean": faith["mean"],
        "all_pass": schema["pass"] and cit["pass"] and num["pass"] and rule["pass"],
        "details": {
            "schema": schema,
            "citations": cit,
            "numbers": num,
            "rules": rule,
            "faithfulness": faith,
        },
    }


def auditor_feedback(result: dict) -> list[str]:
    # Specific, actionable errors for the multi-agent Auditor (14.8 step 4).
    d, msgs = result["details"], []
    msgs += [f"Schema: {e}" for e in d["schema"]["errors"]]
    msgs += [
        f"Citation {c} is not in the retrieved evidence; cite a retrieved ID or SNAPSHOT."
        for c in d["citations"]["invalid"]
    ]
    msgs += [
        f"Number {u['number']} in {u['where']} does not appear in SNAPSHOT or its cited chunk"
        + (f" (expected {u['expected']})." if "expected" in u else ".")
        for u in d["numbers"]["unsupported"]
    ]
    msgs += [f"Rule {v['rule']}: {v['reason']}." for v in d["rules"]["violations"]]
    for i, s in enumerate(d["faithfulness"]["scores"]):
        if s <= 2:
            msgs.append(f"evidence[{i}] is weakly supported by its citations (judge {s}/5).")
    return msgs


# Retrieval (check 5)
def retrieval_metrics(ranked: dict[str, list[str]], gold: dict[str, set[str] | list[str]]) -> dict:
    # ranked: query_id -> ranked chunk IDs; gold: query_id -> relevant IDs.
    r3 = r5 = mrr = 0.0
    for q, rel in gold.items():
        rel = set(rel)
        got = ranked.get(q, [])
        r3 += len(rel & set(got[:3])) / len(rel)
        r5 += len(rel & set(got[:5])) / len(rel)
        rr = next((1 / (i + 1) for i, c in enumerate(got) if c in rel), 0.0)
        mrr += rr
    n = max(1, len(gold))
    return {"recall@3": r3 / n, "recall@5": r5 / n, "mrr": mrr / n, "n_queries": len(gold)}


def score_run(items: list[dict], judge=None) -> tuple[dict, list[dict]]:
    # Aggregate over a run (one row of the three-run / single-vs-multi table).
    # items: [{"brief":..., "snapshot":..., "evidence":[...]}, ...]
    # Returns the aggregate row and the per-brief results.
    res = [score_brief(i["brief"], i["snapshot"], i["evidence"], judge=judge) for i in items]
    n = len(res)

    def agg(k):
        return sum(r[k] for r in res) / n if n else float("nan")

    return {
        "n_briefs": n,
        "schema_valid": agg("schema_valid"),
        "citation_validity": agg("citation_valid_rate"),
        "number_accuracy": agg("number_accuracy_rate"),
        "rule_applicability": agg("rule_applicability_rate"),
        "faithfulness_mean": agg("faithfulness_mean"),
        "all_pass_share": agg("all_pass"),
    }, res
