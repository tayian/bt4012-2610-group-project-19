"""Interface double only. These random scores do not validate a brief."""
from random import Random


def score_brief(brief, snapshot, evidence) -> dict:
    """Match the section 4.4 signature without LLM calls or real evaluation."""
    rng = Random("STUB:" + str(brief.get("client_id", "")))
    return {
        "is_stub": True,
        "citation_validity": round(rng.random(), 4),
        "number_accuracy": round(rng.random(), 4),
        "rule_applicability": round(rng.random(), 4),
        "faithfulness": round(rng.uniform(1, 5), 4),
        "retrieval": None,
        "needs_human_review": True,
    }
