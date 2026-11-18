# Run logger for the agent comparison: one JSON log per system and client in
# results/agent_logs/<system>/<client_id>.json, used by src/llm/agent.py.

from __future__ import annotations

import glob
import json
import os
import time

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
        # usage is a token dict or an object with input_tokens and output_tokens attributes.
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
        with open(self.path, "w") as f:
            json.dump(self.rec, f, indent=2, default=str)
        return False


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def load_logs(system, log_dir=LOG_DIR):
    return [_read_json(p) for p in sorted(glob.glob(os.path.join(log_dir, system, "*.json")))]
