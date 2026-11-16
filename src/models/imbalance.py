# Compare imbalance handling while holding LightGBM parameters and saved folds fixed.

import json
from functools import partial
from pathlib import Path

import pandas as pd

from src.eval.oof import write_frames
from src.models.experiments import evaluate, get_context, parser, publish, write_json
from src.models.sampling import STRATEGIES
from src.models.trees import model_fn


def run(ctx, n_jobs=2, base=None, prefix="M3"):
    # base: LightGBM overrides held fixed across strategies (e.g. the tuned M3 parameters).
    # The default M3 ablation keeps its file names; a tuned run writes imbalance_tuned_*.
    table = "imbalance" if prefix == "M3" else "imbalance_tuned"
    completed = []
    for strategy in STRATEGIES:
        name = f"{prefix}_{strategy}"
        params = {**(base or {}), "imbalance": strategy, "n_jobs": n_jobs}
        oof, row, _ = evaluate(ctx, partial(model_fn, **params), name, params)
        publish(ctx, oof, row, name)
        completed.append(row)
        write_frames({ctx.results / f"tables/{table}_{ctx.suffix}.csv": pd.DataFrame(completed)})
    winner = max(completed, key=lambda row: row["pr_auc_mean"])
    write_json(ctx.results / f"tables/{table}_selection_{ctx.suffix}.json", winner)
    print(f"Best imbalance strategy by CV PR-AUC: {winner['imbalance']}", flush=True)
    return pd.DataFrame(completed)


def main():
    p = parser("M3 imbalance ablations on saved folds.")
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--params", type=Path, help="JSON of tuned M3 parameters (src.models.tune)")
    args = p.parse_args()
    try:
        base = json.loads(args.params.read_text()) if args.params else None
        run(get_context(args), args.n_jobs, base, "M3_tuned" if args.params else "M3")
    except (ValueError, OSError) as exc:
        p.exit(2, f"imbalance: {exc}\n")


if __name__ == "__main__":
    main()
