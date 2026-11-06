# Compare imbalance handling while holding LightGBM parameters and saved folds fixed.

from functools import partial

import pandas as pd

from src.eval.oof import write_frames
from src.models.experiments import evaluate, get_context, parser, publish, write_json
from src.models.sampling import STRATEGIES
from src.models.trees import model_fn


def run(ctx, n_jobs=2):
    completed = []
    for strategy in STRATEGIES:
        name = f"M3_{strategy}"
        params = {"imbalance": strategy, "n_jobs": n_jobs}
        oof, row, _ = evaluate(ctx, partial(model_fn, **params), name, params)
        publish(ctx, oof, row, name)
        completed.append(row)
        write_frames({ctx.results / f"tables/imbalance_{ctx.suffix}.csv": pd.DataFrame(completed)})
    winner = max(completed, key=lambda row: row["pr_auc_mean"])
    write_json(ctx.results / f"tables/imbalance_selection_{ctx.suffix}.json", winner)
    print(f"Best imbalance strategy by CV PR-AUC: {winner['imbalance']}", flush=True)
    return pd.DataFrame(completed)


def main():
    p = parser("M3 imbalance ablations on saved folds.")
    p.add_argument("--n-jobs", type=int, default=2)
    args = p.parse_args()
    try:
        run(get_context(args), args.n_jobs)
    except (ValueError, OSError) as exc:
        p.exit(2, f"imbalance: {exc}\n")


if __name__ == "__main__":
    main()
