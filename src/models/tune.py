# Tune M3 LightGBM with Optuna on the saved folds (objective: mean CV PR-AUC) and publish M3_tuned.
# Run: python3 -m src.models.tune --features <path> [--params <json> to refit without searching]

import json
from functools import partial
from pathlib import Path

import optuna
import pandas as pd

from src.eval.oof import write_frames
from src.models.experiments import evaluate, get_context, parser, publish, write_json
from src.models.trees import model_fn

SEED = 4012


def space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def run(ctx, n_trials=30, n_jobs=4):
    rows = []

    def objective(trial):
        params = space(trial)
        _, row, _ = evaluate(ctx, partial(model_fn, **params, n_jobs=n_jobs), "M3_trial", params)
        rows.append({"trial": trial.number, **params, "pr_auc_mean": row["pr_auc_mean"]})
        return row["pr_auc_mean"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    # Refitted through evaluate() like the trials; its CV score is slightly optimistic.
    best = {**study.best_params, "subsample_freq": 1}
    oof, row, _ = evaluate(ctx, partial(model_fn, **best, n_jobs=n_jobs), "M3_tuned", best)
    publish(ctx, oof, row, "M3_tuned")
    write_frames({ctx.results / f"tables/M3_tuning_{ctx.suffix}.csv": pd.DataFrame(rows)})
    write_json(ctx.results / f"tables/M3_tuned_params_{ctx.suffix}.json", best)
    print(f"Best of {n_trials} trials: PR-AUC {row['pr_auc_mean']:.4f} with {best}", flush=True)
    return best


def refit(ctx, params, n_jobs=4):
    oof, row, _ = evaluate(ctx, partial(model_fn, **params, n_jobs=n_jobs), "M3_tuned", params)
    publish(ctx, oof, row, "M3_tuned")
    print(f"M3_tuned on {ctx.feature_set}: PR-AUC {row['pr_auc_mean']:.4f}", flush=True)


def main():
    p = parser("Tune M3 LightGBM with Optuna on the saved folds.")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--params", type=Path, help="refit these saved parameters instead of searching")
    args = p.parse_args()
    try:
        if args.params:
            refit(get_context(args), json.loads(args.params.read_text()), args.n_jobs)
        else:
            run(get_context(args), args.trials, args.n_jobs)
    except (ValueError, OSError) as exc:
        p.exit(2, f"tune: {exc}\n")


if __name__ == "__main__":
    main()
