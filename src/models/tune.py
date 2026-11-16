# Tune M3 LightGBM with Optuna on the saved training folds (objective: mean CV PR-AUC).
#
# Each trial is a full 5-fold run through the shared evaluate(), so finished trials are cached
# and a rerun resumes. The best parameters are refitted the same way and published as M3_tuned:
#   results/oof/M3_tuned_<version>.parquet and its main_results.csv row
#   results/tables/M3_tuning_<version>.csv            every trial and its CV PR-AUC
#   results/tables/M3_tuned_params_<version>.json     the selected LightGBM parameters
# As with any search, the selected CV score is slightly optimistic; the test set stays unused.

from functools import partial

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
    best = {**study.best_params, "subsample_freq": 1}
    oof, row, _ = evaluate(ctx, partial(model_fn, **best, n_jobs=n_jobs), "M3_tuned", best)
    publish(ctx, oof, row, "M3_tuned")
    write_frames({ctx.results / f"tables/M3_tuning_{ctx.suffix}.csv": pd.DataFrame(rows)})
    write_json(ctx.results / f"tables/M3_tuned_params_{ctx.suffix}.json", best)
    print(f"Best of {n_trials} trials: PR-AUC {row['pr_auc_mean']:.4f} with {best}", flush=True)
    return best


def main():
    p = parser("Tune M3 LightGBM with Optuna on the saved folds.")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--n-jobs", type=int, default=4)
    args = p.parse_args()
    try:
        run(get_context(args), args.trials, args.n_jobs)
    except (ValueError, OSError) as exc:
        p.exit(2, f"tune: {exc}\n")


if __name__ == "__main__":
    main()
