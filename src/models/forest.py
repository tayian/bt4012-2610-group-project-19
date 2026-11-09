# M2 random-forest grid using the same saved folds and fold-fitted preprocessing.

from functools import partial

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.split import SEED
from src.eval.oof import write_frames
from src.models.experiments import evaluate, get_context, parser, publish, write_json
from src.models.linear import model_fn as linear_pipeline


def model_fn(max_depth=None, min_samples_leaf=5, n_estimators=500, n_jobs=2):
    pipeline = linear_pipeline()
    # Trees use the numeric imputer and one-hot encoder without scaling.
    pipeline.named_steps["preprocess"].transformers[0][1].set_params(scale="passthrough")
    pipeline.set_params(
        classifier=RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=SEED,
            n_jobs=n_jobs,
        )
    )
    return pipeline


def search(ctx, n_jobs=2, n_estimators=500):
    rows, completed = [], []
    for depth in (None, 12):
        for leaf in (5, 20):
            name = f"M2_depth{depth}_leaf{leaf}"
            params = dict(
                max_depth=depth, min_samples_leaf=leaf, n_estimators=n_estimators, n_jobs=n_jobs
            )
            oof, row, _ = evaluate(ctx, partial(model_fn, **params), name, params)
            rows.append(row)
            completed.append((oof, row))
            write_frames({ctx.results / f"tables/M2_grid_{ctx.suffix}.csv": pd.DataFrame(rows)})
    oof, row = max(completed, key=lambda item: item[1]["pr_auc_mean"])
    publish(ctx, oof, row, "M2")
    write_json(
        ctx.results / f"tables/M2_selection_{ctx.suffix}.json", dict(row, selected_model="M2")
    )
    print(f"Selected {row['model']}; wrote M2 OOF.", flush=True)
    return oof.assign(model="M2"), row


def main():
    p = parser("Random-forest grid on saved training folds.")
    p.add_argument("--n-jobs", type=int, default=2)
    args = p.parse_args()
    try:
        search(get_context(args), args.n_jobs)
    except (ValueError, OSError) as exc:
        p.exit(2, f"random-forest: {exc}\n")


if __name__ == "__main__":
    main()
