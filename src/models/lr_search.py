# Saved-fold L1, L2 and elastic-net search; converged candidates are checkpointed.

import warnings
from functools import partial

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.eval.oof import write_frames
from src.models.experiments import coefficients, evaluate, get_context, parser, publish, write_json
from src.models.linear import model_fn

CS = (0.01, 0.1, 1.0, 10.0)
PENALTIES = (("l2", None), ("elasticnet", 0.5), ("l1", None))


def search(ctx, max_iter=15000, Cs=CS):
    rows, completed, failed = [], [], []
    for C in Cs:
        for penalty, ratio in PENALTIES:
            name = f"M1_{penalty}_C{C:g}".replace(".", "p")
            params = {"C": float(C), "penalty": penalty, "max_iter": max_iter}
            if ratio is not None:
                params["l1_ratio"] = ratio
            factory = partial(model_fn, **params)
            try:
                oof, row, coef = evaluate(ctx, factory, name, params, collect_coefficients=True)
                completed.append((oof, row, coef, factory))
                rows.append(dict(row, status="complete"))
            except ConvergenceWarning:
                failed.append(name)
                rows.append(dict(model=name, **params, status="not_converged"))
                print(f"NOT CONVERGED: {name}; raise --max-iter and rerun to resume.", flush=True)
            write_frames({ctx.results / f"tables/M1_grid_{ctx.suffix}.csv": pd.DataFrame(rows)})
    if failed:
        raise ValueError(
            f"LR grid incomplete: {', '.join(failed)}. Converged candidates are cached."
        )
    oof, row, coef, factory = max(completed, key=lambda item: item[1]["pr_auc_mean"])
    cache = ctx.cache(
        "M1_selected_refit",
        {k: row[k] for k in ("C", "penalty", "max_iter", "l1_ratio") if k in row},
    )
    if not (cache / "coefficients.csv").exists():
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted = factory().fit(ctx.X.drop(columns="client_id"), ctx.y)
        write_frames({cache / "coefficients.csv": coefficients(fitted, -1, "all_train_refit")})
    full = pd.read_csv(cache / "coefficients.csv")
    all_coefficients = pd.concat(
        [
            candidate_coef.assign(
                candidate=candidate_row["model"],
                C=candidate_row["C"],
                penalty=candidate_row["penalty"],
                l1_ratio=candidate_row.get("l1_ratio", float("nan")),
            )
            for _, candidate_row, candidate_coef, _ in completed
        ],
        ignore_index=True,
    )
    publish(ctx, oof, row, "M1_grid")
    write_frames(
        {
            ctx.results / f"coefficients/M1_grid_{ctx.suffix}.csv": full,
            ctx.results / f"coefficients/M1_grid_{ctx.suffix}_folds.csv": all_coefficients,
        }
    )
    write_json(
        ctx.results / f"tables/M1_selection_{ctx.suffix}.json", dict(row, selected_model="M1_grid")
    )
    print(f"Selected {row['model']}; wrote M1_grid OOF and coefficients.", flush=True)
    return oof.assign(model="M1_grid"), row


def main():
    p = parser("LR regularisation grid on saved training folds.")
    p.add_argument("--max-iter", type=int, default=15000)
    args = p.parse_args()
    try:
        search(get_context(args), args.max_iter)
    except (ValueError, OSError, ConvergenceWarning) as exc:
        p.exit(2, f"lr-grid: {exc}\n")


if __name__ == "__main__":
    main()
