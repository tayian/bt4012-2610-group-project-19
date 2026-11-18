# Nested isotonic calibration: outer saved folds evaluate, inner folds fit the calibrator.

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

from src.data.split import SEED
from src.eval.oof import oof_frame, write_frames
from src.models.experiments import evaluate, get_context, metric_row, parser, publish
from src.models.linear import model_fn as lr_model
from src.models.trees import model_fn as tree_model
from src.viz import style


class IsotonicClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, kind="M3", max_iter=15000, n_jobs=2, inner_folds=3):
        self.kind = kind
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.inner_folds = inner_folds

    def fit(self, X, y):
        if min(np.bincount(np.asarray(y, dtype=int))) < self.inner_folds:
            raise ValueError("Not enough training examples per class for inner calibration folds.")
        base = (
            lr_model(max_iter=self.max_iter)
            if self.kind == "M1"
            else tree_model(n_jobs=self.n_jobs)
        )
        cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.estimator_ = CalibratedClassifierCV(
                estimator=base, method="isotonic", cv=cv, ensemble=False, n_jobs=1
            ).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        # Also keeps the uncalibrated scores for the raw-versus-isotonic comparison.
        base = self.estimator_.calibrated_classifiers_[0].estimator
        self.raw_indices_ = X.index.to_numpy()
        self.raw_scores_ = base.predict_proba(X)[:, 1]
        return self.estimator_.predict_proba(X)


def reliability(y, scores, model, bins=10):
    # Equal-count bins (score deciles), so every point averages about a tenth of the clients;
    # equal-width bins leave the top bins with a handful of clients for well-calibrated scores.
    y, scores = np.asarray(y), np.asarray(scores)
    group = pd.qcut(pd.Series(scores).rank(method="first"), bins, labels=False).to_numpy()
    rows = []
    for b in range(bins):
        mask = group == b
        if mask.any():
            rows.append(
                dict(
                    model=model,
                    bin=b,
                    n=int(mask.sum()),
                    mean_score=float(scores[mask].mean()),
                    fraud_share=float(y[mask].mean()),
                )
            )
    return pd.DataFrame(rows)


def run(ctx, max_iter=15000, n_jobs=2, inner_folds=3):
    comparisons, bins = [], []
    for kind in ("M1", "M3"):
        name = f"{kind}_isotonic"
        params = dict(
            base_model=kind,
            calibration="isotonic",
            inner_folds=inner_folds,
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
        cache = ctx.cache(name, params)
        raw_path = cache / "raw.parquet"
        fitted = []

        def make():
            estimator = IsotonicClassifier(kind, max_iter, n_jobs, inner_folds)
            fitted.append(estimator)
            return estimator

        # evaluate() caches only the calibrated scores, so a cache without raw scores is redone.
        if not raw_path.exists():
            (cache / "complete.json").unlink(missing_ok=True)
        oof, row, _ = evaluate(ctx, make, name, params)
        if fitted:
            raw_scores = np.full(len(ctx.X), np.nan)
            for model in fitted:
                raw_scores[model.raw_indices_] = model.raw_scores_
            if not np.isfinite(raw_scores).all():
                raise ValueError("Missing raw calibration comparison scores.")
            raw = oof_frame(
                ctx.X.client_id, ctx.fold_ids, raw_scores, f"{kind}_calibration_raw", ctx.version
            )
            write_frames({raw_path: raw})
        else:
            raw = pd.read_parquet(raw_path)
        raw_row = metric_row(ctx, raw.score.to_numpy(), f"{kind}_calibration_raw")
        comparisons.extend([raw_row, row])
        bins.extend(
            [reliability(ctx.y, raw.score, f"{kind}_raw"), reliability(ctx.y, oof.score, name)]
        )
        publish(ctx, oof, row, name)
        write_frames({ctx.results / f"oof/{kind}_calibration_raw_{ctx.suffix}.parquet": raw})
    table, bin_table = pd.DataFrame(comparisons), pd.concat(bins, ignore_index=True)
    write_frames({ctx.results / f"tables/calibration_{ctx.suffix}.csv": table})
    plot_reliability(ctx, bin_table)
    print(table[["model", "brier_mean", "brier_std"]].to_string(index=False))
    return table, bin_table


def replot(ctx):
    # Rebuild the reliability table and figure from the published OOF files, without refitting.
    order = ctx.X.client_id
    bins = []
    for kind in ("M1", "M3"):
        for label, file in ((f"{kind}_raw", "calibration_raw"), (f"{kind}_isotonic", "isotonic")):
            oof = pd.read_parquet(ctx.results / f"oof/{kind}_{file}_{ctx.suffix}.parquet")
            scores = oof.set_index("client_id").loc[order, "score"]
            bins.append(reliability(ctx.y, scores, label))
    plot_reliability(ctx, pd.concat(bins, ignore_index=True))


def plot_reliability(ctx, bin_table):
    write_frames({ctx.results / f"tables/reliability_{ctx.suffix}.csv": bin_table})
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for kind, ax in zip(("M1", "M3"), axes):
        ax.plot(
            [0, 1], [0, 1], color=style.TEXT_SECONDARY, linestyle="--", label="Ideal calibration"
        )
        for label, color in ((f"{kind}_raw", style.NON_FRAUD), (f"{kind}_isotonic", style.FRAUD)):
            d = bin_table.loc[bin_table.model.eq(label)]
            ax.plot(d.mean_score, d.fraud_share, marker="o", color=color, label=label)
        ax.set(
            title=f"{kind}: outer-fold predictions",
            xlabel="Mean predicted score",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Observed fraud share")
    fig.tight_layout()
    output = ctx.results / f"figures/reliability_{ctx.suffix}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    fig.savefig(temp, format="png")
    plt.close(fig)
    temp.replace(output)


def main():
    p = parser("Isotonic calibration of M1 and M3, evaluated on saved outer folds.")
    p.add_argument("--max-iter", type=int, default=15000)
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument(
        "--replot", action="store_true", help="redraw the reliability plot from saved OOF files"
    )
    args = p.parse_args()
    try:
        if args.replot:
            replot(get_context(args))
        else:
            run(get_context(args), args.max_iter, args.n_jobs, args.inner_folds)
    except (ValueError, OSError, ConvergenceWarning) as exc:
        p.exit(2, f"calibrate: {exc}\n")


if __name__ == "__main__":
    main()
