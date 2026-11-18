# Time robustness: tuned M3 scored on features rebuilt from invoices up to a cutoff date, against
# the full-history features, on the same saved folds. Run: python3 -m src.eval.time_robustness

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean import final_year_status
from src.eval.metrics import budget_metrics, cv_evaluate
from src.features import aggregate
from src.features.behavioural import INVOICE_COLUMNS, build_features
from src.models.linear import load_training
from src.models.trees import model_fn

CUTOFF = "2016-12-31"
METRICS = ["pr_auc", "roc_auc", "recall@5%", "precision@5%", "recall@1%", "recall@10%"]


def history_up_to(inv, cutoff):
    # Invoices dated on or before the cutoff. The partial-final-year flag is recomputed for the
    # shorter history, because the full table flags 2019, which is no longer present.
    out = inv.loc[inv["invoice_date"].le(pd.Timestamp(cutoff))].copy()
    final, complete, _, _ = final_year_status(out["invoice_date"])
    out["flag_partial_year"] = out["invoice_date"].dt.year.eq(final) & (not complete)
    return out.reset_index(drop=True), final, complete


def fold_summary(y, scores, fold_ids, mask=None):
    # budget_metrics per saved fold, optionally on the rows in mask only, as mean and sample std.
    y, scores = np.asarray(y), np.asarray(scores)
    mask = np.ones(len(y), bool) if mask is None else np.asarray(mask)
    rows = [
        budget_metrics(y[(fold_ids == f) & mask], scores[(fold_ids == f) & mask])
        for f in np.unique(fold_ids)
    ]
    table = pd.DataFrame(rows)
    out = {"n_clients": int(mask.sum()), "n_fraud": int(y[mask].sum())}
    for metric in METRICS:
        out[f"{metric}_mean"] = float(table[metric].mean())
        out[f"{metric}_std"] = float(table[metric].std(ddof=1))
    return out


def run(interim, features_path, pre_path, clients_path, split_dir, params, cutoff, n_jobs):
    inv = pd.read_parquet(Path(interim) / "invoice_clean.parquet", columns=INVOICE_COLUMNS)
    clients = pd.read_parquet(Path(interim) / "client.parquet", columns=aggregate.PROFILE_COLUMNS)
    early, final, complete = history_up_to(inv, cutoff)
    print(
        f"{len(early):,} of {len(inv):,} invoices dated <= {cutoff}; "
        f"final year {final} {'complete' if complete else 'partial'}",
        flush=True,
    )
    pre_path = Path(pre_path)
    pre_path.parent.mkdir(parents=True, exist_ok=True)
    build_features(early, clients).to_parquet(pre_path, index=False)

    variants = {"full history": Path(features_path), f"invoices <= {cutoff}": pre_path}
    oof, n_invoices = {}, {"full history": len(inv), f"invoices <= {cutoff}": len(early)}
    for name, path in variants.items():
        X, y, folds = load_training(path, clients_path, split_dir)
        print(f"Fitting tuned M3 on {name} ({path.name})", flush=True)
        scores, _ = cv_evaluate(
            partial(model_fn, **params, n_jobs=n_jobs),
            X,
            y,
            folds,
            model="M3_tuned",
            feature_version=path.stem,
        )
        oof[name] = (X, y, scores)

    # Every variant uses the same training clients in the same order.
    X, y, _ = oof["full history"]
    fold_ids = oof["full history"][2]["fold"].to_numpy(int)
    has_history = X["client_id"].isin(early["client_id"]).to_numpy()
    rows = []
    for scope, mask in (
        ("all training clients", None),
        (f"clients with invoices <= {cutoff}", has_history),
    ):
        for name, (Xv, yv, scores) in oof.items():
            if not Xv["client_id"].equals(X["client_id"]):
                raise ValueError("Variants must score the same clients in the same order.")
            row = {"history": name, "scope": scope, "n_invoices": n_invoices[name]}
            row.update(fold_summary(yv, scores["score"], fold_ids, mask))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare tuned M3 on full-history features_v1 with features built before a cutoff."
    )
    parser.add_argument("--interim", type=Path, default=Path("data/interim"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_v1.parquet"))
    parser.add_argument(
        "--pre-features", type=Path, default=Path("data/processed/features_v1_pre2017.parquet")
    )
    parser.add_argument("--clients", type=Path, default=Path("data/raw/client_train.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--params", type=Path, default=Path("results/tables/M3_tuned_params_v1.json")
    )
    parser.add_argument("--cutoff", default=CUTOFF)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("results/tables/time_robustness.csv"))
    args = parser.parse_args()
    try:
        table = run(
            args.interim,
            args.features,
            args.pre_features,
            args.clients,
            args.split_dir,
            json.loads(args.params.read_text()),
            args.cutoff,
            args.n_jobs,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"time_robustness: {exc}\n")
    cols = ["history", "scope", "n_clients", "pr_auc_mean", "roc_auc_mean", "recall@5%_mean"]
    print(table[cols].to_string(index=False))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
