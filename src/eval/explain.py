# Top five signed Tree SHAP drivers per training client, from the model that held it out.
# Run: python3 -m src.eval.explain --features <path> [--params <tuned M3 json>]

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.oof import oof_frame, write_frames
from src.models.experiments import get_context, parser, write_json
from src.models.trees import fit_m3

SHAP_COLUMNS = ["client_id", "rank", "feature", "value", "shap_value"]


def positive_shap(explainer, prepared):
    values = explainer.shap_values(prepared)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != prepared.shape or not np.isfinite(values).all():
        raise ValueError("Unexpected SHAP shape or non-finite contributions.")
    return values


def top_rows(ids, X, values):
    if X.shape[1] < 5:
        raise ValueError("The top-five SHAP contract needs at least five input features.")
    # Ties break by feature order; contributions keep their signs, in log-odds.
    top = np.argsort(-np.abs(values), axis=1, kind="stable")[:, :5]
    rows = []
    for i, choices in enumerate(top):
        for rank, j in enumerate(choices, 1):
            raw = X.iloc[i, j]
            # Values must be numeric (v0 categories are codes); text categories fail here.
            value = np.nan if pd.isna(raw) else float(raw)
            rows.append((str(ids[i]), rank, str(X.columns[j]), value, float(values[i, j])))
    return pd.DataFrame(rows, columns=SHAP_COLUMNS)


def run(ctx, n_jobs=2, batch_size=4096, params=None, name="M3"):
    # params are LightGBM overrides (e.g. the tuned M3); name is the published OOF model to match.
    import shap

    params_m3 = params or {}
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    frames, score_frames = [], []
    for fold in range(5):
        settings = {"fold": fold, "n_jobs": n_jobs, "shap_version": shap.__version__}
        if params_m3:
            settings["m3_params"] = params_m3
        cache = ctx.cache("M3_shap", settings)
        if (cache / "complete.json").exists():
            print(f"REUSE SHAP fold {fold + 1}/5", flush=True)
            frame = pd.read_parquet(cache / "top5.parquet")
            scores = pd.read_parquet(cache / "scores.parquet")
        else:
            train, valid = ctx.fold_ids != fold, ctx.fold_ids == fold
            Xtr = ctx.X.loc[train].drop(columns="client_id")
            Xva = ctx.X.loc[valid].drop(columns="client_id")
            ids = ctx.X.loc[valid, "client_id"].to_numpy()
            model = fit_m3(Xtr, ctx.y[train], **{**params_m3, "n_jobs": n_jobs})
            explainer = shap.TreeExplainer(
                model.estimator_, feature_perturbation="tree_path_dependent", model_output="raw"
            )
            batches = []
            for start in range(0, len(Xva), batch_size):
                raw = Xva.iloc[start : start + batch_size]
                contributions = positive_shap(explainer, model._prepare(raw))
                batches.append(top_rows(ids[start : start + len(raw)], raw, contributions))
            frame = pd.concat(batches, ignore_index=True)
            scores = oof_frame(
                ids,
                np.full(len(ids), fold),
                model.predict_proba(Xva)[:, 1],
                "M3_shap_reference",
                ctx.version,
            )
            write_frames({cache / "top5.parquet": frame, cache / "scores.parquet": scores})
            write_json(cache / "complete.json", settings)
            print(f"SHAP fold {fold + 1}/5 complete", flush=True)
        frames.append(frame)
        score_frames.append(scores)
    top = pd.concat(frames, ignore_index=True)
    reference = (
        pd.concat(score_frames, ignore_index=True)
        .set_index("client_id")
        .loc[ctx.X.client_id]
        .reset_index()
    )
    baseline_path = ctx.results / f"oof/{name}_{ctx.suffix}.parquet"
    if not baseline_path.exists():
        raise ValueError(f"{baseline_path} is missing: publish the {name} model before its SHAP.")
    baseline = pd.read_parquet(baseline_path).set_index("client_id").loc[ctx.X.client_id]
    if baseline.fold.tolist() != reference.fold.tolist() or not np.allclose(
        baseline.score, reference.score, rtol=1e-8, atol=1e-10
    ):
        raise ValueError(
            f"{name} OOF scores differ from the SHAP model. Rebuild {baseline_path} first."
        )
    if len(top) != 5 * len(ctx.X) or set(top.client_id) != set(ctx.X.client_id):
        raise ValueError("SHAP export must contain five rows for every training client.")
    order = dict(zip(ctx.X.client_id, range(len(ctx.X))))
    top = (
        top.assign(_order=top.client_id.map(order))
        .sort_values(["_order", "rank"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    output = ctx.results / f"shap_top5_{ctx.suffix}.parquet"
    write_frames(
        {output: top, ctx.results / f"oof/{name}_shap_reference_{ctx.suffix}.parquet": reference}
    )
    write_json(
        ctx.results / f"tables/shap_metadata_{ctx.suffix}.json",
        {
            "model": name,
            "feature_set": ctx.feature_set,
            "data_fingerprint": ctx.fingerprint,
            "scope": "training clients, out-of-fold explanations",
            "units": "raw log-odds",
            "rank": "descending absolute SHAP, signs preserved",
            "value": "raw numeric value or numeric category code; NaN for missing",
            "columns": SHAP_COLUMNS,
            "n_clients": len(ctx.X),
            "n_rows": len(top),
        },
    )
    print(f"Saved {len(top):,} SHAP rows to {output}", flush=True)
    return top


def main():
    p = parser("Export the top five out-of-fold M3 SHAP drivers per training client.")
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--params", type=Path, help="JSON of tuned M3 parameters (src.models.tune)")
    args = p.parse_args()
    try:
        params = json.loads(args.params.read_text()) if args.params else None
        name = "M3_tuned" if args.params else "M3"
        run(get_context(args), args.n_jobs, args.batch_size, params, name)
    except (ValueError, OSError) as exc:
        p.exit(2, f"shap: {exc}\n")


if __name__ == "__main__":
    main()
