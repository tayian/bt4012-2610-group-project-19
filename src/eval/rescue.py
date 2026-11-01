"""W3-C2 Unsupervised rescue analysis: among true frauds that M3 ranks BELOW
its top 10%, what share do M5 (isolation forest) or M6 (autoencoder) rank in
their own top 10%? This number justifies parallel (not series) fusion.

Ranks are computed within each CV fold (OOF scores are only comparable inside
the fold that produced them). A random scorer would "rescue" ~10% per model and
~19% for "M5 or M6", so lift is reported against that baseline.

    python -m src.eval.rescue --m3 v0 --anomaly v0
"""
import argparse
import numpy as np
import pandas as pd

TOP = 0.10


def top_flag(df, col):
    r = df.groupby("fold")[col].rank(pct=True, ascending=False)
    return r <= TOP


def rescue(m3_v, an_v, extra=("M7", "M4")):
    cli = pd.read_parquet("data/interim/client.parquet")[["client_id", "target"]]
    cli["client_id"] = cli["client_id"].astype(str)
    d = pd.read_parquet(f"results/oof/M3_{m3_v}.parquet")[["client_id", "fold", "score"]]
    d = d.rename(columns={"score": "M3"}).merge(cli, on="client_id")
    models = []
    for m in ["M5", "M6"] + list(extra):
        try:
            o = pd.read_parquet(f"results/oof/{m}_{an_v}.parquet")[["client_id", "score"]]
        except FileNotFoundError:
            continue
        d = d.merge(o.rename(columns={"score": m}), on="client_id")
        models.append(m)
    for m in ["M3"] + models:
        d[f"top_{m}"] = top_flag(d, m)
    fr = d[d.target == 1]
    missed = fr[~fr["top_M3"]]
    rows = [{"group": "frauds missed by M3 top 10%", "n": len(missed),
             "share_of_all_frauds": len(missed) / len(fr)}]
    for m in models:
        rows.append({"group": f"rescued by {m} top 10%", "n": int(missed[f"top_{m}"].sum()),
                     "share_of_missed": missed[f"top_{m}"].mean(), "random_baseline": TOP})
    both = missed["top_M5"] | missed["top_M6"]
    rows.append({"group": "rescued by M5 or M6", "n": int(both.sum()),
                 "share_of_missed": both.mean(), "random_baseline": 1 - (1 - TOP) ** 2})
    # same question for non-frauds: anomaly models also flag honest clients
    hon = d[(d.target == 0) & (~d["top_M3"])]
    hb = hon["top_M5"] | hon["top_M6"]
    rows.append({"group": "non-frauds outside M3 top 10% flagged by M5 or M6", "n": int(hb.sum()),
                 "share_of_missed": hb.mean(), "random_baseline": 1 - (1 - TOP) ** 2})
    # Spearman correlation of OOF scores (diversity for fusion)
    corr = d[["M3"] + models].corr(method="spearman")
    out = pd.DataFrame(rows)
    out["m3_version"], out["anomaly_version"] = m3_v, an_v
    return out, corr


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--m3", default="v0")
    p.add_argument("--anomaly", default="v0")
    a = p.parse_args()
    out, corr = rescue(a.m3, a.anomaly)
    out.to_csv(f"results/tables/rescue_M3{a.m3}_anom{a.anomaly}.csv", index=False)
    corr.to_csv(f"results/tables/oof_spearman_{a.anomaly}.csv")
    print(out.round(3).to_string(index=False))
    print("\nSpearman correlation of OOF scores\n", corr.round(2).to_string())
