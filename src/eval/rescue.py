# Among frauds M3 ranks below its top 10%, the share that M5 or M6 rank in their own top 10%.
# Run: python3 -m src.eval.rescue --m3 v0 --anomaly v0

import argparse
from pathlib import Path

import pandas as pd

from src.eval.oof import write_frames

TOP = 0.10


def top_flag(df, col):
    # Ranked within each fold, since OOF scores are only comparable inside one fold.
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
    rows = [
        {
            "group": "frauds missed by M3 top 10%",
            "n": len(missed),
            "share_of_all_frauds": len(missed) / len(fr),
        }
    ]
    for m in models:
        rows.append(
            {
                "group": f"rescued by {m} top 10%",
                "n": int(missed[f"top_{m}"].sum()),
                "share_of_missed": missed[f"top_{m}"].mean(),
                "random_baseline": TOP,
            }
        )
    both = missed["top_M5"] | missed["top_M6"]
    rows.append(
        {
            "group": "rescued by M5 or M6",
            "n": int(both.sum()),
            "share_of_missed": both.mean(),
            "random_baseline": 1 - (1 - TOP) ** 2,
        }
    )
    # The same question for non-frauds, since anomaly models also flag honest clients.
    hon = d[(d.target == 0) & (~d["top_M3"])]
    hb = hon["top_M5"] | hon["top_M6"]
    rows.append(
        {
            "group": "non-frauds outside M3 top 10% flagged by M5 or M6",
            "n": int(hb.sum()),
            "share_of_nonfrauds": hb.mean(),
            "random_baseline": 1 - (1 - TOP) ** 2,
        }
    )
    # Spearman correlation of the OOF scores shows how diverse the models are for fusion.
    corr = d[["M3"] + models].corr(method="spearman")
    out = pd.DataFrame(rows)
    out["m3_version"], out["anomaly_version"] = m3_v, an_v
    return out, corr


def main():
    p = argparse.ArgumentParser(description="Share of M3's missed frauds that M5/M6 rank highly.")
    p.add_argument("--m3", default="v0")
    p.add_argument("--anomaly", default="v0")
    a = p.parse_args()
    out, corr = rescue(a.m3, a.anomaly)
    path = Path("results/tables/rescue.csv")
    if path.exists():
        old = pd.read_csv(path)
        keep = ~((old["m3_version"] == a.m3) & (old["anomaly_version"] == a.anomaly))
        out = pd.concat([old[keep], out], ignore_index=True)
    write_frames({path: out})
    print(out[(out.m3_version == a.m3) & (out.anomaly_version == a.anomaly)].round(3).to_string())
    print(
        "\nSpearman correlation of OOF scores (diversity for fusion)\n", corr.round(2).to_string()
    )


if __name__ == "__main__":
    main()
