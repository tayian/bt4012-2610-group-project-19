# Build features_v2: features_v1 plus peer residuals and graph features, minus one feature of
# each highly correlated pair (corr_pairs.csv) and the LOW_IMPORTANCE columns.
# Run: python3 -m src.features.combine

import argparse
from pathlib import Path

import pandas as pd

from src.features.behavioural import INSPECTION_COLUMNS
from src.features.graph import FEATURES as GRAPH_FEATURES
from src.features.peer import COLUMNS as PEER_FEATURES

# district never appears in any client's top-5 SHAP drivers (region, which it is nested in,
# does); has_elec is 1 for 99.5% of clients and carries 0.07% of the SHAP weight.
LOW_IMPORTANCE = ["district", "has_elec"]


def combine(v1, peer, graph, drop):
    out = v1.drop(columns=[c for c in [*drop, *LOW_IMPORTANCE] if c in v1.columns])
    for extra in (peer[["client_id", *PEER_FEATURES]], graph[["client_id", *GRAPH_FEATURES]]):
        out = out.merge(extra, on="client_id", how="left", validate="one_to_one")
    return out


def main():
    parser = argparse.ArgumentParser(description="Build features_v2 (v1 + F8 + F10).")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--pairs", type=Path, default=Path("results/tables/corr_pairs.csv"))
    args = parser.parse_args()
    p = args.processed
    try:
        v1 = pd.read_parquet(p / "features_v1_with_inspection.parquet")
        frames = [
            v1,
            pd.read_parquet(p / "peer_residual.parquet"),
            pd.read_parquet(p / "graph_features.parquet"),
        ]
        for frame in frames:
            frame["client_id"] = frame["client_id"].astype("string")
        drop = sorted(set(pd.read_csv(args.pairs)["drop"]))
        full = combine(*frames, drop)
        inspection = [c for c in INSPECTION_COLUMNS if c in full.columns]
        full.drop(columns=inspection).to_parquet(p / "features_v2.parquet", index=False)
        full.to_parquet(p / "features_v2_with_inspection.parquet", index=False)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"combine: {exc}\n")
    n = full.shape[1] - 1
    print(
        f"Dropped {len(drop) + len(LOW_IMPORTANCE)} redundant or unused v1 columns: {', '.join(drop + LOW_IMPORTANCE)}"
    )
    print(
        f"Saved features_v2 ({n - len(inspection)} features) and features_v2_with_inspection ({n})"
    )


if __name__ == "__main__":
    main()
