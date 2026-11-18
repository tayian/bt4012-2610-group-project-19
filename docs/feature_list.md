# Feature list

Every feature table has one row per client (train and test, 135,493 rows) keyed by
`client_id`. Rebuild them with `python run.py features-v2` and `python run.py features-v3` (each
runs the steps it needs first; `python run.py all` runs both).

| File | Built by | Contents | Use |
|---|---|---|---|
| `features_v0.parquet` | `src/features/aggregate.py` | 15 quick features (F1, F2 means, F3 counts) | Baseline only |
| `features_v1.parquet` | `src/features/behavioural.py` | 54 features: F1 to F7 | Main table before F12; base of v3 and v4 |
| `features_v1_with_inspection.parquet` | same | v1 + F9 (67 features) | Leakage comparison only |
| `features_v2.parquet` | `src/features/combine.py` | v1 minus 14 redundant or unused columns, + F8 and F10 (49 features) | Graph models (M8sage) and the v2 comparison |
| `features_v2_with_inspection.parquet` | same | v2 + F9 (61 features) | Leakage comparison only |
| `features_v3.parquet` | `src/features/meter_tariff.py` | v1 + all 27 F12 features (81 features) | Leakage comparison only |
| `features_v4.parquet` | same | v1 + the 18 F12 features a meter or tariff change cannot move (72 features) | Final model (M10) and its components |
| `knn_edges.parquet` | `src/features/graph.py` | Client graph edges (`src`, `dst`, `kind`, `similarity`) | Graph models (M8, M8sage) |

**Leakage rules.** F9 comes from the inspector-written columns (`counter_statue`,
`reading_remarque`), which may only be filled in at or after an inspection, so it never goes
in the default tables. F8 and F10 use labels, always out-of-fold: a training client in fold k
only sees the labels of training clients in other folds, and a test client sees every
training label. Nobody sees their own label. (Training rows of one fold can still carry
features computed with another fold's labels, the usual trade-off of out-of-fold encoding.)

**Effect sizes** below are the rank-biserial correlation from the Mann-Whitney U test on
training clients (positive = larger for fraud clients); the full set, with Welch t-tests,
Cohen's d, chi-square, Cramér's V and information gain, is in `results/tables/stat_tests.csv`
(`python run.py feature-stats`). With about 108,000 training clients nearly every test is
significant, so features are ranked by effect size, not p-value.

## F1 Profile (v1 to v4)

`client_catg`, `district`, `region` (kept as text; encoded inside each fold), `tenure_years`
(last invoice minus creation date), `has_elec`, `has_gaz`, `n_counters` (distinct meters).
`n_counters` has the largest effect of any feature (r = 0.42): fraud clients have more meters.

## F2 Consumption level (v1 to v4)

Statistics of `cons_pm` (consumption per valid month) for all meters together and for
ELEC and GAZ separately (suffixes `_elec`, `_gaz`): `cons_pm_mean`, `_median`, `_std`,
`_min`, `_max`, `_mad`, `_skew`, `_kurt`, and `share_level_2_plus` (share of consumption
above tariff level 1). Fraud clients have higher and more variable consumption
(`cons_pm_max` r = 0.34), not lower.

## F3 Billing regularity (v1 to v4)

`n_invoices`, `invoices_per_year`, `months_number_mean`, `months_number_max` (from
`months_valid`), `days_since_last_invoice_mean`, `days_since_last_invoice_max` (gaps per
meter) and `has_missing_months` (a gap over twice the billed period). `n_invoices` r = 0.33.

## F4 Trajectory (v1 to v4)

Judged per meter type (ELEC and GAZ have different scales), then combined per client.

| Feature | Definition | r |
|---|---|---|
| `yoy_drop_max` | Largest fall between consecutive calendar years in mean `cons_pm`, as 1 - this year / last year (negative = no fall). The incomplete final year (2019) is left out | 0.20 |
| `slope_last3y` | Least-squares slope of `cons_pm` per year over the last three years; the steepest across meter types. NaN with fewer than 3 bills | -0.05 |
| `sudden_spike_share` | Share of bills above twice the client's median `cons_pm` (catch-up bills) | 0.21 |
| `zero_run_max` | Longest run of zero-consumption bills on one meter | 0.16 |
| `zero_share` | Share of zero-consumption bills | 0.09 |

## F5 Robust self-deviation (v1 to v4)

z = (x - client median) / (1.4826 x client MAD) per meter type; NaN where the MAD is 0.
`robust_z_max` (largest |z|, r = 0.15) and `n_robust_z_gt3` (bills with |z| > 3, r = 0.16).

## F6 Data validity (v1 to v4)

`n_index_backwards` (new index below old, excluding exact meter rollovers past 99999),
`n_recon_mismatch` and `recon_gap_mean` (index difference x coefficient differs from `cons`),
`n_bad_months_number` (`months_number` <= 0 or > 12). All are rare and carry almost no signal
(|r| <= 0.02); only 12 real backwards readings exist in the whole data.

## F7 Benford and digits (v1 to v4)

`benford_mad`: mean absolute deviation of the first-digit shares of `cons` from Benford's law;
NaN with fewer than 20 positive readings. `round_reading_share`: share of `new_index` values
ending in 0 or 5 (about 0.20 if honest). `benford_mad` is lower for fraud clients
(r = -0.20), mostly because they have more readings, and the deviation shrinks with more
readings; `round_reading_share` carries no signal (r = 0.01).

## F8 Peer residual (v2)

A LightGBM regressor predicts each bill's log1p(`cons_pm`) from `tarif_type`, `client_catg`,
`region`, invoice month and `counter_type`, trained on non-fraud clients only (out-of-fold, as
above). `peer_resid_mean` and `peer_resid_min` are the client's mean and lowest residual
(actual minus predicted; on the log scale, so -1 is about a third of the peer level).
Fraud clients have a lower minimum (r = -0.19): at least one bill far below similar customers.

## F10 Graph (v2)

The graph has two kinds of edges. kNN edges link each client to its 10 nearest clients
(cosine) on standardised F2 to F5 features (1,354,930 edges). Meter edges link clients billed
on the same real meter number (27,604; placeholder numbers such as `0000`, shared by 5,063
clients, and groups over 10 clients are dropped). About 21% of clients share a meter number
with another client, but pairs sharing one are fraud together only slightly more often than
chance (7.2% vs 5.6%).

| Feature | Definition | r |
|---|---|---|
| `nbr_fraud_rate_k10` | Fraud rate among the client's kNN neighbours with a visible label | 0.22 |
| `degree` | Distinct neighbours in the undirected graph (both edge kinds) | 0.06 |
| `community_size` | Size of the client's Louvain community (38 communities) | 0.19 |
| `community_fraud_rate` | Visible fraud rate in the community, shrunk towards the overall rate with a prior weight of 10 clients | 0.36 |
| `n_meter_neighbours` | Clients sharing a real meter number | 0.02 |
| `meter_fraud_rate` | Visible fraud rate among them | 0.02 |

`community_id` is saved in `graph_features.parquet` for reference but is not a model feature
(the IDs are arbitrary labels).

## F9 Meter status (with_inspection files only)

`statue_share_0` to `statue_share_5` and `statue_share_other`, `n_statue_changes` (status
changes on the same meter), `remarque_share_6` to `remarque_share_9` and
`remarque_share_other`. On LightGBM (default M3 settings, 5-fold CV PR-AUC), adding F9 raises
v1 from 0.261 to 0.315. That gap is above the 0.05 limit in the plan, so the headline uses
the tables without F9 and the gap is reported as a limitation.

## F12 Meter, tariff and recency (v3; v4 keeps 18 of 27)

Built by `src/features/meter_tariff.py` (`python run.py features-v3`). "Last N years" is
relative to the client's own last invoice.

| Group | Features |
|---|---|
| Recency | `cons_pm_mean_last1y`, `cons_pm_mean_last2y`, `ratio_last1y_lifetime`, `ratio_last2y_lifetime` (recent mean against the lifetime mean), `cons_pm_last_invoice`, `n_invoices_last2y`, `first_invoice_year`, `last_invoice_year`, `active_years` |
| Tariff | `n_tarif_types`, `tarif_share_11`, `tarif_share_40`, `tarif_share_10`, `tarif_share_15` |
| Meter | `n_counter_codes`, `counter_code_mode`, `has_coefficient_not_1`, `n_counter_numbers_elec`, `n_counter_numbers_gaz`, `new_index_max`, `index_growth_pm_mean`, `index_growth_pm_max` (index difference per valid month) |
| Bands | `level_1_pm_mean` to `level_4_pm_mean` (consumption per month in each tariff level), `share_level_3_plus` (share of bills with any level 3 or 4 consumption) |

`CHANGE_SENSITIVE` (left out of v4): `first_invoice_year`, `last_invoice_year`,
`active_years`, `n_tarif_types`, `n_counter_codes`, `counter_code_mode`,
`n_counter_numbers_elec`, `n_counter_numbers_gaz`, `new_index_max`. A meter replacement or
tariff change after a detected fraud moves these, and labels have no date, so they leak
(`docs/decisions.md`, section 8; `results/tables/f12_ablation.csv`, `f12_pre_cutoff.csv`).

## Dropped from v2

From each pair of features with |Spearman rho| > 0.9 (`results/tables/corr_pairs.csv`,
heatmap in `results/figures/corr_heatmap.png`), the one with the weaker univariate signal is
dropped: `cons_pm_mean`, `cons_pm_std`, `cons_pm_max_elec`, `cons_pm_median_elec`,
`cons_pm_std_elec`, `cons_pm_mean_gaz`, `cons_pm_median_gaz`, `cons_pm_std_gaz`,
`share_level_2_plus`, `n_recon_mismatch`, `zero_share` (and `statue_share_0` from the
with_inspection table). Two more have near-zero importance in the M3 SHAP v0 drivers:
`district` (never in any client's top 5; `region`, which contains it, is) and `has_elec`
(1 for 99.5% of clients). That leaves 49 features, below the plan's target of 60 to 90,
because removing near-duplicates mattered more than reaching a count.

## How much each version helps

LightGBM with the default M3 settings, 5-fold CV PR-AUC on training clients (mean +/- std):

| Features | PR-AUC |
|---|---|
| v0 | 0.2445 +/- 0.0142 |
| v1 | 0.2610 +/- 0.0135 |
| v1_with_inspection | 0.3150 +/- 0.0110 |

`main_results.csv` keeps no default-settings row for v2; tuned M3 scores 0.283 +/- 0.014 on v2
against 0.281 on v1. Tuned M3 on v1, v3 and
v4 is compared in `docs/decisions.md`, section 8 (v4: 0.300 +/- 0.011).

The graph models (5-fold CV PR-AUC, rows in `results/tables/main_results.csv`):

| Model | Inputs | PR-AUC |
|---|---|---|
| M8 label spreading (`python run.py label-spreading`) | graph and training labels only | 0.107 +/- 0.002 |
| M8sage GraphSAGE (`python run.py graphsage`, about 11 minutes on a laptop CPU) | graph + 39 behaviour columns of v2 | 0.226 +/- 0.011 |

Label spreading alone is about twice the 5.6% base rate; GraphSAGE, which also sees each
client's own behaviour, comes close to LightGBM on the same table.
