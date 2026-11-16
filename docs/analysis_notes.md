# Analysis notes: Benford, EDA part 2 and time robustness

Every number below comes from a file in `results/tables/` and uses training clients only
(108,394 clients, 6,053 fraud, 5.58%). The test clients are never read.

## Benford's law on consumption (`python run.py benford`)

Code: `src/features/benford_analysis.py`, shown in `notebooks/02_benford.ipynb`. Tables:
`benford_first_digit.csv`, `benford_district.csv`, `benford_drilldown.csv`, `benford_last_digit.csv`,
`benford_caveats.csv`. Figures: `benford_first_digit.png` (report F3), `benford_two_digit.png`.

The test uses positive per-invoice consumption (`cons` > 0). MAD bands are Nigrini's: first digit
close <= 0.006, acceptable <= 0.012, marginal <= 0.015, otherwise nonconformity; first two digits
close <= 0.0012, acceptable <= 0.0018, marginal <= 0.0022.

| Group | Invoices | First-digit MAD | Nigrini band |
|---|---|---|---|
| Non-fraud | 2,952,305 | 0.0157 | nonconformity |
| Fraud | 252,464 | 0.0075 | acceptable |

- Both chi-square tests have p ≈ 0 (chi-square 69,917 and 1,472 on 8 df). With this many invoices
  any deviation is significant, so the MAD band is the number to read.
- Both classes have too few bills starting with 1 or 2 (non-fraud 26.3% vs 30.1% expected for 1)
  and too many starting with 4 to 6. Non-fraud invoices deviate more than fraud invoices, so a
  large deviation is not a fraud marker here. This matches F7: `benford_mad` is lower for fraud
  clients.
- By district, ranked by MAD: the non-fraud clients of district 63 (0.0201) and district 62
  (0.0183) are in nonconformity. District 60 non-fraud is marginal (0.0122). Every other group is
  acceptable or close. In every district the fraud group is closer to Benford than the non-fraud
  group (fraud MAD 0.0054 to 0.0094).
- Drill-down on the worst group (district 63, non-fraud, 695,286 invoices): the first-digit MAD is
  0.0201 (nonconformity) and the first-two-digit MAD is 0.0020 (marginal). Values from 10 to 30 are
  too rare. Values from about 31 to 60 are too common, most of all 42 to 50 (about +0.3 points each).
  The excess is a broad plateau, not spikes at single values. That looks like many households billed
  in the same narrow range of a few hundred units per bill, not like invented numbers.
- Last digit of `new_index` (readings > 0): 21.19% of non-fraud readings and 21.15% of fraud
  readings end in 0 or 5, against 20% expected (95% CI 21.15–21.24% and 20.99–21.30%). The excess is
  small and the same in both classes. Dropping zero-consumption bills, whose reading repeats, gives
  21.25% and 21.13%.
- Caveats:
  - Consumption spans only about two orders of magnitude.
  - Tariff level 1 is capped per billing period: 16.5% of positive bills reach level 2 or above, so
    they start at or just above a band limit.
  - The median client has 27 positive bills. Even a sample that follows Benford exactly has an
    expected MAD of 0.045 at that size (nonconformity), so a per-client MAD is mostly noise.
  - 39% of clients have fewer than 20 positive bills, which is why F7 leaves `benford_mad` empty
    below that count.

## Outliers, validity and EDA figures (`python run.py eda-figures`)

Code: `src/viz/eda_figures.py`. Tables: `eda_outliers.csv`, `eda_validity_by_class.csv`,
`eda_fraud_rate_heatmap.csv`, `eda_cons_over_time.csv`. Figures: `eda_fraud_rate_heatmap.png`
(report F2), `eda_cons_over_time.png` (report F1), `eda_top_features_boxplots.png`.

- Outliers. Fences and z-scores use both classes pooled. Fraud clients are flagged 2 to 4 times as
  often as non-fraud clients by both rules:

  | Variable | Boxplot rule, non-fraud | Boxplot rule, fraud | abs(z) > 3, non-fraud | abs(z) > 3, fraud |
  |---|---|---|---|---|
  | `cons_pm_max` | 8.5% | 19.6% | 0.37% | 0.74% |
  | `n_counters` | 2.4% | 9.7% | 0.42% | 1.42% |
  | per-invoice `cons_pm` | 7.1% | 12.3% | 0.70% | 1.35% |

  The heavy right tail of consumption is where fraud clients sit, so these outliers are kept, not
  removed.
- Invalid combinations (F6 flags):
  - 17.6% of fraud clients and 14.4% of non-fraud clients have at least one invalid reading
    (chi-square p = 6e-12).
  - The difference comes from reconciliation mismatches (6.6% vs 4.6% of clients) and
    `months_number` outside 1 to 12 (12.4% vs 10.9%).
  - Per invoice the rates are about the same (73 vs 85 flagged per 10,000 invoices), so fraud clients
    have more of these readings only because they have more invoices.
  - Only 6 real backwards readings exist, all from non-fraud clients.
- Fraud rate by district and category (F2): catg 51 has the highest rate in every district, from
  8.9% in district 60 to 24.0% in district 69 (n = 551). Among catg 11, which holds most clients, the
  rate rises from 3.6% (district 60) to 6.8% (district 69). Catg 12 in district 60 is lowest (0.6%,
  n = 462).
- Consumption over time (F1, electricity invoices, quarterly median and IQR of `cons_pm`): the class
  medians track each other from 2005 to about 2014 (for example 2010: 111 vs 113). After that the
  fraud median pulls ahead: 2016 is 112 (non-fraud) vs 131 (fraud), 2018 is 108 vs 135. Fraud labels
  have no date, so this may reflect when clients were caught rather than when they changed. The fraud
  upper quartile spikes in the first quarter of most years; Q1 holds many catch-up bills.
- Top six features by rank-biserial effect size (from `stat_tests.csv`); all are higher for fraud:
  - `n_counters` 0.42
  - `cons_pm_max` 0.34
  - `cons_pm_max_elec` 0.34
  - `n_invoices` 0.33
  - `cons_pm_std_elec` 0.31
  - `cons_pm_mad_elec` 0.29

## Time robustness (`python run.py time-robustness`)

Code: `src/eval/time_robustness.py`. Table: `time_robustness.csv`.

features_v1 is rebuilt by the same `build_features` from invoices dated up to 2016-12-31 (3,474,804
of 4,476,749 invoices). The partial-final-year flag is recomputed on the shorter history, and 2016 is
a complete year. Tuned M3 (`M3_tuned_params_v1.json`, `trees.model_fn`) is trained on the same five
saved folds. Metrics are the fold mean ± sample standard deviation from `budget_metrics`. The
full-history row matches `M3_tuned_features_v1` in `main_results.csv`.

| History | Scope | PR-AUC | ROC-AUC | Recall@5% |
|---|---|---|---|---|
| full | all 108,394 training clients | 0.281 ± 0.011 | 0.844 ± 0.004 | 0.302 ± 0.013 |
| up to 2016 | all 108,394 training clients | 0.204 | 0.784 | 0.238 |
| full | 98,828 clients with pre-2017 invoices | 0.285 | 0.840 | 0.291 |
| up to 2016 | 98,828 clients with pre-2017 invoices | 0.209 | 0.785 | 0.229 |

- Without the last three years, PR-AUC falls by about 27% (0.281 to 0.204), ROC-AUC by 0.06 and
  recall at a 5% inspection budget from 30% to 24%.
- The drop is almost the same on the clients that do have older invoices, so it is not only the
  9,566 clients with no history before 2017 (1.4% of them are fraud).
- Recent behaviour carries much of the signal. This fits F1, where the fraud median only pulls
  ahead after about 2014.
- Limits: labels have no date, so we cannot tell whether a fraud happened before or after 2016. The
  model should be retrained as new invoices arrive.
