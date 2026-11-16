# Decisions

Modelling decisions, the evidence for each, and what they mean for the report. All numbers are
5-fold CV on the 108,394 training clients (mean +/- sample std over folds), from
`results/tables/main_results.csv` unless stated. The test set has not been used.

## 1. Leakage A/B: headline models use features without the inspection columns

`features_v1_with_inspection` adds F9 (meter status and reading remark shares), built from
`counter_statue` and `reading_remarque`, which inspectors may fill in at or after a visit.

| Model | features_v1 | features_v1_with_inspection | Gap |
|---|---|---|---|
| M1 logistic regression (L2, C = 1) | 0.189 +/- 0.006 | 0.201 +/- 0.005 | +0.012 |
| M3 LightGBM (default settings) | 0.261 +/- 0.013 | 0.315 +/- 0.011 | **+0.054** |

The plan's rule: if the full table beats the leak-free one by more than 0.05 PR-AUC, the
leak-free model is the headline and the gap is a limitation. The LightGBM gap is 0.054, so
**every headline model uses `features_v1` / `features_v2` (no F9)**. Report the gap as a
limitation: part of what predicts fraud in this data may be recorded during the inspection
itself, so a model using it would look better on paper than it can be before a visit.

The rows are in `main_results.csv` under the run IDs `M1_features_v1_*`,
`M1_features_v1_with_inspection_*`, `M3_features_v1_*` and `M3_features_v1_with_inspection_*`
(their `feature_version` column reads `v1` for both; the run ID tells them apart).

## 2. Reference model: tuned LightGBM on features_v1

Optuna (TPE, seed 4012, 30 trials, objective = mean CV PR-AUC) tuned M3 on v1
(`python -m src.models.tune`; trials in `results/tables/M3_tuning_v1.csv`, chosen parameters in
`M3_tuned_params_v1.json`). Trial scores ranged 0.242 to 0.281.

| Model | PR-AUC | ROC-AUC | Recall@5% | Precision@5% |
|---|---|---|---|---|
| M3 default, v0 | 0.245 +/- 0.014 | 0.834 | 0.274 | 0.306 |
| M3 default, v1 | 0.261 +/- 0.013 | 0.837 | 0.288 | 0.322 |
| **M3 tuned, v1** | **0.281 +/- 0.011** | 0.844 | 0.303 | 0.338 |
| M9 CatBoost, v1 | 0.265 +/- 0.014 | 0.836 | 0.289 | 0.323 |
| M2 random forest, v1 (best of 4) | 0.249 +/- 0.013 | 0.830 | 0.280 | 0.313 |
| M1 LR, v1 | 0.189 +/- 0.006 | 0.814 | 0.210 | 0.234 |

The tuned score is the best of 30 candidates on the same folds, so it is slightly optimistic;
the test set will give the unbiased number. CatBoost (untuned, 800 trees) is within one fold
std of default LightGBM and below the tuned one, so M3 tuned stays the reference. The SHAP
drivers for v1 (`results/shap_top5_v1.parquet`) explain this tuned model.

The LR grid was not rerun on v1. On v0 all 12 candidates were within 0.003 PR-AUC of each
other (`results/tables/M1_grid_v0.csv`), and each v1 candidate takes about 20 minutes, so the
v1 grid would almost certainly confirm the default (L2, C = 1) already reported above.

**features_v2 adds nothing measurable for the tree models.** With the same settings, tuned M3
scores 0.283 +/- 0.014 on v2 against 0.281 on v1, and CatBoost 0.261 against 0.265: both well
inside one fold std. The graph does carry signal on its own (GraphSAGE 0.226, label spreading
0.107), but the trees already get it from the behaviour features. For the final model either
table is defensible; v1 is simpler to explain, v2 only matters if the fusion ablation shows the
graph models add something.

## 3. Imbalance handling: keep scale_pos_weight, never SMOTE

On the tuned M3 (`results/tables/imbalance_tuned_v1.csv`):

| Strategy | PR-AUC | Brier |
|---|---|---|
| none | 0.279 +/- 0.010 | 0.046 |
| scale_pos_weight (negatives / positives) | 0.281 +/- 0.011 | 0.085 |
| undersample to 1:3 | 0.274 +/- 0.012 | 0.079 |
| SMOTE | 0.214 +/- 0.012 | 0.056 |

None, weighting and undersampling are within one fold std of each other: **no meaningful
difference**, so the selection file's "winner" should not be reported as one. SMOTE is clearly
worse on both v0 and v1, also after synthetic frauds were kept to whole numbers in count
columns. The Brier column is not comparable across rows: weighting and undersampling shift the
scores upwards on purpose (see 4).

The same applies to the v0 selections: the imbalance "winner" (none, 0.2452 vs 0.2445) and the
LR grid winner (C = 10 vs C = 1 by 0.00001) are ties, and M2 random forest (0.248) is not
better than M3 (0.245), especially as M2 had a 4-candidate search and M3 none.

## 4. Scores are rankings, not probabilities

Class weighting makes raw scores far higher than the real fraud rate (the M1 average score is
0.39 against a 5.6% fraud rate). Isotonic calibration, fitted inside each training fold, fixes
this almost exactly (`results/figures/reliability_v0.png`, score deciles), but most of the
Brier gain only undoes the weighting: M3 without weighting already has the same Brier
(0.047) as M3 isotonic, and calibration costs some PR-AUC (0.245 -> 0.236). So:

- every ranking metric (PR-AUC, precision/recall@k, net recovery) uses the raw scores;
- anything shown as a probability, such as `risk_score` in the Inspector Briefs, must come
  from a calibrated model, or the brief shows rank and tier only.

## 5. Unsupervised models: report as a negative result

M5 isolation forest is no better than random (v1 PR-AUC 0.050, ROC-AUC 0.47 against a 0.056
base rate), on all features and on behaviour columns only; flipping the score sign does not
help either. M6 autoencoder is weakly useful (0.094). Fraud clients in this data look ordinary
(higher, more variable consumption, longer histories), so "anomalous" does not mean
"fraudulent". The rescue analysis (`results/tables/rescue.csv`, tuned M3 v1 against v1
anomaly scores) looks at the 3,131 frauds that tuned M3 ranks below its top 10%:

- M5 puts 5.4% of them in its own top 10%, below the 10% a random scorer would.
- M6 puts 14.4% there. That looks like a lift, but M5 or M6 together reach 18.0%, against a
  19% chance rate, and they flag 18.1% of the non-frauds outside M3's top 10% as well.

So the anomaly models should only enter the fusion if the fusion ablation shows a gain. On v0
the isolation forest now uses the behaviour columns only (as on v1) and scores 0.046.

## 5b. Neural models: close to default LightGBM, below the tuned one

| Model (v1) | PR-AUC | ROC-AUC | Recall@5% | Spearman with tuned M3 |
|---|---|---|---|---|
| M7wd wide-and-deep (calendar grid + static) | 0.248 +/- 0.010 | 0.832 | 0.283 | 0.85 |
| M7 1D-CNN (last-invoice aligned + static) | 0.245 +/- 0.013 | 0.833 | 0.276 | 0.87 |
| M4 MLP, focal loss | 0.243 +/- 0.012 | 0.831 | 0.273 | 0.88 |
| M4 MLP, BCE | 0.242 +/- 0.010 | 0.830 | 0.275 | 0.88 |

The sequence models gain about 0.005 over the MLP on the same static features, inside one
fold std. All of them correlate 0.85 to 0.88 with tuned M3 in rank, and they rescue 13 to 16%
of its missed frauds (chance: 10%). So they are the most promising fusion partners, but they
are not expected to beat M3 alone. The earlier v1 rows (CNN 0.180, wide-and-deep 0.185, MLP
0.152) came from a stale `seq.npy` and an older table; they have been replaced. The M7 v0 run
was not repeated on the rebuilt sequences.

## 6. PU learning: no reliable hidden-fraud estimate

Elkan and Noto on the tuned M3 scores gives c = 0.10 (a known fraud's average score), and the
two standard estimates of the true fraud rate disagree by a factor of three (18.5% from the
mean adjusted score, 53.4% from labelled rate / c; `results/tables/pu_estimate.csv`). A small
c mostly reflects how weak any model's scores are here, not how many frauds are hidden. Report
it as "the data are consistent with unlabelled fraud, but its size cannot be estimated from
these scores", not as a number.

## 7. Robustness results: detection, not robustness

The defence rows of `results/tables/robustness.csv` were trained and tested on the same attack
families, and some show attacked recall above clean recall: the model learns to spot that
attack. Describe them as "the attack is detectable", and add a held-out-attack check (train on
one attack, test on the other) before claiming robustness. The study so far ran on v0; v1 adds
reconciliation and trajectory features that a consumption-lowering attacker also moves.
