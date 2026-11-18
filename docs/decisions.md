# Decisions

Modelling decisions, the evidence for each, and what they mean for the report. All numbers are
5-fold CV on the 108,394 training clients (mean +/- sample std over folds), from
`results/tables/main_results.csv` unless stated. Sections 1 to 9 use cross-validation only;
section 10 is the single test-set run.

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

M5 isolation forest is no better than random (v1 PR-AUC 0.050, ROC-AUC 0.48 against a 0.056
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

## 7. Robustness: a realistic attack costs little; augmentation learns the attack's trace

The attack lowers each training fraud's declared consumption by 30% (uniformly, or by a
smooth ramp) and the features are rebuilt from the attacked invoices. Two flaws in the first
version each made attacked bills impossible, and a defence trained on such copies simply
learned to spot them (attacked recall far above clean):

- every tariff level was scaled by the same factor, leaving level 2+ consumption while level 1
  was under its band limit. The units removed now come off the highest level first.
- the meter readings were left as they were, so the meter kept advancing while consumption
  fell. On v3, whose meter-growth features see this, the augmented model reached 0.80 attacked
  recall. The readings now fall with the consumption: each invoice's index difference drops by
  the units removed, later readings on the same meter drop by the running total (so old_index
  still equals the previous new_index), and a meter that would read below zero starts higher.

Recall in the top 5% of clients, 5-fold CV, tuned M3 settings
(`results/tables/robustness.csv`, `results/figures/robustness_v4.png`; v1 and v3 rows are in the
same table):

| Model | v4 clean | v4 uniform | v4 smooth | v1 clean | v1 uniform | v1 smooth |
|---|---|---|---|---|---|---|
| **M3 (tree part of the final model)** | 0.324 | 0.303 | 0.275 | 0.302 | 0.285 | 0.283 |
| M3 without consumption features | 0.302 | 0.290 | 0.292 | 0.278 | 0.277 | 0.287 |
| M3 + augmentation (both attacks) | 0.308 | 0.354 | 0.376 | 0.297 | 0.344 | 0.357 |
| M3 + augmentation, uniform only | 0.314 | 0.358 | **0.337** (held out) | 0.297 | 0.344 | **0.341** (held out) |
| M3 + augmentation, smooth only | 0.298 | **0.340** (held out) | 0.398 | 0.293 | **0.337** (held out) | 0.363 |

- On v4, M3 loses 0.021 recall to the uniform attack and 0.049 (15%) to the smooth one. That is
  more than on v3 (0.006 and 0.022), because v4's recency features (last one or two years
  against the lifetime mean) are exactly what a gradual drop moves, and v3's meter counts,
  which the attack cannot touch, are gone. Hiding 30% of consumption still leaves most of the
  recall: without any consumption features the model keeps 0.302.
- Augmentation costs 0.016 clean recall (0.308 against 0.324) and its gain carries over to the
  attack it was not trained on (0.337 and 0.340 held out, both above the unaugmented model's
  attacked recall). Attacked recall above clean recall still means the model learns the trace
  of the attack (a sustained fall in declared consumption), so report it as "a drop of this
  kind is detectable", not as "the model is robust". In the study only frauds are attacked,
  so any trace of the attack is a fraud signal by construction. The final model is not
  augmented: the clean cost is certain, the attack is hypothetical.
- The v0 rows of the table come from the first attack and are kept only for the record.

## 8. features_v4: F12 without the features a meter replacement moves

v1 summarises consumption over the whole history and never uses the tariff, meter-code,
coefficient or meter-index fields. The time-robustness check (`results/tables/time_robustness.csv`)
showed that recent invoices matter: cutting every client's history at 2016 drops tuned M3 from
0.281 to 0.204 PR-AUC. F12 (`src/features/meter_tariff.py`, 27 features) adds recency
(last 1 and 2 years against the lifetime mean), tariff mix, meter codes and numbers, meter
index growth and per-band consumption. features_v3 = v1 + all of F12.

**v3 looked like a large gain, and most of it is leakage.** Tuned M3 on v3 scores 0.372 against
0.281 on v1, with the gain coming from the counts of meters, meter codes and tariffs
(`results/tables/f12_ablation.csv`). Labels have no date, so a meter replaced after a detected
fraud would also raise those counts. Two checks:

1. *When do meter replacements happen?* (`results/tables/f12_change_year.csv`,
   `results/figures/meter_change_year.png`) Among clients whose last meter replacement was in
   2006 to 2012, 3 to 5% are frauds, close to the 2.8% of clients with no replacement. From
   2013 the rate jumps to 14 to 34%. Within a client's history, a single replacement in the
   first 40% goes with a 5% fraud rate, one in the last 20% with 36 to 41%. That is what
   replacements after detections in a 2013 to 2019 campaign would look like. (An earlier check
   that only compared the share of changes in each client's final year missed this, because
   the campaign years, not the final year, are what matter.)
2. *Does the signal exist before 2013?* (`results/tables/f12_pre_cutoff.csv`) The same tuned
   M3, PR-AUC on the 82,387 clients with invoices before 2013:

   | Features | Full history | Invoices up to 2012 only |
   |---|---|---|
   | v1 | 0.293 | 0.153 |
   | v4 (v3 minus change-sensitive) | 0.314 | 0.159 |
   | v3 | 0.379 | 0.170 |

   The change-sensitive columns add 0.065 on the full history but only 0.011 before 2013.

So, by the same rule as decision 1, **the final table is features_v4**: v1 plus the 18 F12
features a meter or tariff change cannot move (recency, tariff shares, coefficient flag, index
growth, per-band consumption). The nine left out (`CHANGE_SENSITIVE`) are the meter, meter-code
and tariff counts, the most common meter code, the largest meter index, and the first and last
invoice years and the span between them. v3 is kept only as the leakage comparison, like
`features_v1_with_inspection`. A feature measuring years since the last meter change, tried
afterwards, added 0.10 PR-AUC on its own and was rejected for the same reason.

| Model | v1 | v4 | v3 (leakage comparison) |
|---|---|---|---|
| M3 tuned (30 Optuna trials per table) | 0.281 +/- 0.011 | **0.300 +/- 0.011** | 0.372 +/- 0.012 |
| M9 CatBoost | 0.265 +/- 0.014 | 0.285 +/- 0.011 | 0.353 +/- 0.005 |
| M7wd wide-and-deep | 0.248 +/- 0.010 | 0.256 +/- 0.008 | 0.296 +/- 0.011 |

The caveat that remains for the report: v1 itself loses half its PR-AUC when built from
pre-2013 invoices (0.293 to 0.153). Part of that is simply less history, but consumption can
also change after a detected fraud (a repaired meter reads the true consumption), and no
feature built from undated labels can fully rule that out. The same applies to v1's `n_counters`
(distinct meters on the account), which was kept because it is part of the base table, and is
the top SHAP driver for most clients in the final model: a meter replaced after a detection
raises it too.

## 9. Fusion: tuned M3 plus wide-and-deep on v4

Each model's out-of-fold scores become within-fold percentile ranks; a logistic-regression
stacker is fitted on four folds and scored on the fifth (`results/tables/fusion_ablation.csv`).
The rule fixed in advance: adopt the smallest stack whose gain over tuned M3 is positive in
every fold.

| Feature table | M3 tuned | + M7wd (LR stack) | gain, mean / worst fold | folds improved |
|---|---|---|---|---|
| v1 | 0.281 | 0.288 | +0.007 | 5 / 5 |
| v3 (leaky, comparison only) | 0.372 | 0.372 | +0.0003 / -0.005 | 4 / 5 |
| **v4 (final)** | 0.3005 | **0.3058** | **+0.0053 / +0.0023** | **5 / 5** |

- On v4, stacking tuned M3 with CatBoost alone gains only +0.0008. Adding CatBoost to M3 +
  M7wd gives +0.0059, 0.0006 more for a third model, so the two-model stack is kept.
- Rank averaging is worse than M3 alone in almost every combination, because the weaker models
  get equal weight. The graph (GraphSAGE, label spreading) and anomaly models add nothing.
- On v3, the meter-change features gave the tree the recent-behaviour signal the sequence
  model was adding, so fusion had nothing left to add. Once those features are dropped, the
  wide-and-deep model's monthly sequence is useful again.

**The final model is M10: the LR stack of tuned M3 and wide-and-deep on features_v4**
(0.306 +/- 0.011 CV PR-AUC, recall at the top 5% 0.326). SHAP drivers in the briefs come from
tuned M3, which carries most of the stack's weight.

## 10. Test set: scored once, on the CV-chosen model

The models above were refitted on all 108,394 training clients and scored once on the
27,099 held-out test clients (`results/tables/test_results.csv`, 95% bootstrap intervals).
Nothing was changed after this run.

| Model | Test PR-AUC [95% CI] | ROC-AUC | Recall @5% | Precision @5% |
|---|---|---|---|---|
| M1 logistic regression | 0.193 [0.179, 0.210] | 0.815 | 0.209 | 0.233 |
| M3 LightGBM, default | 0.276 [0.254, 0.299] | 0.844 | 0.306 | 0.342 |
| M9 CatBoost | 0.273 [0.253, 0.296] | 0.843 | 0.299 | 0.334 |
| M7wd wide-and-deep | 0.249 [0.231, 0.270] | 0.832 | 0.285 | 0.318 |
| M3 tuned | 0.294 [0.272, 0.318] | 0.850 | 0.316 | 0.353 |
| **M10 final stack** | **0.294 [0.272, 0.318]** | **0.852** | **0.316** | **0.353** |

- Test PR-AUC (0.294) is close to CV (0.306), so the CV estimates were not optimistic.
- The stack's gain over tuned M3 on the test set is +0.0002 (bootstrap 95% interval -0.008 to
  +0.007, positive in 52% of resamples; `test_fusion_gain.csv`). The +0.005 CV gain did not
  carry over: tuned M3 alone would have done as well, and the report should say so.
- Fairness (`fairness.csv`): at the cost-optimal budget no district or region is selected at
  more than twice its share of frauds. Calibration (`calibration_by_group.csv`): expected
  calibration error 0.011 overall on both test and training OOF scores; the worst district is
  0.014 on test and 0.018 on training OOF.
- SHAP stability (`explanation_stability.csv`): across three seeds the top driver is the same
  for 98% of the top 5% of clients; score rank correlation 0.99.
- Net recovery (`cost_model.csv`, `net_recovery.png`): with recovery worth 10 inspections, the
  best budget is about 15% of clients (precision 0.23, recall 0.64); inspecting everyone loses
  money at value-to-cost ratios of 5 and 10.

