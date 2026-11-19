# Model experiments and SHAP

The defaults use current features_v0 and the committed training clients/folds. All model
inputs are loaded through linear.load_training(), which calls read_splits(). Every scored
model comparison uses the shared cv_evaluate() and OOF writer. SHAP refits the same saved
fold models for explanations. The separate test set is never
fitted, scored, explained or used for calibration here.

## Run commands

```bash
python run.py shap
python run.py imbalance random-forest lr-grid calibrate
```

This order produces the SHAP drivers first. Each invocation builds its data/feature
dependencies once. `python run.py model-experiments` runs the same full set in one call.
The core `python run.py all` also builds baseline SHAP and runs the tests, while the longer
searches run explicitly. Both workflows work from the two raw files in data/raw/.

With current features already built, the individual modules avoid repeating data cleaning:

```bash
python -m src.eval.explain
python -m src.models.imbalance
python -m src.models.forest
python -m src.models.lr_search
python -m src.eval.calibration
```

Each module accepts --features, --clients, --split-dir and --results-dir. These experiments
can also use a later feature file via --features; the default filenames below are for v0.
The numeric SHAP value contract additionally requires category values to be numeric codes,
as they are in the current v0 feature table.

## LR grid

- C in 0.01, 0.1, 1, 10, crossed with L2, elastic net (l1_ratio=0.5) and L1: 12 candidates.
- saga, balanced class weights and the existing median-impute, scale, one-hot pipeline.
  All learned preprocessing is fitted inside each training fold.
- Five saved-fold CV fits per candidate, selected by mean fold PR-AUC. Exact ties retain
  grid order. This selected CV estimate has the usual model-selection optimism; the final
  untouched test set is still reserved for the frozen model.
- Outputs: tables/M1_grid_v0.csv, tables/M1_selection_v0.json, oof/M1_grid_v0.parquet,
  coefficients/M1_grid_v0.csv and coefficients/M1_grid_v0_folds.csv, all under results/.
- The first coefficient file is a refit on all training clients for interpretation only.
  Its numeric coefficients are in standardised units; categorical coefficients are on
  one-hot indicators. The fold coefficient file includes every grid candidate's coefficients,
  C/penalty and each fold's own encoded names.
- Existing M1_v0.parquet is the original L2 C=1 baseline; M1_grid_v0 is the selected grid run.

No winner is published until every candidate converges and the selected all-training
coefficient refit converges. A failure prints the candidates that need more iterations.
For example, rerun `python run.py lr-grid --lr-max-iter 30000`. Already converged candidates
can be reused with a larger cap; a failed candidate restarts its five CV fits. Interrupting
a candidate also restarts that candidate, rather than the whole grid.

## Random forest

M2 uses 500 trees and balanced_subsample class weights, with max_depth in None/12 and
min_samples_leaf in 5/20. Numeric median imputation and categorical one-hot encoding fit
inside each fold; trees do not use the LR scaler. The four candidates use 20 CV fits.
The selected mean-PR-AUC winner writes oof/M2_v0.parquet, tables/M2_grid_v0.csv and
tables/M2_selection_v0.json under results/.

## Imbalance study

The four arms hold the default M3 tree parameters, features and saved folds fixed:

| Strategy | Inside each training fit |
|---|---|
| none | All training rows, scale_pos_weight=1 |
| scale_pos_weight | All training rows, weight = training negatives / training positives |
| undersample | Keep every positive and sample exactly three negatives per positive |
| smote | Add synthetic positives to match the negative count; no class weighting |

The mixed numeric/categorical inputs use SMOTENC, the categorical-aware SMOTE variant.
It avoids fractional region or client-category codes. Numeric median imputation and
standardisation for neighbour search are fitted only on the training fold. The imputed
space generates synthetic rows; numeric values are transformed back to their original
units and original real rows retain their original missing values. Category missing
markers become missing again. LightGBM continues to handle numeric and categorical missing
values natively. Sampler seeds are 4012.

Only training rows are resampled. Validation rows keep their natural fraud rate. The
scale_pos_weight arm reproduces the current M3 baseline for identical inputs/threads.
Each arm writes oof/M3_<strategy>_v0.parquet and a main_results.csv row. Comparison and
selection are saved in tables/imbalance_v0.csv and tables/imbalance_selection_v0.json.
The winning configuration is recorded for model selection; the shared fit_m3 default
continues to identify the weighted baseline. Adopt a different strategy explicitly when
the team selects the final reference model.

## Isotonic calibration

This experiment calibrates the original M1 L2 C=1 balanced baseline and the original M3
weighted baseline. It does not silently substitute the selected LR grid winner or select
an imbalance strategy.

Each outer saved fold is used only for evaluation. Inside its training partition,
CalibratedClassifierCV(method="isotonic", ensemble=False) uses three shuffled stratified
inner folds, seed 4012. Their out-of-fold scores fit the calibrator, then the underlying
classifier is refitted on the full outer training partition. The outer validation labels
never fit the model or calibrator. Preprocessing and weights are refitted within the inner
models as well. This is 20 base fits per family (3 inner fits plus a refit, over 5 outer folds).

The outer validation predictions are used to compare raw and isotonic scores. Fitting an
isotonic regressor directly on the full OOF file and assessing it on those same rows would
not be an independent evaluation. Isotonic mappings can introduce ties and change ranking
metrics; the shared tie rules remain in force.

Outputs under results/:

- oof/M1_isotonic_v0.parquet and oof/M3_isotonic_v0.parquet, with main_results.csv rows.
- oof/M1_calibration_raw_v0.parquet and oof/M3_calibration_raw_v0.parquet: paired raw
  scores from the same outer-training refits for an auditable comparison.
- tables/calibration_v0.csv: raw versus calibrated fold metrics, including Brier and net
  recovery from the shared evaluator.
- tables/reliability_v0.csv and figures/reliability_v0.png: ten equal-width probability
  bins, bin counts, average scores and observed fraud shares, using outer-fold scores.

Neither the name isotonic nor a lower Brier alone proves calibration. Inspect the reliability
curve and independent Brier results. Raw weighted baseline scores remain uncalibrated risk
scores for the report and LLM briefs.

## SHAP file contract

`results/shap_top5_v0.parquet` contains exactly these columns:

| Column | Type and meaning |
|---|---|
| client_id | Original string ID; training clients only |
| rank | int64, 1 to 5, descending absolute SHAP contribution |
| feature | Original feature name |
| value | float64 raw feature value or numeric category code; NaN for missing |
| shap_value | float64 signed contribution in raw LightGBM log-odds units |

TreeExplainer uses tree_path_dependent and model_output="raw". For each client's saved
validation fold, its explanation comes from the default weighted M3 fitted on the other
four folds. Categories are mapped exactly as in that fitted model. Full feature names
and order are preserved, including categorical codes. Feature-order ties are stable.
SHAP explains the raw baseline scorer, not the later isotonic transformation and not a
causal effect or a change in fraud probability.

The producer checks its scores against M3_v0.parquet when that file is present. A mismatch
stops publication and asks for a current baseline rebuild. The exact reference scores are
also written to oof/M3_shap_reference_v0.parquet; tables/shap_metadata_v0.json records the
feature set, training fingerprint, scope and units. Five rows are produced for each of the
108,394 training clients: 541,970 rows on the current course split.

The values are in log-odds units and explain the M3 scorer only.

## Checkpoints and results

Completed candidates, calibration pairs and SHAP folds are cached below results/runs/
(gitignored). Reuse requires matching training features, labels, saved folds, source code,
library versions and settings. Held-out values/labels are excluded from the data fingerprint.
Changing the relevant source or data invalidates the cache. A partially fitted candidate
is never reported as complete. Only successful model fits publish their named OOF files.

Each published run upserts its own main_results.csv row and preserves other runs. Selected
grid, isotonic and imbalance files have distinct model names. Do not substitute them for
the baseline files without choosing the corresponding model deliberately.

The budget and cost metrics come from the shared code in `src/eval/metrics.py`, and these
experiments never touch the test set.

References:

- https://scikit-learn.org/1.7/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- https://imbalanced-learn.org/stable/over_sampling.html
- https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html
