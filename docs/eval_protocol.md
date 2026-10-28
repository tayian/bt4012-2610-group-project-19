# Evaluation protocol and cost model (fixed rules)

These rules are frozen on 9 Oct and copied into docs/eval_protocol.md; every model is judged by them, so results are comparable. Changing them later invalidates every earlier result.

## 1. Unit of analysis. One row = one client. The client is who gets inspected, so the client is what we score.

## 2. Split
1. Stratified 80/20 split of clients on target, random_state=4012. Save the client IDs to data/splits/train_ids.csv and test_ids.csv. Never re-split.
2. Inside train: 5-fold StratifiedKFold(shuffle=True, random_state=4012). Save fold assignment to data/splits/folds.csv.
3. All tuning, model selection and ablations use 5-fold CV on train only.
4. The test set is scored once, in week 5 (task W5-B1), for the final table.

## 3. Leakage rules (non-negotiable)
- Resampling (SMOTE, undersampling) happens inside each training fold, never before the split.
- Target or frequency encoding is fitted on the training fold only.
- Graph neighbour features use out-of-fold labels only (a client never sees its own label or its fold’s labels).
- RAG precedent cases come from training clients only.
- Scalers are fitted on the training fold only.

## 4. Metrics reported for every model
Accuracy and F1 at a 0.5 threshold are not headline metrics. One sentence in the report explains why: “predicting no fraud for everyone scores about 94% accuracy.” Every model is compared at the same inspection budget, never at its own default threshold, because a lower threshold alone inflates recall.

## 5. Cost model, built up from one example
Start concrete. Suppose one confirmed fraud recovers 10 units of revenue and one inspection costs 1 unit. If we inspect 1,000 clients and 300 turn out to be fraud (precision 30%):
- Value recovered = 300 × 10 = 3,000
- Cost = 1,000 × 1 = 1,000
- Net = 2,000 units
Now the rule. Going down the ranked list, each extra visit is worth doing while the chance it finds fraud is at least cost ÷ value. Here that is 1 ÷ 10 = 10%. So the best budget is the point where precision on the ranked list falls to 10%.

In formula form, for the top k clients with TP(k) frauds found, value V per fraud and cost C per visit:

Net(k) = V × TP(k) − C × k, and the optimal budget k* is the k that maximises Net(k).

We do not know STEG’s true V/C, so we report results for V/C = 5, 10 and 20 (sensitivity analysis). Report figure: net recovery against budget k, one line per ratio, one panel per final model.

## 6. Comparing models fairly
- A model counts as better only if its PR-AUC gain exceeds the fold-to-fold standard deviation.
- Bootstrap: 1,000 resamples of test clients, report 2.5th and 97.5th percentiles.
- Every ablation changes exactly one thing against the reference model (LightGBM on features v1).
- Any proposed multi-stage pipeline is evaluated as a whole on the test set, never stage by stage.
