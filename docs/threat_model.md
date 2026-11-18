# Threat model (report section 4.4)

1. **Attacker:** a fraudulent client who already sits in the model's top-5% risk list.
2. **Capability:** can lower the *declared* consumption on their own invoices (`consommation_level_1..4`), in any month and on ELEC or GAZ, and the meter index readings move with it (for example by tampering with the meter or colluding with the reader), so the bills stay internally consistent.
3. **Cannot change:** district, region, category, tenure, counter numbers, or meter status/remark history (`counter_statue`, `reading_remarque`).
4. **Budget:** total declared consumption over the client's history falls by at most 30%.
5. **Goal:** drop below the top-5% inspection cut-off, i.e. reduce recall@5% for fraud clients.

Attacks implemented in `src/eval/adversarial.py` (`python run.py robustness` for M3 and M7 on v0, `python run.py robustness-final` for M3 on the final table):

| Attack | Target | Knowledge | How |
|---|---|---|---|
| Uniform cut | M3, M7 | none | every invoice's consumption x 0.70 |
| Smooth-decline mimicry | M3, M7 | black-box | multiplier ramps linearly from 1 to f_end across the client's invoices (honest-looking slow decline), f_end chosen so the total falls exactly 30% |
| PGD | M7 | white-box | 10 steps on the sequence input, perturbation <= 0 per month, projected back onto the 30% budget; static branch held fixed |

How an attacked invoice is built, so that it looks like a real bill:

- The units removed come off the highest tariff level first, so lower levels stay within their band limits.
- The index readings fall with the consumption: each invoice's index difference drops by the units removed, later readings on the same meter drop by the running total (so `old_index` still equals the previous `new_index`), and a meter that would read below zero starts higher.
- `cons` and `cons_pm` are recomputed as `src/data/clean.py` does, and the features and sequence are rebuilt with the same pipeline code as everyone else (`aggregate.py` for v0, `behavioural.py` for v1, `meter_tariff.py` for v3 and v4, `sequence.py`).

An earlier version scaled every tariff level by the same factor and left the index readings unchanged. Both made attacked bills impossible, and a defence trained on them simply learned to spot them; the v0 rows of `results/tables/robustness.csv` still come from that version and are kept only for the record.

Defences: adversarial augmentation for M3 (one attacked copy of each training-fold fraud, with both attacks, uniform only or smooth only), dropping the consumption features, and PGD adversarial training and mimicry augmentation for M7.

Results: `results/tables/robustness.csv` has rows for v0, v1, v3 and v4 (per fold in `robustness_by_fold.csv`, figure `results/figures/robustness_v4.png`). The numbers and their reading are in [decisions.md](decisions.md), section 7.

How to read the defence rows: attacked recall can come out *above* clean recall, because the model learns the trace of the attack (a sustained fall in declared consumption), and in the study only frauds are attacked. The held-out-attack check (train on uniform, test on smooth, and the reverse) keeps most of the gain, so a drop of this kind is detectable; it does not show the model is robust to attacks unlike these. PGD only perturbs the sequence branch with the static features held fixed, so its drop is an upper bound for an attacker who must also keep the static features consistent.
