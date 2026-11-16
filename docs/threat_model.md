# Threat model (report section 4.4)

1. **Attacker:** a fraudulent client who already sits in the model's top-5% risk list.
2. **Capability:** can lower the *declared* consumption on their own invoices (`consommation_level_1..4`), in any month and on ELEC or GAZ.
3. **Cannot change:** district, region, category, tenure, counter numbers, meter status/remark history (`counter_statue`, `reading_remarque`), or the meter index readings taken by STEG staff (`old_index`, `new_index`).
4. **Budget:** total declared consumption over the client's history falls by at most 30%.
5. **Goal:** drop below the top-5% inspection cut-off, i.e. reduce recall@5% for fraud clients.

Attacks implemented in `src/eval/adversarial.py` (`python run.py robustness`):

| Attack | Target | Knowledge | How |
|---|---|---|---|
| Uniform cut | M3, M7 | none | every invoice × 0.70 |
| Smooth-decline mimicry | M3, M7 | black-box | multiplier ramps linearly from 1 to f_end across the client's invoices (honest-looking slow decline), f_end chosen so the total falls exactly 30% |
| PGD | M7 | white-box | 10 steps on the sequence input, perturbation ≤ 0 per month, projected back onto the 30% budget; static branch held fixed |

After the uniform and smooth attacks, `cons` and `cons_pm` are recomputed as `src/data/clean.py` does, and the features and sequence are rebuilt with the same pipeline code as everyone else (`aggregate.py` for v0, `behavioural.py` for v1, `sequence.py`).

Because the attacker cannot change the index readings, lowering declared consumption makes `idx_diff × counter_coefficient` stop matching `cons`. v0 has no reconciliation features, so the v0 study (the one in `results/tables/robustness.csv`) cannot see this. v1 does (`n_recon_mismatch`, `recon_gap_mean`, `n_index_backwards`), and its trajectory features (`yoy_drop_max`, `sudden_spike_share`, `zero_*`, `robust_z_*`) also move when declared consumption drops, so a v1 run of the study measures a harder target for the attacker. An earlier run on stand-in features showed that a tree model can rank clients with many mismatches as *low* risk, because almost no training client has that many; a monotone constraint on such features would prevent it. An index-consistent attack (the attacker also rewrites `new_index`, e.g. by colluding with the meter reader) only differs from the smooth attack once those features exist.

Defences: adversarial augmentation for M3 (one attacked copy of each training-fold fraud), PGD adversarial training and mimicry augmentation for M7. Results: `results/tables/robustness.csv`.

How to read the defence rows: the defences are trained and tested on the same attack families, so attacked recall can come out *above* clean recall (the model learns to spot the attack's pattern). That shows the attack is detectable, not that the model is robust to attacks it has not seen; a held-out-attack check (train on uniform, test on smooth) is needed before claiming robustness. PGD only perturbs the sequence branch with the static features held fixed, so its drop is an upper bound for an attacker who must also keep the static features consistent.
