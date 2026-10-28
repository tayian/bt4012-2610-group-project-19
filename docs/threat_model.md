# Threat model (W4-C1, report section 4.4)

1. **Attacker:** a fraudulent client who already sits in the model's top-5% risk list.
2. **Capability:** can lower the *declared* consumption on their own invoices (`consommation_level_1..4`), in any month and on ELEC or GAZ.
3. **Cannot change:** district, region, category, tenure, counter numbers, meter status/remark history (`counter_statue`, `reading_remarque`), or the meter index readings taken by STEG staff (`old_index`, `new_index`).
4. **Budget:** total declared consumption over the client's history falls by at most 30%.
5. **Goal:** drop below the top-5% inspection cut-off, i.e. reduce recall@5% for fraud clients.

Attacks implemented in `src/eval/adversarial.py`:

| Attack | Target | Knowledge | How |
|---|---|---|---|
| Uniform cut | M3, M7 | none | every invoice × 0.70 |
| Smooth-decline mimicry | M3, M7 | black-box | multiplier ramps linearly from 1 to f_end across the client's invoices (honest-looking slow decline), f_end chosen so the total falls exactly 30%; features and sequence rebuilt from the edited invoices |
| PGD | M7 | white-box | 10 steps on the sequence input, perturbation ≤ 0 per month, projected back onto the 30% budget; static branch held fixed |
| Index-consistent mimicry (stronger, outside the model) | M3 | black-box | smooth mimicry **plus** `new_index` rewritten to match: shows what happens if the attacker also controls readings (e.g. reader collusion) |
