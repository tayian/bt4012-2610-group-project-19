# Precedent Case Cards Specification (`data/corpus/specs/case_cards_spec.md`)

**Document ID:** `DOC-04-CASE-CARDS-SPEC`  
**Workstream:** LLM, RAG & agents  
**Corpus Layer:** Case precedents for analogous reasoning  
**Status:** Draft specification (generator not yet written)  

---

## 1. Overview & Non-Leakage Contract

To help inspectors, the RAG engine retrieves confirmed fraud clients from the training data whose consumption profile, region or data-quality flags resemble the client under review.

* **Non-leakage rule:**  
  Case cards are sampled **only from fraud clients in the 80% training split (`target == 1` and client ID in `data/splits/train_ids.csv`)**. They are never drawn from held-out test clients.
* **Observed data only:** The dataset records whether a client was found fraudulent (`target`) and nothing else about the outcome. There are no inspection reports, on-site findings, tamper methods, recovered amounts or penalties. A card must therefore contain **only** observed fields and features computed from them. It must not contain invented outcomes such as on-site findings, recovered revenue, a "confirmed tamper method" or a sanction status. Any typology a card is linked to is a pattern match made by the generator, not a finding.
* **Volume:** 300 cards, stratified across `client_catg`, region and the consumption patterns listed in §4.
* **Storage format:** line-delimited JSON (`data/corpus/case_cards.jsonl`).

---

## 2. Case Card Data Schema

Each card follows this schema. The values below are a format example; the generator fills them from the data.

```json
{
  "id": "CASE-train_Client_10023",
  "client_id": "train_Client_10023",
  "split": "train",
  "target": 1,
  "client_catg": "11",
  "district": 60,
  "region": "101",
  "has_elec": 1,
  "has_gaz": 0,
  "tenure_years": 8.4,
  "n_invoices": 32,
  "first_invoice": "2009-03-12",
  "last_invoice": "2019-06-20",
  "consumption": {
    "cons_pm_mean": 312.5,
    "cons_pm_median": 290.0,
    "cons_pm_std": 140.2,
    "share_level_2_plus": 0.18,
    "trajectory_summary": "median cons_pm 410 in the first half of the history, 120 in the second half"
  },
  "data_quality": {
    "n_index_backwards": 0,
    "n_index_rollover": 1,
    "n_recon_mismatch": 2,
    "n_bad_months_number": 0,
    "has_missing_months": 1
  },
  "codes_seen": {
    "counter_statue": ["0", "5"],
    "reading_remarque": ["6", "8"],
    "tarif_type": ["11"]
  },
  "matched_typologies": ["TYP-01"]
}
```

Field sources:

| Card field | Source |
| :--- | :--- |
| `client_catg`, `district` (raw column `disrict`), `region`, `target` | `data/interim/client.parquet` |
| `has_elec`, `has_gaz`, `tenure_years`, `n_invoices` | `src/features/aggregate.py` / `src/features/behavioural.py` |
| `cons_pm_*`, `share_level_2_plus`, `has_missing_months` | `src/features/behavioural.py` |
| `n_index_backwards`, `n_index_rollover`, `n_recon_mismatch`, `n_bad_months_number` | per-client counts of `flag_index_backwards & ~flag_index_rollover`, `flag_index_rollover`, `flag_recon_mismatch`, `flag_bad_months_number` from `data/interim/invoice_clean.parquet` (computed by the generator) |
| `trajectory_summary` | computed by the generator from `cons_pm` by date, excluding `flag_partial_year` rows |
| `codes_seen` | distinct raw values on the client's invoices (meanings in the code book are assumptions) |
| `matched_typologies` | rule-based match of the card's features to playbook signatures (a hint for retrieval, not a finding) |

A meter rollover (index passing 99999 and restarting) is not a backwards reading; `src/data/clean.py` separates the two, and cards count them separately.

---

## 3. Template for Text Search Embedding

In `data/corpus/chunks.jsonl` each card becomes one text chunk:

```text
[CASE-train_Client_10023] Region 101, District 60 | client_catg 11 | Meters: ELEC
Tenure: 8.4 years | Invoices: 32 (2009-2019) | Split: train | target = 1 (fraud confirmed)
Consumption: mean cons_pm 312.5, median 290.0; median 410 in the first half of the history, 120 in the second half.
Data quality: 0 backwards readings, 1 meter rollover, 2 reconciliation mismatches, missing months present.
Codes seen: counter_statue 0, 5; reading_remarque 6, 8; tarif_type 11.
Pattern match (not a finding): TYP-01.
```

The card says nothing about how the fraud was committed or what was recovered, because the data do not record it.

---

## 4. Sampling & Generation Pipeline
1. Read `data/splits/train_ids.csv` and keep only training clients.
2. Select 300 fraud clients (`target == 1`) stratified across:
   - large drops in consumption over the history (`TYP-01`, `TYP-02` analogues)
   - backwards index readings that are not rollovers (`TYP-03` analogues)
   - invoices under `counter_statue` 1 with nonzero consumption (`TYP-05` analogues; the code meaning is assumed)
   - long runs of `reading_remarque` 8 (`TYP-08` analogues; the code meaning is assumed)
3. Write `data/corpus/case_cards.jsonl` with the planned generator script `src/llm/build_case_cards.py`.
