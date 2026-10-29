# STEG Precedent Fraud Case Cards Specification (`data/corpus/case_cards_spec.md`)

**Document ID:** `DOC-04-CASE-CARDS-SPEC`  
**Author / Workstream:** Role D (LLM, RAG & Agent Lead)  
**Corpus Layer:** Empirical Case Precedents & Analogous Reasoning  
**Status:** Outline & Generator Architecture (Week 1 Milestone)  

---

## 1. Overview & Non-Leakage Contract

To aid human inspectors, the RAG engine retrieves historical precedent dossiers of confirmed fraud cases sharing similar consumption profiles, geographic regions, or meter anomalies.

* **Strict Non-Leakage Rule (Section 5.3 & 14.2 of Project Plan):**  
  Precedent case cards are sampled **exclusively from confirmed positive fraud clients in the 80% training split (`target == 1` and `split == 'train'`)**. Precedents are strictly **never** drawn from held-out test clients.
* **Volume:** 300 representative fraud cases covering diverse client categories (`client_catg`), regions, and tamper typologies.
* **Storage Format:** Stored as line-delimited JSON (`data/corpus/case_cards.jsonl`) for programmatic indexing.

---

## 2. Case Card Data Schema

Each precedent case card is structured under the following Pydantic / JSON schema:

```json
{
  "id": "CASE-train_Client_10023",
  "client_id": "train_Client_10023",
  "split": "train",
  "client_catg": 11,
  "district": 60,
  "region": 101,
  "counter_type": "ELEC",
  "tenure_years": 8.4,
  "invoice_count": 32,
  "primary_tamper_signature": "Severe year-on-year consumption drop (82%) accompanied by negative meter index readings.",
  "key_metrics": {
    "cons_pm_drop_pct": 0.82,
    "negative_delta_count": 3,
    "reconciliation_mismatches": 5,
    "predominant_remarque": 9
  },
  "on_site_findings": "Field inspection confirmed physical shunt across terminal phase pins 1 and 2, diverting 85% of load current around the induction measuring coil.",
  "recovered_revenue_tnd": 2450.0,
  "outcome_status": "Confirmed Fraud — Meter Seized & Penalties Assessed"
}
```

---

## 3. Template for Text Search Embedding

When encoded into `data/corpus/chunks.jsonl` for dense and sparse retrieval, the case is formatted into this natural language chunk:

```text
[CASE-train_Client_10023] Region 101, District 60 | Category: Residential (11) | Meter: ELEC
Tenure: 8.4 years | Total Invoices: 32 | Split: train
Observed Behaviour: Severe 82% drop in monthly consumption with 3 negative index anomalies.
On-Site Findings: Field inspection verified direct phase shunt bypassing the current coil.
Outcome: Confirmed fraud; 2,450 TND recovered under SOP §3.1 priority escalation.
```

---

## 4. Sampling & Generation Pipeline (Built in Week 2 / Week 3)
1. Read `data/splits/train_ids.csv` to ensure 100% adherence to training clients.
2. Select 300 confirmed positive cases (`target == 1`) stratified across:
   - High YoY drop clients (`TYP-01`, `TYP-02` analogues)
   - Negative index progression clients (`TYP-03` analogues)
   - Disconnected / unauthorized usage clients (`TYP-05` analogues)
   - High Benford deviation / estimated reading clients (`TYP-04`, `TYP-08` analogues)
3. Populate `data/corpus/case_cards.jsonl` using the templater script `src/llm/build_case_cards.py`.

