# Synthetic Inspection SOP Specification (`data/corpus/specs/sop_spec.md`)

**Document ID:** `DOC-03-SOP-SPEC`  
**Workstream:** LLM, RAG & agents  
**Corpus Layer:** Synthetic operational dispatch rules  
**Status:** Draft specification  

---

## 1. Overview & Experimental Role

The two SOP documents (`source_docs/sop_v2023.md`, `source_docs/sop_v2025.md`) are **synthetic: written for coursework, not STEG policy**. STEG has not published inspection rules. The clauses exist so the RAG pipeline can be tested on a realistic problem: picking the rule version that was in force on a given date.

* **Lecture concept demonstrated: temporal metadata filtering.**  
  Rules change over time. An alert dated in 2024 must be evaluated under `v2023`; an alert dated on or after 1 January 2025 must be evaluated under `v2025`. Unfiltered semantic search often retrieves the superseded version, with outdated thresholds or escalation tiers.
* **Validity windows are half-open, `[valid_from, valid_to)`.** A clause applies when `valid_from <= alert_date < valid_to`; `valid_to: None` means open-ended. So `v2023` covers 2023-01-01 up to but not including 2025-01-01, and `v2025` starts on 2025-01-01.
* **`alert_date`** is the date the risk list is scored and the brief is written. It is not an invoice date: the invoice history ends in 2019, before either SOP version starts. Experiments set `alert_date` explicitly.
* **Metadata schema:** every clause carries
  ```yaml
  doc: sop
  jurisdiction: TN
  version: v2023 | v2025
  valid_from: 2023-01-01   # v2023; 2025-01-01 for v2025
  valid_to: 2025-01-01     # v2023 (exclusive); None for v2025
  synthetic: true
  ```

---

## 2. Clause Taxonomy (13 Clauses in Each Version)

Feature names in backticks are pipeline columns (all are built: F4, F7 and F8 in `docs/feature_list.md`). Code meanings are assumptions (see the code book).

| Clause Number | Focus Area & Suspicion Trigger | `v2023` Thresholds (`[2023-01-01, 2025-01-01)`) | `v2025` Thresholds (`[2025-01-01, open)`) |
| :--- | :--- | :--- | :--- |
| **§1.1** | General Inspection Dispatch Priority | Top 10% risk rank dispatches within 45 days. | Top 5% risk rank dispatches within 14 days (tighter budget). |
| **§1.2** | Dual-Meter Premises | Visual check of the second meter at the same address. | Mandatory audit of both meters within 48 hours (`has_elec == 1` and `has_gaz == 1`). |
| **§2.1** | Sudden Year-on-Year Drop (`yoy_drop_max`) | $\ge 70\%$ drop across 3 consecutive cycles. | $\ge 50\%$ drop across 2 consecutive cycles. |
| **§2.2** | Persistent Zero Consumption (`zero_run_max`) | 4 or more consecutive zero-consumption bills. | 2 or more consecutive zero-consumption bills. |
| **§2.3** | Index Delta vs Declared Consumption (`recon_gap`) | Discrepancy $> 200$ kWh on an invoice (`recon_gap > 200`). | Discrepancy $> 50$ kWh on an invoice (`recon_gap > 50`); smaller nonzero gaps are logged only. |
| **§3.1** | Backwards Index Reading (`flag_index_backwards` and not `flag_index_rollover`) | Supervisor sign-off before field visit. | Immediate meter seizure and laboratory examination. |
| **§3.2** | `counter_statue == 5` (assumed: suspected anomaly) | Site inspection within 30 calendar days. | Priority inspection within 7 days. |
| **§3.3** | Consumption under `counter_statue == 1` (assumed: cut-off), confirmed by service-order history | Level 1 administrative warning letter. | Service termination with police escort, and referral for back-billing. |
| **§4.1** | Consecutive `reading_remarque == 8` (assumed: estimated) | 5 or more consecutive invoices. | 3 or more consecutive invoices. |
| **§4.2** | Consecutive `reading_remarque == 9` (assumed: not read; doubtful) | 4 consecutive invoices trigger an access notice. | 2 consecutive invoices trigger an access notice. |
| **§5.1** | Benford's Law Digit Screening (`benford_mad`, `round_reading_share`) | Advisory only; not grounds for action. | `benford_mad > 0.015` (Nigrini first-digit nonconformity) and `round_reading_share > 0.40` (project heuristic) trigger a reader-route audit. |
| **§5.2** | Consumption Far Above Category Peers | Annual consumption $> 3\times$ the regional median for `client_catg 11`. | Consumption $> 2\times$ the regional median for `client_catg 11` and `peer_resid_mean > 2.0`. |
| **§6.1** | Required On-Site Evidence | Two photographs (dial face and terminal seal). | Dual clamp-ammeter reading, Gaussmeter reading, four geo-tagged photos, completed brief checklist. |

---

## 3. Standard SOP Chunk Template

```markdown
### [SOP-vYYYY-X.X] Section Title: Clause Description
* **Metadata:** `doc: sop` | `version: vYYYY` | `valid_from: YYYY-MM-DD` | `valid_to: YYYY-MM-DD` | `jurisdiction: TN` | `synthetic: true`
* **Clause Text:** Clause language giving the threshold conditions and required action.
* **Mandatory Action:** Tier 1 (Notice) | Tier 2 (Priority Inspection) | Tier 3 (Immediate Seizure & Police Escort).
* **Evidence Required:** Records that must be attached.
```
