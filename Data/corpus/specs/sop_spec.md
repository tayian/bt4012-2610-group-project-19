# STEG Synthetic Inspection Standard Operating Procedures (SOP) Specification (`data/corpus/sop_spec.md`)

**Document ID:** `DOC-03-SOP-SPEC`  
**Author / Workstream:** Role D (LLM, RAG & Agent Lead)  
**Corpus Layer:** Regulatory & Operational Dispatch Rules  
**Status:** Outline & Clause Architecture (Week 1 Milestone)  

---

## 1. Overview & Experimental Role

The Inspection SOP defines the legally binding operational dispatch rules for STEG field inspectors.

* **Course Concept Demonstrated (Lecture 6 & Proposal Issue #5):** **Temporal Metadata Filtering.**  
  Regulations change over time. If an alert occurs in 2024, the agent must evaluate the customer under the 2023 SOP (`v2023`). If an alert occurs in late 2025 or 2026, the agent must evaluate under the updated 2025 SOP (`v2025`). Unfiltered semantic search frequently retrieves superseded policies with outdated penalty thresholds or incorrect escalation tiers.
* **Metadata Schema:** Every clause is filtered against the client's alert date:
  ```yaml
  jurisdiction: TN
  valid_from: 2023-01-01
  valid_to: 2025-01-01 # for v2023; None for v2025
  version: v2023 | v2025
  ```

---

## 2. Clause Taxonomy (12 Standard Clauses Across Both Versions)

| Clause Number | Focus Area & Suspicion Trigger | `v2023` Thresholds (`valid_to: 2025-01-01`) | `v2025` Thresholds (`valid_from: 2025-01-01`) |
| :--- | :--- | :--- | :--- |
| **§1.1** | General Inspection Dispatch Priority | Top 10% risk rank dispatches within 45 days. | Top 5% risk rank dispatches within 14 days (tightened budget). |
| **§2.1** | Sudden Year-on-Year Drop (`yoy_drop_max`) | $\ge 70\%$ drop across 3 consecutive cycles. | $\ge 50\%$ drop across 2 consecutive cycles. |
| **§2.2** | Persistent Zero Consumption (`zero_run_max`) | 4 or more consecutive zero-consumption bills. | 2 or more consecutive zero-consumption bills. |
| **§2.3** | Mismatch Between Index Delta & Declared Cons | Index delta discrepancy $> 200 \text{ kWh}$. | Any nonzero reconciliation discrepancy (`recon_mismatch >= 1`). |
| **§3.1** | Negative Index Progression (`is_negative_delta`) | Requires supervisor sign-off before field visit. | Immediate meter confiscation and forensic seal audit. |
| **§3.2** | Suspected Anomaly Flag (`counter_statue == 5`) | Site inspection within 30 calendar days. | High-priority emergency inspection within 7 days. |
| **§3.3** | Unauthorized Reconnection (`statue == 1` + usage) | Level 1 administrative penalty notice. | Immediate power cut, police escort, and penalty assessment. |
| **§4.1** | Prolonged Estimated Billing (`remarque == 8`) | 5 or more consecutive estimated invoices. | 3 or more consecutive estimated invoices. |
| **§4.2** | Repeated Inaccessible Meter (`remarque == 9`) | 4 consecutive inaccessible visits. | 2 consecutive inaccessible visits trigger registered notice. |
| **§5.1** | Benford's Law Digital Anomaly (`benford_mad`) | Not recognized as formal evidentiary basis. | `benford_mad > 0.05` triggers internal reader audit. |
| **§5.2** | Extreme Tariff Category Disparity | Total consumption $> 3\times$ district median. | Total consumption $> 2\times$ district median. |
| **§6.1** | Required On-Site Evidence Standards | Photographs of meter cabinet and physical seal. | Dual ammeter clamp reading + terminal block photos. |

---

## 3. Standard SOP Chunk Template

```markdown
### [SOP-vYYYY-X.X] Section Title: Clause Description
* **Metadata:** `doc: sop` | `version: vYYYY` | `valid_from: YYYY-MM-DD` | `valid_to: YYYY-MM-DD` | `jurisdiction: TN`
* **Rule Text:** Exact formal clause language outlining the threshold conditions and required administrative action.
* **Mandatory Operational Action:** Tier 1 (Notice) | Tier 2 (Priority Inspection) | Tier 3 (Immediate Confiscation & Police Escort).
```

