# Literature Sources & Grounding Bibliography (`docs/sources.md`)

**Project:** BT4012 Fraud Analytics — The Tunisian Company of Electricity and Gas (STEG)  
**Workstream:** LLM, RAG & agents  
**Status:** Draft; bibliographic details checked against Crossref / publisher pages (see notes under each entry)  
**Last Updated:** November 2026  

---

## 1. Overview & Purpose

This document lists the **10 academic and industry references** behind the project's fraud analytics pipeline and RAG corpus:
1. Every section of the **NTL Typology Playbook** (`TYP-01` to `TYP-12`) cites at least one of these sources. The sources support each typology in general terms; detailed mechanism descriptions and most numeric thresholds in the playbook are the team's own and are labelled "project heuristic" there.
2. The **cost model and evaluation protocol** draw on the inspection-targeting literature.
3. The **Benford's Law features** and the **sequence model** follow methods from these works.
4. The **RAG architecture** follows the retrieval-augmented generation framework.

What a source says and how this project uses it are kept apart below: "Core contribution" describes the paper; "Project mapping" describes our use of it.

---

## 2. Curated Sources Catalogue

### [SRC-01] Deep Learning for Electricity-Theft Detection (Wide & Deep CNN)
* **Citation:** Zheng, Z., Yang, Y., Niu, X., Dai, H.-N., & Zhou, Y. (2018). *Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids.* **IEEE Transactions on Industrial Informatics**, 14(4), 1606–1615.
* **DOI:** [10.1109/TII.2017.2785963](https://doi.org/10.1109/TII.2017.2785963) (checked on Crossref)
* **Core Contribution:**
  * Observes that normal consumption is largely periodic while theft introduces non-periodic anomalies.
  * Proposes a wide component (memorising global features) combined with a deep CNN over consumption data arranged by week.
* **Project Mapping:**
  * Design reference for the sequence model over the 120-month `seq.npy` tensor (built by the deep-learning workstream).
  * Motivates trajectory features such as `yoy_drop_max` and `slope_last3y` (planned).
  * Playbook: `TYP-01`, `TYP-09`.

---

### [SRC-02] Survey of Non-Technical Loss Detection
* **Citation:** Glauner, P., Meira, J. A., Valtchev, P., State, R., & Bettinger, F. (2017). *The Challenge of Non-Technical Loss Detection Using Artificial Intelligence: A Survey.* **International Journal of Computational Intelligence Systems**, 10(1), 760–775. (Preprint: arXiv:1606.00626.)
* **DOI:** [10.2991/ijcis.2017.10.1.51](https://doi.org/10.2991/ijcis.2017.10.1.51) (checked on Crossref; pages from the Atlantis Press record)
* **Core Contribution:**
  * Lists causes of non-technical loss (NTL): meter tampering, bypassing meters by rigging lines from the power source, false readings arranged by bribing meter readers, faulty or broken meters, unmetered supply, and errors in meter reading, data processing and billing.
  * Reports estimates of NTL of up to 40% of electricity distributed in countries such as Brazil, India, Malaysia and Lebanon.
  * Surveys expert-system and machine-learning detection methods and open challenges, including class imbalance and biased inspection labels.
* **Project Mapping:**
  * Problem framing and loss context.
  * Playbook: `TYP-05`, `TYP-06`, `TYP-07`, `TYP-08`, `TYP-12`.

---

### [SRC-03] Improving Inspection Accuracy in a Power Utility
* **Citation:** Guerrero, J. I., Monedero, I., Biscarri, F., Biscarri, J., Millán, R., & León, C. (2018). *Non-Technical Losses Reduction by Improving the Inspections Accuracy in a Power Utility.* **IEEE Transactions on Power Systems**, 33(2), 1209–1218.
* **DOI:** [10.1109/TPWRS.2017.2721435](https://doi.org/10.1109/TPWRS.2017.2721435) (checked on Crossref)
* **Core Contribution:**
  * Addresses the practical problem that utilities can only inspect a limited number of customers, and aims to raise the share of inspections that find an NTL.
* **Project Mapping:**
  * Motivates evaluating models by precision and recall within a fixed inspection budget (`Precision@k%`, `Recall@k%`, $k \in \{1\%, 5\%, 10\%\}$) as well as `PR-AUC`.
  * The net-recovery formula used in the cost model, $\text{Net}(k) = V \times \text{TP}(k) - C \times k$, is this project's own simplification; it is not taken from this paper.

---

### [SRC-04] World Bank Background Paper on Power-Sector Losses
* **Citation:** Antmann, P. (2009). *Reducing Technical and Non-Technical Losses in the Power Sector.* Background paper for the World Bank Group Energy Sector Strategy. The World Bank, Washington, DC.
* **Stable Link:** [World Bank Open Knowledge Repository](https://openknowledge.worldbank.org/entities/publication/70bb9d7e-1249-58aa-bc54-ee9f2cfb66cf/full) ([PDF](https://documents1.worldbank.org/curated/en/829751468326689826/pdf/926390WP0Box3800in0the0power0sector.pdf))
* **Core Contribution:**
  * A policy and strategy paper, not an engineering or meter-audit manual. It defines technical and non-technical losses, reviews regional experience (Latin America, East and South Asia, former Soviet Union, Sub-Saharan Africa) and case studies (India, El Salvador, Chile), and sets out a strategy for loss reduction.
  * Names illegal connections to the grid and tampering with or bypassing meters as classic forms of theft; notes that theft by large consumers usually involves collusion with meter readers and that corruption is likely in disconnection operations; discusses automated meter reading and AMI as remedies.
* **Project Mapping:**
  * Playbook: general grounding for `TYP-01`, `TYP-04`, `TYP-05`, `TYP-12`. The paper does not describe magnet, neutral, register-rollback or CT/PT tampering techniques, so it is not cited for those.

---

### [SRC-05] Benford's Law for Electricity-Theft Detection
* **Citation:** Wei, L., Sundararajan, A., Sarwat, A. I., Biswas, S., & Ibrahim, E. (2017). *A Distributed Intelligent Framework for Electricity Theft Detection Using Benford's Law and Stackelberg Game.* **2017 Resilience Week (RWS)**, Wilmington, DE, USA, pp. 5–11.
* **DOI:** [10.1109/RWEEK.2017.8088640](https://doi.org/10.1109/RWEEK.2017.8088640) (checked on Crossref)
* **Core Contribution:**
  * Uses conformity of consumption data with Benford's Law (leading digits, $P(d=1) \approx 30.1\%$) to flag likely theft, combined with a game-theoretic model of utility and thief behaviour.
* **Project Mapping:**
  * Justifies the planned `benford_mad` feature and the `benford_check` agent tool.
  * Playbook: `TYP-04`, `TYP-08`.

---

### [SRC-06] Multi-Sensor Energy-Theft Detection in AMI
* **Citation:** McLaughlin, S., Holbert, B., Fawaz, A., Berthier, R., & Zonouz, S. (2013). *A Multi-Sensor Energy Theft Detection Framework for Advanced Metering Infrastructures.* **IEEE Journal on Selected Areas in Communications**, 31(7), 1319–1330.
* **DOI:** [10.1109/JSAC.2013.130714](https://doi.org/10.1109/JSAC.2013.130714) (checked on Crossref)
* **Core Contribution:**
  * Proposes a theft detector for advanced metering infrastructure that combines several information sources (including physical tamper evidence and consumption data) instead of relying on one.
* **Project Mapping:**
  * Supports combining data-quality evidence with consumption patterns.
  * The index-validity rules used in the pipeline (`new_index - old_index` times `counter_coefficient` should equal billed consumption; `flag_index_backwards`, `flag_index_rollover`, `flag_recon_mismatch`, `recon_gap` in `src/data/clean.py`) are **this project's own logic**, not taken from the paper.
  * Playbook: `TYP-02`, `TYP-03`, `TYP-10`, `TYP-11` (supporting role only; the paper does not analyse those physical mechanisms).

---

### [SRC-07] Benford's Law in Forensic Accounting (MAD Test)
* **Citation:** Nigrini, M. J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection.* **John Wiley & Sons**, Hoboken, NJ. ISBN 978-1-118-15285-0. DOI [10.1002/9781119203094](https://doi.org/10.1002/9781119203094).
* **Core Contribution:**
  * Defines the mean absolute deviation (MAD) test for digit analysis, with conformity bands for the **first-digit** test:
    * $\text{MAD} \le 0.006$: close conformity
    * $0.006 < \text{MAD} \le 0.012$: acceptable conformity
    * $0.012 < \text{MAD} \le 0.015$: marginally acceptable conformity
    * $\text{MAD} > 0.015$: nonconformity
  * The bands are intended for reasonably large data sets; small samples give noisy MAD values.
* **Project Mapping:**
  * The project uses **`benford_mad > 0.015`** (first-digit nonconformity) as its single threshold, in the playbook and in SOP v2025 §5.1.
  * `round_reading_share > 0.40` is a **project heuristic**, not from this book: if last digits were uniform, about 0.20 of readings would end in 0 or 5, and the threshold is twice that.
  * Playbook: `TYP-04`, `TYP-08`.

---

### [SRC-08] Retrieval-Augmented Generation
* **Citation:** Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* **Advances in Neural Information Processing Systems 33 (NeurIPS 2020)**, 9459–9474.
* **Stable Link:** [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)
* **Core Contribution:**
  * Combines a pre-trained generator (parametric memory) with a retrieved document index (non-parametric memory), improving factuality on knowledge-intensive tasks and allowing knowledge to be updated without retraining.
* **Project Mapping:**
  * Basis for the RAG design in `docs/llm_stack.md`: Inspector Briefs are generated from retrieved chunks rather than from the LLM alone.
  * Not cited in the playbook.

---

### [SRC-09] Consumption-Pattern-Based Theft Detection in AMI
* **Citation:** Jokar, P., Arianpoo, N., & Leung, V. C. M. (2016). *Electricity Theft Detection in AMI Using Customers' Consumption Patterns.* **IEEE Transactions on Smart Grid**, 7(1), 216–226.
* **DOI:** [10.1109/TSG.2015.2425222](https://doi.org/10.1109/TSG.2015.2425222) (checked on Crossref)
* **Core Contribution:**
  * Detects theft from abnormal changes in customers' consumption patterns, using distribution-transformer meters to narrow the search area; trains on simulated malicious patterns (for example readings scaled down, or set to zero for some periods).
* **Project Mapping:**
  * Playbook: `TYP-02`, `TYP-03`, `TYP-09`.

---

### [SRC-10] Overview of Electricity Theft
* **Citation:** Depuru, S. S. S. R., Wang, L., & Devabhaktuni, V. (2011). *Electricity theft: Overview, issues, prevention and a smart meter based approach to control theft.* **Energy Policy**, 39(2), 1007–1015.
* **DOI:** [10.1016/j.enpol.2010.11.037](https://doi.org/10.1016/j.enpol.2010.11.037) (checked on Crossref)
* **Core Contribution:**
  * Reviews electricity theft as a major problem in developing countries, its causes and impact, and proposes a smart-meter-based architecture to control it, with a cost-benefit analysis for India.
* **Project Mapping:**
  * Playbook: general background for `TYP-02`, `TYP-07`, `TYP-10`, `TYP-11`. The full text has not been checked page by page, so the playbook cites it only for a general overview.

---

## 3. Typology Cross-Reference Matrix

Generated from the citations in `data/corpus/source_docs/playbook.md`. Keep the two in sync.

| Typology ID | Mechanism | Sources cited in the playbook | Key features (planned features marked) |
| :--- | :--- | :--- | :--- |
| **`TYP-01`** | Meter Bypass / Shunt | `[SRC-01]`, `[SRC-04]` | `cons_pm_mean` collapse; `yoy_drop_max`, `zero_run_max` (planned); `share_level_2_plus` |
| **`TYP-02`** | Magnet Slowdown | `[SRC-06]`, `[SRC-09]`, `[SRC-10]` | Lower `cons_pm_median` without zeros; low `cons_pm_std`; `robust_z_max` (planned) |
| **`TYP-03`** | Index Rollback | `[SRC-06]`, `[SRC-09]` | `flag_index_backwards` without `flag_index_rollover`; `flag_recon_mismatch`, `recon_gap` |
| **`TYP-04`** | Meter-Reader Collusion | `[SRC-04]`, `[SRC-05]`, `[SRC-07]` | `benford_mad > 0.015`, `round_reading_share > 0.40` (both planned) |
| **`TYP-05`** | Illegal Reconnection | `[SRC-02]`, `[SRC-04]` | Consumption under `counter_statue 1` (assumed cut-off); `n_statue_changes` (planned) |
| **`TYP-06`** | Jammed / Defective Meter | `[SRC-02]` | `cons == 0` runs; `counter_statue` 3/4 (assumed); `flag_bad_months_number`, `has_missing_months` |
| **`TYP-07`** | Tariff Misclassification | `[SRC-02]`, `[SRC-10]` | High `cons_pm_mean` within `client_catg 11`; `peer_resid_mean` (planned) |
| **`TYP-08`** | Estimated Reading Abuse | `[SRC-02]`, `[SRC-05]`, `[SRC-07]` | Runs of `reading_remarque 8` (assumed estimated); low `cons_pm_std`; `benford_mad` (planned) |
| **`TYP-09`** | Catch-Up Billing | `[SRC-01]`, `[SRC-09]` | `sudden_spike_share` (planned); `cons_pm_max / (cons_pm_min + 1)`; `recon_gap` |
| **`TYP-10`** | Neutral Disconnection / Inversion | `[SRC-06]`, `[SRC-10]` | ELEC zeros while `cons_pm_mean_gaz` continues; non-rollover backwards readings |
| **`TYP-11`** | CT/PT Tampering | `[SRC-06]`, `[SRC-10]` | `counter_coefficient > 1` accounts; step drop in `cons_pm_mean` |
| **`TYP-12`** | Service Drop Tapping | `[SRC-02]`, `[SRC-04]` | Near-zero `cons_pm_mean`, long `tenure_years`; `nbr_fraud_rate_k10` (planned) |
