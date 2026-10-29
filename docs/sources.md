# Literature Sources & Grounding Bibliography (`docs/sources.md`)

**Project:** BT4012 Fraud Analytics — The Tunisian Company of Electricity and Gas (STEG)  
**Author / Owner:** Role D (LLM, RAG & Agent Lead)  
**Status:** Approved & Frozen for Milestone 1  
**Last Updated:** October 2026  

---

## 1. Overview & Purpose

This document compiles the **8 authoritative academic and industry references** underpinning Group 19's fraud analytics pipeline. In accordance with the project plan:
1. Every section in the **NTL Typology Playbook** (`TYP-01` through `TYP-12`) must directly cite the physical tampering mechanisms and data signatures identified in these sources.
2. The **Cost Model & Evaluation Protocol** (Roles B & D) cites empirical findings on field inspection costs and revenue recovery.
3. The **Benford's Law Features** (Role A, Feature Group `F7`) and **1D-CNN Sequence Architecture** (Role C, Model `M7`) directly implement methodology from these peer-reviewed works.
4. The **RAG Architecture** (Role D) implements the verified non-parametric grounding framework established in forensic AI literature.

---

## 2. Curated Sources Catalogue

### [SRC-01] Deep Learning for Non-Technical Loss (1D-CNN Benchmark)
* **Citation:** Zheng, Z., Yang, Y., Niu, X., Dai, H. N., & Zhou, Y. (2018). *Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids.* **IEEE Transactions on Industrial Informatics**, 14(4), 1606–1615.
* **DOI:** [10.1109/TII.2017.2785963](https://doi.org/10.1109/TII.2017.2785963)
* **Core Contribution:**
  * Demonstrates that electricity theft is characterized by non-periodic anomalies overlaid on periodic seasonal baselines.
  * Proposes a dual-input architecture combining a Deep CNN (extracting temporal features from 2D/1D consumption sequences) with a Wide component (memorizing global categorical and statistical features).
* **Physical Tampering Insights:**
  * Analyzes how physical tampering (bypasses, slowdowns) causes structural discontinuities in temporal load curves rather than simple random noise.
* **Direct Project Mapping:**
  * **Model M7:** Architecture design for the dual-input 1D-CNN on the 120-month `seq.npy` tensor.
  * **Feature Group F4 (Trajectory):** Informs `yoy_drop_max` and `slope_last3y` trend features.
  * **Playbook:** Cites temporal consumption drop signatures in `TYP-01` (Bypass) and `TYP-02` (Magnet).

---

### [SRC-02] Comprehensive NTL Taxonomy & Machine Learning Survey
* **Citation:** Glauner, P., Meira, J. A., Valtchev, P., State, R., & Bettinger, F. (2017). *The Challenge of Non-Technical Loss Detection Using Artificial Intelligence: A Survey.* **International Journal of Computational Intelligence Systems**, 10(1), 760–775. (Preprint: arXiv:1606.00626).
* **DOI:** [10.2991/ijcis.10.1.51](https://doi.org/10.2991/ijcis.10.1.51)
* **Core Contribution:**
  * Establishes the formal definition and taxonomy of Non-Technical Losses (NTLs): distinguishing between **physical electricity theft**, **meter defects/breakdowns**, and **administrative/billing errors**.
  * Documents that NTLs account for up to 40% of generated electricity in developing utilities, severely destabilizing grid financing.
  * Surveys the shift from static boolean rule-based expert systems to automated machine learning (SVM, tree ensembles, neural nets).
* **Key Challenges Highlighted:**
  * Extreme class imbalance (typically <5% confirmed positive rates).
  * Noisy negatives (uninspected clients are merely "unlabeled", not confirmed honest).
* **Direct Project Mapping:**
  * **Report Section 1 & Proposal:** Problem formulation and macro-economic loss context for STEG.
  * **Model Escalation (M1–M10):** Justification for combining supervised models with unsupervised anomaly detectors (Autoencoder / Isolation Forest) to rescue unlabeled positives.
  * **Playbook:** General grounding for `TYP-06` (Defective Meters) and `TYP-07` (Billing Discrepancies).

---

### [SRC-03] Operational Inspection Budgeting & Cost-Sensitive Allocation
* **Citation:** Guerrero, A. E., León, C., Monedero, Í., Biscarri, F., & Biscarri, J. (2014). *Non-Technical Losses Reduction by Improving the Inspections Accuracy in a Power Utility.* **IEEE Transactions on Power Systems**, 29(3), 1209–1218.
* **DOI:** [10.1109/TPWRS.2013.2293140](https://doi.org/10.1109/TPWRS.2013.2293140)
* **Core Contribution:**
  * Proves that raw classification metrics (e.g. Accuracy or default 0.5-threshold F1-score) are operationally meaningless for utility management due to finite inspection capacity.
  * Formulates inspection dispatching as an operational ranking optimization problem under a constrained monthly inspection budget $k$.
  * Demonstrates that false positives directly consume labor and transportation overhead ($C$), while true positives yield recovered energy revenue and regulatory penalties ($V$).
* **Direct Project Mapping:**
  * **Evaluation Protocol (Section 5.4 & 5.5):** Foundation of our primary evaluation metrics (`PR-AUC`, `Precision@k%`, `Recall@k%` for $k \in \{1\%, 5\%, 10\%\}$).
  * **Cost Curve:** Mathematical derivation of expected net recovery: $\text{Net}(k) = V \times \text{TP}(k) - C \times k$.

---

### [SRC-04] World Bank Technical Guidelines on Physical Meter Tampering
* **Citation:** World Bank / ESMAP. (2009). *Reducing Technical and Non-Technical Losses in the Power Sector: Best Practice and International Experience.* **Energy Sector Management Assistance Program (ESMAP) Technical Paper 040/09**, The World Bank Group, Washington, DC.
* **Stable Link:** [World Bank Document Repository](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/856321468163777590/)
* **Core Contribution:**
  * Authoritative engineering audit manual detailing mechanical, magnetic, and electrical tampering mechanisms across induction (electromechanical) and solid-state meters.
  * Categorizes primary on-site attack vectors:
    1. **Direct Bridging / Shunts:** Low-resistance copper jumper bypassing current coils.
    2. **Magnetic Tampering:** Applying external neodymium magnets to induce eddy-current drag or saturate current transformers (CTs).
    3. **Neutral Disconnection / Inversion:** Reversing phase/neutral terminal wiring to halt current measurement.
    4. **Register / Gear Rollback:** Physical alteration of internal number drums.
    5. **Meter-Reader Collusion:** Systematic manual under-reporting in exchange for bribes.
* **Direct Project Mapping:**
  * **NTL Typology Playbook:** Primary technical reference for `TYP-01` (Bypass), `TYP-02` (Magnet), `TYP-03` (Index Rollback), `TYP-04` (Collusion), and `TYP-05` (Illegal Reconnection).
  * **Inspector Brief On-Site Checklists:** Supplies the exact physical inspection procedures (e.g., terminal block seal check, clamp ammeter line comparison, magnet discoloration inspection).

---

### [SRC-05] Forensic Benford's Law for Smart Grid Anomaly Screening
* **Citation:** Wei, L., Sundararajan, A., Sarwat, A. I., Biswas, S., & Ibrahim, E. (2018). *A Distributed Intelligent Framework for Electricity Theft Detection Using Benford’s Law and Stackelberg Game.* **Proceedings of IEEE Resilience Week (RWS)**, Denver, CO, pp. 177–183.
* **DOI:** [10.1109/RWS.2018.8473489](https://doi.org/10.1109/RWS.2018.8473489)
* **Core Contribution:**
  * Establishes that unmanipulated electricity consumption data across customer populations naturally obeys Benford's Law (logarithmic distribution of leading digits, with $P(d=1) \approx 30.1\%$).
  * Proves that fraudulent consumers manipulating readings or corrupt meter-readers fabricating estimates inject unnatural uniformity or clustering, creating statistically significant deviations from Benford's distribution.
  * Demonstrates that goodness-of-fit screening (MAD, Chi-square) operates with extremely low computational complexity, making it an ideal first-stage anomaly screen.
* **Direct Project Mapping:**
  * **Feature Group F7:** Theoretical justification for computing client-level `benford_mad`.
  * **Agent Tool (`benford_check`):** Informs the forensic check tool in the multi-agent investigation workflow.
  * **Playbook:** Supplies diagnostic evidence for `TYP-04` (Collusion) and `TYP-08` (Estimated Reading Abuse).

---

### [SRC-06] Multi-Sensor Physical Meter Tampering & Data Validity
* **Citation:** McLaughlin, S., Holbert, B., Fawaz, A., Berthier, R., & Zonouz, S. (2013). *A Multi-Sensor Energy Theft Detection Framework for Advanced Metering Infrastructure.* **IEEE Journal on Selected Areas in Communications**, 31(7), 1319–1330.
* **DOI:** [10.1109/JSAC.2013.130718](https://doi.org/10.1109/JSAC.2013.130718)
* **Core Contribution:**
  * Analyzes data anomalies resulting from physical attacks on electronic meters, focusing on register-inversion, index deltas, and multi-sensor reconciliation gaps.
  * Formulates data validity invariants: in an honest meter, $\text{new\_index} - \text{old\_index}$ must equal declared cumulative consumption, and negative deltas ($\text{new\_index} < \text{old\_index}$) represent physically impossible events caused by rollback or hardware resets.
* **Direct Project Mapping:**
  * **Feature Group F6 (Data Validity):** Direct foundation for `n_index_backwards` ($\text{new} < \text{old}$), `n_recon_mismatch`, and `recon_gap_mean`.
  * **Playbook:** Grounding for `TYP-03` (Index Rollback & Impossible Readings).

---

### [SRC-07] Forensic Accounting & Auditing Methodology (Benford MAD)
* **Citation:** Nigrini, M. J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection.* **John Wiley & Sons**, Hoboken, NJ. ISBN: 978-1-118-15285-0.
* **Core Contribution:**
  * The definitive treatise on statistical forensic auditing.
  * Formalizes the **Mean Absolute Deviation (MAD)** metric for digit analysis, establishing standard conformity thresholds:
    * $\text{MAD} \le 0.006$: Close conformity
    * $0.006 < \text{MAD} \le 0.012$: Acceptable conformity
    * $0.012 < \text{MAD} \le 0.015$: Marginally acceptable
    * $\text{MAD} > 0.015$: Nonconformity (indicative of manual manipulation or estimation)
  * Analyzes digit rounding heuristics: fraudsters fabricating bills disproportionately choose numbers ending in $0$ or $5$.
* **Direct Project Mapping:**
  * **Feature Group F7:** Supplies the mathematical formula and threshold scale for `benford_mad` and `round_reading_share` (honest $\approx 0.20$).
  * **Playbook:** Analytical foundation for auditing suspicious flat-rate or rounded usage in `TYP-08`.

---

### [SRC-08] Retrieval-Augmented Generation for Verifiable Grounding
* **Citation:** Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W. T., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* **Advances in Neural Information Processing Systems (NeurIPS 2020)**, 33, 9459–9474.
* **Stable Link:** [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)
* **Core Contribution:**
  * Proposes the seminal RAG framework combining pre-trained parametric memory (LLM) with non-parametric retrieval memory (dense vector index).
  * Demonstrates that grounding generation on explicitly retrieved evidence documents dramatically cuts factual hallucinations, provides source provenance, and allows real-time updates to domain knowledge without retraining.
* **Direct Project Mapping:**
  * **Subsystem Architecture (`docs/llm_stack.md`):** Theoretical foundation of our RAG pipeline.
  * **Investigation Layer:** Direct justification for generating Inspector Briefs using retrieved policy chunks and precedent case cards rather than raw LLM generation.
  * **Faithfulness Evaluation:** Framework for measuring citation existence and number accuracy against retrieved non-parametric contexts.

---

## 3. Typology Cross-Reference Matrix

The table below indicates which sources are cited in each section of the upcoming **NTL Typology Playbook**:

| Typology ID | Mechanism Description | Primary Sources | Key Invariants / Features |
| :--- | :--- | :--- | :--- |
| **`TYP-01`** | Meter Bypass / Shunt Bridging | `[SRC-01]`, `[SRC-04]` | Sudden permanent consumption drop; zero-run bills (`zero_run_max`); clamp-ammeter differential. |
| **`TYP-02`** | Magnet Slowdown / CT Saturation | `[SRC-01]`, `[SRC-04]`, `[SRC-06]` | Uniform percentage drop (40–70%); seasonal pattern preserved but scaled down; missing zero runs. |
| **`TYP-03`** | Index Rollback & Gear Inversion | `[SRC-04]`, `[SRC-06]` | Impossible negative deltas (`n_index_backwards`); reconciliation gap between index delta and billed units. |
| **`TYP-04`** | Meter Reader Collusion / Falsification | `[SRC-04]`, `[SRC-05]`, `[SRC-07]` | Benford nonconformity (`benford_mad > 0.015`); rounded readings (`round_reading_share`); catch-up spikes. |
| **`TYP-05`** | Illegal Reconnection after Cutoff | `[SRC-04]` | Usage while account status is suspended/cut; bypassed utility security seals; zero billed units. |
| **`TYP-06`** | Defective / Jammed Meter | `[SRC-02]`, `[SRC-06]` | Mechanical stop; consumption flatlines to absolute zero across consecutive quarters; status code transitions. |
| **`TYP-07`** | Tariff Misclassification | `[SRC-02]`, `[SRC-04]` | Discrepancy between declared category (`client_catg`) and usage magnitude; tariff rate exploitation. |
| **`TYP-08`** | Estimated Reading Abuse | `[SRC-05]`, `[SRC-07]` | Repetitive remark codes (`reading_remarque`); low variance in consecutive bills; high Benford MAD. |
