# Non-Technical Loss (NTL) Typology Playbook (`data/corpus/source_docs/playbook.md`)

**Document ID:** `DOC-01-PLAYBOOK`  
**Workstream:** LLM, RAG & agents  
**Corpus Layer:** Fraud typology reference  
**Status:** Draft (sections TYP-01 to TYP-12)  

---

## Reading this playbook

* **Feature names.** Names in backticks are either columns that exist now or features marked **(planned)**, which the pipeline does not compute yet. Existing columns: raw invoice fields; cleaned-invoice columns from `src/data/clean.py` (`cons`, `cons_pm`, `idx_diff`, `recon_gap`, `flag_index_backwards`, `flag_index_rollover`, `flag_recon_mismatch`, `flag_bad_months_number`, `flag_partial_year`); client features from `src/features/behavioural.py` (`cons_pm_mean/median/std/min/max/mad/skew/kurt`, `share_level_2_plus`, `n_invoices`, `invoices_per_year`, `months_number_mean/max`, `days_since_last_invoice_mean/max`, `has_missing_months`, with `_elec`/`_gaz` variants) and `src/features/aggregate.py` (`client_catg`, `district`, `region`, `tenure_years`, `has_elec`, `has_gaz`, `n_counters`).
* **Backwards readings vs rollovers.** `flag_index_backwards` marks `new_index < old_index`. Most such rows are meter rollovers (the register wraps past its maximum, e.g. 99999 back to 0), flagged by `flag_index_rollover`; only 12 of 2,264 backwards rows are not rollovers. A rollover is not evidence of tampering.
* **Code meanings are assumed.** STEG has not published code definitions; see the code book `[CODE-<field>-<value>]`.
* **Thresholds.** Numbers marked "project heuristic" are the team's choices, not taken from a cited source. Mechanism descriptions are general engineering background written by the team; cited sources support the typology in general terms, not every technical detail.

## Master Table of Contents & Typology Matrix

| Section ID | Typology Name | Applicable Utility | Primary Feature Signatures | Sources Cited |
| :--- | :--- | :--- | :--- | :--- |
| **`[TYP-01]`** | Direct Meter Bypass / Phase Shunt | ELEC, GAZ | `cons_pm_mean` collapse, `yoy_drop_max` (planned), `zero_run_max` (planned), `share_level_2_plus` | `[SRC-01]`, `[SRC-04]` |
| **`[TYP-02]`** | External Magnet Slowdown | ELEC, GAZ | Drop in `cons_pm_median` without zeros, low `cons_pm_std`, `robust_z_max` (planned) | `[SRC-06]`, `[SRC-09]`, `[SRC-10]` |
| **`[TYP-03]`** | Internal Register / Index Rollback | ELEC, GAZ | `flag_index_backwards` without `flag_index_rollover`, `flag_recon_mismatch`, `recon_gap` | `[SRC-06]`, `[SRC-09]` |
| **`[TYP-04]`** | Meter-Reader Collusion & Under-Reporting | ELEC, GAZ | `benford_mad > 0.015` (planned), `round_reading_share > 0.40` (planned), low `cons_pm_std` | `[SRC-04]`, `[SRC-05]`, `[SRC-07]` |
| **`[TYP-05]`** | Unauthorised Self-Reconnection | ELEC, GAZ | Consumption under `counter_statue 1` (assumed cut-off), `n_statue_changes` (planned) | `[SRC-02]`, `[SRC-04]` |
| **`[TYP-06]`** | Meter Jamming & Mechanical Breakdown | ELEC, GAZ | Zero `cons` while occupied, `counter_statue` 3/4 (assumed), `flag_bad_months_number` | `[SRC-02]` |
| **`[TYP-07]`** | Tariff Misclassification | ELEC | High `cons_pm_mean` in `client_catg 11` vs peers, `peer_resid_mean` (planned) | `[SRC-02]`, `[SRC-10]` |
| **`[TYP-08]`** | Estimated Reading Abuse | ELEC, GAZ | Runs of `reading_remarque 8` (assumed estimated), low `cons_pm_std`, catch-up `cons_pm_max` | `[SRC-02]`, `[SRC-05]`, `[SRC-07]` |
| **`[TYP-09]`** | Catch-Up Billing After Intermittent Tampering | ELEC, GAZ | `sudden_spike_share` (planned), high `cons_pm_max / (cons_pm_min + 1)`, `recon_gap` | `[SRC-01]`, `[SRC-09]` |
| **`[TYP-10]`** | Neutral Disconnection & Terminal Inversion | ELEC | Sudden zeros on ELEC while GAZ continues, `flag_index_backwards` (non-rollover) | `[SRC-06]`, `[SRC-10]` |
| **`[TYP-11]`** | Instrument Transformer (CT/PT) Tampering | ELEC | `counter_coefficient > 1` accounts, step drop in `cons_pm_mean` | `[SRC-06]`, `[SRC-10]` |
| **`[TYP-12]`** | Direct Service Drop Tapping (Hooking) | ELEC | Near-zero `cons_pm_mean` with long `tenure_years`, `nbr_fraud_rate_k10` (planned) | `[SRC-02]`, `[SRC-04]` |

---

## Detailed Fraud Typology Specifications

### [TYP-01] Direct Meter Bypass / Phase Shunt (Bridging)
* **Metadata:** `doc: playbook` | `section: meter_bypass` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  A low-resistance conductor or hidden jumper is fitted between the incoming and outgoing terminals of the meter's current coil, or across the service cable before it reaches the meter. Current divides in inverse proportion to resistance ($I_{\text{meter}} = I_{\text{total}} \cdot \frac{R_{\text{shunt}}}{R_{\text{coil}} + R_{\text{shunt}}}$), so most of the load current (70% to 95%, project heuristic) bypasses the coil. In gas installations, an illegal pipe with a valve is tapped upstream of the meter inlet, diverting gas around the measuring diaphragm.
* **2. Data Signature:**  
  * Lasting fall in monthly consumption: `cons_pm_mean` and `cons_pm_median` drop and stay low.
  * Large year-on-year drop: `yoy_drop_max > 0.65` (planned; project heuristic) and a negative recent trend `slope_last3y` (planned).
  * Runs of zero or near-zero bills: `zero_run_max` and `zero_share` (both planned); today visible as invoices with `cons == 0`.
  * Usage capped in the first tariff band: `share_level_2_plus` near zero.
* **3. On-Site Inspection Protocol:**  
  1. Inspect terminal-cover seals for cutting, re-crimping or non-matching serial numbers.
  2. Remove the terminal cover and look for copper bridges, hidden wires behind the board, or scoring on terminal screws.
  3. Clamp-ammeter check: measure current on the cable entering the meter ($I_{\text{in}}$) and leaving it ($I_{\text{out}}$). A ratio $I_{\text{out}} / I_{\text{in}} < 0.95$ (project heuristic) suggests a parallel bypass.
  4. For gas, run a flow test and apply soap solution to joints between the riser and meter inlet to find hidden T-taps.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-01]` (Zheng et al., 2018): Electricity theft shows up as non-periodic anomalies in otherwise periodic consumption, which motivates trajectory features.
  * `[SRC-04]` (Antmann, 2009): Names tampering with or bypassing meters as a classic form of non-technical loss.

---

### [TYP-02] External Magnet Slowdown
* **Metadata:** `doc: playbook` | `section: magnet_slowdown` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Strong permanent magnets (rare-earth magnets of roughly 1.0 to 1.5 tesla, project heuristic) are placed against the meter casing. In electromechanical induction meters the magnetic field induces eddy currents in the rotating aluminium disc, creating a braking torque that slows the disc (by roughly 30% to 80%, project heuristic) without stopping it. In electronic meters a strong external field can saturate the core of the internal current transformer and distort the measurement. In gas meters a magnet can interfere with a magnetic coupling between the measuring unit and the counter.
* **2. Data Signature:**  
  * A sustained plateau of reduced consumption without complete zeros: `cons_pm_median` drops while zero-consumption invoices stay rare.
  * A break from the client's own history: `robust_z_max > 3.0` and `n_robust_z_gt3` (both planned).
  * Unusually flat bills: low `cons_pm_std` and `cons_pm_mad` compared with peers.
  * Avoidance of the higher tariff bands: `share_level_2_plus` falls to zero.
* **3. On-Site Inspection Protocol:**  
  1. Inspect the casing for discoloration, circular scratches or adhesive residue from mounting a magnet.
  2. Use a Gaussmeter or magnetic viewing film to detect residual magnetism in the meter.
  3. Load test: connect a known resistive load (e.g. a 2 kW heater) and time the disc against the meter constant ($C_{\text{meter}}$ rev/kWh). A slower disc suggests braking.
  4. For gas meters, check for jerky or frozen dials while appliances are drawing gas.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-09]` (Jokar et al., 2016): Detects theft from changes in customers' consumption patterns; its simulated attacks include readings scaled down by a constant factor, which matches a slowdown.
  * `[SRC-10]` (Depuru et al., 2011): Overview of electricity-theft methods and their impact in distribution networks.
  * `[SRC-06]` (McLaughlin et al., 2013): Argues for combining physical tamper evidence with consumption data rather than relying on either alone.

---

### [TYP-03] Internal Register / Index Rollback
* **Metadata:** `doc: playbook` | `section: index_rollback` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  The register wheels are turned back to erase recorded consumption. Access is gained by picking seals, drilling small holes in the casing, or removing the glass; a needle or wire engages the drum gears and reverses the digits (e.g. rolling 14,850 kWh back to 11,200 kWh). In gas meters the drums can be uncoupled and turned back before the reader's visit.
* **2. Data Signature:**  
  * A reading lower than the previous one: `flag_index_backwards` **without** `flag_index_rollover`. A rollover (register wrapping past its maximum, e.g. 99999 to 0) is normal and not a rollback. In this data only 12 invoice rows are backwards and not rollovers.
  * A reconciliation failure: `flag_recon_mismatch` with a large `recon_gap`, where billed consumption does not equal the index difference (`idx_diff`) times `counter_coefficient`. Client-level counts `n_index_backwards` and `recon_gap_mean` are planned.
  * A backwards jump followed by a catch-up spike, or a meter exchange (`counter_statue 2`, assumed meaning) that could explain the discontinuity innocently.
* **3. On-Site Inspection Protocol:**  
  1. Inspect the register window under magnification for drill holes, needle punctures, cracked glass or resin repairs.
  2. Check the drum digits for misalignment, scratched paint or broken gear teeth.
  3. Check seals with a loupe for cut and re-twisted wires or mismatched crimp marks.
  4. Compare the dial reading on site with the last invoiced `new_index`.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-09]` (Jokar et al., 2016): Flags theft from abnormal changes in a customer's consumption pattern; a rollback appears in billed consumption as an abrupt drop.
  * `[SRC-06]` (McLaughlin et al., 2013): Combines tamper evidence with consumption data in a multi-sensor theft detector. The index-reconciliation rule used here is this project's own logic (`src/data/clean.py`), not taken from the paper.

---

### [TYP-04] Meter-Reader Collusion & Under-Reporting
* **Metadata:** `doc: playbook` | `section: reader_collusion` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  A meter reader records false, low readings in exchange for bribes, so the customer is billed for far less than they use. The physical register keeps advancing, so an unbilled balance builds up inside the meter. When routes change or a supervisor audits the meter, the fraud surfaces as a large catch-up bill or a sudden claim that the meter is broken or cannot be read.
* **2. Data Signature:**  
  * Departure from Benford's Law in the first digits of `cons_pm`: `benford_mad > 0.015` (planned), Nigrini's first-digit nonconformity band. These bands assume large samples; a single client has only a few dozen invoices, so a client-level MAD is noisy and is more reliable aggregated over a reader route or area.
  * Rounding bias: `round_reading_share > 0.40` (planned; project heuristic). If last digits were uniform, about 0.20 of readings would end in 0 or 5; the threshold is twice that.
  * Repeated `reading_remarque 6` (assumed: normal reading) with unusually low `cons_pm_std` that hides seasonal peaks.
  * A spike (`sudden_spike_share`, planned) or a switch to `reading_remarque 9` (assumed: not read; meaning doubtful) when the route changes.
* **3. On-Site Inspection Protocol:**  
  1. Make an unannounced visit by a supervisory team, without the usual route reader.
  2. Record the dial reading and compute the unbilled balance ($\Delta_{\text{unbilled}} = \text{Index}_{\text{actual}} - \text{Index}_{\text{last\_billed}}$).
  3. Compare neighbouring meters on the same route; widespread digit anomalies point to route-level collusion.
  4. Check whether the meter is in fact accessible, which would contradict past "not read" remarks.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-04]` (Antmann, 2009): Notes that theft by large consumers usually involves collusion with meter readers.
  * `[SRC-05]` (Wei et al., 2017): Uses Benford's Law on consumption data to flag likely electricity theft.
  * `[SRC-07]` (Nigrini, 2012): Defines the mean absolute deviation (MAD) test and its conformity bands.

---

### [TYP-05] Unauthorised Self-Reconnection
* **Metadata:** `doc: playbook` | `section: illegal_reconnection` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  A customer whose supply was cut off for arrears, breach of contract or detected fraud restores it without authorisation. The customer removes locking tags or clips, replaces removed fuses with copper wire, bypasses the open breaker, or splices the disconnected service cable back into the meter terminals. In gas systems, customers remove blanking plugs or bypass locked valves with flexible hoses.
* **2. Data Signature:**  
  * Consumption (`cons_pm > 0`, `idx_diff > 0`) on invoices carrying `counter_statue 1`, if that code does mean cut-off (assumed). Caution: about 72% of all status-1 invoice rows show nonzero consumption and the code is absent after 2013, so consumption under status 1 is common and weak evidence on its own. Confirm against the account's service-order history.
  * Frequent status changes: `n_statue_changes` (planned).
  * A jump from zero consumption back to normal levels without a reconnection order.
  * Arrears or a cut-off order on the account alongside continuing metered consumption.
* **3. On-Site Inspection Protocol:**  
  1. Confirm in the customer database that an active, unreversed cut-off order exists.
  2. Inspect the isolation point for broken padlock seals, blanking discs or locking clips.
  3. Examine the fuse carrier for copper bridges, solid slugs or non-utility fuses.
  4. Measure voltage on the load side of the main breaker; photograph live circuits and unauthorised connections.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-02]` (Glauner et al., 2017): Defines non-technical losses (meter tampering, bypassing, bribed readings, faulty meters, billing errors) and surveys their detection; it does not single out reconnection.
  * `[SRC-04]` (Antmann, 2009): Notes that corruption is also likely in disconnection operations for unpaid bills, and lists disconnection-reconnection management among loss-reduction measures.

---

### [TYP-06] Meter Jamming & Mechanical Breakdown
* **Metadata:** `doc: playbook` | `section: meter_jamming` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  The meter mechanism is deliberately obstructed, or a genuine fault is left unreported. In electromechanical meters, fine wires, hair or debris pushed through casing seams create friction against the disc. Customers may also stay silent about real failures (seized bearings, dry or stripped gears) and benefit from unmetered use. In gas diaphragm meters, contaminants or pins can freeze the valve or block the diaphragm stroke without cutting off supply.
* **2. Data Signature:**  
  * A switch to long runs of zero consumption while the premises are occupied: invoices with `cons == 0`; `zero_run_max` and `zero_share` are planned.
  * A static index over several cycles: `idx_diff == 0` despite past consumption.
  * Status codes `counter_statue 3` or `4` (assumed: under repair, blocked/failed); in training data neither is associated with a higher fraud rate, so they do not by themselves separate fraud from honest faults.
  * Irregular billing: `flag_bad_months_number`, `has_missing_months`, long `days_since_last_invoice_max`.
* **3. On-Site Inspection Protocol:**  
  1. Inspect seams, terminal covers and windows for inserted objects, pinholes or seal damage.
  2. Tap the casing while appliances run and watch whether the disc or drum jumps.
  3. Switch on a high-draw appliance and watch the disc or diaphragm. No movement while voltage or gas flow is present means the meter is jammed.
  4. Photograph the register and send the meter to the laboratory for examination.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-02]` (Glauner et al., 2017): Treats faulty and broken meters as a category of non-technical loss distinct from deliberate theft, which matters here because the data cannot always tell them apart.

---

### [TYP-07] Tariff Misclassification & Category Arbitrage
* **Metadata:** `doc: playbook` | `section: tariff_misclassification` | `counter_type: ELEC` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  A business (bakery, workshop, shop, cold store) is supplied under a cheaper residential contract instead of a commercial or professional one. No wiring is altered; the loss comes from the lower tariff. In this data the residential/commercial split is itself assumed: `client_catg 11` is assumed residential and `client_catg 12` commercial, and `tarif_type 15` is assumed to be a professional electricity tariff. `tarif_type 40` is the main **gas** tariff, not a professional electricity tariff.
* **2. Data Signature:**  
  * Monthly consumption far above category peers: high `cons_pm_mean` and `cons_pm_max` within `client_catg 11` and the same `region`.
  * Large positive peer residuals: `peer_resid_mean` and `peer_resid_min` (planned).
  * Heavy use of the upper tariff bands: `share_level_2_plus > 0.60` (project heuristic).
  * Note: in training data `client_catg 12` has a lower fraud rate (3.8%) than the base rate (5.6%), while `client_catg 51` (meaning unknown) has a much higher one (17.2%).
* **3. On-Site Inspection Protocol:**  
  1. Check what activity takes place on the premises against the account name and category.
  2. Look for commercial refrigeration, motors or three-phase equipment on a residential supply.
  3. Check whether the service breaker rating exceeds the residential contract.
  4. Record signage, trading licences and business activity to support reclassification and back-billing.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-02]` (Glauner et al., 2017): Lists errors in meter reading, data processing and billing among the causes of non-technical loss, alongside physical theft; misclassification is closest to this billing category.
  * `[SRC-10]` (Depuru et al., 2011): Overview of electricity theft and related irregularities in distribution networks, with emphasis on developing countries.

---

### [TYP-08] Estimated Reading Abuse & Fake Index Estimation
* **Metadata:** `doc: playbook` | `section: estimated_reading_abuse` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Estimated billing is used to hide rising consumption or tampering for long periods. When a meter is not read, the billing system bills an estimate based on history. A customer can keep estimates going by blocking access (locked gates, hidden enclosures), or an agent can enter estimate remarks falsely. Actual use then drifts away from billed use until a physical read finally takes place.
* **2. Data Signature:**  
  * Long unbroken runs of `reading_remarque 8` (assumed: estimated reading). In training data clients with at least one remark-8 invoice have a fraud rate of 7.3% against a 5.6% base rate.
  * Artificially regular bills: low `cons_pm_std` over consecutive periods, consistent with a formula rather than real seasonal demand.
  * Departure from Benford's Law: `benford_mad > 0.015` (planned), with the small-sample caution given in `[TYP-04]`.
  * A large catch-up bill when a real reading is taken: high `cons_pm_max` or `recon_gap`.
* **3. On-Site Inspection Protocol:**  
  1. Check whether the meter is physically accessible and document what, if anything, blocked the reader.
  2. Compare reader logs with the dial reading to work out how long, and how much, consumption went unbilled.
  3. Inspect seals and terminals to make sure the estimation period did not hide a bypass or rollback.
  4. Order relocation of the meter to the property boundary so it can be read without entry.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-02]` (Glauner et al., 2017): Counts errors in meter reading and billing among the sources of non-technical loss.
  * `[SRC-05]` (Wei et al., 2017): Applies Benford's Law to consumption data as a theft screen.
  * `[SRC-07]` (Nigrini, 2012): Gives the MAD conformity bands used for `benford_mad`.

---

### [TYP-09] Catch-Up Billing After Intermittent Tampering
* **Metadata:** `doc: playbook` | `section: catchup_billing` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Tampering is switched on and off. After running a bypass or magnet for months, the customer removes it before an expected inspection or reading round. When the meter records normally again, consumption jumps, producing alternating periods of near-zero bills and sharp peaks.
* **2. Data Signature:**  
  * Volatile consumption: `sudden_spike_share` (planned; share of invoices with `cons_pm > 2.0 × median`, project heuristic). High `cons_pm_skew` and `cons_pm_kurt` are existing proxies.
  * A wide gap between low and high bills: high `cons_pm_max / (cons_pm_min + 1)`, computable from existing features.
  * A saw-tooth pattern in the 120-month sequence `seq.npy` (built by the deep-learning workstream): multi-month lows followed by single-period spikes.
  * Reconciliation gaps (`recon_gap`, `flag_recon_mismatch`) when the index advances quickly after flat periods.
* **3. On-Site Inspection Protocol:**  
  1. Inspect terminal screws and conductors for clip marks, scratches or insulation damage from temporary jumpers.
  2. Test the casing with a Gaussmeter for residual magnetism from a recently removed magnet.
  3. Compare spike dates with reading-route schedules to see whether spikes follow reader visits.
  4. Install a temporary check meter on the service cable and compare it with the customer meter.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-01]` (Zheng et al., 2018): Theft appears as non-periodic anomalies in consumption series, which a CNN over the sequence can pick up.
  * `[SRC-09]` (Jokar et al., 2016): Its simulated attacks include readings switched to zero for some periods and reported normally for others, which matches intermittent tampering.

---

### [TYP-10] Neutral Disconnection & Terminal Inversion
* **Metadata:** `doc: playbook` | `section: neutral_inversion` | `counter_type: ELEC` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  The neutral conductor at the meter is disconnected, weakened or swapped. In a single-phase electromechanical meter the voltage coil needs a potential difference between phase and neutral to drive the disc. If the utility neutral is disconnected and the household neutral is earthed locally (a ground rod or water pipe), the voltage across the coil falls towards zero and the disc stops while power still flows. Swapping phase and neutral connections can, in some meter types, make the register run backwards.
* **2. Data Signature:**  
  * Consumption suddenly falls to zero or near zero while neighbouring meters on the same feeder record normal use.
  * Backwards readings that are not rollovers (`flag_index_backwards` without `flag_index_rollover`) during periods of inversion.
  * In dual-fuel households (`has_elec == 1` and `has_gaz == 1`), gas consumption (`cons_pm_mean_gaz`) continues normally while electricity (`cons_pm_mean_elec`) stops.
  * Long zero runs (`zero_run_max`, planned) in clients with long `tenure_years` (e.g. > 5 years, project heuristic).
* **3. On-Site Inspection Protocol:**  
  1. Measure the voltage between the neutral terminal and true earth; more than 5 V (project heuristic) suggests a missing or lifted neutral.
  2. Inspect neutral connections for loose screws, cut pigtails or non-utility earth wires to household pipes.
  3. Check polarity with an approved tester to confirm phase and neutral enter the correct terminals.
  4. Check the service neutral from the pole to the building entry.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-10]` (Depuru et al., 2011): Overview of electricity-theft methods in distribution networks.
  * `[SRC-06]` (McLaughlin et al., 2013): Supports cross-checking meter evidence against consumption data; it does not analyse neutral faults specifically.

---

### [TYP-11] Instrument Transformer (CT/PT) Tampering
* **Metadata:** `doc: playbook` | `section: ct_pt_tampering` | `counter_type: ELEC` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Large accounts are metered through current and voltage transformers (CTs/PTs). The meter sees a scaled-down current, and billed consumption is multiplied back up by the transformer ratio. In this data the multiplier appears as `counter_coefficient`, which is greater than 1 on only about 1,600 of 3.08 million ELEC invoice rows. Tampering includes shorting or loading the CT secondary, disconnecting PT voltage leads, or fitting a lower-ratio CT. Recorded consumption then drops by a fixed fraction (e.g. 50% or 75%) while production continues.
* **2. Data Signature:**  
  * Consumption low relative to the account's apparent size and activity.
  * Changes in `counter_coefficient` over a client's history, or a coefficient that does not reconcile with the index (`flag_recon_mismatch`).
  * Missing use of the upper bands (`consommation_level_3`, `consommation_level_4`).
  * A permanent step drop in `cons_pm_mean` by a simple fraction ($1/2$, $1/3$, $1/4$).
* **3. On-Site Inspection Protocol:**  
  1. Inspect the CT enclosure and test terminal block for broken or replaced seals.
  2. Measure primary current with a high-voltage clamp ($I_{\text{primary}}$) and secondary current at the meter ($I_{\text{secondary}}$); check that the ratio matches the nameplate.
  3. Inspect CT secondary terminals for shorting links, extra burden resistors or cut voltage leads.
  4. Run a CT ratio and burden test with an approved test set.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-10]` (Depuru et al., 2011): Overview of electricity-theft methods, including tampering with metering.
  * `[SRC-06]` (McLaughlin et al., 2013): Supports combining meter evidence with consumption data; it does not analyse CT/PT circuits specifically.

---

### [TYP-12] Direct Service Drop Tapping (Overhead Hooking)
* **Metadata:** `doc: playbook` | `section: direct_tapping` | `counter_type: ELEC` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  The network is tapped before the meter. Hooks are thrown over low-voltage overhead lines, insulation-piercing connectors are attached, or hidden underground cables are spliced into service mains. Unmetered power runs into the premises through concealed conduits. The legal meter records nothing or only a small load, while heavy appliances draw power directly from the grid.
* **2. Data Signature:**  
  * Little or no recorded consumption (e.g. `cons_pm_mean < 10` kWh, project heuristic), indistinguishable from an empty property.
  * Long `tenure_years` (e.g. > 5 years, project heuristic) with zero consumption across years (`zero_share`, planned).
  * Spatial clustering: a high fraud rate among neighbouring clients, `nbr_fraud_rate_k10` (planned), which must be computed from training labels only.
  * A visibly occupied or active property that records almost nothing.
* **3. On-Site Inspection Protocol:**  
  1. Inspect the overhead service line from the pole to the building, visually or by drone.
  2. Look for non-standard piercing connectors, spliced cables or extra wires on the overhead lines.
  3. Use a thermal camera along walls and conduits to find hidden conductors carrying current.
  4. Open the main breaker at the meter and check with a clamp ammeter whether power still flows into the internal panel.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-02]` (Glauner et al., 2017): Lists bypassing meters by rigging lines from the power source among the causes of non-technical loss.
  * `[SRC-04]` (Antmann, 2009): Cites illegal connections to the grid and the tapping of power lines as classic forms of theft.
