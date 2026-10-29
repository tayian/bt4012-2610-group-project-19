# STEG Non-Technical Loss (NTL) Typology Playbook (`data/corpus/playbook.md`)

**Document ID:** `DOC-01-PLAYBOOK`  
**Author / Workstream:** Role D (LLM, RAG & Agent Lead)  
**Corpus Layer:** Operational Fraud Typology Reference  
**Status:** Complete (Sections TYP-01 to TYP-12)  

---

## Master Table of Contents & Typology Matrix

| Section ID | Typology Name | Applicable Utility | Primary STEG Feature Signatures | Primary Source Citation |
| :--- | :--- | :--- | :--- | :--- |
| **`[TYP-01]`** | Direct Meter Bypass / Phase Shunt | ELEC, GAZ | `yoy_drop_max`, `zero_run_max`, `cons_pm_mean` collapse, `recon_mismatch` | `[SRC-01]`, `[SRC-04]` (World Bank §3.1) |
| **`[TYP-02]`** | External Neodymium Magnet Slowdown | ELEC, GAZ | Sudden drop in `cons_pm_median`, increased `robust_z_max`, low `share_level_2_plus` | `[SRC-04]` (World Bank §3.4), `[SRC-06]` |
| **`[TYP-03]`** | Internal Register / Index Rollback | ELEC, GAZ | `is_negative_delta == 1`, `n_index_backwards > 0`, reconciliation discrepancy | `[SRC-04]` (World Bank §3.3), `[SRC-07]` |
| **`[TYP-04]`** | Meter-Reader Collusion & Under-Reporting | ELEC, GAZ | `benford_mad` anomaly, `round_reading_share > 0.40`, uniform `cons_pm` variance | `[SRC-04]` (World Bank §4.2), `[SRC-05]` |
| **`[TYP-05]`** | Unauthorized / Illegal Self-Reconnection | ELEC, GAZ | Active consumption while `counter_statue == 1` (cut-off), sudden surge in `cons_pm` | `[SRC-02]`, `[SRC-04]` |
| **`[TYP-06]`** | Meter Jamming & Mechanical Breakdown | ELEC, GAZ | Persistent `cons_pm == 0` without account closure, `counter_statue == 3` (repair) | `[SRC-02]`, `[SRC-04]` |
| **`[TYP-07]`** | Tariff Misclassification & Category Arbitrage | ELEC | High `total_cons_mean` under residential `client_catg 11` vs commercial `tarif_type` | `[SRC-02]`, `[SRC-08]` |
| **`[TYP-08]`** | Estimated Reading Abuse & Fake Index Estimation | ELEC, GAZ | Long consecutive runs of `reading_remarque == 8`, followed by sharp correction | `[SRC-02]`, `[SRC-05]` |
| **`[TYP-09]`** | Catch-Up Billing & Post-Tampering Reconciliation | ELEC, GAZ | Extreme `sudden_spike_share`, anomalous index gap `recon_gap_mean` | `[SRC-01]`, `[SRC-07]` |
| **`[TYP-10]`** | Neutral Disconnection & Terminal Inversion | ELEC | Complete zero reading on single phase, negative voltage drop | `[SRC-04]` (World Bank §3.2), `[SRC-06]` |
| **`[TYP-11]`** | Instrument Transformer (CT/PT) Tampering | ELEC (Industrial) | Multi-ratio discrepancy, low secondary current, `counter_coefficient` distortion | `[SRC-04]` (World Bank §3.5), `[SRC-06]` |
| **`[TYP-12]`** | Direct Service Drop Tapping (Overhead Hooking) | ELEC | Zero meter consumption alongside documented regional grid energy losses | `[SRC-02]`, `[SRC-04]` |

---

## Detailed Fraud Typology Specifications

### [TYP-01] Direct Meter Bypass / Phase Shunt (Bridging)
* **Metadata:** `doc: playbook` | `section: meter_bypass` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  A low-resistance conductor or concealed jumper wire is installed directly between the incoming line terminal and the outgoing load terminal of the electricity meter's current measuring coil, or across the external service drop prior to meter entry. Following Ohm's and Kirchhoff's current laws ($I_{\text{meter}} = I_{\text{total}} \cdot \frac{R_{\text{shunt}}}{R_{\text{coil}} + R_{\text{shunt}}}$), the vast majority of load current (typically 70% to 95%) bypasses the current sensor coil, leaving only a negligible fraction to register on the counting mechanism. In natural gas installations, an illegal bypass pipe with an illicit manual valve is tapped upstream of the gas meter inlet, diverting fuel flow around the diaphragm measuring bellows directly into internal household appliances.
* **2. STEG Data Signature:**  
  * Structural, permanent collapse in monthly consumption: sharp fall in `cons_pm_mean` and `cons_pm_median`.
  * Extreme values in trajectory drop metrics: `yoy_drop_max > 0.65` and steep negative slope in `slope_last3y`.
  * Prolonged streaks of zero or near-zero billing periods: elevated `zero_run_max` and `zero_share`.
  * Severe suppression of progressive billing bands: `share_level_2_plus` falls close to zero as usage stays artificially capped in Tier 1.
  * Disparity between actual physical load and declared index advance (`recon_gap_mean` divergence).
* **3. On-Site Inspection Protocol:**  
  1. Inspect security wire seals on the meter terminal cover for evidence of cutting, re-crimping, or non-matching utility serial numbers.
  2. Remove the terminal cover and examine the phase terminal blocks for unauthorized copper bridge links, hidden bridging wires behind the mounting board, or scoring marks on terminal screws.
  3. Conduct dual-clamp ammeter audit: measure current simultaneously on the service drop cable entering the meter ($I_{\text{in}}$) and the internal cable exiting the meter ($I_{\text{out}}$). Any ratio $I_{\text{out}} / I_{\text{in}} < 0.95$ confirms an active parallel bypass.
  4. For gas installations, perform an acoustic flow test and apply soapy bubble solution across all pipe joints between the service riser and meter inlet to identify hidden T-taps.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-01]` (Zheng et al., 2018): Characterizes structural temporal discontinuities and non-periodic load drops caused by physical bypasses.
  * `[SRC-04]` (World Bank ESMAP Paper 040/09, §3.1): Detailed technical breakdown of line-to-load shunting and internal terminal bridging across residential induction meters.

---

### [TYP-02] External Neodymium Magnet Slowdown
* **Metadata:** `doc: playbook` | `section: magnet_slowdown` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Fraudsters affix high-grade rare-earth neodymium (NdFeB) permanent magnets (producing magnetic fields of 1.0 to 1.5 Tesla) directly against the exterior polycarbonate casing of the meter. In electromechanical Ferraris induction meters, the external magnetic flux penetrates the casing and induces intense eddy currents in the rotating aluminum rotor disc. This creates an artificial magnetic braking torque that counteracts driving torque, slowing disc rotation by 30% to 80% without completely halting it. In modern solid-state/static meters, the intense external flux saturates the ferromagnetic core of the internal current transformer (CT), causing severe magnetic saturation and non-linear attenuation of secondary current measurement. In gas meters, the external magnet clamps the internal magnetic coupling drive connecting the bellows to the counter.
* **2. STEG Data Signature:**  
  * Sustained, artificial plateau of suppressed consumption without complete zeros: `cons_pm_mean` drops significantly while `zero_run_max` remains 0.
  * Significant shift in client self-deviation: high `robust_z_max > 3.0` and elevated `n_robust_z_gt3` during months of magnetic application.
  * Unnaturally flat, low variance across consecutive billing periods (`cons_pm_std` severely diminished relative to peer seasonal baselines in feature group `F8`).
  * Total avoidance of higher tariff brackets: `share_level_2_plus` drops from historic normal levels to zero.
* **3. On-Site Inspection Protocol:**  
  1. Inspect the exterior polycarbonate enclosure for localized discoloration, circular scratch patterns, or adhesive residue indicative of magnet mounting.
  2. Utilize a portable Gaussmeter or magnetic viewing film across the meter housing to detect localized residual magnetism in internal iron pole pieces, disc spindles, or CT cores.
  3. Conduct calibrated load test: engage a certified resistive load (e.g., 2.0 kW heating element) and time disc revolutions against the nameplate constant ($C_{\text{meter}} \text{ rev/kWh}$). Disc rotation speed slower than rated constant confirms mechanical braking.
  4. For gas meters, check for jerky dial movement or complete dial freezing while gas appliances are actively drawing fuel.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-04]` (World Bank ESMAP Paper 040/09, §3.4): Details magnetic flux interaction with eddy-current disc braking systems in electromechanical meters.
  * `[SRC-06]` (McLaughlin et al., 2013): Formal sensor vulnerability analysis demonstrating current transformer core saturation under external magnetic tampering.

---

### [TYP-03] Internal Register / Index Rollback
* **Metadata:** `doc: playbook` | `section: index_rollback` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Direct mechanical manipulation of the internal odometer register wheels to rewind recorded cumulative consumption. Fraudsters access the interior register by picking lead security seals, drilling microscopic puncture holes through the polycarbonate enclosure, or removing the meter glass. A fine needle, heated wire, or vacuum suction is inserted to physically engage the numeral drum gear teeth and reverse the counter digits backwards (e.g. rolling 14,850 kWh back to 11,200 kWh). In gas meters, reverse air pressure or physical pinion gear uncoupling is employed to manually roll back the mechanical register drums before routine meter-reader rounds.
* **2. STEG Data Signature:**  
  * Definitive physical violation of monotonic index progression: occurrence of `new_index < old_index`.
  * Deterministic binary indicators: `is_negative_delta == 1` and `n_index_backwards > 0`.
  * Severe index reconciliation failure: `is_reconciliation_mismatch == 1` and anomalous magnitude in `recon_gap_mean`, where declared consumption does not equate to dial difference.
  * Abnormal billing patterns: negative index jumps followed by compensatory catch-up spikes or sudden tariff reclassifications.
* **3. On-Site Inspection Protocol:**  
  1. High-magnification optical inspection of the register window for hairline drill holes, needle punctures, cracked glass, or melted resin repairs.
  2. Examine internal odometer drum digits for mechanical misalignment, scratched numeral paint, or broken plastic gear teeth caused by forceful reverse rotation.
  3. Audit meter cabinet and utility lead seals with a magnifying loupe: check for cut and re-twisted copper wires, stretched lead seals, or mismatched crimp die impressions.
  4. Compare current on-site physical dial reading directly with the historical invoice `new_index` on the handheld terminal to verify chronological forward progression.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-04]` (World Bank ESMAP Paper 040/09, §3.3): Documents physical register drum reversal, gear tampering, and unmetered rollbacks in induction meters.
  * `[SRC-07]` (Jokar et al., 2016): Analyzes anomalous consumption drops and index trajectory reversals characteristic of mechanical register fraud.

---

### [TYP-04] Meter-Reader Collusion & Under-Reporting
* **Metadata:** `doc: playbook` | `section: reader_collusion` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Systemic human fraud arising from operational collusion between corrupt utility meter readers and property owners. The meter reader enters false, artificially low index readings into the handheld terminal in exchange for cash bribes, allowing the customer to consume large amounts of electricity while billing only minimal amounts. Because the physical meter dial advances normally, a growing cumulative deficit accumulates inside the physical counter. When meter readers rotate routes or an unannounced supervisory audit occurs, the fraud is revealed either as a sudden multi-thousand kWh catch-up bill or an abrupt claim that the meter is broken or inaccessible.
* **2. STEG Data Signature:**  
  * Severe violation of Benford's Law: human-fabricated index increments disrupt natural first-digit logarithmic distributions, causing elevated `benford_mad > 0.05` across `cons_pm`.
  * Digital rounding bias: high `round_reading_share >> 0.20` (fabricated readings end abnormally in digits 0 or 5 rather than uniform distributions).
  * Repetitive `reading_remarque == 6` (claimed direct visual reading) exhibiting unnaturally low variance across years, masking true seasonal peaks.
  * Sudden massive spike in consumption (`sudden_spike_share`) or sudden transition to `reading_remarque == 9` (inaccessible) when meter-reader inspection routes change.
* **3. On-Site Inspection Protocol:**  
  1. Perform an unannounced audit visit conducted exclusively by a centralized supervisory task force without the assigned regional route reader.
  2. Record the physical meter index on-site and calculate the true cumulative unbilled balance ($\Delta_{\text{unbilled}} = \text{Index}_{\text{actual}} - \text{Index}_{\text{last\_billed}}$).
  3. Cross-reference historical billing records of neighboring meters along the same reader route; widespread Benford anomalies indicate systemic route collusion.
  4. Inspect property to verify that the meter location is freely accessible, disproving historical claims of meter inaccessibility (`remarque 9`).
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-04]` (World Bank ESMAP Paper 040/09, §4.2): Analyzes utility governance failures, meter-reader collusion mechanisms, and field transcription bribery.
  * `[SRC-05]` (Wei et al., 2018): Establishes the mathematical application of Benford's Law (MAD and Chi-square) to detect human-fabricated utility readings.

---

### [TYP-05] Unauthorized / Illegal Self-Reconnection
* **Metadata:** `doc: playbook` | `section: illegal_reconnection` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  A customer whose electric or gas service has been formally disconnected by STEG for unpaid arrears, contractual violation, or detected fraud restores power on their own initiative. The customer removes the utility's official locking tags or security clips, replaces removed service fuses with heavy copper wires, bypasses the open service circuit breaker, or splices disconnected overhead/underground service drop cables back into the meter input terminals. In gas systems, customers remove blanking plugs or bypass locked service ball valves with flexible high-pressure hoses.
* **2. STEG Data Signature:**  
  * Ongoing, active consumption (`cons_pm > 0`, `index_delta > 0`) during billing periods where administrative status is `counter_statue == 1` (disconnected / cut off).
  * High frequency of administrative state fluctuations: elevated `n_statue_changes`.
  * Sharp transitions from periods of zero consumption back to full residential consumption levels without a corresponding utility re-activation order.
  * Conflict between utility account receivables status (active collection / shut-off order) and active metered energy consumption.
* **3. On-Site Inspection Protocol:**  
  1. Cross-reference central STEG database: confirm that the account holds an active, unreversed "Suspension / Cut-Off" administrative service order.
  2. Inspect the service isolation point: check if utility padlock seals, plastic blanking discs, or locking clips have been broken, cut, or tampered with.
  3. Examine the main cut-out fuse carrier for unauthorized copper wire bridges, solid copper slugs, or non-utility cartridge fuses replacing removed fuses.
  4. Measure line voltage on the load side of the main circuit breaker; photograph live energized circuits and unauthorized connections for police filing.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-02]` (Glauner et al., 2017): Identifies unauthorized self-reconnection after credit disconnection as a major category of non-technical losses.
  * `[SRC-04]` (World Bank ESMAP Paper 040/09, §3.6): Documents physical bypassing of utility service isolation hardware and lockout tags.

---

### [TYP-06] Meter Jamming & Mechanical Breakdown
* **Metadata:** `doc: playbook` | `section: meter_jamming` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Deliberate physical obstruction or exploitation of mechanical friction to arrest or impede the meter mechanism. In electromechanical meters, perpetrators insert fine metallic filaments, animal hair, or insect debris through casing seams or gasket seals into the rotor disc chamber, creating physical friction against the aluminum disc. Alternatively, consumers fail to report legitimate mechanical failures (such as seized spindle bearings, dry gear trains, or stripped worm gears), actively benefiting from unmetered consumption. In natural gas diaphragm meters, contaminants, viscous liquids, or physical pins are introduced to freeze the sliding valve or block the bellows stroke without triggering supply cut-off.
* **2. STEG Data Signature:**  
  * Abrupt transition to and prolonged duration of zero consumption: elevated `zero_run_max` and `zero_share == 1.0` while premises remain occupied.
  * Static meter indices across multiple billing cycles: `index_delta == 0` despite active historical consumption baselines.
  * Administrative anomaly flags: transition of `counter_statue` to state 3 (under repair) or state 4 (blocked / mechanical failure), or recorded `n_statue_changes > 0`.
  * Invalidation indicators: elevated `n_bad_months_number` or irregular billing intervals where meters remain unread.
* **3. On-Site Inspection Protocol:**  
  1. Inspect meter casing seams, terminal covers, and viewing windows for foreign object insertion, drilled pinholes, or seal tampering.
  2. Perform physical tap and acoustic test: gently tap the enclosure while appliances are operating to observe if the disc or index drum releases momentarily.
  3. Conduct active load verification: turn on high-draw appliances (e.g., electric heater or gas stove) and observe rotor disc movement or diaphragm cycling. If zero movement occurs while voltage/gas flow is present, meter is jammed.
  4. Photograph index counter and label meter for removal to laboratory for destructive mechanical autopsy.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-02]` (Glauner et al., 2017): Distinguishes mechanical meter defects and unreported breakdowns as structural Non-Technical Losses distinct from active theft.
  * `[SRC-04]` (World Bank ESMAP Paper 040/09, §3.7): Documents internal physical obstruction, gear decoupling, and disc friction techniques.

---

### [TYP-07] Tariff Misclassification & Category Arbitrage
* **Metadata:** `doc: playbook` | `section: tariff_misclassification` | `counter_type: ELEC` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Administrative and contractual fraud where a commercial, agricultural, or small industrial facility illicitly operates under a subsidized residential electricity tariff. Commercial enterprises (e.g., bakeries, workshops, retail stores, refrigeration units) register under domestic low-voltage categories (`client_catg 11`, `tarif_type 11`) rather than commercial or professional rates (`client_catg 12`, `tarif_type 15` or `40`). This enables commercial operators to evade higher commercial capacity charges, peak demand rates, and commercial energy taxation, imposing severe financial losses on utility billing structures without altering physical meter wiring.
* **2. STEG Data Signature:**  
  * Excessive monthly consumption volumes far beyond domestic norms: extreme `cons_pm_mean` and `cons_pm_max` relative to category peers.
  * Massive positive deviation in peer-residual features (`peer_resid_mean` and `peer_resid_min` in group `F8`), indicating consumption substantially exceeding regression predictions based on tariff and region.
  * High occupancy in Tier 2 and Tier 3 consumption tiers: high `share_level_2_plus` ($>0.60$), reflecting industrial continuous load.
  * Inconsistency between contractual district / client category and empirical power utilization patterns.
* **3. On-Site Inspection Protocol:**  
  1. Conduct physical property audit: verify nature of business activity conducted on premise versus registered account name and category.
  2. Inspect installed machinery, commercial refrigeration, industrial motors, or three-phase equipment connected to a single-phase residential meter.
  3. Audit installed service breaker capacity: check if service breaker amperage has been upgraded beyond residential utility contractual allowances.
  4. Document premises signage, trading license, and business operations for legal reclassification and retroactive commercial tariff back-billing.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-02]` (Glauner et al., 2017): Classifies tariff arbitrage and billing misclassification within the macro-economic NTL taxonomy.
  * `[SRC-08]` (Depuru et al., 2011): Surveys category misclassification and contractual irregularities in developing distribution networks.

---

### [TYP-08] Estimated Reading Abuse & Fake Index Estimation
* **Metadata:** `doc: playbook` | `section: estimated_reading_abuse` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Exploitation of utility estimation protocols to conceal true escalating consumption or physical meter tampering over prolonged periods. When utility readers are unable or unwilling to read meters, utility billing systems apply historical average estimates. Colluding customers or negligent utility agents sustain consecutive estimated billing cycles by deliberately obstructing meter access (locked gates, hidden enclosures) or falsely entering estimate remark codes. This creates an unmetered consumption buffer where actual usage diverges from billed estimates, delaying fraud detection for years until a mandatory supervisory physical read occurs.
* **2. STEG Data Signature:**  
  * Prolonged unbroken sequences of estimated invoices: high frequency and consecutive runs of `reading_remarque == 8`.
  * Artificial regularity in consumption: `cons_pm_std` approaches zero across consecutive quarters, reflecting algorithmic billing software formulas rather than genuine seasonal demand.
  * Statistical distribution anomalies: strong rejection under Benford's Law (`benford_mad > 0.05`) due to synthetic algorithmic estimation.
  * Termination in a massive index catch-up bill: sudden occurrence of `cons_pm_max` or `recon_gap_mean` when a true physical reading is finally captured.
* **3. On-Site Inspection Protocol:**  
  1. Verify physical accessibility of the meter enclosure: document whether gates, locks, or vegetation genuinely prevented meter-reader access.
  2. Compare handheld reader logs with on-site dial reading to determine the exact duration and volume of unbilled accrued energy.
  3. Conduct full meter seal and terminal inspection to ensure the prolonged estimation period was not used to execute an unobserved physical bypass or rollback.
  4. Issue a formal directive for meter relocation to the external property boundary wall to ensure unimpeded bi-monthly utility visual reading.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-02]` (Glauner et al., 2017): Analyzes administrative estimation errors and systemic utility meter-reading failure modes.
  * `[SRC-05]` (Wei et al., 2018): Demonstrates that repeated estimated billing algorithms create unnatural digit clustering detectable via Benford goodness-of-fit screening.

---

### [TYP-09] Catch-Up Billing & Post-Tampering Reconciliation
* **Metadata:** `doc: playbook` | `section: catchup_billing` | `counter_type: both` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Transient or intermittent meter tampering where perpetrators temporarily restore normal meter operation prior to anticipated utility inspection rounds or seasonal reader cycles. After operating an illegal bypass or magnet slowdown for months, the perpetrator removes the jumper or magnet to avoid immediate on-site detection by meter readers. When the meter resumes normal recording, a sudden surge in metered consumption occurs as accumulated unbilled load or seasonal catch-up consumption registers on the index dials, producing violent oscillations between near-zero billing and massive consumption peaks.
* **2. STEG Data Signature:**  
  * Severe consumption volatility: elevated `sudden_spike_share` (share of invoices with `cons_pm > 2.0 * median`).
  * Extreme discrepancy between minimum and maximum consumption: high ratio of `cons_pm_max / (cons_pm_min + 1)`.
  * Distinct saw-tooth consumption trajectory in time-series tensor `seq.npy`: alternating multi-month drops followed by sharp single-period spikes.
  * Reconciliation mismatches: elevated `recon_gap_mean` as index advances rapidly following long flat intervals.
* **3. On-Site Inspection Protocol:**  
  1. Inspect terminal block screws and phase conductors for tell-tale mechanical indentation marks, metal scratching, or insulation scoring caused by temporary jumper crocodile clips.
  2. Test meter polycarbonate casing with Gaussmeter for residual magnetic fields from recently detached magnets.
  3. Cross-reference consumption spike dates against utility meter-reader inspection route schedules to identify correlation with reader visit cycles.
  4. Install a temporary secondary check meter on the pole-mounted service drop to monitor real-time load versus customer terminal readings.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-01]` (Zheng et al., 2018): Details intermittent load curve fluctuations and periodic tamper-removal cycles in smart grid load curves.
  * `[SRC-07]` (Jokar et al., 2016): Mathematical modeling of saw-tooth consumption anomalies and transient theft behavior in electricity distribution.

---

### [TYP-10] Neutral Disconnection & Terminal Inversion
* **Metadata:** `doc: playbook` | `section: neutral_inversion` | `counter_type: ELEC` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Electrical circuit manipulation involving the disconnection, attenuation, or inversion of the utility neutral reference conductor at the meter terminal block. In single-phase electromechanical meters, the voltage coil requires a potential difference between the live phase and utility neutral to generate the driving magnetic flux. By disconnecting the utility neutral wire from the meter and grounding the household neutral circuit through an independent local earth electrode (e.g. copper ground rod or water pipe), the potential across the meter's voltage coil drops to near zero, stopping disk rotation while power continues to flow through the phase conductor. Alternatively, reversing incoming phase and neutral wires inverses current direction, driving mechanical index dials backward.
* **2. STEG Data Signature:**  
  * Instantaneous collapse of declared consumption to zero or near-zero while neighboring premises on the same low-voltage feeder record normal usage.
  * Occurrence of reverse index readings (`n_index_backwards > 0`, `is_negative_delta == 1`) during periods of phase/neutral terminal inversion.
  * Disparity between electricity and gas usage in dual-fuel households (`has_elec == 1` and `has_gaz == 1`), where gas consumption continues normally while electric consumption ceases.
  * Elevated `zero_run_max` accompanied by historical customer tenure (`tenure_years > 5`).
* **3. On-Site Inspection Protocol:**  
  1. Measure voltage potential between neutral terminal and true earth ground using a digital multimeter; potential $>5\text{ V}$ indicates missing or lifted utility neutral.
  2. Inspect neutral terminal block connections for loose clamping screws, cut neutral pigtails, or non-utility grounding wires connected to household piping.
  3. Verify phase polarity: use an approved phase-rotation tester or polarity pen to confirm phase conductor enters terminal 1 and neutral enters terminal 3.
  4. Audit physical integrity of utility service drop neutral conductor from distribution pole to weatherhead service entrance.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-04]` (World Bank ESMAP Paper 040/09, §3.2): Details neutral terminal disconnection, artificial local earthing, and polarity reversal techniques.
  * `[SRC-06]` (McLaughlin et al., 2013): Formal circuit analysis of terminal inversion and missing neutral detection in distribution systems.

---

### [TYP-11] Instrument Transformer (CT/PT) Tampering
* **Metadata:** `doc: playbook` | `section: ct_pt_tampering` | `counter_type: ELEC` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  High-value industrial and commercial fraud executed on secondary measurement circuits of Current Transformers (CTs) and Potential Transformers (PTs). For high-amperage commercial accounts (`client_catg 12` or `tarif_type 40`), meters measure stepped-down current proportional to the primary line current governed by the transformer transformation ratio (`counter_coefficient`). Perpetrators install unauthorized low-resistance burden resistors across CT secondary terminals, short-circuit CT secondary windings, disconnect PT voltage sensing leads, or replace nameplate CTs with lower-ratio units. This reduces the recorded current by exact ratios (e.g., 50% or 75%) without interrupting high-voltage three-phase customer production lines.
* **2. STEG Data Signature:**  
  * Disproportionately low consumption readings relative to declared contract power capacity and commercial operations.
  * Discrepancies involving `counter_coefficient` multipliers: abnormal shifts in consumption scaling factors.
  * Severe skew across consumption bands: uncharacteristic suppression of Tier 3 and Tier 4 commercial blocks (`consommation_level_3` and `_4`).
  * Sudden permanent ratio shift: step-change drop in `cons_pm_mean` by clean integer fractions ($1/2$, $1/3$, or $1/4$) visible in trajectory features.
* **3. On-Site Inspection Protocol:**  
  1. Inspect instrument transformer enclosure, CT mounting cabinet, and test terminal block (TTB) security seals for unauthorized entry or broken seals.
  2. Measure primary line current with a high-voltage clamp ammeter ($I_{\text{primary}}$) and compare against secondary current ($I_{\text{secondary}}$) at the meter terminals to verify actual ratio matches the specified nameplate ratio.
  3. Inspect CT secondary terminal blocks for unauthorized shorting links, parallel burden resistors, or severed voltage leads.
  4. Perform CT ratio and burden testing using an approved instrument transformer test set to detect internal winding short circuits or core saturation.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-04]` (World Bank ESMAP Paper 040/09, §3.5): Comprehensive technical audit manual for CT/PT secondary circuit tampering, shorting links, and ratio manipulation.
  * `[SRC-06]` (McLaughlin et al., 2013): Multi-sensor framework analyzing transformer secondary circuit degradation and non-linear ratio distortions.

---

### [TYP-12] Direct Service Drop Tapping (Overhead Hooking)
* **Metadata:** `doc: playbook` | `section: direct_tapping` | `counter_type: ELEC` | `jurisdiction: TN`
* **1. Physical Mechanism & Modus Operandi:**  
  Complete electrical bypass executed externally to the meter installation by tapping directly into the utility distribution network prior to the customer's meter connection. Fraudsters throw physical metal hooks, attach piercing insulation connectors, or splice concealed underground cables directly into low-voltage overhead lines or underground service mains. Unmetered electricity is routed directly into the premises main switchboard through hidden conduit buried inside walls or underground. The legal meter remains completely dormant or registers only negligible standby load, while heavy electrical loads (air conditioning, machinery, heating) draw unmetered power straight from the grid.
* **2. STEG Data Signature:**  
  * Meter record indicates total inactivity or minimal baseline consumption (`cons_pm_mean < 10 \text{ kWh}`), indistinguishable from an abandoned property.
  * High customer account tenure (`tenure_years > 5`) with near-total `zero_share` across consecutive years.
  * Extreme spatial clustering: elevated neighbor fraud rate (`nbr_fraud_rate_k10` in group `F10`) as informal tapping spreads socially within specific geographic districts.
  * Contrast between physical premises status (occupied, active machinery, illuminated facility) and zero utility meter registration.
* **3. On-Site Inspection Protocol:**  
  1. Conduct visual and optical drone inspection of the overhead service line from the low-voltage distribution pole to the property roof entrance.
  2. Inspect service drop conductors for non-standard insulation piercing connectors (IPCs), spliced cables, or auxiliary wires tapped into overhead phase lines.
  3. Utilize a thermal imaging camera (FLIR) along service walls and conduits to detect heat signatures emitted by concealed conductors carrying high unmetered currents.
  4. Disconnect main breaker at customer meter and verify with clamp ammeter if power continues to flow into internal distribution panels.
* **4. Literature Citation & Evidence Grounding:**  
  * `[SRC-02]` (Glauner et al., 2017): Documents direct network tapping and illegal overhead hooks as dominant NTL vectors in developing electrical distribution grids.
  * `[SRC-04]` (World Bank ESMAP Paper 040/09, §3.1): Detailed physical diagrams of illegal service drop tapping, underground cable splicing, and overhead line hooking.

