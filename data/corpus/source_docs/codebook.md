# STEG Code Book (`data/corpus/source_docs/codebook.md`)

**Document ID:** `DOC-02-CODEBOOK`  
**Corpus layer:** Categorical code reference  
**Status:** Data-derived facts; code meanings are assumptions  
**Entries:** 82 (one per value)

---

## About this code book

STEG has never published definitions for the codes in this dataset. Everything in an entry's **Observed in data** line is computed from the project data. Everything in an **Assumed meaning** line is a guess made by the project team. The guesses come from field names and common utility practice, are unverified, and are kept only where the data do not contradict them. Entries carrying a guess are tagged `assumed: true` in their metadata. An Inspector Brief that cites a code meaning must say that the meaning is assumed.

There is one entry for every value of `counter_statue`, `reading_remarque`, `tarif_type` and `counter_code` listed in `results/tables/code_values.csv`, and one for each `client_catg` value (11, 12, 51). Each entry is one retrieval chunk, identified by `[CODE-<field>-<value>]`. BM25 matches these exact IDs and code strings better than dense embeddings do.

## How the facts were computed

* **Rows and share:** raw values from `data/interim/invoice.parquet` (4,476,749 invoice rows). Raw values are used because `invoice_clean.parquet` maps unexpected `counter_statue` / `reading_remarque` values to `other`.
* **ELEC/GAZ split:** `counter_type` of the same rows (stripped, upper-cased).
* **Zero-consumption share:** rows where `consommation_level_1..4` sum to 0.
* **Fraud rate:** training clients only (`data/splits/train_ids.csv`, 108,394 clients, base fraud rate 5.58% from `target` in `data/interim/client.parquet`). It is the share of fraud clients among training clients who carry the value on at least one invoice. Clients with longer histories are more likely to carry a rare value at some point, so this is a descriptive association, not a causal or per-invoice rate. Rates based on fewer than 30 clients are flagged.
* For `client_catg` the counts are clients from `client.parquet`, not invoice rows.

## Index

| Field | Values | Notes |
| :--- | :--- | :--- |
| `counter_statue` | 12 | 0–5 are the normal set; 46, A, 618, 769, 269375, 420 are invalid (47 rows) |
| `reading_remarque` | 8 | 6–9 are the normal set; 203, 413, 207, 5 are invalid (34 rows) |
| `tarif_type` | 17 | each value is used with only one counter type |
| `counter_code` | 42 | each value is used with only one counter type; no meanings assumed; 0 and 1 are invalid (34 rows) |
| `client_catg` | 3 | client-level field |

---

## counter_statue (meter status)

The field name suggests a meter or contract status. The values 0–5 have assumed meanings.

### [CODE-counter_statue-0] counter_statue = 0
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: 0` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 4,379,008 invoice rows (97.82% of 4,476,749); 3,022,136 ELEC / 1,356,872 GAZ (69.0% ELEC); 9.9% of these rows have zero total consumption; seen 1977-2019. Training clients carrying it at least once: 107,488, of which fraud: 6,047 (5.63%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Normal active service.
* **Investigation relevance:** Baseline state carried by almost every invoice; on its own it says nothing about fraud.

### [CODE-counter_statue-1] counter_statue = 1
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: 1` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 74,036 invoice rows (1.65% of 4,476,749); 40,423 ELEC / 33,613 GAZ (54.6% ELEC); 28.3% of these rows have zero total consumption; seen 1977-2013. Training clients carrying it at least once: 27,617, of which fraud: 1,892 (6.85%; base rate 5.58%). Most of these rows (about 72%) still record nonzero consumption, and the code does not appear after 2013, which sits uneasily with a plain 'service is off' reading.
* **Assumed meaning (unverified — STEG has not published code definitions):** Disconnected / cut-off service.
* **Investigation relevance:** If the assumed meaning holds, consumption recorded under this status could point to self-reconnection ([TYP-05]). Because most status-1 rows still show consumption, treat it as a weak signal and confirm the account's real service-order history before acting.

### [CODE-counter_statue-5] counter_statue = 5
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: 5` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 20,639 invoice rows (0.46% of 4,476,749); 14,110 ELEC / 6,529 GAZ (68.4% ELEC); 66.2% of these rows have zero total consumption; seen 1981-2017. Training clients carrying it at least once: 13,643, of which fraud: 1,422 (10.42%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Suspected anomaly or meter defect flagged by the reader.
* **Investigation relevance:** Among the common status codes this one shows the largest gap between its training fraud rate and the base rate. That is an association in historical data, not proof of tampering; use it as a reason to look closer, together with consumption evidence.

### [CODE-counter_statue-4] counter_statue = 4
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: 4` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 2,729 invoice rows (0.06% of 4,476,749); 2,494 ELEC / 235 GAZ (91.4% ELEC); 13.2% of these rows have zero total consumption; seen 1978-2012. Training clients carrying it at least once: 476, of which fraud: 30 (6.30%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Meter blocked / mechanical failure.
* **Investigation relevance:** Possibly related to jammed or defective meters ([TYP-06]); the training fraud rate is close to the base rate, so it is not a strong signal by itself.

### [CODE-counter_statue-3] counter_statue = 3
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: 3` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 258 invoice rows (0.0058% of 4,476,749); 187 ELEC / 71 GAZ (72.5% ELEC); 25.6% of these rows have zero total consumption; seen 1979-2011. Training clients carrying it at least once: 50, of which fraud: 1 (2.00%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Meter under repair / calibration.
* **Investigation relevance:** Possibly related to defective or jammed meters ([TYP-06]). Too few training clients to interpret the fraud rate.

### [CODE-counter_statue-2] counter_statue = 2
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: 2` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 32 invoice rows (0.0007% of 4,476,749); 10 ELEC / 22 GAZ (31.2% ELEC); 90.6% of these rows have zero total consumption; seen 1994-2008. Training clients carrying it at least once: 17, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Assumed meaning (unverified — STEG has not published code definitions):** Meter replaced / exchanged.
* **Investigation relevance:** A meter exchange can explain index discontinuities; check for one before reading an index jump as a rollback ([TYP-03]). Too few training clients to say anything about fraud rates.

## counter_statue: invalid or rare values

The same 34 invoice rows (5 clients; 33 ELEC rows and 1 GAZ row) carry an odd `counter_statue` (46, 618, 769, 269375 or 420), an odd `reading_remarque` (203, 207, 413 or 5), and `counter_code` 0 or 1. In these rows `reading_remarque` holds a number that looks like a real `counter_code`, `counter_coefficient` holds what looks like a remark (6, 8 or 9) and the index and consumption fields look moved along as well. That suggests each row was shifted by one column during data entry or export. The value `A` is different: 13 rows of a single client that are not shifted. `src/data/clean.py` maps all of these statue values to `other`. They are kept here so that every raw value has an entry.

### [CODE-counter_statue-46] counter_statue = 46
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: 46` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 14 invoice rows (0.0003% of 4,476,749); 14 ELEC / 0 GAZ (100.0% ELEC); 0.0% of these rows have zero total consumption; seen 2005-2011. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret. Row appears column-shifted (see the note above), so its consumption and index fields are unreliable.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

### [CODE-counter_statue-A] counter_statue = A
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: A` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 13 invoice rows (0.0003% of 4,476,749); 13 ELEC / 0 GAZ (100.0% ELEC); 92.3% of these rows have zero total consumption; seen 2005-2012. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret. These rows belong to a single client and are not column-shifted (their `counter_code` and remark look normal); the letter has no known meaning.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

### [CODE-counter_statue-618] counter_statue = 618
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: 618` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 12 invoice rows (0.0003% of 4,476,749); 12 ELEC / 0 GAZ (100.0% ELEC); 0.0% of these rows have zero total consumption; seen 2015-2019. No training client carries this value, so there is no fraud rate. Row appears column-shifted (see the note above), so its consumption and index fields are unreliable.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

### [CODE-counter_statue-769] counter_statue = 769
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: 769` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 6 invoice rows (0.0001% of 4,476,749); 6 ELEC / 0 GAZ (100.0% ELEC); 0.0% of these rows have zero total consumption; seen 2010-2011. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret. Row appears column-shifted (see the note above), so its consumption and index fields are unreliable.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

### [CODE-counter_statue-269375] counter_statue = 269375
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: 269375` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 1 invoice row (<0.0001% of 4,476,749); 1 ELEC / 0 GAZ (100.0% ELEC); 0.0% of these rows have zero total consumption; seen 1996-1996. No training client carries this value, so there is no fraud rate. Row appears column-shifted (see the note above), so its consumption and index fields are unreliable.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

### [CODE-counter_statue-420] counter_statue = 420
* **Metadata:** `doc: codebook` | `field: counter_statue` | `value: 420` | `counter_type: GAZ` | `jurisdiction: TN`
* **Observed in data:** 1 invoice row (<0.0001% of 4,476,749); 0 ELEC / 1 GAZ (0.0% ELEC); 0.0% of these rows have zero total consumption; seen 2012-2012. No training client carries this value, so there is no fraud rate. Row appears column-shifted (see the note above), so its consumption and index fields are unreliable.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

## reading_remarque (reading remark)

The field name suggests a remark made by the meter reader. The values 6–9 have assumed meanings.

### [CODE-reading_remarque-6] reading_remarque = 6
* **Metadata:** `doc: codebook` | `field: reading_remarque` | `value: 6` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 2,230,939 invoice rows (49.83% of 4,476,749); 1,494,098 ELEC / 736,841 GAZ (67.0% ELEC); 18.0% of these rows have zero total consumption; seen 1977-2019. Training clients carrying it at least once: 95,826, of which fraud: 5,728 (5.98%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Normal reading taken by the meter reader.
* **Investigation relevance:** Most frequent remark; its fraud rate is close to the base rate. Not informative on its own.

### [CODE-reading_remarque-9] reading_remarque = 9
* **Metadata:** `doc: codebook` | `field: reading_remarque` | `value: 9` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 1,416,992 invoice rows (31.65% of 4,476,749); 1,054,605 ELEC / 362,387 GAZ (74.4% ELEC); 0.2% of these rows have zero total consumption; seen 1996-2019. Training clients carrying it at least once: 93,263, of which fraud: 5,947 (6.38%; base rate 5.58%). It covers about a third of all invoice rows, a very high share for a 'meter could not be read' meaning, so the assumed meaning may be wrong.
* **Assumed meaning (unverified — STEG has not published code definitions):** Meter inaccessible / not read.
* **Investigation relevance:** Do not treat this code as evidence that the customer obstructed the meter: its frequency makes the assumed meaning doubtful. Check the reader's own logs before relying on it ([TYP-04], [TYP-08]).

### [CODE-reading_remarque-8] reading_remarque = 8
* **Metadata:** `doc: codebook` | `field: reading_remarque` | `value: 8` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 828,123 invoice rows (18.50% of 4,476,749); 530,046 ELEC / 298,077 GAZ (64.0% ELEC); 7.5% of these rows have zero total consumption; seen 1996-2019. Training clients carrying it at least once: 76,428, of which fraud: 5,549 (7.26%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Estimated reading (no physical read).
* **Investigation relevance:** If the assumed meaning holds, long runs of this remark could hide true consumption ([TYP-08]). The training fraud rate is somewhat above the base rate.

### [CODE-reading_remarque-7] reading_remarque = 7
* **Metadata:** `doc: codebook` | `field: reading_remarque` | `value: 7` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 661 invoice rows (0.01% of 4,476,749); 624 ELEC / 37 GAZ (94.4% ELEC); 17.2% of these rows have zero total consumption; seen 1984-2006. Training clients carrying it at least once: 381, of which fraud: 1 (0.26%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Reading declared by the customer.
* **Investigation relevance:** Rare and not seen after 2006; too little data to use as a signal.

## reading_remarque: invalid or rare values

The same 34 invoice rows (5 clients; 33 ELEC rows and 1 GAZ row) carry an odd `counter_statue` (46, 618, 769, 269375 or 420), an odd `reading_remarque` (203, 207, 413 or 5), and `counter_code` 0 or 1. In these rows `reading_remarque` holds a number that looks like a real `counter_code`, `counter_coefficient` holds what looks like a remark (6, 8 or 9) and the index and consumption fields look moved along as well. That suggests each row was shifted by one column during data entry or export. `src/data/clean.py` maps these values to `other`. They are kept here so that every raw value has an entry.

### [CODE-reading_remarque-203] reading_remarque = 203
* **Metadata:** `doc: codebook` | `field: reading_remarque` | `value: 203` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 15 invoice rows (0.0003% of 4,476,749); 15 ELEC / 0 GAZ (100.0% ELEC); 0.0% of these rows have zero total consumption; seen 1996-2011. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret. Row appears column-shifted (see the note above), so its consumption and index fields are unreliable.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

### [CODE-reading_remarque-413] reading_remarque = 413
* **Metadata:** `doc: codebook` | `field: reading_remarque` | `value: 413` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 12 invoice rows (0.0003% of 4,476,749); 12 ELEC / 0 GAZ (100.0% ELEC); 0.0% of these rows have zero total consumption; seen 2015-2019. No training client carries this value, so there is no fraud rate. Row appears column-shifted (see the note above), so its consumption and index fields are unreliable.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

### [CODE-reading_remarque-207] reading_remarque = 207
* **Metadata:** `doc: codebook` | `field: reading_remarque` | `value: 207` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 6 invoice rows (0.0001% of 4,476,749); 6 ELEC / 0 GAZ (100.0% ELEC); 0.0% of these rows have zero total consumption; seen 2010-2011. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret. Row appears column-shifted (see the note above), so its consumption and index fields are unreliable.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

### [CODE-reading_remarque-5] reading_remarque = 5
* **Metadata:** `doc: codebook` | `field: reading_remarque` | `value: 5` | `counter_type: GAZ` | `jurisdiction: TN`
* **Observed in data:** 1 invoice row (<0.0001% of 4,476,749); 0 ELEC / 1 GAZ (0.0% ELEC); 0.0% of these rows have zero total consumption; seen 2012-2012. No training client carries this value, so there is no fraud rate. Row appears column-shifted (see the note above), so its consumption and index fields are unreliable.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

## tarif_type (tariff)

Every tariff value is used with only one counter type, so the ELEC/GAZ split is a hard fact even where the meaning is unknown. Gas rows total 1,397,343.

### [CODE-tarif_type-11] tarif_type = 11
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 11` | `counter_type: ELEC` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 2,679,872 invoice rows (59.86% of 4,476,749); 2,679,872 ELEC / 0 GAZ (100.0% ELEC); 5.4% of these rows have zero total consumption; seen 1977-2019. Training clients carrying it at least once: 98,211, of which fraud: 5,510 (5.61%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Standard (residential) low-voltage electricity tariff.
* **Investigation relevance:** Main electricity tariff; its fraud rate is close to the base rate. Useful as the reference group when comparing consumption with peers ([TYP-07]).

### [CODE-tarif_type-40] tarif_type = 40
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 40` | `counter_type: GAZ` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 1,379,755 invoice rows (30.82% of 4,476,749); 0 ELEC / 1,379,755 GAZ (0.0% ELEC); 17.2% of these rows have zero total consumption; seen 1981-2019. Training clients carrying it at least once: 49,173, of which fraud: 3,487 (7.09%; base rate 5.58%). An earlier draft called it 'small industrial power'; the data contradict that. Every row is GAZ, and it accounts for about 98.7% of all gas invoice rows.
* **Assumed meaning (unverified — STEG has not published code definitions):** The standard gas tariff (most likely residential gas, since nearly all clients are category 11).
* **Investigation relevance:** Being on this tariff simply means the client has a gas meter. Use it to route gas-specific checks; it is not evidence of a professional or industrial activity.

### [CODE-tarif_type-10] tarif_type = 10
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 10` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 276,210 invoice rows (6.17% of 4,476,749); 276,210 ELEC / 0 GAZ (100.0% ELEC); 4.9% of these rows have zero total consumption; seen 1977-2019. Training clients carrying it at least once: 10,272, of which fraud: 760 (7.40%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-tarif_type-15] tarif_type = 15
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 15` | `counter_type: ELEC` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 72,422 invoice rows (1.62% of 4,476,749); 72,422 ELEC / 0 GAZ (100.0% ELEC); 66.7% of these rows have zero total consumption; seen 1980-2019. Training clients carrying it at least once: 3,078, of which fraud: 264 (8.58%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Commercial / professional low-voltage electricity tariff.
* **Investigation relevance:** Relevant to tariff-misclassification checks ([TYP-07]) only if the assumed meaning holds. The high share of zero-consumption rows needs explaining before it is read as a fraud signal.

### [CODE-tarif_type-45] tarif_type = 45
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 45` | `counter_type: GAZ` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 17,552 invoice rows (0.39% of 4,476,749); 0 ELEC / 17,552 GAZ (0.0% ELEC); 9.3% of these rows have zero total consumption; seen 2000-2019. Training clients carrying it at least once: 322, of which fraud: 52 (16.15%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** A commercial or professional gas tariff.
* **Investigation relevance:** Small gas group with a training fraud rate well above the base rate (based on a few hundred clients); interpret with care.

### [CODE-tarif_type-13] tarif_type = 13
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 13` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 11,656 invoice rows (0.26% of 4,476,749); 11,656 ELEC / 0 GAZ (100.0% ELEC); 32.5% of these rows have zero total consumption; seen 1982-2019. Training clients carrying it at least once: 570, of which fraud: 57 (10.00%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-tarif_type-14] tarif_type = 14
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 14` | `counter_type: ELEC` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 11,611 invoice rows (0.26% of 4,476,749); 11,611 ELEC / 0 GAZ (100.0% ELEC); 44.9% of these rows have zero total consumption; seen 1982-2019. Training clients carrying it at least once: 564, of which fraud: 57 (10.11%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Agricultural (pumping) electricity tariff.
* **Investigation relevance:** Small group; a high share of zero-consumption rows may reflect seasonal use rather than fraud.

### [CODE-tarif_type-12] tarif_type = 12
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 12` | `counter_type: ELEC` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 11,345 invoice rows (0.25% of 4,476,749); 11,345 ELEC / 0 GAZ (100.0% ELEC); 70.1% of these rows have zero total consumption; seen 2006-2019. Training clients carrying it at least once: 522, of which fraud: 57 (10.92%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Residential dual-rate (day/night) electricity tariff.
* **Investigation relevance:** Small group with a training fraud rate above the base rate; interpret with care given its size.

### [CODE-tarif_type-29] tarif_type = 29
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 29` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 10,090 invoice rows (0.23% of 4,476,749); 10,090 ELEC / 0 GAZ (100.0% ELEC); 40.7% of these rows have zero total consumption; seen 1979-2019. Training clients carrying it at least once: 542, of which fraud: 45 (8.30%; base rate 5.58%). An earlier draft of this code book called it a domestic natural-gas tariff; the data contradict that, since every row is ELEC.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** An electricity tariff with no known meaning; no specific investigation use.

### [CODE-tarif_type-9] tarif_type = 9
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 9` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 6,039 invoice rows (0.13% of 4,476,749); 6,039 ELEC / 0 GAZ (100.0% ELEC); 7.1% of these rows have zero total consumption; seen 2011-2019. Training clients carrying it at least once: 567, of which fraud: 45 (7.94%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-tarif_type-21] tarif_type = 21
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 21` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 104 invoice rows (0.0023% of 4,476,749); 104 ELEC / 0 GAZ (100.0% ELEC); 16.3% of these rows have zero total consumption; seen 1991-2019. Training clients carrying it at least once: 11, of which fraud: 3 (27.27%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-tarif_type-8] tarif_type = 8
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 8` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 43 invoice rows (0.0010% of 4,476,749); 43 ELEC / 0 GAZ (100.0% ELEC); 97.7% of these rows have zero total consumption; seen 2011-2012. Training clients carrying it at least once: 20, of which fraud: 1 (5.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-tarif_type-30] tarif_type = 30
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 30` | `counter_type: GAZ` | `jurisdiction: TN`
* **Observed in data:** 35 invoice rows (0.0008% of 4,476,749); 0 ELEC / 35 GAZ (0.0% ELEC); 45.7% of these rows have zero total consumption; seen 1981-1989. Training clients carrying it at least once: 10, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-tarif_type-24] tarif_type = 24
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 24` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 9 invoice rows (0.0002% of 4,476,749); 9 ELEC / 0 GAZ (100.0% ELEC); 100.0% of these rows have zero total consumption; seen 2006-2009. No training client carries this value, so there is no fraud rate.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-tarif_type-18] tarif_type = 18
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 18` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 4 invoice rows (<0.0001% of 4,476,749); 4 ELEC / 0 GAZ (100.0% ELEC); 25.0% of these rows have zero total consumption; seen 2007-2008. Training clients carrying it at least once: 1, of which fraud: 1 (100.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-tarif_type-27] tarif_type = 27
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 27` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 1 invoice row (<0.0001% of 4,476,749); 1 ELEC / 0 GAZ (100.0% ELEC); 100.0% of these rows have zero total consumption; seen 2009-2009. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-tarif_type-42] tarif_type = 42
* **Metadata:** `doc: codebook` | `field: tarif_type` | `value: 42` | `counter_type: GAZ` | `jurisdiction: TN`
* **Observed in data:** 1 invoice row (<0.0001% of 4,476,749); 0 ELEC / 1 GAZ (0.0% ELEC); 100.0% of these rows have zero total consumption; seen 2009-2009. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

## counter_code (meter code)

The field name suggests a meter model or family. Every value is used with only one counter type. No per-value meanings are assumed, so these entries record only what the data show.

### [CODE-counter_code-203] counter_code = 203
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 203` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 1,516,836 invoice rows (33.88% of 4,476,749); 1,516,836 ELEC / 0 GAZ (100.0% ELEC); 4.4% of these rows have zero total consumption; seen 1977-2019. Training clients carrying it at least once: 49,780, of which fraud: 3,158 (6.34%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-5] counter_code = 5
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 5` | `counter_type: GAZ` | `jurisdiction: TN`
* **Observed in data:** 1,352,035 invoice rows (30.20% of 4,476,749); 0 ELEC / 1,352,035 GAZ (0.0% ELEC); 17.2% of these rows have zero total consumption; seen 1981-2019. Training clients carrying it at least once: 48,240, of which fraud: 3,398 (7.04%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-207] counter_code = 207
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 207` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 555,628 invoice rows (12.41% of 4,476,749); 555,628 ELEC / 0 GAZ (100.0% ELEC); 7.3% of these rows have zero total consumption; seen 1983-2019. Training clients carrying it at least once: 26,833, of which fraud: 1,559 (5.81%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-413] counter_code = 413
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 413` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 378,917 invoice rows (8.46% of 4,476,749); 378,917 ELEC / 0 GAZ (100.0% ELEC); 9.1% of these rows have zero total consumption; seen 1978-2019. Training clients carrying it at least once: 15,357, of which fraud: 1,031 (6.71%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-202] counter_code = 202
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 202` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 343,251 invoice rows (7.67% of 4,476,749); 343,251 ELEC / 0 GAZ (100.0% ELEC); 17.5% of these rows have zero total consumption; seen 1977-2019. Training clients carrying it at least once: 12,802, of which fraud: 1,000 (7.81%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-420] counter_code = 420
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 420` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 98,273 invoice rows (2.20% of 4,476,749); 98,273 ELEC / 0 GAZ (100.0% ELEC); 8.7% of these rows have zero total consumption; seen 1977-2019. Training clients carrying it at least once: 4,695, of which fraud: 363 (7.73%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-410] counter_code = 410
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 410` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 69,080 invoice rows (1.54% of 4,476,749); 69,080 ELEC / 0 GAZ (100.0% ELEC); 11.4% of these rows have zero total consumption; seen 1980-2019. Training clients carrying it at least once: 2,720, of which fraud: 174 (6.40%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-433] counter_code = 433
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 433` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 34,447 invoice rows (0.77% of 4,476,749); 34,447 ELEC / 0 GAZ (100.0% ELEC); 6.8% of these rows have zero total consumption; seen 1979-2019. Training clients carrying it at least once: 1,392, of which fraud: 127 (9.12%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-10] counter_code = 10
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 10` | `counter_type: GAZ` | `jurisdiction: TN`
* **Observed in data:** 27,744 invoice rows (0.62% of 4,476,749); 0 ELEC / 27,744 GAZ (0.0% ELEC); 16.8% of these rows have zero total consumption; seen 1981-2019. Training clients carrying it at least once: 1,074, of which fraud: 102 (9.50%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-442] counter_code = 442
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 442` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 17,050 invoice rows (0.38% of 4,476,749); 17,050 ELEC / 0 GAZ (100.0% ELEC); 5.0% of these rows have zero total consumption; seen 1993-2019. Training clients carrying it at least once: 654, of which fraud: 65 (9.94%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-25] counter_code = 25
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 25` | `counter_type: GAZ` | `jurisdiction: TN`
* **Observed in data:** 14,934 invoice rows (0.33% of 4,476,749); 0 ELEC / 14,934 GAZ (0.0% ELEC); 8.6% of these rows have zero total consumption; seen 1995-2019. Training clients carrying it at least once: 278, of which fraud: 43 (15.47%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-407] counter_code = 407
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 407` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 13,768 invoice rows (0.31% of 4,476,749); 13,768 ELEC / 0 GAZ (100.0% ELEC); 18.8% of these rows have zero total consumption; seen 1979-2019. Training clients carrying it at least once: 546, of which fraud: 77 (14.10%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-204] counter_code = 204
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 204` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 12,427 invoice rows (0.28% of 4,476,749); 12,427 ELEC / 0 GAZ (100.0% ELEC); 8.1% of these rows have zero total consumption; seen 2002-2019. Training clients carrying it at least once: 701, of which fraud: 106 (15.12%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-453] counter_code = 453
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 453` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 8,290 invoice rows (0.19% of 4,476,749); 8,290 ELEC / 0 GAZ (100.0% ELEC); 4.9% of these rows have zero total consumption; seen 1993-2019. Training clients carrying it at least once: 263, of which fraud: 18 (6.84%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-201] counter_code = 201
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 201` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 7,672 invoice rows (0.17% of 4,476,749); 7,672 ELEC / 0 GAZ (100.0% ELEC); 17.8% of these rows have zero total consumption; seen 1978-2019. Training clients carrying it at least once: 408, of which fraud: 27 (6.62%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-467] counter_code = 467
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 467` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 7,302 invoice rows (0.16% of 4,476,749); 7,302 ELEC / 0 GAZ (100.0% ELEC); 7.3% of these rows have zero total consumption; seen 1985-2019. Training clients carrying it at least once: 238, of which fraud: 12 (5.04%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-506] counter_code = 506
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 506` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 3,389 invoice rows (0.08% of 4,476,749); 3,389 ELEC / 0 GAZ (100.0% ELEC); 6.1% of these rows have zero total consumption; seen 1997-2019. Training clients carrying it at least once: 97, of which fraud: 6 (6.19%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-483] counter_code = 483
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 483` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 2,830 invoice rows (0.06% of 4,476,749); 2,830 ELEC / 0 GAZ (100.0% ELEC); 4.9% of these rows have zero total consumption; seen 1997-2019. Training clients carrying it at least once: 84, of which fraud: 2 (2.38%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-214] counter_code = 214
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 214` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 2,643 invoice rows (0.06% of 4,476,749); 2,643 ELEC / 0 GAZ (100.0% ELEC); 9.8% of these rows have zero total consumption; seen 2008-2019. Training clients carrying it at least once: 201, of which fraud: 24 (11.94%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-40] counter_code = 40
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 40` | `counter_type: GAZ` | `jurisdiction: TN`
* **Observed in data:** 2,482 invoice rows (0.06% of 4,476,749); 0 ELEC / 2,482 GAZ (0.0% ELEC); 13.6% of these rows have zero total consumption; seen 1989-2019. Training clients carrying it at least once: 46, of which fraud: 8 (17.39%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-532] counter_code = 532
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 532` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 1,982 invoice rows (0.04% of 4,476,749); 1,982 ELEC / 0 GAZ (100.0% ELEC); 3.8% of these rows have zero total consumption; seen 1987-2019. Training clients carrying it at least once: 44, of which fraud: 0 (0.00%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-565] counter_code = 565
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 565` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 1,082 invoice rows (0.02% of 4,476,749); 1,082 ELEC / 0 GAZ (100.0% ELEC); 4.9% of these rows have zero total consumption; seen 2002-2019. Training clients carrying it at least once: 29, of which fraud: 1 (3.45%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-403] counter_code = 403
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 403` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 1,070 invoice rows (0.02% of 4,476,749); 1,070 ELEC / 0 GAZ (100.0% ELEC); 16.4% of these rows have zero total consumption; seen 1984-2019. Training clients carrying it at least once: 62, of which fraud: 8 (12.90%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-450] counter_code = 450
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 450` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 1,059 invoice rows (0.02% of 4,476,749); 1,059 ELEC / 0 GAZ (100.0% ELEC); 9.6% of these rows have zero total consumption; seen 1978-2019. Training clients carrying it at least once: 39, of which fraud: 2 (5.13%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-600] counter_code = 600
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 600` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 807 invoice rows (0.02% of 4,476,749); 807 ELEC / 0 GAZ (100.0% ELEC); 7.3% of these rows have zero total consumption; seen 2005-2019. Training clients carrying it at least once: 23, of which fraud: 2 (8.70%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-210] counter_code = 210
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 210` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 789 invoice rows (0.02% of 4,476,749); 789 ELEC / 0 GAZ (100.0% ELEC); 5.6% of these rows have zero total consumption; seen 2005-2019. Training clients carrying it at least once: 59, of which fraud: 10 (16.95%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-102] counter_code = 102
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 102` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 313 invoice rows (0.0070% of 4,476,749); 313 ELEC / 0 GAZ (100.0% ELEC); 5.8% of these rows have zero total consumption; seen 1977-2011. Training clients carrying it at least once: 76, of which fraud: 0 (0.00%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-227] counter_code = 227
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 227` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 178 invoice rows (0.0040% of 4,476,749); 178 ELEC / 0 GAZ (100.0% ELEC); 30.3% of these rows have zero total consumption; seen 2010-2019. Training clients carrying it at least once: 3, of which fraud: 1 (33.33%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-101] counter_code = 101
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 101` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 74 invoice rows (0.0017% of 4,476,749); 74 ELEC / 0 GAZ (100.0% ELEC); 17.6% of these rows have zero total consumption; seen 1977-2011. Training clients carrying it at least once: 22, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-65] counter_code = 65
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 65` | `counter_type: GAZ` | `jurisdiction: TN`
* **Observed in data:** 74 invoice rows (0.0017% of 4,476,749); 0 ELEC / 74 GAZ (0.0% ELEC); 20.3% of these rows have zero total consumption; seen 2005-2019. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-16] counter_code = 16
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 16` | `counter_type: GAZ` | `jurisdiction: TN`
* **Observed in data:** 73 invoice rows (0.0016% of 4,476,749); 0 ELEC / 73 GAZ (0.0% ELEC); 6.8% of these rows have zero total consumption; seen 2007-2019. Training clients carrying it at least once: 2, of which fraud: 1 (50.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-310] counter_code = 310
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 310` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 51 invoice rows (0.0011% of 4,476,749); 51 ELEC / 0 GAZ (100.0% ELEC); 21.6% of these rows have zero total consumption; seen 1979-1991. Training clients carrying it at least once: 17, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-307] counter_code = 307
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 307` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 50 invoice rows (0.0011% of 4,476,749); 50 ELEC / 0 GAZ (100.0% ELEC); 20.0% of these rows have zero total consumption; seen 1981-2011. Training clients carrying it at least once: 20, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-305] counter_code = 305
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 305` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 44 invoice rows (0.0010% of 4,476,749); 44 ELEC / 0 GAZ (100.0% ELEC); 11.4% of these rows have zero total consumption; seen 1978-2010. Training clients carrying it at least once: 13, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-222] counter_code = 222
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 222` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 42 invoice rows (0.0009% of 4,476,749); 42 ELEC / 0 GAZ (100.0% ELEC); 7.1% of these rows have zero total consumption; seen 2003-2019. Training clients carrying it at least once: 5, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-317] counter_code = 317
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 317` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 16 invoice rows (0.0004% of 4,476,749); 16 ELEC / 0 GAZ (100.0% ELEC); 25.0% of these rows have zero total consumption; seen 1979-1989. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-303] counter_code = 303
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 303` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 5 invoice rows (0.0001% of 4,476,749); 5 ELEC / 0 GAZ (100.0% ELEC); 40.0% of these rows have zero total consumption; seen 1986-2011. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-333] counter_code = 333
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 333` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 4 invoice rows (<0.0001% of 4,476,749); 4 ELEC / 0 GAZ (100.0% ELEC); 0.0% of these rows have zero total consumption; seen 1989-1990. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-367] counter_code = 367
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 367` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 3 invoice rows (<0.0001% of 4,476,749); 3 ELEC / 0 GAZ (100.0% ELEC); 0.0% of these rows have zero total consumption; seen 1991-1991. No training client carries this value, so there is no fraud rate.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

### [CODE-counter_code-325] counter_code = 325
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 325` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 1 invoice row (<0.0001% of 4,476,749); 1 ELEC / 0 GAZ (100.0% ELEC); 0.0% of these rows have zero total consumption; seen 1986-1986. Training clients carrying it at least once: 1, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret.
* **Meaning:** Unknown. STEG has not published code definitions and the data give no basis for an assumption.
* **Investigation relevance:** No known meaning; no specific investigation use beyond noting the value in the client snapshot.

## counter_code: invalid or rare values

The same 34 invoice rows (5 clients; 33 ELEC rows and 1 GAZ row) carry an odd `counter_statue` (46, 618, 769, 269375 or 420), an odd `reading_remarque` (203, 207, 413 or 5), and `counter_code` 0 or 1. In these rows `reading_remarque` holds a number that looks like a real `counter_code`, `counter_coefficient` holds what looks like a remark (6, 8 or 9) and the index and consumption fields look moved along as well. That suggests each row was shifted by one column during data entry or export. `counter_code` 0 (33 rows) and 1 (1 row) occur only in these shifted rows, so they are not real meter codes. They are kept here so that every raw value has an entry.

### [CODE-counter_code-0] counter_code = 0
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 0` | `counter_type: ELEC` | `jurisdiction: TN`
* **Observed in data:** 33 invoice rows (0.0007% of 4,476,749); 33 ELEC / 0 GAZ (100.0% ELEC); 0.0% of these rows have zero total consumption; seen 1996-2019. Training clients carrying it at least once: 2, of which fraud: 0 (0.00%; base rate 5.58%). Too few clients to interpret. Row appears column-shifted (see the note above), so its consumption and index fields are unreliable.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

### [CODE-counter_code-1] counter_code = 1
* **Metadata:** `doc: codebook` | `field: counter_code` | `value: 1` | `counter_type: GAZ` | `jurisdiction: TN`
* **Observed in data:** 1 invoice row (<0.0001% of 4,476,749); 0 ELEC / 1 GAZ (0.0% ELEC); 0.0% of these rows have zero total consumption; seen 2012-2012. No training client carries this value, so there is no fraud rate. Row appears column-shifted (see the note above), so its consumption and index fields are unreliable.
* **Meaning:** Invalid value outside the field's normal set (see the note above).
* **Investigation relevance:** Treat as a data-entry error, not as a real code. Do not cite it as evidence of anything.

## client_catg (client category)

A client-level field from `client.parquet`, not an invoice field.

### [CODE-client_catg-11] client_catg = 11
* **Metadata:** `doc: codebook` | `field: client_catg` | `value: 11` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 131,494 clients (97.05% of 135,493); their invoices split 2,956,888 ELEC / 1,348,000 GAZ (68.7% ELEC); 10.4% of their invoice rows have zero total consumption. 105,208 are training clients, of whom 5,751 are fraud (5.47%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Residential (domestic) customer.
* **Investigation relevance:** Reference category for peer comparison; fraud rate close to the base rate.

### [CODE-client_catg-12] client_catg = 12
* **Metadata:** `doc: codebook` | `field: client_catg` | `value: 12` | `counter_type: both` | `jurisdiction: TN` | `assumed: true`
* **Observed in data:** 2,321 clients (1.71% of 135,493); their invoices split 48,711 ELEC / 19,218 GAZ (71.7% ELEC); 18.4% of their invoice rows have zero total consumption. 1,836 are training clients, of whom 70 are fraud (3.81%; base rate 5.58%).
* **Assumed meaning (unverified — STEG has not published code definitions):** Commercial / professional customer.
* **Investigation relevance:** Training fraud rate is below the base rate. A residential-looking consumption profile in this category is not by itself suspicious.

### [CODE-client_catg-51] client_catg = 51
* **Metadata:** `doc: codebook` | `field: client_catg` | `value: 51` | `counter_type: both` | `jurisdiction: TN`
* **Observed in data:** 1,678 clients (1.24% of 135,493); their invoices split 73,807 ELEC / 30,125 GAZ (71.0% ELEC); 6.6% of their invoice rows have zero total consumption. 1,350 are training clients, of whom 232 are fraud (17.19%; base rate 5.58%).
* **Meaning:** Unknown. STEG has not published category definitions and the data give no basis for an assumption.
* **Investigation relevance:** Training fraud rate is about three times the base rate (over a thousand training clients), a strong association in this data. Its meaning is unknown, so a brief may report the rate but should not describe the category (for example as 'industrial').
