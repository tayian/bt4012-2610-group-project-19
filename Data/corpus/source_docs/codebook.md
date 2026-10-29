# STEG Utility Code Book (`data/corpus/codebook.md`)

**Document ID:** `DOC-02-CODEBOOK`  
**Author / Workstream:** Role D (LLM, RAG & Agent Lead)  
**Corpus Layer:** Alphanumeric Variable Decoding & Grounding  
**Status:** Outline & Variable Specifications (Week 1 Milestone)  

---

## 1. Overview & Architectural Role

In STEG billing histories, field events, meter statuses, and inspection notes are encoded as numerical categorical integers (e.g., `counter_statue == 5`, `reading_remarque == 9`). 

* **The Problem:** General dense embedding models (like `MiniLM`) often treat single-digit integer codes ambiguously or drift in high-dimensional vector space.
* **The Solution:** BM25 sparse keyword retrieval maps exact alphanumeric query tokens (`CODE-counter_statue-5`, `CODE-reading_remarque-9`) directly to these operational definitions.
* **Chunking Specification:** One chunk per code value, identified by `[CODE-<field>-<value>]`.

---

## 2. Code Categories & Master Index

```
Categorical Variables in STEG Billing Data:
├── Group 1: counter_statue (Meter Operational & Administrative State)
│   ├── [CODE-counter_statue-0]: Normal Active Service
│   ├── [CODE-counter_statue-1]: Disconnected / Cut-Off
│   ├── [CODE-counter_statue-2]: Meter Replaced / Exchanged
│   ├── [CODE-counter_statue-3]: Under Repair / Calibration
│   ├── [CODE-counter_statue-4]: Blocked / Mechanical Failure
│   └── [CODE-counter_statue-5]: Suspected Anomaly / Tampering Flag
│
├── Group 2: reading_remarque (Meter-Reading Observation Code)
│   ├── [CODE-reading_remarque-6]: Normal Reader Visual Reading
│   ├── [CODE-reading_remarque-7]: Customer Self-Declared Reading
│   ├── [CODE-reading_remarque-8]: Estimated / Historical Average Reading
│   └── [CODE-reading_remarque-9]: Inaccessible / Obstructed Counter
│
├── Group 3: tarif_type (Contractual Tariff Classification)
│   ├── [CODE-tarif_type-11]: Residential Low Voltage Electricity
│   ├── [CODE-tarif_type-12]: Residential Dual-Rate (Day/Night)
│   ├── [CODE-tarif_type-14]: Agricultural Pumping Tariff
│   ├── [CODE-tarif_type-15]: Commercial / Professional Low Voltage
│   ├── [CODE-tarif_type-29]: Domestic Natural Gas (Low Pressure)
│   ├── [CODE-tarif_type-40]: Small Industrial Power
│   └── [CODE-tarif_type-45]: Commercial Gas Supply
│
└── Group 4: counter_code (Meter Hardware & Technology Family)
    └── Technical codes distinguishing electromechanical Ferraris induction disk meters from solid-state electronic static meters.
```

---

## 3. Standard Code Chunk Template

Every entry in the code book follows this standardized schema:

```markdown
### [CODE-field-value] Field Name: Code Value
* **Metadata:** `doc: codebook` | `field: <field_name>` | `value: <int>` | `jurisdiction: TN`
* **Official Operational Meaning:** Formal STEG utility definition of the state.
* **Fraud Investigation Implication:** Why this state appears in high-risk customer records (e.g. repeated `statue 1` with positive consumption indicates illegal self-reconnection; repeated `remarque 9` indicates deliberate obstruction to prevent inspection).
* **Cross-Reference:** Linked typology ID (`[TYP-xx]`) and inspection protocol action.
```

