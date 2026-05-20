# Healthcare Payer/Provider Knowledge Graph

> End-to-end Knowledge Graph platform for US Healthcare — OWL ontology aligned with FHIR R4, ICD-10-CM, CPT, and SNOMED-CT. Covers care coordination, population health, fraud detection, and provider network analytics — with a GCP deployment architecture.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![RDFLib](https://img.shields.io/badge/RDFLib-7.x-green.svg)](https://rdflib.readthedocs.io)
[![OWL](https://img.shields.io/badge/Ontology-OWL%202%20%2F%20Turtle-orange.svg)](https://www.w3.org/TR/owl2-overview/)
[![FHIR](https://img.shields.io/badge/Standard-FHIR%20R4-red.svg)](https://hl7.org/fhir/R4/)

---

## What This Project Demonstrates

| Skill Area | Implementation |
|------------|----------------|
| **Ontology & Knowledge Modeling** | OWL 2 ontology in Turtle — 14 classes, 20 object properties, 18 datatype properties, SNOMED/ICD/CPT seed individuals, disjointness axioms |
| **Healthcare Standards Alignment** | FHIR R4 resource mapping (Patient, Practitioner, Claim, Encounter, Condition, Procedure, Coverage), ICD-10-CM, CPT, SNOMED-CT, LOINC, RxNorm |
| **Knowledge Graph Engineering** | RDFLib triple population, SPARQL query library (10 production queries), NetworkX graph export |
| **GCP Architecture** | Pub/Sub → Dataflow → BigQuery → Dataproc → Vertex AI pipeline design; Stardog/Neo4j KG store options; BigQuery triple store schema |
| **Graph Analytics** | Provider network PageRank, Louvain community detection, referral concentration (Gini), patient risk stratification |
| **Fraud / FWA Detection** | Graph-feature-based classifier; billing z-score, betweenness centrality, referral ring detection |
| **Data Science** | Synthetic FHIR data generation, HCC risk scoring, care gap analysis, payer mix analytics, denial reason breakdown |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              Ontology Layer (OWL / Turtle)                    │
│  ontology/healthcare_kg.ttl                                   │
│  Classes: Patient · Practitioner · Encounter · Claim ·        │
│           Condition · Procedure · Coverage · FraudAlert       │
│  Aligned: FHIR R4 · ICD-10-CM · CPT · SNOMED-CT · LOINC     │
└───────────────────────────┬──────────────────────────────────┘
                            │ instantiated by
┌───────────────────────────▼──────────────────────────────────┐
│              Knowledge Graph (RDF Triple Store)               │
│  kg/healthcare_kg_populated.ttl                               │
│  ~500 patients · ~80 providers · ~3K encounters               │
│  ~3K claims · referral network edges · fraud alerts           │
└──────┬─────────────────────────────────────┬─────────────────┘
       │ SPARQL queries                      │ NetworkX export
┌──────▼────────────────┐          ┌─────────▼─────────────────┐
│  src/sparql_queries.py│          │  src/graph_analytics.py   │
│  10 use-case queries: │          │  - Provider PageRank       │
│  - care gaps          │          │  - Community detection     │
│  - fraud alerts       │          │  - Referral concentration  │
│  - payer mix          │          │  - Patient risk tiers      │
│  - denial analysis    │          │  - Fraud feature matrix    │
│  - comorbidities      │          └─────────┬─────────────────┘
└───────────────────────┘                    │
                                   ┌─────────▼─────────────────┐
                                   │  src/fraud_detector.py    │
                                   │  Graph-boosted FWA model  │
                                   │  Precision · Recall · AUC │
                                   └───────────────────────────┘
```

---

## Project Structure

```
healthcare-knowledge-graph/
├── ontology/
│   └── healthcare_kg.ttl          # OWL 2 ontology (Turtle format)
├── src/
│   ├── data_generator.py          # Synthetic FHIR-aligned data (no real PHI)
│   ├── kg_builder.py              # RDFLib KG construction + NetworkX export
│   ├── sparql_queries.py          # 10 production SPARQL queries
│   ├── graph_analytics.py         # PageRank, community detection, risk tiers
│   └── fraud_detector.py          # Graph-feature FWA classifier
├── notebooks/
│   └── healthcare_kg_demo.ipynb   # End-to-end walkthrough
├── docs/
│   └── gcp_architecture.md        # Full GCP deployment design
├── data/                           # Generated (gitignored)
├── kg/                             # Generated (gitignored)
└── results/                        # Generated (gitignored)
```

---

## Ontology Design

The OWL ontology (`ontology/healthcare_kg.ttl`) defines a formal semantic model:

### Classes (14)
`Person` → `Patient`, `Practitioner`  
`Organization` → `Hospital`, `Clinic`, `Payer`, `PharmacyBenefit`  
`Encounter` → `InpatientEncounter`, `OutpatientEncounter`, `EmergencyEncounter`, `TelehealthEncounter`  
`Condition` → `ChronicCondition`, `AcuteCondition`  
`ClinicalCode` → `ICD10Code`, `CPTCode`, `SNOMEDCode`, `LOINCCode`, `RxNormCode`  
`Claim` → `ProfessionalClaim` (CMS-1500), `InstitutionalClaim` (UB-04)  
`FraudAlert`, `CareTeam`, `Coverage`, `Specialty`

### Key Object Properties
| Property | Type | Description |
|----------|------|-------------|
| `hasCondition` | — | Patient → Condition (ICD-10 coded) |
| `encounterDiagnosis` | — | Encounter → Condition |
| `encounterProcedure` | — | Encounter → Procedure (CPT coded) |
| `referredTo` | — | Practitioner → Practitioner (referral network edge) |
| `triggeredAlert` | — | Claim → FraudAlert |
| `snomedParent` | Transitive | SNOMED IS-A hierarchy |
| `relatedCondition` | Symmetric | Co-morbidity relationship |
| `hasCPT` / `hasICD10` / `hasSNOMED` | Subproperty of `codedAs` | Code cross-reference |

### Seed Individuals
- 8 ICD-10-CM codes (E11 diabetes, I10 hypertension, J44 COPD, I50 heart failure, N18 CKD, F32 depression, ...)
- 10 CPT codes (99213–99215 office visits, 93000 ECG, 45378 colonoscopy, ...)
- 5 SNOMED-CT concepts with IS-A hierarchy
- 10 medical specialties (NUCC taxonomy)
- 10 MIND-aligned topic seed individuals

---

## SPARQL Query Library

Ten production-ready queries covering core payer/provider use cases:

| Query | Use Case |
|-------|----------|
| `high_risk_patients` | Population health — HCC risk > 1.5 with ≥2 chronic conditions |
| `care_gaps_diabetes_nephrology` | Care coordination — diabetics without specialist follow-up |
| `high_cost_inpatient` | Utilisation management — top cost inpatient encounters |
| `fraud_alerts_summary` | FWA detection — alert counts by fraud type |
| `top_billing_providers` | Provider profiling — upcoding screen |
| `denial_analysis` | Revenue integrity — denial reason breakdown |
| `payer_mix` | Analytics — claim volume + payment by payer |
| `comorbidity_pairs` | Population health — frequent diagnosis co-occurrence |
| `specialist_coverage` | Care coordination — heart failure without cardiology |
| `referral_concentration` | FWA — kickback ring detection |

---

## Graph Analytics

### Provider Network
- **PageRank** — identifies central providers in the referral ecosystem
- **Louvain community detection** — surfaces provider clusters (care networks or fraud rings)
- **Referral concentration (Gini coefficient)** — flags providers with disproportionate referral focus
- **Fraud ring labeling** — injected during data generation; validated by community membership

### Patient Risk Stratification
Three-tier segmentation (High / Medium / Low) using:
- HCC risk score (CMS Hierarchical Condition Categories model)
- Annual claim spend
- Chronic condition count

### Fraud Feature Matrix
13 features per claim combining claim-level signals with graph-derived provider metrics:

```
bill_allow_ratio          # upcoding signal
bill_amount_zscore        # anomaly vs provider baseline
provider_pagerank         # network centrality
provider_betweenness      # bridge role in referral network
provider_ref_concentration# Gini of referral targets
provider_is_fraud_ring    # community-level label
```

---

## Setup & Run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate synthetic data (500 patients, ~3K claims)
python src/data_generator.py --patients 500 --fraud-rate 0.04

# 3. Build the knowledge graph
python src/kg_builder.py

# 4. Run SPARQL queries
python src/sparql_queries.py

# 5. Graph analytics + fraud features
python src/graph_analytics.py

# 6. Train and evaluate fraud detector
python src/fraud_detector.py

# 7. Full walkthrough notebook
jupyter notebook notebooks/healthcare_kg_demo.ipynb
```

---

## GCP Deployment

See [`docs/gcp_architecture.md`](docs/gcp_architecture.md) for the full production architecture including:

- **Cloud Pub/Sub** → real-time FHIR event streaming from EHR/EMR systems
- **Cloud Dataflow (Apache Beam)** → claims normalisation and ontology mapping pipeline
- **BigQuery** → KG triple store at scale + analytics layer
- **Dataproc (PySpark + GraphX)** → distributed graph analytics
- **Vertex AI** → fraud model training, serving, and monitoring
- **Stardog / Neo4j on GCP** → SPARQL endpoint with OWL 2 reasoning
- **Cloud Healthcare API** → FHIR R4 store for PHI-compliant resource management
- **Cloud DLP + VPC-SC** → de-identification and HIPAA compliance

---

## Healthcare Standards Coverage

| Standard | Role in This Project |
|----------|---------------------|
| **FHIR R4** | Ontology class mapping (Patient, Practitioner, Claim, Encounter, Condition, Procedure, Coverage, ClaimResponse) |
| **ICD-10-CM** | Diagnosis coding on Condition individuals and Claim triples |
| **CPT** | Procedure coding on Encounter and Claim triples |
| **SNOMED-CT** | Clinical concept hierarchy (IS-A transitive property) |
| **LOINC** | Lab result observation coding (LabResult class) |
| **RxNorm** | Medication normalisation (MedicationRequest class) |
| **NUCC Taxonomy** | Provider specialty classification |
| **HCC** | Risk score model for patient stratification |
| **DRG** | Inpatient encounter grouping (drg datatype property) |

---

## Author

**Yogeshvar Reddy Kallam** — MS Computer Science, Penn State (Class of 2026)  
[yogeshvarreddykallam.github.io](https://yogeshvarreddykallam.github.io) · [GitHub](https://github.com/yogeshvarreddykallam)
