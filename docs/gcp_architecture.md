# GCP Architecture — Healthcare Knowledge Graph Platform

## Overview

Production-grade deployment of the Healthcare Knowledge Graph on Google Cloud Platform, integrating GCP-native services for ingestion, storage, transformation, semantic querying, and ML inference.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES (Payer / Provider Systems)                 │
│  HL7 FHIR APIs  │  EHR/EMR Exports  │  Claims Files (837/835)  │  ADT Feeds │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Cloud Pub/Sub  │  ← Real-time event streaming
                    │  (FHIR events,  │     (encounters, claims, orders)
                    │   ADT messages) │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────────┐
         │                   │                       │
┌────────▼────────┐ ┌────────▼────────┐  ┌──────────▼──────────┐
│  Cloud Dataflow  │ │  Cloud Storage  │  │    Cloud Healthcare  │
│  (Apache Beam)  │ │  (GCS)          │  │    API (FHIR Store)  │
│                 │ │  - Raw FHIR JSON│  │                      │
│  - Parse FHIR   │ │  - Claims files │  │  - FHIR R4 server    │
│  - Validate     │ │  - Ontology TTL │  │  - Patient/Encounter │
│  - Normalise    │ │  - GraphML      │  │  - Claim resources   │
│  - Map to OWL   │ └────────┬────────┘  └──────────┬──────────┘
└────────┬────────┘          │                      │
         │                   └──────────┬───────────┘
         │                              │
┌────────▼──────────────────────────────▼──────────────────────────────────┐
│                         BigQuery (Data Warehouse)                         │
│                                                                           │
│  Dataset: healthcare_kg                                                   │
│  ├── raw_claims          — 837/835 claim transactions                     │
│  ├── fhir_patients       — Patient demographics (de-identified)           │
│  ├── fhir_encounters     — Encounter + diagnosis + procedure records      │
│  ├── fhir_practitioners  — Provider NPI, specialty, affiliations          │
│  ├── kg_triples          — Subject / Predicate / Object triple store      │
│  ├── entity_pagerank     — Pre-computed graph centrality scores           │
│  └── fraud_features      — ML-ready feature table                        │
└────────────┬──────────────────────────────────────────────────────────────┘
             │
    ┌────────┴──────────────────────────────────────────────┐
    │                                                       │
┌───▼──────────────────┐              ┌────────────────────▼───────────────┐
│    Dataproc           │              │         Vertex AI                  │
│   (Apache Spark)      │              │                                    │
│                       │              │  ┌─────────────────────────────┐  │
│  - graph_analytics.py │              │  │  AutoML / Custom Training   │  │
│  - PageRank on        │              │  │  - Fraud detector (GBT)     │  │
│    provider network   │              │  │  - Risk stratification      │  │
│  - Community detect.  │              │  │  - Care gap prediction      │  │
│  - Patient cohorts    │              │  └─────────────────────────────┘  │
│  - Feature engineering│              │  ┌─────────────────────────────┐  │
│  (PySpark + GraphX)   │              │  │  Vertex AI Model Registry   │  │
└───┬───────────────────┘              │  │  Vertex AI Endpoints        │  │
    │                                  └──┼─────────────────────────────┘  │
    │                                     └────────────────────────────────┘
    │
┌───▼──────────────────────────────────────────────────────────────────────┐
│              Knowledge Graph Store (Semantic Layer)                       │
│                                                                           │
│  Option A: Stardog (SPARQL 1.1 + OWL 2 reasoning)                        │
│            Deployed on GKE — supports SPARQL queries in sparql_queries.py │
│                                                                           │
│  Option B: Neo4j on GCP Marketplace (Cypher queries, graph algorithms)   │
│            Nodes: Patient, Provider, Claim, Condition, Procedure          │
│            Relationships: HAS_CONDITION, BILLED_FOR, REFERRED_TO          │
│                                                                           │
│  Option C: RDFLib on Dataproc (current implementation — file-based TTL)  │
└──────────────────────────────────────────────────────────────────────────┘
             │
┌────────────▼─────────────────────────────────────────────────────────────┐
│                    Looker Studio / Data Studio Dashboard                   │
│                                                                           │
│  - Population health dashboard (risk tiers, chronic disease prevalence)  │
│  - Provider network graph visualisation (D3.js / Looker)                 │
│  - Fraud alert queue (flagged claims, anomaly scores)                    │
│  - Care gap closure tracking (HEDIS measures)                            │
│  - Denial reason analysis and revenue recovery pipeline                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## GCP Service Mapping

| Component | GCP Service | Purpose |
|-----------|-------------|---------|
| Real-time ingestion | Cloud Pub/Sub | FHIR event streaming from EHR systems |
| Batch ETL | Cloud Dataflow (Apache Beam) | Claims file parsing, FHIR normalisation |
| Raw storage | Cloud Storage (GCS) | Landing zone for claims, FHIR bundles, ontology files |
| FHIR server | Cloud Healthcare API | Standards-compliant FHIR R4 resource storage |
| Data warehouse | BigQuery | Claims analytics, KG triple store, feature tables |
| Graph compute | Dataproc (PySpark + GraphX) | PageRank, community detection, graph feature engineering |
| ML training | Vertex AI Training | Fraud detector, risk model, care gap prediction |
| ML serving | Vertex AI Endpoints | Real-time fraud scoring (sub-100ms) |
| KG store | Stardog on GKE / Neo4j Marketplace | SPARQL querying, OWL reasoning |
| Orchestration | Cloud Composer (Airflow) | Daily pipeline scheduling |
| Monitoring | Cloud Monitoring + Logging | Pipeline SLAs, model drift detection |
| Security / PHI | Cloud DLP + VPC-SC | De-identification, audit logging (HIPAA) |

---

## BigQuery — KG Triple Store Schema

```sql
-- Triple store in BigQuery (scales to billions of triples)
CREATE TABLE healthcare_kg.kg_triples (
    subject     STRING NOT NULL,  -- e.g. "hkg:patient_PAT_A1B2C3D4"
    predicate   STRING NOT NULL,  -- e.g. "hkg:hasCondition"
    object      STRING NOT NULL,  -- e.g. "hkg:cond_E11_PAT_A1B2C3D4"
    graph_name  STRING,           -- named graph (e.g. "claims_2024Q1")
    source      STRING,           -- originating system
    loaded_at   TIMESTAMP
);

-- Example SPARQL-equivalent in BigQuery SQL:
-- "Find all patients with diabetes AND hypertension"
SELECT DISTINCT s1.subject AS patient
FROM healthcare_kg.kg_triples s1
JOIN healthcare_kg.kg_triples s2 ON s1.subject = s2.subject
JOIN healthcare_kg.kg_triples icd1 ON s1.object = icd1.subject
JOIN healthcare_kg.kg_triples icd2 ON s2.object = icd2.subject
WHERE s1.predicate = 'hkg:hasCondition'
  AND s2.predicate = 'hkg:hasCondition'
  AND icd1.predicate = 'hkg:codeValue' AND icd1.object LIKE 'E11%'
  AND icd2.predicate = 'hkg:codeValue' AND icd2.object LIKE 'I10%';
```

---

## HIPAA Compliance Notes

- All patient identifiers are de-identified via **Cloud DLP** before storage in BigQuery
- PHI access controlled via **VPC Service Controls** and IAM policies
- **Audit logs** captured in Cloud Logging for all KG queries touching PHI
- **Encryption at rest and in transit** enforced via CMEK (Customer-Managed Encryption Keys)
- FHIR store configured with **consent management** per FHIR R4 Consent resource

---

## Data Pipeline DAG (Cloud Composer / Airflow)

```
daily_healthcare_kg_pipeline
├── ingest_claims_files          (GCS → BigQuery raw_claims)
├── fetch_fhir_resources         (Cloud Healthcare API → GCS)
├── run_fhir_normalisation       (Dataflow beam job)
├── populate_rdf_graph           (kg_builder.py on Dataproc)
├── run_graph_analytics          (graph_analytics.py on Dataproc)
├── refresh_entity_pagerank      (BigQuery → entity_pagerank table)
├── score_fraud_candidates       (Vertex AI batch prediction)
├── refresh_looker_extracts      (BigQuery → Looker Studio)
└── alert_care_gaps              (SPARQL → care coordination team)
```
