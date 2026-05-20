"""sparql_queries.py — SPARQL query library for the Healthcare Knowledge Graph.

Demonstrates semantic querying capabilities aligned with real payer/provider
use cases: care coordination, population health, utilisation management,
fraud detection, and provider network analysis.

All queries run against the populated RDF graph (kg/healthcare_kg_populated.ttl).

Usage:
    python src/sparql_queries.py                          # run all queries
    python src/sparql_queries.py --query care_gaps        # run specific query
    python src/sparql_queries.py --kg kg/healthcare_kg_populated.ttl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rdflib import Graph

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("sparql")

PREFIX = """
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX hkg:  <http://yogeshvarreddykallam.github.io/healthcare-kg#>
"""

# ─────────────────────────────────────────────────────────────────────
#  Query library
# ─────────────────────────────────────────────────────────────────────

QUERIES: dict[str, tuple[str, str]] = {}

# ── 1. Population Health — High-risk patients with multiple chronic conditions

QUERIES["high_risk_patients"] = (
    "High-risk patients (HCC risk > 1.5) with ≥2 chronic conditions",
    PREFIX + """
SELECT ?patientId ?riskScore (COUNT(?condition) AS ?conditionCount)
WHERE {
    ?patient  a             hkg:Patient ;
              hkg:patientId  ?patientId ;
              hkg:riskScore  ?riskScore ;
              hkg:hasCondition ?condition .
    ?condition a hkg:ChronicCondition .
    FILTER(?riskScore > 1.5)
}
GROUP BY ?patientId ?riskScore
HAVING (COUNT(?condition) >= 2)
ORDER BY DESC(?riskScore)
LIMIT 20
"""
)

# ── 2. Care Coordination — Patients with diabetes who lack nephrology follow-up

QUERIES["care_gaps_diabetes_nephrology"] = (
    "Care Gap: Diabetic patients (E11) without nephrology encounter",
    PREFIX + """
SELECT ?patientId ?riskScore
WHERE {
    ?patient  a             hkg:Patient ;
              hkg:patientId  ?patientId ;
              hkg:riskScore  ?riskScore ;
              hkg:hasCondition ?cond .
    ?cond  hkg:hasICD10 ?icd .
    ?icd   hkg:codeValue ?code .
    FILTER(STRSTARTS(?code, "E11"))

    FILTER NOT EXISTS {
        ?enc hkg:forPatient ?patient ;
             hkg:renderingProvider ?prov .
        ?prov hkg:hasSpecialty ?spec .
        ?spec rdfs:label "Endocrinology" .
    }
}
ORDER BY DESC(?riskScore)
LIMIT 25
"""
)

# ── 3. Utilisation Management — High-cost inpatient encounters

QUERIES["high_cost_inpatient"] = (
    "Utilisation Management: Top 15 inpatient encounters by billed amount",
    PREFIX + """
SELECT ?claimId ?patientId ?billedAmt ?paidAmt ?icd10 ?status
WHERE {
    ?claim  a               hkg:InstitutionalClaim ;
            hkg:claimId      ?claimId ;
            hkg:claimAmount  ?billedAmt ;
            hkg:paidAmount   ?paidAmt ;
            hkg:claimStatus  ?status ;
            hkg:claimForPatient ?pat ;
            hkg:claimDiagnosisCode ?icd .
    ?pat    hkg:patientId  ?patientId .
    ?icd    hkg:codeValue  ?icd10 .
    FILTER(?billedAmt > 5000)
}
ORDER BY DESC(?billedAmt)
LIMIT 15
"""
)

# ── 4. Fraud Detection — Claims with fraud alerts, grouped by alert type

QUERIES["fraud_alerts_summary"] = (
    "Fraud / Waste / Abuse: Claims with anomaly alerts by type",
    PREFIX + """
SELECT ?alertType (COUNT(?alert) AS ?alertCount)
         (SUM(?billedAmt) AS ?totalBilled)
WHERE {
    ?claim   a                hkg:Claim ;
             hkg:claimAmount  ?billedAmt ;
             hkg:triggeredAlert ?alert .
    ?alert   hkg:alertType   ?alertType .
}
GROUP BY ?alertType
ORDER BY DESC(?alertCount)
"""
)

# ── 5. Provider Network — Top billing providers (potential upcoding screen)

QUERIES["top_billing_providers"] = (
    "Provider Profiling: Top 10 providers by total billed amount",
    PREFIX + """
SELECT ?npi ?providerName
       (COUNT(?claim) AS ?claimCount)
       (SUM(?billed) AS ?totalBilled)
       (AVG(?billed) AS ?avgBilled)
WHERE {
    ?claim  a              hkg:Claim ;
            hkg:claimAmount ?billed ;
            hkg:billingProvider ?prov .
    ?prov   hkg:npi        ?npi ;
            hkg:providerName ?providerName .
}
GROUP BY ?npi ?providerName
ORDER BY DESC(?totalBilled)
LIMIT 10
"""
)

# ── 6. Denied Claims Analysis — Denial reason breakdown

QUERIES["denial_analysis"] = (
    "Revenue Integrity: Claim denial reasons and recovery opportunity",
    PREFIX + """
SELECT ?denialReason
       (COUNT(?claim) AS ?deniedCount)
       (SUM(?billedAmt) AS ?totalDenied)
WHERE {
    ?claim  a               hkg:Claim ;
            hkg:claimStatus  "denied" ;
            hkg:denialReason ?denialReason ;
            hkg:claimAmount  ?billedAmt .
}
GROUP BY ?denialReason
ORDER BY DESC(?deniedCount)
"""
)

# ── 7. Payer Mix — Claim volume and spend by payer

QUERIES["payer_mix"] = (
    "Payer Analysis: Claim volume and payments by payer",
    PREFIX + """
SELECT ?payerLabel
       (COUNT(?claim) AS ?claimCount)
       (SUM(?billed) AS ?totalBilled)
       (SUM(?paid) AS ?totalPaid)
WHERE {
    ?claim  a              hkg:Claim ;
            hkg:claimAmount ?billed ;
            hkg:paidAmount  ?paid ;
            hkg:submittedTo ?payer .
    ?payer  rdfs:label     ?payerLabel .
}
GROUP BY ?payerLabel
ORDER BY DESC(?totalBilled)
"""
)

# ── 8. Chronic Disease Co-morbidity — Which conditions co-occur most

QUERIES["comorbidity_pairs"] = (
    "Population Health: Frequent diagnosis co-morbidity pairs",
    PREFIX + """
SELECT ?code1 ?desc1 ?code2 ?desc2 (COUNT(?patient) AS ?patientCount)
WHERE {
    ?patient a hkg:Patient ;
             hkg:hasCondition ?cond1 , ?cond2 .
    ?cond1   hkg:hasICD10 ?icd1 .
    ?cond2   hkg:hasICD10 ?icd2 .
    ?icd1    hkg:codeValue ?code1 ;
             rdfs:label    ?desc1 .
    ?icd2    hkg:codeValue ?code2 ;
             rdfs:label    ?desc2 .
    FILTER(?code1 < ?code2)   # avoid duplicates
}
GROUP BY ?code1 ?desc1 ?code2 ?desc2
HAVING (COUNT(?patient) >= 3)
ORDER BY DESC(?patientCount)
LIMIT 15
"""
)

# ── 9. Care Team Completeness — Patients whose conditions lack specialist coverage

QUERIES["specialist_coverage"] = (
    "Care Coordination: Patients with heart failure lacking cardiology visits",
    PREFIX + """
SELECT ?patientId ?riskScore
WHERE {
    ?patient a hkg:Patient ;
             hkg:patientId ?patientId ;
             hkg:riskScore  ?riskScore ;
             hkg:hasCondition ?cond .
    ?cond  hkg:hasICD10 ?icd .
    ?icd   hkg:codeValue ?code .
    FILTER(STRSTARTS(?code, "I50"))   # Heart Failure

    FILTER NOT EXISTS {
        ?enc hkg:forPatient ?patient ;
             hkg:renderingProvider ?prov .
        ?prov hkg:hasSpecialty ?spec .
        ?spec rdfs:label "Cardiology" .
    }
}
ORDER BY DESC(?riskScore)
LIMIT 20
"""
)

# ── 10. Fraud Ring — Providers with disproportionate referral concentration

QUERIES["referral_concentration"] = (
    "Fraud Detection: Providers with high self-referral concentration (kickback screen)",
    PREFIX + """
SELECT ?fromNPI ?fromName
       (COUNT(?toProvider) AS ?referralPartners)
WHERE {
    ?fromProv  a                  hkg:Practitioner ;
               hkg:npi            ?fromNPI ;
               hkg:providerName   ?fromName ;
               hkg:referredTo     ?toProvider .
}
GROUP BY ?fromNPI ?fromName
HAVING (COUNT(?toProvider) >= 3)
ORDER BY DESC(?referralPartners)
LIMIT 15
"""
)


# ─────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────

def run_query(g: Graph, name: str) -> None:
    label, sparql = QUERIES[name]
    print(f"\n{'═'*70}")
    print(f"  {label}")
    print(f"{'═'*70}")
    try:
        results = list(g.query(sparql))
        if not results:
            print("  (no results — run data_generator.py + kg_builder.py first)")
            return
        # Print header from first row
        vars_ = [str(v) for v in results[0].labels] if hasattr(results[0], "labels") else []
        if vars_:
            print("  " + " | ".join(f"{v:>18}" for v in vars_))
            print("  " + "-" * (21 * len(vars_)))
        for row in results[:15]:
            vals = [str(v)[:18] if v is not None else "NULL" for v in row]
            print("  " + " | ".join(f"{v:>18}" for v in vals))
        if len(results) > 15:
            print(f"  ... ({len(results)} total rows)")
    except Exception as e:
        print(f"  Query error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg",    type=Path, default=Path("kg/healthcare_kg_populated.ttl"))
    parser.add_argument("--query", choices=list(QUERIES.keys()) + ["all"], default="all")
    args = parser.parse_args()

    if not args.kg.exists():
        log.error("%s not found. Run kg_builder.py first.", args.kg)
        return

    log.info("Loading KG from %s ...", args.kg)
    g = Graph()
    g.parse(str(args.kg), format="turtle")
    log.info("Loaded %d triples", len(g))

    queries_to_run = list(QUERIES.keys()) if args.query == "all" else [args.query]
    for qname in queries_to_run:
        run_query(g, qname)

    print(f"\n✅ Ran {len(queries_to_run)} SPARQL queries against {len(g):,} triples.")


if __name__ == "__main__":
    main()
