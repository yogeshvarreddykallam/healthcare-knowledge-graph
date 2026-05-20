"""kg_builder.py — Populate the Healthcare OWL ontology with synthetic data.

Reads JSON files produced by data_generator.py, creates RDF individuals
for every patient, practitioner, encounter, claim, and referral edge, and
serialises the resulting graph in Turtle format. Also exports a NetworkX
DiGraph for fast graph analytics.

Outputs:
    kg/healthcare_kg_populated.ttl   — full RDF graph
    kg/provider_network.graphml      — provider referral + co-treatment graph
    kg/patient_provider.graphml      — bipartite patient–provider graph

Usage:
    python src/kg_builder.py
    python src/kg_builder.py --data-dir data/ --ontology ontology/healthcare_kg.ttl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import networkx as nx
from rdflib import Graph, Literal, Namespace, RDF, RDFS, XSD, URIRef

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("kg-builder")

HKG = Namespace("http://yogeshvarreddykallam.github.io/healthcare-kg#")

# Property shortcuts
PATIENT_ID   = HKG.patientId
BIRTH_DATE   = HKG.birthDate
GENDER       = HKG.gender
ZIP          = HKG.zipCode
RISK         = HKG.riskScore
NPI          = HKG.npi
PROV_NAME    = HKG.providerName
CLAIM_ID_P   = HKG.claimId
CLAIM_DATE   = HKG.claimDate
CLAIM_BILLED = HKG.claimAmount
CLAIM_ALLOW  = HKG.allowedAmount
CLAIM_PAID   = HKG.paidAmount
CLAIM_STATUS = HKG.claimStatus
DENIAL_RSN   = HKG.denialReason
ENC_DATE     = HKG.encounterDate
ALERT_TYPE   = HKG.alertType
ALERT_SCORE  = HKG.alertScore
CODE_VAL     = HKG.codeValue


def load_json(data_dir: Path, name: str) -> list[dict]:
    p = data_dir / f"{name}.json"
    if not p.exists():
        log.warning("%s not found — skipping", p)
        return []
    with open(p) as f:
        return json.load(f)


def safe_uri(fragment: str) -> URIRef:
    clean = fragment.replace("-", "_").replace(" ", "_").replace("/", "_")
    return HKG[clean]


def build_rdf(
    rdf: Graph,
    practitioners: list[dict],
    patients: list[dict],
    encounters: list[dict],
    claims: list[dict],
    referrals: list[dict],
) -> None:

    # ── Practitioners ─────────────────────────────────────────
    for p in practitioners:
        uri = safe_uri(p["practitioner_id"])
        rdf.add((uri, RDF.type, HKG.Practitioner))
        rdf.add((uri, NPI,      Literal(p["npi"], datatype=XSD.string)))
        rdf.add((uri, PROV_NAME,Literal(p["name"], datatype=XSD.string)))
        spec_uri = safe_uri("spec_" + p["specialty"].replace(" ", "_").replace("/","_"))
        rdf.add((uri, HKG.hasSpecialty, spec_uri))
        rdf.add((spec_uri, RDF.type, HKG.Specialty))
        rdf.add((spec_uri, RDFS.label, Literal(p["specialty"])))
        org_uri = safe_uri("org_" + p["organization"].replace(" ", "_")[:30])
        rdf.add((uri, HKG.affiliatedWith, org_uri))
        rdf.add((org_uri, RDF.type, HKG.Hospital))
        rdf.add((org_uri, RDFS.label, Literal(p["organization"])))

    log.info("Added %d practitioners", len(practitioners))

    # ── Patients ──────────────────────────────────────────────
    for pat in patients:
        uri = safe_uri(pat["patient_id"])
        rdf.add((uri, RDF.type,   HKG.Patient))
        rdf.add((uri, PATIENT_ID, Literal(pat["patient_id"], datatype=XSD.string)))
        rdf.add((uri, BIRTH_DATE, Literal(pat["birth_date"], datatype=XSD.date)))
        rdf.add((uri, GENDER,     Literal(pat["gender"], datatype=XSD.string)))
        rdf.add((uri, ZIP,        Literal(pat["zip_code"], datatype=XSD.string)))
        rdf.add((uri, RISK,       Literal(pat["hcc_risk_score"], datatype=XSD.float)))

        # Coverage
        payer_uri = safe_uri("payer_" + pat["payer"].replace(" ", "_"))
        rdf.add((uri, HKG.hasCoverage, payer_uri))
        rdf.add((payer_uri, RDF.type, HKG.Payer))
        rdf.add((payer_uri, RDFS.label, Literal(pat["payer"])))

        # PCP
        pcp_uri = safe_uri(pat["pcp_id"])
        rdf.add((uri, HKG.hasPCP, pcp_uri))

        # Conditions
        for icd, desc, ctype in pat["conditions"]:
            cond_uri = safe_uri(f"cond_{icd.replace('.','_')}_{pat['patient_id']}")
            cls = HKG.ChronicCondition if ctype == "chronic" else HKG.AcuteCondition
            rdf.add((cond_uri, RDF.type, cls))
            icd_uri = safe_uri(f"ICD_{icd.replace('.','_')}")
            rdf.add((cond_uri, HKG.hasICD10, icd_uri))
            rdf.add((icd_uri,  RDF.type, HKG.ICD10Code))
            rdf.add((icd_uri,  CODE_VAL, Literal(icd, datatype=XSD.string)))
            rdf.add((icd_uri,  RDFS.label, Literal(desc)))
            rdf.add((uri, HKG.hasCondition, cond_uri))

    log.info("Added %d patients", len(patients))

    # ── Encounters ────────────────────────────────────────────
    enc_type_map = {
        "outpatient": HKG.OutpatientEncounter,
        "inpatient":  HKG.InpatientEncounter,
        "emergency":  HKG.EmergencyEncounter,
    }
    for enc in encounters:
        uri = safe_uri(enc["encounter_id"])
        cls = enc_type_map.get(enc["encounter_type"], HKG.Encounter)
        rdf.add((uri, RDF.type, cls))
        rdf.add((uri, ENC_DATE, Literal(enc["encounter_date"], datatype=XSD.date)))
        rdf.add((uri, HKG.forPatient,        safe_uri(enc["patient_id"])))
        rdf.add((uri, HKG.renderingProvider, safe_uri(enc["provider_id"])))
        # Diagnosis
        icd_uri = safe_uri(f"ICD_{enc['icd10_primary'].replace('.','_')}")
        rdf.add((uri, HKG.encounterDiagnosis, icd_uri))
        # Procedures
        for cpt in enc["cpt_codes"]:
            cpt_uri = safe_uri(f"CPT_{cpt}")
            rdf.add((uri, HKG.encounterProcedure, cpt_uri))
            rdf.add((cpt_uri, RDF.type, HKG.CPTCode))
            rdf.add((cpt_uri, CODE_VAL, Literal(cpt, datatype=XSD.string)))

    log.info("Added %d encounters", len(encounters))

    # ── Claims ────────────────────────────────────────────────
    for clm in claims:
        uri = safe_uri(clm["claim_id"])
        cls = HKG.ProfessionalClaim if clm["encounter_type"] == "outpatient" else HKG.InstitutionalClaim
        rdf.add((uri, RDF.type,        cls))
        rdf.add((uri, CLAIM_ID_P,      Literal(clm["claim_id"], datatype=XSD.string)))
        rdf.add((uri, CLAIM_DATE,      Literal(clm["claim_date"], datatype=XSD.date)))
        rdf.add((uri, CLAIM_BILLED,    Literal(clm["billed_amount"], datatype=XSD.decimal)))
        rdf.add((uri, CLAIM_ALLOW,     Literal(clm["allowed_amount"], datatype=XSD.decimal)))
        rdf.add((uri, CLAIM_PAID,      Literal(clm["paid_amount"], datatype=XSD.decimal)))
        rdf.add((uri, CLAIM_STATUS,    Literal(clm["claim_status"], datatype=XSD.string)))
        rdf.add((uri, HKG.claimForPatient,  safe_uri(clm["patient_id"])))
        rdf.add((uri, HKG.billingProvider,  safe_uri(clm["provider_id"])))
        rdf.add((uri, HKG.claimForEncounter,safe_uri(clm["encounter_id"])))

        if clm["denial_reason"]:
            rdf.add((uri, DENIAL_RSN, Literal(clm["denial_reason"], datatype=XSD.string)))

        # Fraud alert triples
        if clm["is_fraud"]:
            alert_uri = safe_uri(f"alert_{clm['claim_id']}")
            rdf.add((alert_uri, RDF.type,    HKG.FraudAlert))
            rdf.add((alert_uri, ALERT_TYPE,  Literal(clm["fraud_type"], datatype=XSD.string)))
            rdf.add((alert_uri, ALERT_SCORE, Literal(round(0.6 + 0.35 * 0.5, 3), datatype=XSD.float)))
            rdf.add((uri, HKG.triggeredAlert, alert_uri))
            rdf.add((alert_uri, HKG.alertInvolvesProvider, safe_uri(clm["provider_id"])))
            rdf.add((alert_uri, HKG.alertInvolvesPatient,  safe_uri(clm["patient_id"])))

    log.info("Added %d claims (%d fraud alerts)",
             len(claims), sum(1 for c in claims if c["is_fraud"]))

    # ── Referrals ─────────────────────────────────────────────
    for ref in referrals:
        rdf.add((
            safe_uri(ref["from_provider"]),
            HKG.referredTo,
            safe_uri(ref["to_provider"]),
        ))

    log.info("Added %d referral triples", len(referrals))
    log.info("Total RDF triples: %d", len(rdf))


def build_provider_network(
    practitioners: list[dict],
    referrals: list[dict],
    claims: list[dict],
) -> nx.DiGraph:
    """Directed provider referral graph with fraud ring labels."""
    G = nx.DiGraph()

    fraud_ring = {p["practitioner_id"] for p in practitioners if p["is_fraud_ring"]}

    for p in practitioners:
        G.add_node(
            p["practitioner_id"],
            name=p["name"],
            specialty=p["specialty"],
            is_fraud_ring=p["is_fraud_ring"],
        )

    # Referral edges (weighted)
    for ref in referrals:
        a, b = ref["from_provider"], ref["to_provider"]
        if G.has_edge(a, b):
            G[a][b]["weight"] += ref["count"]
        else:
            G.add_edge(a, b, weight=ref["count"],
                       is_fraud_ring=ref["is_fraud_ring"])

    # Add claim-based edge weights (co-treatment)
    enc_providers: dict[str, str] = {}
    for clm in claims:
        enc_providers[clm["encounter_id"]] = clm["provider_id"]

    log.info("Provider network: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def build_patient_provider_graph(
    patients: list[dict],
    claims: list[dict],
) -> nx.Graph:
    """Bipartite patient–provider graph for co-visit analysis."""
    G = nx.Graph()
    for pat in patients:
        G.add_node(pat["patient_id"], node_type="patient",
                   risk=pat["hcc_risk_score"])
    for clm in claims:
        pid = clm["patient_id"]
        prv = clm["provider_id"]
        G.add_node(prv, node_type="provider")
        if G.has_edge(pid, prv):
            G[pid][prv]["weight"] += 1
        else:
            G.add_edge(pid, prv, weight=1)
    log.info("Patient-provider bipartite graph: %d nodes, %d edges",
             G.number_of_nodes(), G.number_of_edges())
    return G


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  type=Path, default=Path("data"))
    parser.add_argument("--ontology",  type=Path, default=Path("ontology/healthcare_kg.ttl"))
    parser.add_argument("--out-dir",   type=Path, default=Path("kg"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    practitioners = load_json(args.data_dir, "practitioners")
    patients      = load_json(args.data_dir, "patients")
    encounters    = load_json(args.data_dir, "encounters")
    claims        = load_json(args.data_dir, "claims")
    referrals     = load_json(args.data_dir, "referrals")

    # Load base ontology
    rdf = Graph()
    rdf.parse(str(args.ontology), format="turtle")
    log.info("Base ontology: %d triples", len(rdf))

    # Populate
    build_rdf(rdf, practitioners, patients, encounters, claims, referrals)

    # Serialise
    ttl_out = args.out_dir / "healthcare_kg_populated.ttl"
    rdf.serialize(destination=str(ttl_out), format="turtle")
    log.info("Saved RDF → %s", ttl_out)

    # NetworkX graphs
    G_prov = build_provider_network(practitioners, referrals, claims)
    nx.write_graphml(G_prov, str(args.out_dir / "provider_network.graphml"))

    G_pp = build_patient_provider_graph(patients, claims)
    nx.write_graphml(G_pp, str(args.out_dir / "patient_provider.graphml"))

    log.info("KG build complete.")


if __name__ == "__main__":
    main()
