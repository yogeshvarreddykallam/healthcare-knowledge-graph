"""data_generator.py — Synthetic FHIR-aligned healthcare data generator.

Generates realistic but entirely synthetic US healthcare data including:
  - Patients with demographics and chronic conditions
  - Practitioners with NPI, specialty, and organizational affiliations
  - Encounters (outpatient, inpatient, ED) with diagnoses and procedures
  - Claims with CPT/ICD-10 codes, billed/allowed/paid amounts
  - Referral relationships between providers
  - Injected fraud patterns for anomaly detection benchmarking

All data is synthetic — no real PHI. Designed to populate the OWL ontology
defined in ontology/healthcare_kg.ttl.

Usage:
    python src/data_generator.py                   # default 500 patients
    python src/data_generator.py --patients 2000 --seed 42
    python src/data_generator.py --out-dir data/ --fraud-rate 0.05
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────
#  Vocabulary tables (representative subsets of real code sets)
# ─────────────────────────────────────────────────────────────────────

ICD10_CHRONIC = [
    ("E11",   "Type 2 Diabetes Mellitus",              "chronic"),
    ("I10",   "Essential Hypertension",                "chronic"),
    ("J44.1", "COPD with Acute Exacerbation",          "chronic"),
    ("I50.9", "Heart Failure, Unspecified",            "chronic"),
    ("N18.3", "Chronic Kidney Disease, Stage 3",       "chronic"),
    ("F32.1", "Major Depressive Disorder, Moderate",   "chronic"),
    ("E78.5", "Hyperlipidemia, Unspecified",           "chronic"),
    ("M54.5", "Low Back Pain",                         "acute"),
    ("J06.9", "Acute Upper Respiratory Infection",     "acute"),
    ("I21.9", "Acute Myocardial Infarction",           "acute"),
    ("Z23",   "Encounter for Immunization",            "acute"),
    ("Z00.00","Encounter for General Adult Exam",      "acute"),
    ("R05.9", "Cough, Unspecified",                    "acute"),
    ("K21.0", "GERD with Esophagitis",                 "chronic"),
    ("G47.00","Insomnia, Unspecified",                 "chronic"),
]

CPT_CODES = [
    ("99213", "Office Visit, Established Pt Level 3",  150.0,  95.0),
    ("99214", "Office Visit, Established Pt Level 4",  230.0, 145.0),
    ("99215", "Office Visit, Established Pt Level 5",  320.0, 200.0),
    ("99232", "Subsequent Hospital Care Level 2",       180.0, 110.0),
    ("99283", "ED Visit, Moderate Severity",            450.0, 290.0),
    ("93000", "ECG with Interpretation",                 85.0,  52.0),
    ("71046", "Chest X-Ray, 2 Views",                   120.0,  75.0),
    ("80053", "Comprehensive Metabolic Panel",           60.0,  38.0),
    ("36415", "Routine Venipuncture",                    25.0,  16.0),
    ("45378", "Colonoscopy, Diagnostic",                850.0, 530.0),
    ("43239", "Upper GI Endoscopy with Biopsy",         950.0, 595.0),
    ("27447", "Total Knee Arthroplasty",              12000.0,7500.0),
    ("33533", "CABG, Arterial",                       45000.0,28000.0),
    ("99490", "Chronic Care Management, 20 min",         42.0,  42.0),
    ("G0438", "Annual Wellness Visit",                  185.0, 185.0),
]

SPECIALTIES = [
    "Primary Care / Family Medicine",
    "Internal Medicine",
    "Cardiology",
    "Endocrinology",
    "Nephrology",
    "Pulmonology",
    "Psychiatry",
    "Orthopedics",
    "Emergency Medicine",
    "Gastroenterology",
    "Radiology",
    "Pathology / Lab",
]

PAYERS = [
    "BlueCross BlueShield",
    "UnitedHealthcare",
    "Aetna",
    "Cigna",
    "Humana",
    "Medicaid",
    "Medicare",
    "Molina Healthcare",
]

HOSPITALS = [
    "Penn State Health Milton S. Hershey Medical Center",
    "UPMC Presbyterian",
    "Jefferson University Hospital",
    "Temple University Hospital",
    "Geisinger Medical Center",
]

DENIAL_REASONS = [
    "CO-4: The procedure code is inconsistent with the modifier",
    "CO-50: Non-covered service",
    "CO-97: Payment included in allowance for another service",
    "CO-B15: Authorization expired",
    "PR-1: Deductible amount",
    None,  # most claims paid
]

FRAUD_PATTERNS = [
    "upcoding",          # billing higher E/M level than documented
    "unbundling",        # splitting procedure to inflate reimbursement
    "duplicate_claim",   # same service billed twice
    "phantom_billing",   # service never rendered
    "kickback_referral", # unusually high referral concentration
]

FIRST_NAMES = ["James","Mary","John","Patricia","Robert","Jennifer","Michael",
               "Linda","William","Barbara","David","Susan","Richard","Jessica",
               "Joseph","Sarah","Thomas","Karen","Charles","Lisa","Yogesh","Priya",
               "Raj","Ananya","Wei","Li","Maria","Carlos","Elena","Omar"]

LAST_NAMES  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller",
               "Davis","Wilson","Taylor","Anderson","Thomas","Jackson","White",
               "Harris","Martin","Thompson","Lee","Patel","Kumar","Chen","Nguyen",
               "Rodriguez","Lopez","Hernandez","Kim","Kallam","Reddy","Shah"]


# ─────────────────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────────────────

def _uid() -> str:
    return str(uuid.uuid4())[:8].upper()

def _date(start_year=2020, end_year=2024) -> str:
    start = date(start_year, 1, 1)
    end   = date(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()

def _birth_date(min_age=18, max_age=85) -> str:
    today = date.today()
    days  = random.randint(min_age * 365, max_age * 365)
    return (today - timedelta(days=days)).isoformat()

def _npi() -> str:
    return str(random.randint(1000000000, 1999999999))

def _name() -> str:
    return f"Dr. {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def _patient_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


# ─────────────────────────────────────────────────────────────────────
#  Generator
# ─────────────────────────────────────────────────────────────────────

class HealthcareDataGenerator:
    def __init__(self, n_patients: int = 500, n_providers: int = 80,
                 fraud_rate: float = 0.04, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.n_patients  = n_patients
        self.n_providers = n_providers
        self.fraud_rate  = fraud_rate

    # ── Practitioners ─────────────────────────────────────────────────

    def generate_practitioners(self) -> list[dict]:
        practitioners = []
        for _ in range(self.n_providers):
            specialty = random.choice(SPECIALTIES)
            p = {
                "practitioner_id": f"PRV-{_uid()}",
                "npi":             _npi(),
                "name":            _name(),
                "specialty":       specialty,
                "organization":    random.choice(HOSPITALS),
                "zip_code":        f"{random.randint(15000, 19999):05d}",
                "is_fraud_ring":   False,
            }
            practitioners.append(p)

        # Inject a small fraud ring (5 providers who refer back and forth)
        ring_size = max(3, int(self.n_providers * 0.06))
        ring = random.sample(range(len(practitioners)), ring_size)
        for idx in ring:
            practitioners[idx]["is_fraud_ring"] = True

        return practitioners

    # ── Patients ──────────────────────────────────────────────────────

    def generate_patients(self, practitioners: list[dict]) -> list[dict]:
        patients = []
        pcp_ids = [p["practitioner_id"] for p in practitioners
                   if "Primary Care" in p["specialty"] or "Internal Medicine" in p["specialty"]]
        if not pcp_ids:
            pcp_ids = [p["practitioner_id"] for p in practitioners]

        for _ in range(self.n_patients):
            n_conditions = random.choices([0,1,2,3,4], weights=[10,30,30,20,10])[0]
            conditions   = random.sample(ICD10_CHRONIC, min(n_conditions, len(ICD10_CHRONIC)))
            chronic_count = sum(1 for c in conditions if c[2] == "chronic")
            # HCC risk score: higher for more/worse chronic conditions
            risk_score = round(0.3 + 0.2 * chronic_count + random.gauss(0, 0.1), 3)
            risk_score = max(0.1, min(risk_score, 3.5))

            pat = {
                "patient_id":   f"PAT-{_uid()}",
                "name":          _patient_name(),
                "birth_date":    _birth_date(),
                "gender":        random.choice(["M", "F"]),
                "zip_code":      f"{random.randint(15000, 19999):05d}",
                "payer":         random.choice(PAYERS),
                "pcp_id":        random.choice(pcp_ids),
                "conditions":    [(c[0], c[1], c[2]) for c in conditions],
                "hcc_risk_score": risk_score,
            }
            patients.append(pat)
        return patients

    # ── Encounters & Claims ────────────────────────────────────────────

    def generate_encounters_and_claims(
        self,
        patients: list[dict],
        practitioners: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        encounters: list[dict] = []
        claims:     list[dict] = []

        prov_by_id = {p["practitioner_id"]: p for p in practitioners}
        all_prov_ids = [p["practitioner_id"] for p in practitioners]

        for pat in patients:
            # Number of encounters driven by risk score
            n_enc = max(1, int(np.random.poisson(pat["hcc_risk_score"] * 4)))
            n_enc = min(n_enc, 20)

            for _ in range(n_enc):
                enc_type = random.choices(
                    ["outpatient", "inpatient", "emergency"],
                    weights=[70, 20, 10]
                )[0]

                provider_id = random.choice(all_prov_ids)
                provider    = prov_by_id[provider_id]

                # Choose CPT based on encounter type
                if enc_type == "inpatient":
                    cpt_pool = [c for c in CPT_CODES if c[0] in ("99232","33533","27447")]
                elif enc_type == "emergency":
                    cpt_pool = [c for c in CPT_CODES if c[0] in ("99283",)]
                else:
                    cpt_pool = [c for c in CPT_CODES
                                if c[0] not in ("99232","33533","27447","99283")]

                if not cpt_pool:
                    cpt_pool = CPT_CODES[:5]

                # Primary CPT + 0–2 ancillary codes
                n_cpts = random.randint(1, min(3, len(cpt_pool)))
                selected_cpts = random.sample(cpt_pool, n_cpts)

                # Diagnosis from patient's conditions or random
                if pat["conditions"] and random.random() < 0.75:
                    dx = random.choice(pat["conditions"])
                    icd = dx[0]; dx_desc = dx[1]
                else:
                    icd_row = random.choice(ICD10_CHRONIC)
                    icd = icd_row[0]; dx_desc = icd_row[1]

                enc_date = _date()
                enc_id   = f"ENC-{_uid()}"
                enc      = {
                    "encounter_id":   enc_id,
                    "patient_id":     pat["patient_id"],
                    "provider_id":    provider_id,
                    "organization":   provider["organization"],
                    "encounter_type": enc_type,
                    "encounter_date": enc_date,
                    "icd10_primary":  icd,
                    "diagnosis_desc": dx_desc,
                    "cpt_codes":      [c[0] for c in selected_cpts],
                }
                encounters.append(enc)

                # Build claim
                total_billed  = sum(c[2] for c in selected_cpts)
                total_allowed = sum(c[3] for c in selected_cpts)

                # Fraud injection
                is_fraud = random.random() < self.fraud_rate
                fraud_type = None
                if is_fraud:
                    fraud_type = random.choice(FRAUD_PATTERNS)
                    if fraud_type == "upcoding":
                        total_billed  *= 1.4   # inflated billing
                        total_allowed *= 1.0   # payer pays standard
                    elif fraud_type == "unbundling":
                        total_billed  *= 1.25
                    elif fraud_type == "duplicate_claim":
                        total_billed   = total_billed  # same amount, duplicate
                    elif fraud_type == "phantom_billing":
                        total_billed  *= 1.0   # billed but service never rendered

                paid_ratio   = random.gauss(0.82, 0.08)
                paid_ratio   = max(0.0, min(1.0, paid_ratio))
                total_paid   = round(total_allowed * paid_ratio, 2)
                claim_status = "paid" if total_paid > 0 else "denied"
                denial_reason = None
                if claim_status == "denied" or random.random() < 0.08:
                    denial_reason = random.choice([r for r in DENIAL_REASONS if r])
                    claim_status  = "denied"
                    total_paid    = 0.0

                claim = {
                    "claim_id":       f"CLM-{_uid()}",
                    "encounter_id":   enc_id,
                    "patient_id":     pat["patient_id"],
                    "provider_id":    provider_id,
                    "payer":          pat["payer"],
                    "claim_date":     enc_date,
                    "icd10_primary":  icd,
                    "cpt_codes":      [c[0] for c in selected_cpts],
                    "billed_amount":  round(total_billed, 2),
                    "allowed_amount": round(total_allowed, 2),
                    "paid_amount":    total_paid,
                    "claim_status":   claim_status,
                    "denial_reason":  denial_reason,
                    "is_fraud":       is_fraud,
                    "fraud_type":     fraud_type,
                    "encounter_type": enc_type,
                }
                claims.append(claim)

        return encounters, claims

    # ── Referral network ──────────────────────────────────────────────

    def generate_referrals(
        self,
        practitioners: list[dict],
        encounters: list[dict],
    ) -> list[dict]:
        """
        Build a referral edge list. Fraud ring practitioners have
        disproportionately high mutual referral counts.
        """
        referrals: list[dict] = []
        all_ids = [p["practitioner_id"] for p in practitioners]
        ring_ids = [p["practitioner_id"] for p in practitioners if p["is_fraud_ring"]]

        prov_encounter_count: dict[str, int] = {}
        for enc in encounters:
            prov_encounter_count[enc["provider_id"]] = \
                prov_encounter_count.get(enc["provider_id"], 0) + 1

        # Normal referrals proportional to caseload
        for prov in practitioners:
            n_refs = max(1, int(prov_encounter_count.get(prov["practitioner_id"], 0) * 0.15))
            targets = random.sample(
                [i for i in all_ids if i != prov["practitioner_id"]],
                min(n_refs, len(all_ids) - 1)
            )
            for t in targets:
                referrals.append({
                    "from_provider": prov["practitioner_id"],
                    "to_provider":   t,
                    "count":         random.randint(1, 8),
                    "is_fraud_ring": prov["is_fraud_ring"],
                })

        # Fraud ring: high-density mutual referrals
        if len(ring_ids) >= 2:
            for i, a in enumerate(ring_ids):
                for b in ring_ids[i+1:]:
                    referrals.append({
                        "from_provider": a,
                        "to_provider":   b,
                        "count":         random.randint(25, 60),
                        "is_fraud_ring": True,
                    })
                    referrals.append({
                        "from_provider": b,
                        "to_provider":   a,
                        "count":         random.randint(25, 60),
                        "is_fraud_ring": True,
                    })

        return referrals

    # ── Master generate ───────────────────────────────────────────────

    def generate_all(self) -> dict[str, list[dict]]:
        print("Generating practitioners...")
        practitioners = self.generate_practitioners()
        print(f"  {len(practitioners)} practitioners")

        print("Generating patients...")
        patients = self.generate_patients(practitioners)
        print(f"  {len(patients)} patients")

        print("Generating encounters and claims...")
        encounters, claims = self.generate_encounters_and_claims(patients, practitioners)
        print(f"  {len(encounters)} encounters, {len(claims)} claims")

        print("Generating referral network...")
        referrals = self.generate_referrals(practitioners, encounters)
        print(f"  {len(referrals)} referral edges")

        fraud_claims = [c for c in claims if c["is_fraud"]]
        print(f"\nSummary:")
        print(f"  Total claims:      {len(claims)}")
        print(f"  Fraud claims:      {len(fraud_claims)} ({100*len(fraud_claims)/len(claims):.1f}%)")
        print(f"  Total billed:     ${sum(c['billed_amount'] for c in claims):,.0f}")
        print(f"  Total paid:       ${sum(c['paid_amount'] for c in claims):,.0f}")

        return {
            "practitioners": practitioners,
            "patients":      patients,
            "encounters":    encounters,
            "claims":        claims,
            "referrals":     referrals,
        }


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic healthcare data.")
    parser.add_argument("--patients",    type=int,   default=500)
    parser.add_argument("--providers",   type=int,   default=80)
    parser.add_argument("--fraud-rate",  type=float, default=0.04)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out-dir",     type=Path,  default=Path("data"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gen  = HealthcareDataGenerator(
        n_patients=args.patients,
        n_providers=args.providers,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )
    data = gen.generate_all()

    for key, records in data.items():
        out_path = args.out_dir / f"{key}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Saved {len(records)} {key} → {out_path}")


if __name__ == "__main__":
    main()
