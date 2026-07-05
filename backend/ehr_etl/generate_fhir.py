"""
backend/ehr_etl/generate_fhir.py
---------------------------------
Pure-Python synthetic FHIR R4 bundle generator (no Synthea / Docker / Java needed).

Generates realistic patient bundles with:
  Patient, Practitioner, Organization, Encounter, Condition, Observation,
  MedicationRequest, AllergyIntolerance, Procedure, ImagingStudy,
  DiagnosticReport, CarePlan, Goal, Immunization

Output: one JSON file per patient written to /tmp/synthea_output/fhir/
(same location the ETL expects, keeping generated data out of the repo).

Usage:
    python backend/ehr_etl/generate_fhir.py --count 5
    python backend/ehr_etl/generate_fhir.py --count 170
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

OUTPUT_DIR = Path("/tmp/synthea_output/fhir")

# ---------------------------------------------------------------------------
# Seed data pools
# ---------------------------------------------------------------------------

FIRST_NAMES_M = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Kenneth",
]
FIRST_NAMES_F = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan",
    "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
    "Ashley", "Dorothy", "Kimberly", "Emily", "Donna",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
]
STREETS = [
    "Oak Street", "Maple Avenue", "Cedar Lane", "Pine Road", "Elm Drive",
    "Main Street", "Park Avenue", "Lake View Drive", "River Road", "Hillside Way",
    "Meadow Lane", "Forest Path", "Sunrise Boulevard", "Valley View", "Summit Road",
]
CITIES = [
    ("Boston", "MA", "02101"), ("Springfield", "MA", "01101"),
    ("Worcester", "MA", "01601"), ("Cambridge", "MA", "02139"),
    ("Lowell", "MA", "01850"), ("Brockton", "MA", "02301"),
    ("Quincy", "MA", "02169"), ("Lynn", "MA", "01901"),
]
INSURERS = ["Aetna", "Blue Cross Blue Shield", "Cigna", "UnitedHealthcare", "Humana", "Medicaid", "Medicare"]

CONDITIONS = [
    ("Hypertension", "38341003", "I10"),
    ("Type 2 diabetes mellitus", "44054006", "E11.9"),
    ("Hyperlipidemia", "55822004", "E78.5"),
    ("Chronic kidney disease stage 3", "700379008", "N18.3"),
    ("Obesity", "414916001", "E66.9"),
    ("Coronary artery disease", "53741008", "I25.10"),
    ("Heart failure with reduced ejection fraction", "134401001", "I50.20"),
    ("Chronic obstructive pulmonary disease", "13645005", "J44.9"),
    ("Asthma", "195967001", "J45.909"),
    ("Hypothyroidism", "40930008", "E03.9"),
    ("Osteoarthritis", "396275006", "M17.11"),
    ("Depression", "35489007", "F32.9"),
    ("Anxiety disorder", "197480006", "F41.9"),
    ("Prediabetes", "15777000", "R73.09"),
    ("Anemia", "271737000", "D64.9"),
    ("Atrial fibrillation", "49436004", "I48.91"),
    ("Gastroesophageal reflux disease", "235595009", "K21.0"),
    ("Urinary tract infection", "68566005", "N39.0"),
]

MEDICATIONS = [
    ("Lisinopril 10 MG", "314076", "10 mg", "Oral", "Once daily"),
    ("Metformin 500 MG", "860975", "500 mg", "Oral", "Twice daily"),
    ("Atorvastatin 20 MG", "617312", "20 mg", "Oral", "Once daily at bedtime"),
    ("Amlodipine 5 MG", "197361", "5 mg", "Oral", "Once daily"),
    ("Metoprolol Tartrate 25 MG", "866427", "25 mg", "Oral", "Twice daily"),
    ("Omeprazole 20 MG", "402014", "20 mg", "Oral", "Once daily before meals"),
    ("Levothyroxine 50 MCG", "966787", "50 mcg", "Oral", "Once daily on empty stomach"),
    ("Aspirin 81 MG", "243670", "81 mg", "Oral", "Once daily"),
    ("Sertraline 50 MG", "312940", "50 mg", "Oral", "Once daily"),
    ("Furosemide 40 MG", "977425", "40 mg", "Oral", "Once daily in morning"),
    ("Warfarin 5 MG", "855332", "5 mg", "Oral", "Once daily"),
    ("Albuterol 90 MCG", "745752", "90 mcg", "Inhaled", "Every 4-6 hours as needed"),
    ("Gabapentin 300 MG", "310431", "300 mg", "Oral", "Three times daily"),
    ("Hydrochlorothiazide 25 MG", "310798", "25 mg", "Oral", "Once daily"),
    ("Pantoprazole 40 MG", "261455", "40 mg", "Oral", "Once daily before breakfast"),
]

ALLERGIES = [
    ("Penicillin", "Rash", "mild"),
    ("Sulfa drugs", "Anaphylaxis", "severe"),
    ("Codeine", "Nausea", "mild"),
    ("Ibuprofen", "GI upset", "mild"),
    ("Latex", "Contact dermatitis", "moderate"),
    ("Shellfish", "Hives", "moderate"),
    ("Peanuts", "Anaphylaxis", "severe"),
    ("Aspirin", "Bronchospasm", "severe"),
]

VACCINES = [
    ("Influenza", "141", "Influenza seasonal injectable preservative free"),
    ("COVID-19", "213", "SARS-COV-2 (COVID-19) vaccine, mRNA"),
    ("Tdap", "115", "Tdap"),
    ("Pneumococcal", "33", "pneumococcal polysaccharide vaccine, 23 valent"),
    ("Shingles", "187", "Zoster vaccine, recombinant"),
]

VISIT_TYPES = [
    ("Encounter for check up", "AMB", "outpatient"),
    ("Office visit", "AMB", "outpatient"),
    ("Urgent care visit", "AMB", "outpatient"),
    ("Emergency department visit", "EMER", "emergency"),
    ("Hospital admission", "IMP", "inpatient"),
    ("Follow-up visit", "AMB", "outpatient"),
]

PROCEDURES = [
    ("Blood glucose measurement", "2336009"),
    ("Measurement of blood pressure", "392570002"),
    ("Echocardiography", "40701008"),
    ("Colonoscopy", "73761001"),
    ("Complete blood count", "26604007"),
    ("Chest X-ray", "399208008"),
    ("Mammography", "71651007"),
    ("Upper GI endoscopy", "73761001"),
    ("Urinalysis", "27171005"),
    ("Spirometry", "127783003"),
]

LOINC_PANELS = [
    ("Basic metabolic panel", [
        ("Glucose", "2345-7", "mg/dL", 70, 99, 60, 350),
        ("BUN", "3094-0", "mg/dL", 7, 20, 5, 80),
        ("Creatinine", "2160-0", "mg/dL", 0.6, 1.2, 0.4, 5.0),
        ("Sodium", "2951-2", "mEq/L", 136, 145, 125, 155),
        ("Potassium", "2823-3", "mEq/L", 3.5, 5.0, 2.5, 6.5),
        ("Bicarbonate", "1963-8", "mEq/L", 22, 29, 10, 40),
    ]),
    ("Lipid panel", [
        ("Total Cholesterol", "2093-3", "mg/dL", 0, 200, 100, 350),
        ("LDL Cholesterol", "18262-6", "mg/dL", 0, 100, 50, 250),
        ("HDL Cholesterol", "2085-9", "mg/dL", 40, 60, 20, 120),
        ("Triglycerides", "2571-8", "mg/dL", 0, 150, 50, 600),
    ]),
    ("Complete blood count", [
        ("WBC", "6690-2", "10*3/uL", 4.5, 11.0, 1.0, 20.0),
        ("RBC", "789-8", "10*6/uL", 4.5, 5.5, 2.0, 7.0),
        ("Hemoglobin", "718-7", "g/dL", 12.0, 17.5, 6.0, 20.0),
        ("Hematocrit", "4544-3", "%", 36, 50, 20, 60),
        ("Platelets", "777-3", "10*3/uL", 150, 400, 50, 1000),
    ]),
]

ORGANIZATIONS = [
    "Massachusetts General Hospital",
    "Brigham and Women's Hospital",
    "Boston Medical Center",
    "UMass Memorial Medical Center",
    "Tufts Medical Center",
    "Beth Israel Deaconess Medical Center",
]

PHYSICIAN_FIRST = ["James", "Maria", "David", "Sarah", "Robert", "Jennifer", "Michael", "Emily"]
PHYSICIAN_LAST = ["Chen", "Patel", "Williams", "Kim", "Johnson", "Martinez", "Garcia", "Thompson"]
SPECIALTIES = ["Internal Medicine", "Family Medicine", "Cardiology", "Endocrinology",
               "Pulmonology", "Nephrology", "Gastroenterology", "Geriatrics"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def uid() -> str:
    return str(uuid.uuid4())


def fhir_date(d: date) -> str:
    return d.isoformat()


def fhir_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def rand_date(start_year: int, end_year: int) -> date:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def rand_dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day,
                    random.randint(6, 20), random.randint(0, 59), 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Resource builders
# ---------------------------------------------------------------------------

def build_organization(org_id: str, name: str) -> dict:
    city, state, z = random.choice(CITIES)
    return {
        "resourceType": "Organization",
        "id": org_id,
        "name": name,
        "type": [{"coding": [{"code": "prov", "display": "Healthcare Provider"}]}],
        "address": [{"line": [f"{random.randint(1,999)} {random.choice(STREETS)}"],
                     "city": city, "state": state, "postalCode": z, "country": "US"}],
    }


def build_practitioner(prac_id: str, name: str, specialty: str) -> dict:
    parts = name.split()
    return {
        "resourceType": "Practitioner",
        "id": prac_id,
        "name": [{"family": parts[-1], "given": parts[:-1], "prefix": ["Dr."]}],
        "qualification": [{"code": {"coding": [{"display": specialty}]}}],
    }


def build_patient(pat_id: str, mrn: str, sex: str, dob: date,
                  first: str, last: str, city: str, state: str, zipcode: str,
                  phone: str, email: str, insurer: str) -> dict:
    return {
        "resourceType": "Patient",
        "id": pat_id,
        "identifier": [{"type": {"coding": [{"code": "MR"}]}, "value": mrn}],
        "name": [{"family": last, "given": [first], "use": "official"}],
        "gender": sex,
        "birthDate": fhir_date(dob),
        "address": [{"line": [f"{random.randint(1,999)} {random.choice(STREETS)}"],
                     "city": city, "state": state, "postalCode": zipcode, "country": "US"}],
        "telecom": [{"system": "phone", "value": phone}, {"system": "email", "value": email}],
        "maritalStatus": {"coding": [{"code": random.choice(["M", "S", "D", "W"])}]},
    }


DISCHARGE_DISPOSITIONS = [
    "Discharged to Home",
    "Discharged to Skilled Nursing Facility",
    "Discharged to Rehabilitation Facility",
    "Transferred to Another Hospital",
    "Left Against Medical Advice",
]


def build_encounter(enc_id: str, pat_id: str, prac_id: str, org_id: str,
                    visit_type: str, class_code: str,
                    start: datetime, end: datetime, chief: str) -> dict:
    enc: dict = {
        "resourceType": "Encounter",
        "id": enc_id,
        "status": "finished",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                  "code": class_code},
        "type": [{"text": visit_type, "coding": [{"display": visit_type}]}],
        "subject": {"reference": f"Patient/{pat_id}"},
        "participant": [{"type": [{"coding": [{"code": "ATND", "display": "attender"}]}],
                         "individual": {"reference": f"Practitioner/{prac_id}"}}],
        "serviceProvider": {"reference": f"Organization/{org_id}"},
        "period": {"start": fhir_datetime(start), "end": fhir_datetime(end)},
        "reasonCode": [{"text": chief}] if chief else [],
    }
    if class_code in ("IMP", "EMER"):
        enc["hospitalization"] = {
            "dischargeDisposition": {
                "coding": [{"display": random.choice(DISCHARGE_DISPOSITIONS)}]
            }
        }
    return enc


def build_condition(cond_id: str, pat_id: str, enc_id: str,
                    name: str, snomed: str, icd: str,
                    onset: date, status: str) -> dict:
    return {
        "resourceType": "Condition",
        "id": cond_id,
        "clinicalStatus": {"coding": [{"code": status}]},
        "code": {"text": name,
                 "coding": [{"system": "http://snomed.info/sct", "code": snomed, "display": name},
                             {"system": "http://hl7.org/fhir/sid/icd-10", "code": icd}]},
        "subject": {"reference": f"Patient/{pat_id}"},
        "encounter": {"reference": f"Encounter/{enc_id}"},
        "onsetDateTime": fhir_date(onset),
    }


VITALS_LOINCS = {"8480-6", "8462-4", "8867-4", "8310-5", "9279-1",
                 "2708-6", "8302-2", "29463-7", "39156-5"}


def build_observation(obs_id: str, pat_id: str, enc_id: str,
                      name: str, loinc: str, value: float, unit: str,
                      ref_low: float, ref_high: float, when: datetime) -> dict:
    flag = "H" if value > ref_high else ("L" if value < ref_low else None)
    cat_code = "vital-signs" if loinc in VITALS_LOINCS else "laboratory"
    obs = {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                   "code": cat_code}]}],
        "code": {"text": name, "coding": [{"system": "http://loinc.org", "code": loinc, "display": name}]},
        "subject": {"reference": f"Patient/{pat_id}"},
        "encounter": {"reference": f"Encounter/{enc_id}"},
        "effectiveDateTime": fhir_datetime(when),
        "valueQuantity": {"value": round(value, 1), "unit": unit},
        "referenceRange": [{"low": {"value": ref_low, "unit": unit},
                            "high": {"value": ref_high, "unit": unit}}],
    }
    if flag:
        obs["interpretation"] = [{"coding": [{"code": flag}]}]
    return obs


def build_medication_request(med_id: str, pat_id: str, enc_id: str, prac_id: str,
                              drug: str, rxnorm: str, dose: str, route: str, freq: str,
                              start: date, end: date | None, status: str) -> dict:
    mr: dict = {
        "resourceType": "MedicationRequest",
        "id": med_id,
        "status": status,
        "intent": "order",
        "medicationCodeableConcept": {
            "text": drug,
            "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": rxnorm, "display": drug}],
        },
        "subject": {"reference": f"Patient/{pat_id}"},
        "encounter": {"reference": f"Encounter/{enc_id}"},
        "requester": {"reference": f"Practitioner/{prac_id}"},
        "authoredOn": fhir_date(start),
        "dosageInstruction": [{
            "route": {"text": route},
            "timing": {"code": {"text": freq},
                       "repeat": {"boundsPeriod": {
                           "start": fhir_date(start),
                           "end": fhir_date(end) if end else fhir_date(start + timedelta(days=365)),
                       }}},
            "doseAndRate": [{"doseQuantity": {
                "value": float(dose.split()[0]) if dose.split()[0].replace('.', '').isdigit() else 1,
                "unit": dose.split()[1] if len(dose.split()) > 1 else "mg",
            }}],
        }],
    }
    return mr


def build_allergy(aid: str, pat_id: str, allergen: str, reaction: str, severity: str) -> dict:
    return {
        "resourceType": "AllergyIntolerance",
        "id": aid,
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "patient": {"reference": f"Patient/{pat_id}"},
        "code": {"text": allergen, "coding": [{"display": allergen}]},
        "reaction": [{"manifestation": [{"text": reaction, "coding": [{"display": reaction}]}],
                      "severity": severity}],
        "recordedDate": fhir_date(rand_date(2000, 2020)),
    }


def build_procedure(proc_id: str, pat_id: str, enc_id: str, prac_id: str,
                    name: str, snomed: str, when: datetime) -> dict:
    return {
        "resourceType": "Procedure",
        "id": proc_id,
        "status": "completed",
        "code": {"text": name, "coding": [{"system": "http://snomed.info/sct",
                                           "code": snomed, "display": name}]},
        "subject": {"reference": f"Patient/{pat_id}"},
        "encounter": {"reference": f"Encounter/{enc_id}"},
        "performer": [{"actor": {"reference": f"Practitioner/{prac_id}"}}],
        "performedDateTime": fhir_datetime(when),
    }


def build_imaging(img_id: str, pat_id: str, enc_id: str, prac_id: str,
                  modality: str, body: str, when: datetime) -> dict:
    return {
        "resourceType": "ImagingStudy",
        "id": img_id,
        "status": "available",
        "subject": {"reference": f"Patient/{pat_id}"},
        "encounter": {"reference": f"Encounter/{enc_id}"},
        "referrer": {"reference": f"Practitioner/{prac_id}"},
        "started": fhir_datetime(when),
        "modality": [{"system": "http://dicom.nema.org/resources/ontology/DCM", "code": modality}],
        "series": [{"uid": uid(), "modality": {"code": modality},
                    "bodySite": {"display": body},
                    "instance": [{"uid": uid(), "sopClass": {"code": "1.2.840.10008.5.1.4.1.1.2"}}]}],
    }


def build_immunization(imm_id: str, pat_id: str, enc_id: str,
                       vaccine: str, cvx: str, admin_date: date,
                       dose_num: int) -> dict:
    return {
        "resourceType": "Immunization",
        "id": imm_id,
        "status": "completed",
        "vaccineCode": {
            "text": vaccine,
            "coding": [{"system": "http://hl7.org/fhir/sid/cvx", "code": cvx, "display": vaccine}],
        },
        "patient": {"reference": f"Patient/{pat_id}"},
        "encounter": {"reference": f"Encounter/{enc_id}"},
        "occurrenceDateTime": fhir_date(admin_date),
        "doseQuantity": {"value": 1, "unit": "mL"},
        "protocolApplied": [{"doseNumberPositiveInt": dose_num}],
        "route": {"coding": [{"code": "IM", "display": "Intramuscular"}]},
        "site": {"coding": [{"code": "LA", "display": "Left arm"}]},
        "lotNumber": f"LOT{random.randint(10000, 99999)}",
        "manufacturer": {"display": random.choice(["Pfizer", "Moderna", "GSK", "Seqirus"])},
    }


def build_care_plan(cp_id: str, pat_id: str, enc_id: str,
                    goals: list[str], interventions: list[str],
                    follow_up: str) -> dict:
    goal_refs = []
    goal_resources = []
    for i, g in enumerate(goals):
        gid = uid()
        goal_refs.append({"reference": f"Goal/{gid}"})
        goal_resources.append({
            "resourceType": "Goal",
            "id": gid,
            "lifecycleStatus": "active",
            "description": {"text": g},
            "subject": {"reference": f"Patient/{pat_id}"},
        })
    return {
        "care_plan": {
            "resourceType": "CarePlan",
            "id": cp_id,
            "status": "active",
            "intent": "plan",
            "subject": {"reference": f"Patient/{pat_id}"},
            "encounter": {"reference": f"Encounter/{enc_id}"},
            "note": [{"text": follow_up}],
            "goal": goal_refs,
            "activity": [{"detail": {"description": iv}} for iv in interventions],
        },
        "goals": goal_resources,
    }


# ---------------------------------------------------------------------------
# Patient bundle generator
# ---------------------------------------------------------------------------

def generate_patient_bundle(orgs: list[tuple], practitioners: list[tuple]) -> dict:
    pat_id = uid()
    sex = random.choice(["male", "female"])
    first = random.choice(FIRST_NAMES_M if sex == "male" else FIRST_NAMES_F)
    last = random.choice(LAST_NAMES)
    mrn = str(uuid.uuid4())
    dob = rand_date(1940, 1995)
    city, state, zipcode = random.choice(CITIES)
    phone = f"555-{random.randint(100,999)}-{random.randint(1000,9999)}"
    email = f"{first.lower()}.{last.lower()}{random.randint(1,99)}@example.com"
    insurer = random.choice(INSURERS)

    entries: list[dict] = []

    def add(resource: dict) -> dict:
        entries.append({"resource": resource})
        return resource

    patient = add(build_patient(pat_id, mrn, sex, dob,
                                first, last, city, state, zipcode, phone, email, insurer))

    # Pick 2–4 conditions (first is usually chronic)
    num_conditions = random.randint(2, 4)
    chosen_conditions = random.sample(CONDITIONS, num_conditions)

    # Generate 3–8 encounters over the past 5 years
    num_encounters = random.randint(3, 8)
    encounter_data: list[dict] = []
    for i in range(num_encounters):
        enc_id = uid()
        org_id, _ = random.choice(orgs)
        prac_id, _ = random.choice(practitioners)
        vt, cls, _ = random.choice(VISIT_TYPES)
        enc_start_date = rand_date(2019, 2024)
        duration_hours = random.choice([1, 2, 4, 24, 48, 72]) if cls in ("IMP", "EMER") else random.choice([1, 2])
        enc_start_dt = rand_dt(enc_start_date)
        enc_end_dt = enc_start_dt + timedelta(hours=duration_hours)
        chief = random.choice([c[0] for c in chosen_conditions] + ["Annual checkup", "Follow-up", "Chest pain", "Shortness of breath"])

        enc = add(build_encounter(enc_id, pat_id, prac_id, org_id,
                                  vt, cls, enc_start_dt, enc_end_dt, chief))
        encounter_data.append({
            "id": enc_id, "org_id": org_id, "prac_id": prac_id,
            "start": enc_start_dt, "end": enc_end_dt, "class": cls,
            "date": enc_start_date,
        })

    # Conditions — link to first encounter
    for cond_name, snomed, icd in chosen_conditions:
        cond_id = uid()
        enc = random.choice(encounter_data)
        status = "active" if random.random() < 0.7 else "resolved"
        onset_start = max(dob.year + 30, 2000)
        onset_end = max(onset_start, 2023)
        onset = rand_date(onset_start, onset_end)
        add(build_condition(cond_id, pat_id, enc["id"], cond_name, snomed, icd, onset, status))

    # Observations (labs) — 1–3 panels across encounters
    panels_to_do = random.sample(LOINC_PANELS, min(random.randint(1, 3), len(LOINC_PANELS)))
    for panel_name, tests in panels_to_do:
        enc = random.choice(encounter_data)
        for test_name, loinc, unit, rlo, rhi, vlo, vhi in tests:
            value = round(random.uniform(vlo * 0.85, vhi * 0.85), 1)
            add(build_observation(uid(), pat_id, enc["id"],
                                  test_name, loinc, value, unit, rlo, rhi, enc["start"]))

    # Vital signs as observations (height, weight, BP)
    for enc in encounter_data[:3]:
        height_cm = random.uniform(155, 195)
        weight_kg = random.uniform(55, 120)
        add(build_observation(uid(), pat_id, enc["id"],
                              "Body Height", "8302-2", round(height_cm, 1), "cm", 0, 300, enc["start"]))
        add(build_observation(uid(), pat_id, enc["id"],
                              "Body Weight", "29463-7", round(weight_kg, 1), "kg", 0, 300, enc["start"]))
        systolic = random.uniform(100, 170)
        add(build_observation(uid(), pat_id, enc["id"],
                              "Systolic Blood Pressure", "8480-6", round(systolic, 0), "mmHg",
                              90, 120, enc["start"]))
        diastolic = random.uniform(60, 110)
        add(build_observation(uid(), pat_id, enc["id"],
                              "Diastolic Blood Pressure", "8462-4", round(diastolic, 0), "mmHg",
                              60, 80, enc["start"]))
        hr = random.uniform(55, 105)
        add(build_observation(uid(), pat_id, enc["id"],
                              "Heart rate", "8867-4", round(hr, 0), "/min", 60, 100, enc["start"]))

    # Medications — 2–4 active, link to inpatient/ER for discharge prescriptions
    num_meds = random.randint(2, 4)
    chosen_meds = random.sample(MEDICATIONS, num_meds)
    for drug, rxnorm, dose, route, freq in chosen_meds:
        enc = random.choice(encounter_data)
        start_d = enc["date"]
        end_d = start_d + timedelta(days=random.choice([90, 180, 365])) if random.random() < 0.5 else None
        status = "active" if not end_d else "completed"
        add(build_medication_request(uid(), pat_id, enc["id"], enc["prac_id"],
                                     drug, rxnorm, dose, route, freq, start_d, end_d, status))

    # Allergies — 0–2
    if random.random() < 0.5:
        chosen_allergies = random.sample(ALLERGIES, random.randint(1, 2))
        for allergen, reaction, severity in chosen_allergies:
            add(build_allergy(uid(), pat_id, allergen, reaction, severity))

    # Procedures — 1–3
    num_procs = random.randint(1, 3)
    chosen_procs = random.sample(PROCEDURES, num_procs)
    for proc_name, snomed in chosen_procs:
        enc = random.choice(encounter_data)
        add(build_procedure(uid(), pat_id, enc["id"], enc["prac_id"],
                            proc_name, snomed, enc["start"]))

    # Imaging — 0–1
    if random.random() < 0.4:
        enc = random.choice(encounter_data)
        modality = random.choice(["CT", "MR", "CR", "US"])
        body = random.choice(["Chest", "Abdomen", "Head", "Knee", "Spine"])
        add(build_imaging(uid(), pat_id, enc["id"], enc["prac_id"],
                          modality, body, enc["start"]))

    # Social history observations (smoking, alcohol)
    if encounter_data:
        sh_enc = encounter_data[0]
        smoking_loinc = "72166-2"
        smoking_values = ["Never smoker", "Former smoker", "Light tobacco smoker", "Non-smoker"]
        entries.append({"resource": {
            "resourceType": "Observation",
            "id": uid(),
            "status": "final",
            "category": [{"coding": [{"code": "social-history"}]}],
            "code": {"text": "Tobacco smoking status", "coding": [
                {"system": "http://loinc.org", "code": smoking_loinc}]},
            "subject": {"reference": f"Patient/{pat_id}"},
            "encounter": {"reference": f"Encounter/{sh_enc['id']}"},
            "effectiveDateTime": fhir_datetime(sh_enc["start"]),
            "valueCodeableConcept": {"text": random.choice(smoking_values)},
        }})
        entries.append({"resource": {
            "resourceType": "Observation",
            "id": uid(),
            "status": "final",
            "category": [{"coding": [{"code": "social-history"}]}],
            "code": {"text": "Alcohol Use", "coding": [
                {"system": "http://loinc.org", "code": "74013-4"}]},
            "subject": {"reference": f"Patient/{pat_id}"},
            "encounter": {"reference": f"Encounter/{sh_enc['id']}"},
            "effectiveDateTime": fhir_datetime(sh_enc["start"]),
            "valueCodeableConcept": {"text": random.choice(["Never", "Less than monthly", "Monthly", "Weekly", "Daily or almost daily"])},
        }})

    # Immunizations — 1–3
    vaccines = random.sample(VACCINES, random.randint(1, 3))
    for i, (vaccine_name, cvx, full_name) in enumerate(vaccines):
        enc = random.choice(encounter_data)
        add(build_immunization(uid(), pat_id, enc["id"],
                               full_name, cvx, enc["date"], i + 1))

    # Care plan — 1 from last encounter
    last_enc = sorted(encounter_data, key=lambda e: e["start"])[-1]
    goals = [
        f"Maintain {random.choice(['blood pressure', 'blood glucose', 'weight'])} within target range",
        f"Adhere to prescribed medications for {chosen_conditions[0][0]}",
    ]
    interventions = [
        "Schedule follow-up in 3 months",
        "Dietary counseling referral",
        "Lifestyle modification — 30 min exercise 5x/week",
    ]
    follow_up = f"Follow up with primary care in 3 months. Monitor {chosen_conditions[0][0]}."
    cp_result = build_care_plan(uid(), pat_id, last_enc["id"], goals, interventions, follow_up)
    add(cp_result["care_plan"])
    for g in cp_result["goals"]:
        add(g)

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }
    return bundle, pat_id


# ---------------------------------------------------------------------------
# Shared org / practitioner pool builder
# ---------------------------------------------------------------------------

def generate_shared_resources(num_orgs: int = 6, num_practitioners: int = 15) -> tuple:
    orgs = []
    practitioners = []
    for _ in range(num_orgs):
        oid = uid()
        name = random.choice(ORGANIZATIONS)
        orgs.append((oid, build_organization(oid, name)))

    for _ in range(num_practitioners):
        pid = uid()
        fn = f"{random.choice(PHYSICIAN_FIRST)} {random.choice(PHYSICIAN_LAST)}"
        specialty = random.choice(SPECIALTIES)
        practitioners.append((pid, build_practitioner(pid, fn, specialty)))

    return orgs, practitioners


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic FHIR R4 bundles")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of patient bundles to generate (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} patient bundles → {out}")

    orgs, practitioners = generate_shared_resources()

    # Write a shared bundle with all orgs + practitioners (pass 1 in ETL reads this)
    shared_entries = [{"resource": org} for _, org in orgs] + \
                     [{"resource": prac} for _, prac in practitioners]
    shared_bundle = {"resourceType": "Bundle", "type": "collection", "entry": shared_entries}
    (out / "shared_organizations.json").write_text(json.dumps(shared_bundle, indent=2))

    org_ids = [(oid, None) for oid, _ in orgs]
    prac_ids = [(pid, None) for pid, _ in practitioners]

    generated = 0
    for i in range(args.count):
        bundle, pat_id = generate_patient_bundle(org_ids, prac_ids)
        filename = f"patient_{i:04d}_{pat_id}.json"
        (out / filename).write_text(json.dumps(bundle, indent=2))
        generated += 1
        if (i + 1) % 10 == 0 or i + 1 == args.count:
            print(f"  {i+1}/{args.count} bundles written")

    print(f"Done. {generated} patient bundles in {out}")


if __name__ == "__main__":
    main()
