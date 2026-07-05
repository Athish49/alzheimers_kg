"""
backend/ehr_etl/load_synthea.py
--------------------------------
Standalone ETL script: Synthea FHIR R4 JSON → Neon Postgres EHR schema.

Run from repo root:
    python backend/ehr_etl/load_synthea.py
    python backend/ehr_etl/load_synthea.py --skip-docker
    python backend/ehr_etl/load_synthea.py --schema-only

Design notes:
  - Two-pass load: organizations + practitioners first (shared across bundles),
    then patient bundles one at a time in their own transaction.
  - All top-level resources use FHIR resource id as the Postgres PK for
    deterministic deduplication across re-runs (ON CONFLICT DO NOTHING).
  - Pivoted / child rows are guarded by skipping any bundle whose patient_id
    already exists in the patients table.
"""

import argparse
import base64
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg

# ---------------------------------------------------------------------------
# Paths — everything relative to repo root, resolved from this file's location
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # backend/ehr_etl/
BACKEND_DIR = SCRIPT_DIR.parent                        # backend/
REPO_ROOT   = BACKEND_DIR.parent                       # repo root

ENV_FILE     = BACKEND_DIR / ".env"
SCHEMA_FILE  = SCRIPT_DIR / "schema.sql"
# FHIR JSON files are ephemeral — written by Synthea Docker and read once.
# Use /tmp to keep generated patient bundles out of the repo entirely.
# Synthea writes to /output/fhir/ inside the container.
# We mount /tmp/synthea_output as /output, so files land in /tmp/synthea_output/fhir/.
SYNTHEA_MOUNT = Path("/tmp/synthea_output")   # docker volume mount point
OUTPUT_FHIR   = SYNTHEA_MOUNT / "fhir"        # where Synthea actually writes bundles


# ---------------------------------------------------------------------------
# Read DATABASE_URL from backend/.env
# ---------------------------------------------------------------------------
def load_database_url():
    if not ENV_FILE.exists():
        sys.exit(f"ERROR: {ENV_FILE} not found")
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip()
    sys.exit("ERROR: DATABASE_URL not found in backend/.env")


# ---------------------------------------------------------------------------
# LOINC → vitals column, lab panel, and social-history LOINC codes
# ---------------------------------------------------------------------------

VITALS_LOINC = {
    "8480-6":  "systolic_bp",
    "8462-4":  "diastolic_bp",
    "8867-4":  "heart_rate",
    "8310-5":  "temperature_c",
    "9279-1":  "respiratory_rate",
    "2708-6":  "spo2_pct",
    "8302-2":  "height_cm",
    "29463-7": "weight_kg",
    "39156-5": "bmi",
}

# LOINC → panel name mapping
LAB_PANEL_LOINC = {}
for code in ["6690-2","789-8","718-7","4544-3","777-3","32623-1","28539-5","28540-3","28542-9"]:
    LAB_PANEL_LOINC[code] = "CBC"
for code in ["2951-2","2823-3","2075-0","20565-8","3094-0","2160-0","33914-3","2345-7","17861-6"]:
    LAB_PANEL_LOINC[code] = "Metabolic"
for code in ["2093-3","18262-6","2085-9","2571-8"]:
    LAB_PANEL_LOINC[code] = "Lipid Panel"
for code in ["4548-4","3016-3","11579-0","2986-8"]:
    LAB_PANEL_LOINC[code] = "Endocrine"
for code in ["1742-6","1920-8","6768-6","1975-2","1751-7"]:
    LAB_PANEL_LOINC[code] = "Liver Function"
for code in ["5767-9","5778-6","2514-8","5804-0","20454-5","25428-4","5802-4"]:
    LAB_PANEL_LOINC[code] = "Urinalysis"
for code in ["5902-2","6301-6","3173-2"]:
    LAB_PANEL_LOINC[code] = "Coagulation"

# Social-history LOINC codes we recognize
SOCIAL_LOINC = {
    "72166-2": "smoking_status",
    "74013-4": "alcohol_frequency",
    "21843-8": "occupation",
    "71802-3": "housing_status",
    "63504-5": "education_level",
}

# Encounter class → visit_type enum
ENCOUNTER_CLASS_MAP = {
    "AMB":    "outpatient",
    "EMER":   "emergency",
    "IMP":    "inpatient",
    "SS":     "surgical",
    "VR":     "telehealth",
    "OBSENC": "observation",
}

# AllergyIntolerance severity → allergy_severity enum
ALLERGY_SEVERITY_MAP = {
    "mild":     "mild",
    "moderate": "moderate",
    "severe":   "severe",
}

# Interpretation codes → lab_flag enum
INTERP_MAP = {
    "N":  "normal",
    "H":  "high",
    "L":  "low",
    "HH": "critical",
    "LL": "critical",
    "A":  "critical",
}

# note_type enum values
NOTE_TYPE_MAP = {
    "progress note":        "progress_note",
    "discharge summary":    "discharge_summary",
    "consultation note":    "consultation_note",
    "operative report":     "operative_report",
    "nursing note":         "nursing_note",
}

# Discharge disposition mapping
DISPOSITION_MAP = {
    "home":              "home",
    "home health care":  "home",
    "skilled nursing":   "skilled_nursing",
    "transferred":       "transferred",
    "against medical advice": "ama",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def strip_ref(ref_str):
    """'Practitioner/abc' or 'urn:uuid:abc' → 'abc'"""
    if not ref_str:
        return None
    if ref_str.startswith("urn:uuid:"):
        return ref_str[9:]
    if "/" in ref_str:
        return ref_str.split("/")[-1]
    return ref_str


def get_loinc(resource):
    """Return first LOINC code from resource.code.coding, or None."""
    for coding in resource.get("code", {}).get("coding", []):
        if "loinc" in coding.get("system", "").lower():
            return coding.get("code")
    return None


def get_text(resource, field="code"):
    val = resource.get(field, {})
    if isinstance(val, list):
        val = val[0] if val else {}
    if not isinstance(val, dict):
        return None
    return val.get("text") or val.get("display")


def parse_datetime(dt_str):
    """Parse an ISO datetime string (including ±HH:MM offsets) to a timezone-aware datetime."""
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def parse_date(dt_str):
    if not dt_str:
        return None
    return dt_str[:10]  # YYYY-MM-DD


def safe_numeric(val):
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def decode_b64(data_str):
    """Decode base64-encoded string (DocumentReference, DiagnosticReport attachments)."""
    if not data_str:
        return None
    try:
        return base64.b64decode(data_str).decode("utf-8", errors="replace")
    except Exception:
        return None


def classify_note_type(raw_type):
    if not raw_type:
        return "progress_note"
    lower = raw_type.lower()
    for key, val in NOTE_TYPE_MAP.items():
        if key in lower:
            return val
    return "progress_note"


def classify_disposition(raw):
    if not raw:
        return None
    lower = raw.lower()
    for key, val in DISPOSITION_MAP.items():
        if key in lower:
            return val
    return None


# ---------------------------------------------------------------------------
# Schema application
# ---------------------------------------------------------------------------

def apply_schema(conn):
    """Apply schema.sql to the database if the EHR patients table does not yet exist."""
    cur = conn.cursor()
    cur.execute("SELECT to_regclass('ehr.patients')")
    if cur.fetchone()[0] is not None:
        print("Schema already applied — skipping.")
        return

    print("Applying schema...")
    sql = SCHEMA_FILE.read_text()

    # Strip semicolons inside -- comment lines so they don't produce false splits.
    sql_clean = re.sub(r"(--[^\n]*);", r"\1", sql)
    statements = [s.strip() for s in sql_clean.split(";") if s.strip()]
    for stmt in statements:
        try:
            conn.execute(stmt)
        except Exception as e:
            # Ignore harmless "already exists" errors on ENUMs / extensions
            msg = str(e).lower()
            if "already exists" in msg or "duplicate" in msg:
                conn.rollback()
            else:
                conn.rollback()
                raise
    conn.commit()
    print("Schema applied.")


# ---------------------------------------------------------------------------
# Synthea FHIR bundle parsing helpers
# ---------------------------------------------------------------------------

def index_bundle(bundle):
    """Return a dict of {resource_type: [resource, ...]} from a FHIR bundle."""
    idx = {}
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        rt = res.get("resourceType")
        if rt:
            idx.setdefault(rt, []).append(res)
    return idx


# ---------------------------------------------------------------------------
# PASS 1: Load organizations and practitioners from all bundle files
# ---------------------------------------------------------------------------

def load_orgs_practitioners(conn, fhir_files):
    print("Pass 1: Loading organizations and practitioners...")
    org_count = 0
    prac_count = 0

    for fhir_file in fhir_files:
        try:
            bundle = json.loads(fhir_file.read_text())
        except Exception as e:
            print(f"  WARNING: Could not parse {fhir_file.name}: {e}")
            continue

        idx = index_bundle(bundle)

        for org in idx.get("Organization", []):
            oid = org.get("id")
            if not oid:
                continue
            name = org.get("name", "Unknown")
            otype = get_text(org, "type") or (org.get("type", [{}])[0].get("text") if org.get("type") else None)
            addr_parts = []
            for a in org.get("address", []):
                parts = a.get("line", []) + [a.get("city",""), a.get("state",""), a.get("postalCode","")]
                addr_parts.append(", ".join(p for p in parts if p))
            address = "; ".join(addr_parts) or None
            phone = next((t.get("value") for t in org.get("telecom", []) if t.get("system") == "phone"), None)

            conn.execute("""
                INSERT INTO organizations (organization_id, name, type, address, phone)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (organization_id) DO NOTHING
            """, (oid, name, otype, address, phone))
            org_count += 1

        for prac in idx.get("Practitioner", []):
            pid = prac.get("id")
            if not pid:
                continue
            name_obj = (prac.get("name") or [{}])[0]
            given = " ".join(name_obj.get("given", []))
            family = name_obj.get("family", "")
            full_name = f"{given} {family}".strip() or "Unknown"
            npi = next((i.get("value") for i in prac.get("identifier", [])
                        if "npi" in i.get("system", "").lower()), None)
            qual = (prac.get("qualification") or [{}])[0].get("code", {}).get("text")

            conn.execute("""
                INSERT INTO practitioners (practitioner_id, full_name, npi, specialty)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (practitioner_id) DO NOTHING
            """, (pid, full_name, npi, qual))
            prac_count += 1

    conn.commit()
    print(f"  Loaded {org_count} org rows, {prac_count} practitioner rows (may include duplicates pre-conflict).")


# ---------------------------------------------------------------------------
# PASS 2: Load patient bundles
# ---------------------------------------------------------------------------

def load_patient_bundle(conn, bundle):
    """Load a single patient FHIR bundle. Returns True if loaded, False if skipped."""
    idx = index_bundle(bundle)

    patients = idx.get("Patient", [])
    if not patients:
        return False
    patient = patients[0]
    patient_id = patient.get("id")
    if not patient_id:
        return False

    # Idempotency guard: skip entire bundle if patient already loaded
    cur = conn.execute("SELECT 1 FROM patients WHERE patient_id = %s", (patient_id,))
    if cur.fetchone():
        return False

    # --- patients -----------------------------------------------------------
    name_obj = next((n for n in patient.get("name", []) if n.get("use") == "official"),
                    (patient.get("name") or [{}])[0])
    given = " ".join(name_obj.get("given", []))
    family = name_obj.get("family", "")
    full_name = f"{given} {family}".strip()

    preferred_name = None
    for n in patient.get("name", []):
        if n.get("use") == "nickname":
            preferred_name = " ".join(n.get("given", []))
            break

    dob = patient.get("birthDate")
    sex = patient.get("gender", "unknown")

    # Extensions
    race = ethnicity = gender_identity = None
    for ext in patient.get("extension", []):
        url = ext.get("url", "")
        if "us-core-race" in url:
            for sub in ext.get("extension", []):
                if sub.get("url") == "text":
                    race = sub.get("valueString")
        elif "us-core-ethnicity" in url:
            for sub in ext.get("extension", []):
                if sub.get("url") == "text":
                    ethnicity = sub.get("valueString")
        elif "genderIdentity" in url or "gender-identity" in url:
            gender_identity = ext.get("valueCodeableConcept", {}).get("text") or ext.get("valueString")

    lang = next((c.get("language", {}).get("coding", [{}])[0].get("display")
                 for c in patient.get("communication", [])
                 if c.get("preferred")), None)

    addr = (patient.get("address") or [{}])[0]
    phone = next((t.get("value") for t in patient.get("telecom", []) if t.get("system") == "phone"), None)
    email = next((t.get("value") for t in patient.get("telecom", []) if t.get("system") == "email"), None)

    # MRN
    mrn = next((i.get("value") for i in patient.get("identifier", [])
                if i.get("type", {}).get("coding", [{}])[0].get("code") == "MR"), patient_id[:8])

    # General practitioner
    gp_ref = (patient.get("generalPractitioner") or [{}])[0].get("reference")
    gp_id = strip_ref(gp_ref)

    # Look up GP name from practitioners table if possible
    pcp_name = None
    if gp_id:
        row = conn.execute("SELECT full_name FROM practitioners WHERE practitioner_id = %s", (gp_id,)).fetchone()
        if row:
            pcp_name = row[0]

    conn.execute("""
        INSERT INTO patients (
            patient_id, mrn, full_name, preferred_name, date_of_birth, biological_sex,
            gender_identity, primary_care_physician, race, ethnicity, primary_language,
            address_line1, city, state, zip, phone, email
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (patient_id) DO NOTHING
    """, (
        patient_id, mrn, full_name, preferred_name, dob, sex,
        gender_identity, pcp_name, race, ethnicity, lang,
        ", ".join(addr.get("line", [])) or None,
        addr.get("city"), addr.get("state"), addr.get("postalCode"),
        phone, email,
    ))

    # --- patient_emergency_contacts -----------------------------------------
    for contact in patient.get("contact", []):
        c_name_obj = contact.get("name", {})
        c_given = " ".join(c_name_obj.get("given", []))
        c_family = c_name_obj.get("family", "")
        c_name = f"{c_given} {c_family}".strip() or "Unknown"
        relationship = next(
            (r.get("coding", [{}])[0].get("display") for r in [contact.get("relationship", [{}])[0]] if contact.get("relationship")),
            None
        ) if contact.get("relationship") else None
        c_phone = next((t.get("value") for t in contact.get("telecom", []) if t.get("system") == "phone"), None)

        conn.execute("""
            INSERT INTO patient_emergency_contacts (patient_id, contact_name, relationship, phone)
            VALUES (%s, %s, %s, %s)
        """, (patient_id, c_name, relationship, c_phone))

    # --- encounters ---------------------------------------------------------
    # Build a map of encounter_id → encounter resource for later joins
    enc_map = {}  # fhir_id → encounter resource
    for enc in idx.get("Encounter", []):
        eid = enc.get("id")
        if not eid:
            continue
        enc_map[eid] = enc

        period = enc.get("period", {})
        admit_dt = parse_datetime(period.get("start"))
        discharge_dt = parse_datetime(period.get("end"))
        enc_date = parse_date(period.get("start"))

        # Encounter class → visit_type
        enc_class_code = enc.get("class", {}).get("code", "AMB")
        visit_type = ENCOUNTER_CLASS_MAP.get(enc_class_code, "outpatient")

        dept = None
        for t in enc.get("type", []):
            dept = t.get("text") or (t.get("coding") or [{}])[0].get("display")
            if dept:
                break

        facility_id = strip_ref(enc.get("serviceProvider", {}).get("reference"))

        # attending physician: participant with type ATND
        attending_id = None
        for part in enc.get("participant", []):
            for pt in part.get("type", []):
                for coding in pt.get("coding", []):
                    if coding.get("code") == "ATND":
                        attending_id = strip_ref(part.get("individual", {}).get("reference"))
        if not attending_id and enc.get("participant"):
            # Fallback: first participant
            attending_id = strip_ref(enc["participant"][0].get("individual", {}).get("reference"))

        chief_complaint = (enc.get("reasonCode") or [{}])[0].get("text")
        primary_diag = None
        for diag in enc.get("diagnosis", []):
            if diag.get("rank") == 1 or not primary_diag:
                primary_diag = diag.get("condition", {}).get("display") or diag.get("use", {}).get("text")
                if diag.get("rank") == 1:
                    break

        los_hours = None
        if admit_dt and discharge_dt:
            diff = discharge_dt - admit_dt
            los_hours = round(diff.total_seconds() / 3600, 2)

        status_raw = enc.get("status", "finished")
        status = "completed" if status_raw in ("finished", "completed") else ("active" if status_raw == "in-progress" else "cancelled")

        conn.execute("""
            INSERT INTO encounters (
                encounter_id, patient_id, encounter_date, visit_type, department,
                facility_id, attending_physician_id, chief_complaint, primary_diagnosis,
                status, admission_datetime, discharge_datetime, length_of_stay_hours
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (encounter_id) DO NOTHING
        """, (
            eid, patient_id, enc_date, visit_type, dept,
            facility_id, attending_id, chief_complaint, primary_diag,
            status, admit_dt, discharge_dt, los_hours,
        ))

        # --- encounter_care_team (non-attending participants) ---------------
        for part in enc.get("participant", []):
            part_id = strip_ref(part.get("individual", {}).get("reference"))
            if not part_id or part_id == attending_id:
                continue
            role = (part.get("type") or [{}])[0].get("coding", [{}])[0].get("display") or "clinician"
            conn.execute("""
                INSERT INTO encounter_care_team (encounter_id, practitioner_id, role)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (eid, part_id, role))

        # --- encounter_discharge (inpatient / ER) ---------------------------
        hosp = enc.get("hospitalization")
        if hosp and visit_type in ("inpatient", "emergency"):
            disp_text = hosp.get("dischargeDisposition", {}).get("coding", [{}])[0].get("display")
            disp = classify_disposition(disp_text)
            conn.execute("""
                INSERT INTO encounter_discharge (encounter_id, disposition)
                VALUES (%s, %s)
                ON CONFLICT (encounter_id) DO NOTHING
            """, (eid, disp))

    # --- conditions ---------------------------------------------------------
    cond_map = {}  # fhir_id → condition_id (same as fhir_id)
    for cond in idx.get("Condition", []):
        cid = cond.get("id")
        if not cid:
            continue
        cond_map[cid] = cid

        name = cond.get("code", {}).get("text") or (cond.get("code", {}).get("coding") or [{}])[0].get("display", "Unknown")
        snomed = next((c.get("code") for c in cond.get("code", {}).get("coding", [])
                       if "snomed" in c.get("system", "").lower()), None)
        icd10 = next((c.get("code") for c in cond.get("code", {}).get("coding", [])
                      if "icd-10" in c.get("system", "").lower() or "icd10" in c.get("system", "").lower()), None)

        # category
        cat_code = (cond.get("category") or [{}])[0].get("coding", [{}])[0].get("code", "")
        if "chronic" in cat_code.lower() or cat_code == "55607006":
            category = "chronic"
        elif "encounter-diagnosis" in cat_code.lower() or cat_code == "439401001":
            category = "acute"
        else:
            category = "chronic"  # default to chronic for problem-list items

        onset = parse_date(cond.get("onsetDateTime") or cond.get("onsetPeriod", {}).get("start"))
        resolution = parse_date(cond.get("abatementDateTime") or cond.get("abatementPeriod", {}).get("end"))

        clin_status = cond.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "active")
        if clin_status in ("resolved", "inactive", "remission"):
            status = "resolved"
        elif clin_status == "recurrence":
            status = "recurrence"
        else:
            status = "active"

        first_enc_ref = strip_ref(cond.get("encounter", {}).get("reference"))
        recorder_ref = strip_ref(cond.get("recorder", {}).get("reference"))

        conn.execute("""
            INSERT INTO conditions (
                condition_id, patient_id, first_encounter_id, condition_name,
                snomed_code, icd10_code, category, onset_date, resolution_date,
                status, treating_physician_id
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (condition_id) DO NOTHING
        """, (
            cid, patient_id, first_enc_ref, name,
            snomed, icd10, category, onset, resolution,
            status, recorder_ref,
        ))

    # --- encounter_conditions -----------------------------------------------
    for enc in idx.get("Encounter", []):
        eid = enc.get("id")
        if not eid:
            continue
        for diag in enc.get("diagnosis", []):
            cond_ref = strip_ref(diag.get("condition", {}).get("reference"))
            if not cond_ref or cond_ref not in cond_map:
                continue
            rank = diag.get("rank")
            is_primary = (rank == 1)
            conn.execute("""
                INSERT INTO encounter_conditions (encounter_id, condition_id, is_primary, diagnosis_rank)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (encounter_id, condition_id) DO NOTHING
            """, (eid, cond_ref, is_primary, rank))

    # --- vitals -------------------------------------------------------------
    # Group vital-sign observations by (encounter_id, effectiveDateTime) for pivot
    vitals_groups = {}  # (enc_id, dt_str) → {col: value}
    for obs in idx.get("Observation", []):
        cat_codes = [c.get("code") for cat in obs.get("category", []) for c in cat.get("coding", [])]
        if "vital-signs" not in cat_codes:
            continue

        loinc = get_loinc(obs)
        enc_ref = strip_ref(obs.get("encounter", {}).get("reference"))
        dt_str = obs.get("effectiveDateTime", "")
        if not enc_ref or not dt_str:
            continue

        key = (enc_ref, dt_str)
        if key not in vitals_groups:
            vitals_groups[key] = {"recorded_by": None}

        perfs = obs.get("performer", [])
        if perfs and not vitals_groups[key]["recorded_by"]:
            vitals_groups[key]["recorded_by"] = perfs[0].get("display")

        # Handle compound BP observation (55284-4) which carries component LOINCs
        if obs.get("component"):
            for comp in obs["component"]:
                comp_loinc = None
                for coding in comp.get("code", {}).get("coding", []):
                    if "loinc" in coding.get("system", "").lower():
                        comp_loinc = coding.get("code")
                if comp_loinc and comp_loinc in VITALS_LOINC:
                    val = comp.get("valueQuantity", {}).get("value")
                    col = VITALS_LOINC[comp_loinc]
                    vitals_groups[key][col] = safe_numeric(val)
        elif loinc and loinc in VITALS_LOINC:
            val = obs.get("valueQuantity", {}).get("value")
            col = VITALS_LOINC[loinc]
            vitals_groups[key][col] = safe_numeric(val)

    for (enc_id, dt_str), vals in vitals_groups.items():
        if enc_id not in enc_map:
            continue
        recorded_at = parse_datetime(dt_str)
        conn.execute("""
            INSERT INTO vitals (
                patient_id, encounter_id, recorded_at, recorded_by,
                systolic_bp, diastolic_bp, heart_rate, temperature_c,
                respiratory_rate, spo2_pct, height_cm, weight_kg, bmi
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            patient_id, enc_id, recorded_at, vals.get("recorded_by"),
            vals.get("systolic_bp"), vals.get("diastolic_bp"), vals.get("heart_rate"),
            vals.get("temperature_c"), vals.get("respiratory_rate"), vals.get("spo2_pct"),
            vals.get("height_cm"), vals.get("weight_kg"), vals.get("bmi"),
        ))

    # --- lab_results --------------------------------------------------------
    # Pre-fetch panel_id lookup from the already-seeded lab_panels table
    panel_rows = conn.execute("SELECT panel_id, panel_name FROM lab_panels").fetchall()
    panel_id_map = {name: pid for pid, name in panel_rows}

    for obs in idx.get("Observation", []):
        cat_codes = [c.get("code") for cat in obs.get("category", []) for c in cat.get("coding", [])]
        if "laboratory" not in cat_codes:
            continue

        obs_id = obs.get("id")
        if not obs_id:
            continue

        loinc = get_loinc(obs)
        enc_ref = strip_ref(obs.get("encounter", {}).get("reference"))
        if not enc_ref or enc_ref not in enc_map:
            continue

        test_name = obs.get("code", {}).get("text") or (obs.get("code", {}).get("coding") or [{}])[0].get("display", "Unknown")
        collected_at = parse_datetime(obs.get("effectiveDateTime"))
        if not collected_at:
            continue

        # Numeric vs text result
        vq = obs.get("valueQuantity", {})
        result_value = safe_numeric(vq.get("value"))
        result_value_text = obs.get("valueString") or obs.get("valueCodeableConcept", {}).get("text")
        unit = vq.get("unit") or vq.get("code")

        ref = (obs.get("referenceRange") or [{}])[0]
        ref_low = safe_numeric(ref.get("low", {}).get("value"))
        ref_high = safe_numeric(ref.get("high", {}).get("value"))

        interp_code = (obs.get("interpretation") or [{}])[0].get("coding", [{}])[0].get("code")
        interp_flag = INTERP_MAP.get(interp_code)
        if interp_flag is None and result_value is not None and ref_low is not None and ref_high is not None:
            if result_value < ref_low:
                interp_flag = "low"
            elif result_value > ref_high:
                interp_flag = "high"
            else:
                interp_flag = "normal"

        panel_name = LAB_PANEL_LOINC.get(loinc, "Other")
        panel_id = panel_id_map.get(panel_name, panel_id_map.get("Other"))

        ordering_id = strip_ref((obs.get("performer") or [{}])[0].get("reference"))

        conn.execute("""
            INSERT INTO lab_results (
                lab_result_id, patient_id, encounter_id, panel_id, ordering_physician_id,
                test_name, loinc_code, result_value, result_value_text, unit,
                reference_range_low, reference_range_high, interpretation_flag, collected_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (lab_result_id) DO NOTHING
        """, (
            obs_id, patient_id, enc_ref, panel_id, ordering_id,
            test_name, loinc, result_value, result_value_text, unit,
            ref_low, ref_high, interp_flag, collected_at,
        ))

    # --- medications --------------------------------------------------------
    # Synthea never emits dispenseRequest; use encounter class as the heuristic
    # for whether a medication was sent as a discharge prescription.
    inpatient_enc_ids = {
        eid for eid, enc in enc_map.items()
        if ENCOUNTER_CLASS_MAP.get(enc.get("class", {}).get("code", "AMB"), "outpatient")
        in ("inpatient", "emergency")
    }

    for med_req in idx.get("MedicationRequest", []):
        mid = med_req.get("id")
        if not mid:
            continue

        drug_name = (med_req.get("medicationCodeableConcept") or {}).get("text") or \
                    (med_req.get("medicationCodeableConcept", {}).get("coding") or [{}])[0].get("display", "Unknown")
        rxnorm = next((c.get("code") for c in med_req.get("medicationCodeableConcept", {}).get("coding", [])
                       if "rxnorm" in c.get("system", "").lower()), None)

        enc_ref = strip_ref(med_req.get("encounter", {}).get("reference"))
        requester_ref = strip_ref(med_req.get("requester", {}).get("reference"))
        ordered_at = parse_datetime(med_req.get("authoredOn"))

        dosage = (med_req.get("dosageInstruction") or [{}])[0]
        dose_val = dosage.get("doseAndRate", [{}])[0].get("doseQuantity", {})
        dose_str = f"{dose_val.get('value','')} {dose_val.get('unit','')}".strip() or None
        route = dosage.get("route", {}).get("text")
        frequency = dosage.get("timing", {}).get("code", {}).get("text")

        bounds = dosage.get("timing", {}).get("repeat", {}).get("boundsPeriod", {})
        start_date = parse_date(bounds.get("start"))
        end_date = parse_date(bounds.get("end"))

        status_raw = med_req.get("status", "active")
        status_map = {"active": "active", "completed": "completed",
                      "stopped": "discontinued", "on-hold": "on_hold",
                      "cancelled": "discontinued", "entered-in-error": "discontinued"}
        med_status = status_map.get(status_raw, "active")

        if enc_ref and enc_ref in inpatient_enc_ids:
            delivery_type = "sent_as_prescription"
        elif med_req.get("dispenseRequest"):
            delivery_type = "sent_as_prescription"
        else:
            delivery_type = "administered_during_visit"

        refill_count = (med_req.get("dispenseRequest") or {}).get("numberOfRepeatsAllowed", 0)

        conn.execute("""
            INSERT INTO medications (
                medication_id, patient_id, ordering_encounter_id, prescribed_by_id,
                drug_name, rxnorm_code, dose, route, frequency,
                ordered_at, start_date, end_date, status, delivery_type, refill_count
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (medication_id) DO NOTHING
        """, (
            mid, patient_id, enc_ref, requester_ref,
            drug_name, rxnorm, dose_str, route, frequency,
            ordered_at, start_date, end_date, med_status, delivery_type, refill_count,
        ))

    # --- allergies ----------------------------------------------------------
    for allergy in idx.get("AllergyIntolerance", []):
        aid = allergy.get("id")
        if not aid:
            continue

        allergen = allergy.get("code", {}).get("text") or (allergy.get("code", {}).get("coding") or [{}])[0].get("display", "Unknown")
        reaction_list = allergy.get("reaction", [])
        reaction_detail = None
        severity_raw = None
        if reaction_list:
            reaction_detail = (reaction_list[0].get("manifestation") or [{}])[0].get("text") or \
                              (reaction_list[0].get("manifestation") or [{}])[0].get("coding", [{}])[0].get("display")
            severity_raw = reaction_list[0].get("severity")

        severity = ALLERGY_SEVERITY_MAP.get(severity_raw)
        recorded_date = parse_date(allergy.get("recordedDate"))
        recorder_ref = strip_ref(allergy.get("recorder", {}).get("reference"))

        conn.execute("""
            INSERT INTO allergies (
                allergy_id, patient_id, recorder_id, allergen,
                reaction_detail, severity, recorded_date
            ) VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (allergy_id) DO NOTHING
        """, (aid, patient_id, recorder_ref, allergen, reaction_detail, severity, recorded_date))

    # --- procedures ---------------------------------------------------------
    for proc in idx.get("Procedure", []):
        pid_proc = proc.get("id")
        if not pid_proc:
            continue

        enc_ref = strip_ref(proc.get("encounter", {}).get("reference"))
        if not enc_ref or enc_ref not in enc_map:
            continue

        proc_name = proc.get("code", {}).get("text") or (proc.get("code", {}).get("coding") or [{}])[0].get("display", "Unknown")
        cpt = next((c.get("code") for c in proc.get("code", {}).get("coding", [])
                    if "cpt" in c.get("system", "").lower()), None)
        snomed = next((c.get("code") for c in proc.get("code", {}).get("coding", [])
                       if "snomed" in c.get("system", "").lower()), None)

        performed_at = parse_datetime(proc.get("performedDateTime"))
        if not performed_at:
            pp = proc.get("performedPeriod", {})
            performed_at = parse_datetime(pp.get("start"))

        performer_ref = strip_ref((proc.get("performer") or [{}])[0].get("actor", {}).get("reference"))

        conn.execute("""
            INSERT INTO procedures (
                procedure_id, patient_id, encounter_id, performing_physician_id,
                procedure_name, cpt_code, snomed_code, performed_at, status
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (procedure_id) DO NOTHING
        """, (
            pid_proc, patient_id, enc_ref, performer_ref,
            proc_name, cpt, snomed, performed_at, proc.get("status", "completed"),
        ))

    # --- imaging_studies ----------------------------------------------------
    modality_map = {
        "CR": "x_ray", "DX": "x_ray", "RG": "x_ray",
        "CT": "ct",
        "MR": "mri",
        "US": "ultrasound",
        "ES": "echocardiogram",
        "PT": "pet",
        "MG": "mammogram",
    }
    lateral_map = {
        "left": "left", "right": "right", "bilateral": "bilateral",
    }

    for img in idx.get("ImagingStudy", []):
        img_id = img.get("id")
        if not img_id:
            continue

        enc_ref = strip_ref(img.get("encounter", {}).get("reference"))
        if not enc_ref or enc_ref not in enc_map:
            continue

        series = (img.get("series") or [{}])[0]
        mod_code = series.get("modality", {}).get("code", "")
        modality = modality_map.get(mod_code, "other")
        body_region = series.get("bodySite", {}).get("display")
        lat_text = (series.get("laterality") or {}).get("display", "").lower()
        laterality = lateral_map.get(lat_text, "not_applicable")

        indication = (img.get("reasonCode") or [{}])[0].get("text")
        date_performed = parse_date(img.get("started"))

        ordering_ref = strip_ref(img.get("referrer", {}).get("reference"))
        facility_ref = strip_ref(img.get("location", {}).get("reference"))

        conn.execute("""
            INSERT INTO imaging_studies (
                imaging_study_id, patient_id, encounter_id, ordering_physician_id,
                performing_facility_id, modality, body_region, laterality,
                clinical_indication, date_performed, status
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (imaging_study_id) DO NOTHING
        """, (
            img_id, patient_id, enc_ref, ordering_ref,
            facility_ref, modality, body_region, laterality,
            indication, date_performed, "completed",
        ))

    # --- diagnostic_reports -------------------------------------------------
    for dr in idx.get("DiagnosticReport", []):
        dr_id = dr.get("id")
        if not dr_id:
            continue

        enc_ref = strip_ref(dr.get("encounter", {}).get("reference"))
        if not enc_ref or enc_ref not in enc_map:
            continue

        report_type = dr.get("code", {}).get("text")
        issued_at = parse_datetime(dr.get("issued"))
        authored_by = (dr.get("performer") or [{}])[0].get("display")

        # Decode narrative from presentedForm
        narrative = None
        for form in dr.get("presentedForm", []):
            if form.get("data"):
                narrative = decode_b64(form["data"])
                break
        if not narrative:
            narrative = dr.get("conclusion")

        conn.execute("""
            INSERT INTO diagnostic_reports (
                report_id, patient_id, encounter_id, authored_by,
                report_type, report_narrative, issued_at, status
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (report_id) DO NOTHING
        """, (
            dr_id, patient_id, enc_ref, authored_by,
            report_type, narrative, issued_at, dr.get("status"),
        ))

    # --- clinical_notes (generated from encounter data) ---------------------
    # Synthea does not emit DocumentReference resources. Generate one note per
    # encounter from the available FHIR data (encounter type, chief complaint,
    # conditions, and procedures linked to that encounter).
    enc_conditions: dict[str, list[str]] = {}
    for cond in idx.get("Condition", []):
        cond_enc = strip_ref(cond.get("encounter", {}).get("reference"))
        if cond_enc:
            name = cond.get("code", {}).get("text") or \
                   (cond.get("code", {}).get("coding") or [{}])[0].get("display", "")
            if name:
                enc_conditions.setdefault(cond_enc, []).append(name)

    enc_procedures: dict[str, list[str]] = {}
    for proc in idx.get("Procedure", []):
        proc_enc = strip_ref(proc.get("encounter", {}).get("reference"))
        if proc_enc:
            name = proc.get("code", {}).get("text") or \
                   (proc.get("code", {}).get("coding") or [{}])[0].get("display", "")
            if name:
                enc_procedures.setdefault(proc_enc, []).append(name)

    for enc_id, enc in enc_map.items():
        period = enc.get("period", {})
        enc_date = parse_date(period.get("start"))
        admit_dt = parse_datetime(period.get("start"))
        discharge_dt = parse_datetime(period.get("end"))

        enc_class_code = enc.get("class", {}).get("code", "AMB")
        vtype = ENCOUNTER_CLASS_MAP.get(enc_class_code, "outpatient")
        note_type = "discharge_summary" if vtype in ("inpatient", "emergency") else "progress_note"
        authored_at = discharge_dt or admit_dt

        enc_type_text = None
        for t in enc.get("type", []):
            enc_type_text = t.get("text") or (t.get("coding") or [{}])[0].get("display")
            if enc_type_text:
                break
        enc_type_text = enc_type_text or vtype.replace("_", " ").title()

        chief_complaint = (enc.get("reasonCode") or [{}])[0].get("text")

        lines = [f"{enc_type_text} — {enc_date}"]
        if chief_complaint:
            lines.append(f"Chief complaint: {chief_complaint}")
        diagnoses = enc_conditions.get(enc_id, [])
        if diagnoses:
            lines.append(f"Diagnoses: {', '.join(diagnoses)}")
        procedures = enc_procedures.get(enc_id, [])
        if procedures:
            lines.append(f"Procedures: {', '.join(procedures)}")
        note_text = "\n".join(lines)

        # attending_id from enc participant (already extracted when building enc_map)
        attending_id = None
        for part in enc.get("participant", []):
            for pt in part.get("type", []):
                for coding in pt.get("coding", []):
                    if coding.get("code") == "ATND":
                        attending_id = strip_ref(part.get("individual", {}).get("reference"))
            if attending_id:
                break
        if not attending_id and enc.get("participant"):
            attending_id = strip_ref(enc["participant"][0].get("individual", {}).get("reference"))

        conn.execute("""
            INSERT INTO clinical_notes (
                note_id, patient_id, encounter_id, author_id,
                note_type, note_text, authored_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (note_id) DO NOTHING
        """, (str(uuid.uuid4()), patient_id, enc_id, attending_id, note_type, note_text, authored_at))

    # --- care_plans ---------------------------------------------------------
    for cp in idx.get("CarePlan", []):
        cp_id = cp.get("id")
        if not cp_id:
            continue

        enc_ref = strip_ref(cp.get("encounter", {}).get("reference"))
        if not enc_ref or enc_ref not in enc_map:
            continue

        follow_up = (cp.get("note") or [{}])[0].get("text")

        conn.execute("""
            INSERT INTO care_plans (care_plan_id, patient_id, encounter_id, follow_up_instructions)
            VALUES (%s,%s,%s,%s)
            ON CONFLICT (care_plan_id) DO NOTHING
        """, (cp_id, patient_id, enc_ref, follow_up))

        # Goals
        for goal_ref in cp.get("goal", []):
            goal_id = strip_ref(goal_ref.get("reference"))
            # Goals are separate Goal resources — get from idx
            goal_text = f"Goal: {goal_id}"
            for g in idx.get("Goal", []):
                if g.get("id") == goal_id:
                    goal_text = g.get("description", {}).get("text", goal_text)
                    break
            conn.execute("""
                INSERT INTO care_plan_goals (care_plan_id, goal_description)
                VALUES (%s, %s)
            """, (cp_id, goal_text))

        # Interventions from activity
        for act in cp.get("activity", []):
            detail = act.get("detail", {})
            desc = detail.get("description") or detail.get("code", {}).get("text")
            if desc:
                conn.execute("""
                    INSERT INTO care_plan_interventions (care_plan_id, intervention_description)
                    VALUES (%s, %s)
                """, (cp_id, desc))

    # --- immunizations ------------------------------------------------------
    for imm in idx.get("Immunization", []):
        imm_id = imm.get("id")
        if not imm_id:
            continue

        vaccine_name = imm.get("vaccineCode", {}).get("text") or \
                       (imm.get("vaccineCode", {}).get("coding") or [{}])[0].get("display", "Unknown")
        cvx = next((c.get("code") for c in imm.get("vaccineCode", {}).get("coding", [])
                    if "cvx" in c.get("system", "").lower()), None)
        date_admin = parse_date(imm.get("occurrenceDateTime"))
        if not date_admin:
            continue

        enc_ref = strip_ref(imm.get("encounter", {}).get("reference"))
        if enc_ref and enc_ref not in enc_map:
            enc_ref = None

        admin_by = (imm.get("performer") or [{}])[0].get("actor", {}).get("display")
        facility_ref = strip_ref(imm.get("location", {}).get("reference"))
        if facility_ref and not conn.execute(
            "SELECT 1 FROM organizations WHERE organization_id=%s", (facility_ref,)
        ).fetchone():
            facility_ref = None

        dose_num = None
        series_total = None
        for pa in imm.get("protocolApplied", []):
            dose_num = pa.get("doseNumberPositiveInt") or pa.get("doseNumberString")
            series_total = pa.get("seriesDosesPositiveInt") or pa.get("seriesDosesString")
            try:
                dose_num = int(dose_num) if dose_num else None
                series_total = int(series_total) if series_total else None
            except (TypeError, ValueError):
                dose_num = series_total = None

        status_raw = imm.get("status", "completed")
        imm_status = "completed" if status_raw == "completed" else \
                     ("not_done" if status_raw == "not-done" else "refused")

        conn.execute("""
            INSERT INTO immunizations (
                immunization_id, patient_id, encounter_id, administered_by,
                administered_at_facility_id, vaccine_name, cvx_code, date_administered,
                dose_number, series_total, route, site, lot_number, manufacturer, status
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (immunization_id) DO NOTHING
        """, (
            imm_id, patient_id, enc_ref, admin_by,
            facility_ref, vaccine_name, cvx, date_admin,
            dose_num, series_total,
            imm.get("route", {}).get("text"),
            imm.get("site", {}).get("text"),
            imm.get("lotNumber"),
            imm.get("manufacturer", {}).get("display"),
            imm_status,
        ))

    # --- social_history -----------------------------------------------------
    social = {}
    housing_map = {
        "stable": "stable", "unstable": "unstable",
        "homeless": "homeless", "houseless": "homeless",
    }

    for obs in idx.get("Observation", []):
        cat_codes = [c.get("code") for cat in obs.get("category", []) for c in cat.get("coding", [])]
        if "social-history" not in cat_codes and "survey" not in cat_codes:
            continue
        loinc = get_loinc(obs)
        if not loinc or loinc not in SOCIAL_LOINC:
            continue

        col = SOCIAL_LOINC[loinc]
        val = obs.get("valueCodeableConcept", {}).get("text") or \
              obs.get("valueString") or \
              str(obs.get("valueQuantity", {}).get("value", ""))

        if col == "housing_status":
            val = housing_map.get(val.lower(), None)
        social[col] = val

    if social:
        conn.execute("""
            INSERT INTO social_history (patient_id, smoking_status, alcohol_frequency,
                occupation, housing_status, education_level)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (patient_id) DO UPDATE SET
                smoking_status = EXCLUDED.smoking_status,
                alcohol_frequency = EXCLUDED.alcohol_frequency,
                occupation = EXCLUDED.occupation,
                housing_status = EXCLUDED.housing_status,
                education_level = EXCLUDED.education_level,
                last_updated_at = NOW()
        """, (
            patient_id,
            social.get("smoking_status"), social.get("alcohol_frequency"),
            social.get("occupation"), social.get("housing_status"),
            social.get("education_level"),
        ))

    return True


# ---------------------------------------------------------------------------
# Post-processing steps
# ---------------------------------------------------------------------------

def run_post_processing(conn):
    print("Running post-processing...")

    # 1. discharge_prescriptions: link medications to encounter_discharge records
    #    for prescriptions with delivery_type='sent_as_prescription'
    print("  Post-processing: discharge_prescriptions...")
    conn.execute("""
        INSERT INTO discharge_prescriptions (discharge_id, medication_id)
        SELECT ed.discharge_id, m.medication_id
        FROM encounter_discharge ed
        JOIN medications m ON m.ordering_encounter_id = ed.encounter_id
        WHERE m.delivery_type = 'sent_as_prescription'
        ON CONFLICT (discharge_id, medication_id) DO NOTHING
    """)

    # 2. medications.renewal_due_date: UPDATE for active prescriptions with an end_date
    print("  Post-processing: medications renewal_due_date...")
    conn.execute("""
        UPDATE medications
        SET renewal_due_date = (end_date - INTERVAL '30 days')::date
        WHERE end_date IS NOT NULL
          AND status = 'active'
          AND renewal_due_date IS NULL
    """)

    # 3. patient_alerts: derive from severe allergies
    #    patient_alerts has no unique constraint on (patient_id, alert_text), so we
    #    guard with NOT EXISTS instead of ON CONFLICT to avoid duplicates on re-run.
    print("  Post-processing: patient_alerts (severe allergies)...")
    conn.execute("""
        INSERT INTO patient_alerts (patient_id, alert_text, severity, is_active, recorded_at)
        SELECT
            a.patient_id,
            'Severe allergy: ' || a.allergen,
            'high',
            TRUE,
            NOW()
        FROM allergies a
        WHERE a.severity = 'severe'
          AND NOT EXISTS (
              SELECT 1 FROM patient_alerts pa
              WHERE pa.patient_id = a.patient_id
                AND pa.alert_text = 'Severe allergy: ' || a.allergen
          )
    """)

    # 4. condition_lab_associations: LOINC-based heuristic links
    #    Map known condition SNOMED/names to relevant lab LOINC codes
    print("  Post-processing: condition_lab_associations...")
    # Diabetes → HbA1c (4548-4), Glucose (2345-7)
    conn.execute("""
        INSERT INTO condition_lab_associations (condition_id, lab_result_id)
        SELECT DISTINCT c.condition_id, lr.lab_result_id
        FROM conditions c
        JOIN lab_results lr ON lr.patient_id = c.patient_id
            AND lr.loinc_code IN ('4548-4', '2345-7')
        WHERE c.condition_name ILIKE '%diabetes%'
           OR c.snomed_code IN ('44054006','73211009','11687002')
        ON CONFLICT (condition_id, lab_result_id) DO NOTHING
    """)
    # Hypertension → BMP / metabolic panel markers
    conn.execute("""
        INSERT INTO condition_lab_associations (condition_id, lab_result_id)
        SELECT DISTINCT c.condition_id, lr.lab_result_id
        FROM conditions c
        JOIN lab_results lr ON lr.patient_id = c.patient_id
            AND lr.loinc_code IN ('2951-2','2823-3','2160-0','3094-0')
        WHERE c.condition_name ILIKE '%hypertension%'
           OR c.snomed_code IN ('38341003','59621000')
        ON CONFLICT (condition_id, lab_result_id) DO NOTHING
    """)
    # Hyperlipidemia / high cholesterol → Lipid panel
    conn.execute("""
        INSERT INTO condition_lab_associations (condition_id, lab_result_id)
        SELECT DISTINCT c.condition_id, lr.lab_result_id
        FROM conditions c
        JOIN lab_results lr ON lr.patient_id = c.patient_id
            AND lr.loinc_code IN ('2093-3','18262-6','2085-9','2571-8')
        WHERE c.condition_name ILIKE '%lipid%'
           OR c.condition_name ILIKE '%cholesterol%'
           OR c.snomed_code IN ('13644009','55822004')
        ON CONFLICT (condition_id, lab_result_id) DO NOTHING
    """)
    # Hypothyroidism → TSH
    conn.execute("""
        INSERT INTO condition_lab_associations (condition_id, lab_result_id)
        SELECT DISTINCT c.condition_id, lr.lab_result_id
        FROM conditions c
        JOIN lab_results lr ON lr.patient_id = c.patient_id
            AND lr.loinc_code IN ('3016-3','11579-0')
        WHERE c.condition_name ILIKE '%thyroid%'
           OR c.snomed_code IN ('40930008','44054006')
        ON CONFLICT (condition_id, lab_result_id) DO NOTHING
    """)

    conn.commit()
    print("Post-processing complete.")


# ---------------------------------------------------------------------------
# Synthea Docker generation
# ---------------------------------------------------------------------------

def generate_fhir_data(count: int = 200, seed: int = 42) -> None:
    """Generate synthetic FHIR R4 bundles using the built-in Python generator.
    Writes to OUTPUT_FHIR (/tmp/synthea_output/fhir) — no Docker or Java needed.
    """
    import shutil
    if SYNTHEA_MOUNT.exists():
        shutil.rmtree(SYNTHEA_MOUNT)
    SYNTHEA_MOUNT.mkdir(parents=True, exist_ok=True)
    OUTPUT_FHIR.mkdir(parents=True, exist_ok=True)

    # Import and run the bundled generator
    sys.path.insert(0, str(SCRIPT_DIR))
    from generate_fhir import generate_shared_resources, generate_patient_bundle
    import random as _random
    import json as _json
    _random.seed(seed)

    print(f"Generating {count} synthetic FHIR R4 bundles → {OUTPUT_FHIR}")
    orgs, practitioners = generate_shared_resources()

    # Shared bundle (orgs + practitioners) for ETL pass 1
    shared_entries = [{"resource": org} for _, org in orgs] + \
                     [{"resource": prac} for _, prac in practitioners]
    shared_bundle = {"resourceType": "Bundle", "type": "collection", "entry": shared_entries}
    (OUTPUT_FHIR / "shared_organizations.json").write_text(_json.dumps(shared_bundle))

    org_ids = [(oid, None) for oid, _ in orgs]
    prac_ids = [(pid, None) for pid, _ in practitioners]

    for i in range(count):
        bundle, pat_id = generate_patient_bundle(org_ids, prac_ids)
        (OUTPUT_FHIR / f"patient_{i:04d}_{pat_id}.json").write_text(_json.dumps(bundle))
        if (i + 1) % 25 == 0 or i + 1 == count:
            print(f"  {i+1}/{count} bundles written")

    print("FHIR generation complete.")


# ---------------------------------------------------------------------------
# Row count summary
# ---------------------------------------------------------------------------

def print_counts(conn):
    tables = [
        "patients", "encounters", "conditions", "vitals", "lab_results",
        "medications", "allergies", "procedures", "imaging_studies",
        "diagnostic_reports", "clinical_notes", "care_plans", "immunizations",
        "social_history", "discharge_prescriptions", "condition_lab_associations",
        "patient_alerts",
    ]
    print("\nRow counts:")
    for t in tables:
        row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
        print(f"  {t:<40} {row[0]:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Load Synthea FHIR data into Neon Postgres EHR schema")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip FHIR generation; use existing files in /tmp/synthea_output/fhir/")
    parser.add_argument("--schema-only", action="store_true",
                        help="Apply schema only; do not load any FHIR data")
    parser.add_argument("--count", type=int, default=200,
                        help="Number of patients to generate (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Stop after loading this many patients into Postgres")
    args = parser.parse_args()

    db_url = load_database_url()
    print(f"Connecting to Neon Postgres...")

    conn = psycopg.connect(db_url, autocommit=False)
    # All EHR tables live in the 'ehr' schema; set search_path so unqualified
    # table names in every INSERT/SELECT resolve there, not to public.
    conn.execute("SET search_path TO ehr, public")

    try:
        apply_schema(conn)

        if args.schema_only:
            print("--schema-only: done.")
            return

        # Generate FHIR data if needed
        if not args.skip_generate:
            generate_fhir_data(count=args.count, seed=args.seed)
        else:
            print("--skip-generate: using existing FHIR files in /tmp/synthea_output/fhir/.")

        if not OUTPUT_FHIR.exists() or not any(OUTPUT_FHIR.glob("*.json")):
            sys.exit(f"ERROR: No FHIR JSON files found in {OUTPUT_FHIR}. "
                     "Run without --skip-docker or generate data first.")

        fhir_files = sorted(OUTPUT_FHIR.glob("*.json"))
        print(f"Found {len(fhir_files)} FHIR JSON files.")

        # Pass 1: organizations + practitioners (shared across all bundles)
        load_orgs_practitioners(conn, fhir_files)

        # Pass 2: one patient bundle at a time
        print("Pass 2: Loading patient bundles...")
        loaded = 0
        skipped = 0
        errors = 0

        for fhir_file in fhir_files:
            try:
                bundle = json.loads(fhir_file.read_text())
            except Exception as e:
                print(f"  WARNING: Could not parse {fhir_file.name}: {e}")
                errors += 1
                continue

            # Only process bundles that contain a Patient resource
            has_patient = any(
                e.get("resource", {}).get("resourceType") == "Patient"
                for e in bundle.get("entry", [])
            )
            if not has_patient:
                skipped += 1
                continue

            try:
                result = load_patient_bundle(conn, bundle)
                if result:
                    conn.commit()
                    loaded += 1
                    if loaded % 10 == 0:
                        print(f"  Loaded {loaded} patients...")
                    if args.max_patients:
                        total = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
                        if total >= args.max_patients:
                            print(f"  Reached --max-patients={args.max_patients} total, stopping.")
                            break
                else:
                    conn.rollback()
                    skipped += 1
            except Exception as e:
                conn.rollback()
                print(f"  ERROR loading {fhir_file.name}: {e}")
                errors += 1

        print(f"Patient bundles: {loaded} loaded, {skipped} skipped, {errors} errors.")

        run_post_processing(conn)

        print_counts(conn)

    finally:
        conn.close()

    print("\nETL complete.")


if __name__ == "__main__":
    main()
