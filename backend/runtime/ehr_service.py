"""
runtime.ehr_service
-------------------
Read-only queries against the ehr.* schema (Synthea FHIR data).
No PDP / RLS — all data is synthetic and non-session-scoped.
Each function gets and closes its own connection.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from runtime.seed.db import get_conn


def _ehr_conn():
    conn = get_conn()
    conn.execute("SET search_path TO ehr, public")
    return conn


def _s(v: Any) -> Any:
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def _row(cur, cols: list[str]) -> dict:
    row = cur.fetchone()
    return {c: _s(v) for c, v in zip(cols, row)} if row else {}


def _rows(cur, cols: list[str]) -> list[dict]:
    return [{c: _s(v) for c, v in zip(cols, row)} for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Patient list
# ---------------------------------------------------------------------------

def get_ehr_assignments(user_id: str) -> list[str]:
    """Return EHR patient UUIDs (as strings) assigned to this practitioner."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT patient_id::text FROM ehr_patient_assignments WHERE practitioner_id = %s",
            (user_id,),
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def list_patients(page: int = 1, limit: int = 20, search: str = "") -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        offset = (page - 1) * limit
        if search:
            cur.execute(
                """
                SELECT p.patient_id::text, p.full_name, p.date_of_birth::text,
                       p.biological_sex, p.mrn,
                       COALESCE(
                           (SELECT string_agg(c.condition_name, ', ' ORDER BY c.onset_date DESC)
                            FROM conditions c
                            WHERE c.patient_id = p.patient_id AND c.status = 'active'
                            LIMIT 3),
                           ''
                       ) AS headline
                FROM patients p
                WHERE p.full_name ILIKE %s OR p.mrn ILIKE %s
                ORDER BY p.full_name
                LIMIT %s OFFSET %s
                """,
                (f"%{search}%", f"%{search}%", limit, offset),
            )
        else:
            cur.execute(
                """
                SELECT p.patient_id::text, p.full_name, p.date_of_birth::text,
                       p.biological_sex, p.mrn,
                       COALESCE(
                           (SELECT string_agg(c.condition_name, ', ' ORDER BY c.onset_date DESC)
                            FROM conditions c
                            WHERE c.patient_id = p.patient_id AND c.status = 'active'
                            LIMIT 3),
                           ''
                       ) AS headline
                FROM patients p
                ORDER BY p.full_name
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
        cols = ["patient_id", "name", "dob", "sex", "mrn", "headline"]
        return _rows(cur, cols)
    finally:
        conn.close()


def count_patients(search: str = "") -> int:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        if search:
            cur.execute(
                "SELECT COUNT(*) FROM patients WHERE full_name ILIKE %s OR mrn ILIKE %s",
                (f"%{search}%", f"%{search}%"),
            )
        else:
            cur.execute("SELECT COUNT(*) FROM patients")
        return cur.fetchone()[0]
    finally:
        conn.close()


def list_patients_filtered(
    patient_ids: list[str],
    page: int = 1,
    limit: int = 20,
    search: str = "",
) -> list[dict]:
    """List EHR patients restricted to the given patient_ids (UUID strings)."""
    if not patient_ids:
        return []
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        offset = (page - 1) * limit
        uuid_list = [str(pid) for pid in patient_ids]
        base = (
            "SELECT p.patient_id::text, p.full_name, p.date_of_birth::text,"
            " p.biological_sex, p.mrn,"
            " COALESCE("
            "  (SELECT string_agg(c.condition_name, ', ' ORDER BY c.onset_date DESC)"
            "   FROM conditions c WHERE c.patient_id = p.patient_id AND c.status = 'active' LIMIT 3),"
            " '') AS headline"
            " FROM patients p"
            " WHERE p.patient_id = ANY(%s::uuid[])"
        )
        if search:
            cur.execute(
                base + " AND (p.full_name ILIKE %s OR p.mrn ILIKE %s)"
                " ORDER BY p.full_name LIMIT %s OFFSET %s",
                (uuid_list, f"%{search}%", f"%{search}%", limit, offset),
            )
        else:
            cur.execute(
                base + " ORDER BY p.full_name LIMIT %s OFFSET %s",
                (uuid_list, limit, offset),
            )
        cols = ["patient_id", "name", "dob", "sex", "mrn", "headline"]
        return _rows(cur, cols)
    finally:
        conn.close()


def count_patients_filtered(patient_ids: list[str], search: str = "") -> int:
    """Count EHR patients restricted to the given patient_ids (UUID strings)."""
    if not patient_ids:
        return 0
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        uuid_list = [str(pid) for pid in patient_ids]
        if search:
            cur.execute(
                "SELECT COUNT(*) FROM patients"
                " WHERE patient_id = ANY(%s::uuid[]) AND (full_name ILIKE %s OR mrn ILIKE %s)",
                (uuid_list, f"%{search}%", f"%{search}%"),
            )
        else:
            cur.execute(
                "SELECT COUNT(*) FROM patients WHERE patient_id = ANY(%s::uuid[])",
                (uuid_list,),
            )
        return cur.fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Patient banner / snapshot summary
# ---------------------------------------------------------------------------

def get_patient_banner(patient_id: str) -> dict:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.patient_id::text, p.mrn, p.full_name, p.preferred_name,
                   p.date_of_birth::text, p.biological_sex, p.gender_identity,
                   p.primary_care_physician, p.race, p.ethnicity, p.primary_language,
                   p.address_line1, p.city, p.state, p.zip, p.phone, p.email,
                   p.insurance_payer_name, p.insurance_member_id, p.insurance_group_number,
                   p.created_at::text
            FROM patients p
            WHERE p.patient_id = %s::uuid
            """,
            (patient_id,),
        )
        cols = [
            "patient_id", "mrn", "full_name", "preferred_name", "date_of_birth",
            "biological_sex", "gender_identity", "primary_care_physician",
            "race", "ethnicity", "primary_language",
            "address_line1", "city", "state", "zip", "phone", "email",
            "insurance_payer_name", "insurance_member_id", "insurance_group_number",
            "created_at",
        ]
        return _row(cur, cols)
    finally:
        conn.close()


def get_patient_alerts(patient_id: str) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT alert_text, severity, is_active FROM patient_alerts WHERE patient_id = %s::uuid AND is_active",
            (patient_id,),
        )
        return _rows(cur, ["alert_text", "severity", "is_active"])
    finally:
        conn.close()


def get_patient_emergency_contacts(patient_id: str) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT contact_name, relationship, phone FROM patient_emergency_contacts WHERE patient_id = %s::uuid",
            (patient_id,),
        )
        return _rows(cur, ["contact_name", "relationship", "phone"])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Encounters
# ---------------------------------------------------------------------------

def get_encounters(patient_id: str, limit: int = 50) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.encounter_id::text, e.encounter_date::text, e.visit_type,
                   e.department, o.name AS facility, p.full_name AS attending_physician,
                   e.chief_complaint, e.primary_diagnosis, e.status,
                   e.admission_datetime::text, e.discharge_datetime::text,
                   e.length_of_stay_hours, e.payer_name, e.total_charges
            FROM encounters e
            LEFT JOIN organizations o ON o.organization_id = e.facility_id
            LEFT JOIN practitioners p ON p.practitioner_id = e.attending_physician_id
            WHERE e.patient_id = %s::uuid
            ORDER BY e.encounter_date DESC
            LIMIT %s
            """,
            (patient_id, limit),
        )
        cols = [
            "encounter_id", "encounter_date", "visit_type", "department", "facility",
            "attending_physician", "chief_complaint", "primary_diagnosis", "status",
            "admission_datetime", "discharge_datetime", "length_of_stay_hours",
            "payer_name", "total_charges",
        ]
        return _rows(cur, cols)
    finally:
        conn.close()


def get_encounter_detail(patient_id: str, encounter_id: str) -> dict:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        # Header
        cur.execute(
            """
            SELECT e.encounter_id::text, e.encounter_date::text, e.visit_type,
                   e.department, o.name AS facility, p.full_name AS attending_physician,
                   e.chief_complaint, e.primary_diagnosis, e.status,
                   e.admission_datetime::text, e.discharge_datetime::text,
                   e.length_of_stay_hours, e.payer_name, e.claim_status,
                   e.drg_code, e.total_charges
            FROM encounters e
            LEFT JOIN organizations o ON o.organization_id = e.facility_id
            LEFT JOIN practitioners p ON p.practitioner_id = e.attending_physician_id
            WHERE e.patient_id = %s::uuid AND e.encounter_id = %s::uuid
            """,
            (patient_id, encounter_id),
        )
        header = _row(cur, [
            "encounter_id", "encounter_date", "visit_type", "department", "facility",
            "attending_physician", "chief_complaint", "primary_diagnosis", "status",
            "admission_datetime", "discharge_datetime", "length_of_stay_hours",
            "payer_name", "claim_status", "drg_code", "total_charges",
        ])
        if not header:
            return {}

        # Care team
        cur.execute(
            """
            SELECT pr.full_name, ect.role
            FROM encounter_care_team ect
            JOIN practitioners pr ON pr.practitioner_id = ect.practitioner_id
            WHERE ect.encounter_id = %s::uuid
            """,
            (encounter_id,),
        )
        header["care_team"] = _rows(cur, ["name", "role"])

        # Diagnoses
        cur.execute(
            """
            SELECT c.condition_name, c.icd10_code, c.snomed_code,
                   ec.is_primary, ec.diagnosis_rank, ec.is_new_this_visit,
                   ec.confirmed_date::text, p.full_name AS confirmed_by
            FROM encounter_conditions ec
            JOIN conditions c ON c.condition_id = ec.condition_id
            LEFT JOIN practitioners p ON p.practitioner_id = ec.confirmed_by_id
            WHERE ec.encounter_id = %s::uuid
            ORDER BY ec.diagnosis_rank NULLS LAST
            """,
            (encounter_id,),
        )
        header["diagnoses"] = _rows(cur, [
            "condition_name", "icd10_code", "snomed_code",
            "is_primary", "diagnosis_rank", "is_new_this_visit",
            "confirmed_date", "confirmed_by",
        ])

        # Vitals
        cur.execute(
            """
            SELECT recorded_at::text, systolic_bp, diastolic_bp, heart_rate,
                   temperature_c, respiratory_rate, spo2_pct, weight_kg
            FROM vitals
            WHERE encounter_id = %s::uuid
            ORDER BY recorded_at
            """,
            (encounter_id,),
        )
        header["vitals"] = _rows(cur, [
            "recorded_at", "systolic_bp", "diastolic_bp", "heart_rate",
            "temperature_c", "respiratory_rate", "spo2_pct", "weight_kg",
        ])

        # Procedures
        cur.execute(
            """
            SELECT pr2.procedure_name, pr2.cpt_code, pr2.performed_at::text,
                   pr2.status, p.full_name AS performing_physician
            FROM procedures pr2
            LEFT JOIN practitioners p ON p.practitioner_id = pr2.performing_physician_id
            WHERE pr2.encounter_id = %s::uuid
            ORDER BY pr2.performed_at
            """,
            (encounter_id,),
        )
        header["procedures"] = _rows(cur, [
            "procedure_name", "cpt_code", "performed_at", "status", "performing_physician",
        ])

        # Medications ordered
        cur.execute(
            """
            SELECT drug_name, dose, route, frequency, ordered_at::text,
                   start_date::text, end_date::text, status, delivery_type
            FROM medications
            WHERE ordering_encounter_id = %s::uuid
            ORDER BY ordered_at
            """,
            (encounter_id,),
        )
        header["medications"] = _rows(cur, [
            "drug_name", "dose", "route", "frequency", "ordered_at",
            "start_date", "end_date", "status", "delivery_type",
        ])

        # Clinical notes
        cur.execute(
            """
            SELECT n.note_type, p.full_name AS author, n.authored_at::text, n.note_text
            FROM clinical_notes n
            LEFT JOIN practitioners p ON p.practitioner_id = n.author_id
            WHERE n.encounter_id = %s::uuid
            ORDER BY n.authored_at
            """,
            (encounter_id,),
        )
        header["notes"] = _rows(cur, ["note_type", "author", "authored_at", "note_text"])

        # Discharge (inpatient/ER only)
        cur.execute(
            """
            SELECT ed.disposition, ed.instructions_summary,
                   array_agg(m.drug_name) FILTER (WHERE m.drug_name IS NOT NULL) AS discharge_meds
            FROM encounter_discharge ed
            LEFT JOIN discharge_prescriptions dp ON dp.discharge_id = ed.discharge_id
            LEFT JOIN medications m ON m.medication_id = dp.medication_id
            WHERE ed.encounter_id = %s::uuid
            GROUP BY ed.discharge_id
            """,
            (encounter_id,),
        )
        drow = cur.fetchone()
        if drow:
            header["discharge"] = {
                "disposition": drow[0],
                "instructions_summary": drow[1],
                "discharge_prescriptions": drow[2] or [],
            }

        return header
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Vitals
# ---------------------------------------------------------------------------

def get_vitals(patient_id: str, limit: int = 100) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT v.recorded_at::text, v.encounter_id::text,
                   v.systolic_bp, v.diastolic_bp, v.heart_rate, v.temperature_c,
                   v.respiratory_rate, v.spo2_pct, v.height_cm, v.weight_kg, v.bmi
            FROM vitals v
            WHERE v.patient_id = %s::uuid
            ORDER BY v.recorded_at DESC
            LIMIT %s
            """,
            (patient_id, limit),
        )
        return _rows(cur, [
            "recorded_at", "encounter_id", "systolic_bp", "diastolic_bp",
            "heart_rate", "temperature_c", "respiratory_rate", "spo2_pct",
            "height_cm", "weight_kg", "bmi",
        ])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def get_conditions(patient_id: str) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.condition_id::text, c.condition_name, c.snomed_code, c.icd10_code,
                   c.category, c.onset_date::text, c.resolution_date::text, c.status,
                   c.first_encounter_id::text, p.full_name AS treating_physician
            FROM conditions c
            LEFT JOIN practitioners p ON p.practitioner_id = c.treating_physician_id
            WHERE c.patient_id = %s::uuid
            ORDER BY c.status, c.onset_date DESC NULLS LAST
            """,
            (patient_id,),
        )
        return _rows(cur, [
            "condition_id", "condition_name", "snomed_code", "icd10_code",
            "category", "onset_date", "resolution_date", "status",
            "first_encounter_id", "treating_physician",
        ])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Medications + Allergies
# ---------------------------------------------------------------------------

def get_allergies(patient_id: str) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT allergen, reaction_detail, severity, recorded_date::text
            FROM allergies
            WHERE patient_id = %s::uuid
            ORDER BY severity DESC NULLS LAST
            """,
            (patient_id,),
        )
        return _rows(cur, ["allergen", "reaction_detail", "severity", "recorded_date"])
    finally:
        conn.close()


def get_medications(patient_id: str) -> dict:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.medication_id::text, m.drug_name, m.rxnorm_code, m.dose,
                   m.route, m.frequency, m.ordered_at::text,
                   m.start_date::text, m.end_date::text,
                   m.status, m.delivery_type, m.refill_count,
                   m.ordering_encounter_id::text,
                   p.full_name AS prescribing_physician
            FROM medications m
            LEFT JOIN encounters e ON e.encounter_id = m.ordering_encounter_id
            LEFT JOIN practitioners p ON p.practitioner_id = e.attending_physician_id
            WHERE m.patient_id = %s::uuid
            ORDER BY m.status, m.ordered_at DESC NULLS LAST
            """,
            (patient_id,),
        )
        medications = _rows(cur, [
            "medication_id", "drug_name", "rxnorm_code", "dose", "route", "frequency",
            "ordered_at", "start_date", "end_date", "status", "delivery_type",
            "refill_count", "ordering_encounter_id", "prescribing_physician",
        ])

        cur.execute(
            """
            SELECT allergen, reaction_detail, severity, recorded_date::text
            FROM allergies
            WHERE patient_id = %s::uuid
            ORDER BY severity DESC NULLS LAST
            """,
            (patient_id,),
        )
        allergies = _rows(cur, ["allergen", "reaction_detail", "severity", "recorded_date"])

        return {"medications": medications, "allergies": allergies}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lab results
# ---------------------------------------------------------------------------

def get_labs(patient_id: str, limit: int = 200) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT lr.lab_result_id::text, lp.panel_name, lr.loinc_code, lr.test_name,
                   lr.result_value, lr.unit, lr.reference_range_low,
                   lr.reference_range_high, lr.interpretation_flag,
                   lr.collected_at::text, lr.encounter_id::text,
                   p.full_name AS ordering_physician
            FROM lab_results lr
            LEFT JOIN lab_panels lp ON lp.panel_id = lr.panel_id
            LEFT JOIN practitioners p ON p.practitioner_id = lr.ordering_physician_id
            WHERE lr.patient_id = %s::uuid
            ORDER BY lr.collected_at DESC
            LIMIT %s
            """,
            (patient_id, limit),
        )
        return _rows(cur, [
            "lab_result_id", "panel_name", "loinc_code", "test_name",
            "result_value", "unit", "reference_range_low", "reference_range_high",
            "interpretation_flag", "collected_at", "encounter_id", "ordering_physician",
        ])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------

def get_procedures(patient_id: str) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT pr.procedure_id::text, pr.procedure_name, pr.cpt_code,
                   pr.performed_at::text, pr.status, pr.encounter_id::text,
                   p.full_name AS performing_physician
            FROM procedures pr
            LEFT JOIN practitioners p ON p.practitioner_id = pr.performing_physician_id
            WHERE pr.patient_id = %s::uuid
            ORDER BY pr.performed_at DESC NULLS LAST
            """,
            (patient_id,),
        )
        return _rows(cur, [
            "procedure_id", "procedure_name", "cpt_code", "performed_at",
            "status", "encounter_id", "performing_physician",
        ])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Imaging
# ---------------------------------------------------------------------------

def get_imaging(patient_id: str) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT img.imaging_study_id::text, img.modality, img.body_region, img.laterality,
                   img.date_ordered::text, img.date_performed::text,
                   img.status, img.clinical_indication, img.encounter_id::text,
                   p.full_name AS ordering_physician,
                   o.name AS performing_facility,
                   dr.report_narrative
            FROM imaging_studies img
            LEFT JOIN practitioners p ON p.practitioner_id = img.ordering_physician_id
            LEFT JOIN organizations o ON o.organization_id = img.performing_facility_id
            LEFT JOIN diagnostic_reports dr ON dr.imaging_study_id = img.imaging_study_id
            WHERE img.patient_id = %s::uuid
            ORDER BY img.date_performed DESC NULLS LAST
            """,
            (patient_id,),
        )
        return _rows(cur, [
            "imaging_study_id", "modality", "body_region", "laterality",
            "date_ordered", "date_performed", "status", "clinical_indication",
            "encounter_id", "ordering_physician", "performing_facility", "report_narrative",
        ])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Clinical notes
# ---------------------------------------------------------------------------

def get_notes(patient_id: str) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT n.note_id::text, n.note_type, p.full_name AS author,
                   n.authored_at::text, n.encounter_id::text, n.note_text
            FROM clinical_notes n
            LEFT JOIN practitioners p ON p.practitioner_id = n.author_id
            WHERE n.patient_id = %s::uuid
            ORDER BY n.authored_at DESC
            """,
            (patient_id,),
        )
        return _rows(cur, [
            "note_id", "note_type", "author", "authored_at", "encounter_id", "note_text",
        ])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Care plans
# ---------------------------------------------------------------------------

def get_care_plans(patient_id: str) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT cp.care_plan_id::text, cp.encounter_id::text, cp.follow_up_instructions,
                   array_agg(DISTINCT cpg.goal_description) FILTER (WHERE cpg.goal_description IS NOT NULL) AS goals,
                   array_agg(DISTINCT cpi.intervention_description) FILTER (WHERE cpi.intervention_description IS NOT NULL) AS interventions
            FROM care_plans cp
            LEFT JOIN care_plan_goals cpg ON cpg.care_plan_id = cp.care_plan_id
            LEFT JOIN care_plan_interventions cpi ON cpi.care_plan_id = cp.care_plan_id
            WHERE cp.patient_id = %s::uuid
            GROUP BY cp.care_plan_id, cp.encounter_id, cp.follow_up_instructions
            ORDER BY cp.created_at DESC
            """,
            (patient_id,),
        )
        return _rows(cur, ["care_plan_id", "encounter_id", "follow_up_instructions", "goals", "interventions"])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Immunizations
# ---------------------------------------------------------------------------

def get_immunizations(patient_id: str) -> list[dict]:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT i.immunization_id::text, vf.family_name AS vaccine_family,
                   i.cvx_code, i.vaccine_name, i.date_administered::text,
                   i.dose_number, i.route, i.site, i.lot_number,
                   i.manufacturer, i.status, i.encounter_id::text
            FROM immunizations i
            LEFT JOIN vaccine_families vf ON vf.cvx_code = i.cvx_code
            WHERE i.patient_id = %s::uuid
            ORDER BY i.date_administered DESC NULLS LAST
            """,
            (patient_id,),
        )
        return _rows(cur, [
            "immunization_id", "vaccine_family", "cvx_code", "vaccine_name",
            "date_administered", "dose_number", "route", "site", "lot_number",
            "manufacturer", "status", "encounter_id",
        ])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Social history
# ---------------------------------------------------------------------------

def get_social_history(patient_id: str) -> dict:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT smoking_status, pack_years, quit_date::text, alcohol_frequency,
                   alcohol_drinks_per_week, occupation, housing_status, education_level,
                   last_updated_at::text
            FROM social_history
            WHERE patient_id = %s::uuid
            """,
            (patient_id,),
        )
        return _row(cur, [
            "smoking_status", "pack_years", "quit_date", "alcohol_frequency",
            "alcohol_drinks_per_week", "occupation", "housing_status", "education_level",
            "last_updated_at",
        ])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Snapshot summary (all at once for the landing card)
# ---------------------------------------------------------------------------

def get_snapshot(patient_id: str) -> dict:
    conn = _ehr_conn()
    try:
        cur = conn.cursor()

        # Active problems (top 5)
        cur.execute(
            """
            SELECT condition_name, icd10_code, onset_date::text, status
            FROM conditions
            WHERE patient_id = %s::uuid AND status = 'active'
            ORDER BY onset_date DESC NULLS LAST
            LIMIT 5
            """,
            (patient_id,),
        )
        active_problems = _rows(cur, ["condition_name", "icd10_code", "onset_date", "status"])

        # Active medications (top 5)
        cur.execute(
            """
            SELECT drug_name, dose, route, frequency
            FROM medications
            WHERE patient_id = %s::uuid AND status = 'active'
            ORDER BY ordered_at DESC NULLS LAST
            LIMIT 5
            """,
            (patient_id,),
        )
        active_meds = _rows(cur, ["drug_name", "dose", "route", "frequency"])

        # Latest vitals (most recent single reading)
        cur.execute(
            """
            SELECT recorded_at::text, systolic_bp, diastolic_bp, heart_rate,
                   temperature_c, respiratory_rate, spo2_pct, height_cm, weight_kg, bmi
            FROM vitals
            WHERE patient_id = %s::uuid
            ORDER BY recorded_at DESC
            LIMIT 1
            """,
            (patient_id,),
        )
        latest_vitals = _row(cur, [
            "recorded_at", "systolic_bp", "diastolic_bp", "heart_rate",
            "temperature_c", "respiratory_rate", "spo2_pct", "height_cm", "weight_kg", "bmi",
        ])

        # Recent encounters (last 3)
        cur.execute(
            """
            SELECT e.encounter_id::text, e.encounter_date::text, e.visit_type,
                   e.department, p.full_name AS attending, e.chief_complaint, e.primary_diagnosis
            FROM encounters e
            LEFT JOIN practitioners p ON p.practitioner_id = e.attending_physician_id
            WHERE e.patient_id = %s::uuid
            ORDER BY e.encounter_date DESC
            LIMIT 3
            """,
            (patient_id,),
        )
        recent_encounters = _rows(cur, [
            "encounter_id", "encounter_date", "visit_type",
            "department", "attending", "chief_complaint", "primary_diagnosis",
        ])

        # Recent/abnormal labs (last 5 abnormal or last 5)
        cur.execute(
            """
            SELECT test_name, result_value, unit, interpretation_flag,
                   collected_at::text
            FROM lab_results
            WHERE patient_id = %s::uuid
            ORDER BY interpretation_flag NULLS LAST, collected_at DESC
            LIMIT 5
            """,
            (patient_id,),
        )
        recent_labs = _rows(cur, [
            "test_name", "result_value", "unit", "interpretation_flag", "collected_at",
        ])

        # Allergies
        cur.execute(
            "SELECT allergen, severity FROM allergies WHERE patient_id = %s::uuid ORDER BY severity DESC NULLS LAST LIMIT 5",
            (patient_id,),
        )
        allergies = _rows(cur, ["allergen", "severity"])

        # Alerts
        cur.execute(
            "SELECT alert_text, severity FROM patient_alerts WHERE patient_id = %s::uuid AND is_active",
            (patient_id,),
        )
        alerts = _rows(cur, ["alert_text", "severity"])

        # Immunization status
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'completed') AS completed,
                COUNT(*) FILTER (WHERE status != 'completed') AS not_done
            FROM immunizations
            WHERE patient_id = %s::uuid
            """,
            (patient_id,),
        )
        r = cur.fetchone()
        imm_status = {"completed": r[0], "not_done": r[1]} if r else {}

        return {
            "active_problems": active_problems,
            "active_medications": active_meds,
            "latest_vitals": latest_vitals,
            "recent_encounters": recent_encounters,
            "recent_labs": recent_labs,
            "allergies": allergies,
            "alerts": alerts,
            "immunization_status": imm_status,
        }
    finally:
        conn.close()
