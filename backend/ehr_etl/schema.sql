-- =============================================================================
-- EHR DEMO PROJECT — PostgreSQL Schema
-- Mapped to: Doctor-Facing EHR Layout and Navigation Spec
-- Data source: Synthea FHIR R4 output (100 patients)
-- =============================================================================
-- Conventions:
--   - All PKs are UUIDs matching Synthea's FHIR resource IDs where possible
--   - FKs enforce referential integrity; cascade deletes from patient downward
--   - ENUM types used for low-cardinality controlled vocabularies
--   - Every table has created_at for audit; data tables add last_updated_at
--   - Synthea FHIR source field noted in comments on each column
-- =============================================================================

-- ---------------------------------------------------------------------------
-- SCHEMA ISOLATION
-- ---------------------------------------------------------------------------
-- All EHR tables live in the 'ehr' schema so they don't collide with the
-- security-demo tables (patients, conditions, etc.) that live in 'public'.
CREATE SCHEMA IF NOT EXISTS ehr;
SET search_path TO ehr, public;

-- ---------------------------------------------------------------------------
-- EXTENSIONS
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- gen_random_uuid()


-- ---------------------------------------------------------------------------
-- ENUM TYPES
-- ---------------------------------------------------------------------------

CREATE TYPE visit_type AS ENUM (
    'inpatient', 'outpatient', 'emergency', 'telehealth', 'surgical', 'observation'
);

CREATE TYPE encounter_status AS ENUM (
    'completed', 'active', 'cancelled'
);

CREATE TYPE condition_status AS ENUM (
    'active', 'resolved', 'recurrence'
);

CREATE TYPE condition_category AS ENUM (
    'chronic', 'acute', 'historical'
);

CREATE TYPE medication_status AS ENUM (
    'active', 'completed', 'discontinued', 'on_hold'
);

CREATE TYPE medication_delivery AS ENUM (
    'administered_during_visit', 'sent_as_prescription'
);

CREATE TYPE lab_flag AS ENUM (
    'normal', 'high', 'low', 'critical'
);

CREATE TYPE imaging_modality AS ENUM (
    'x_ray', 'ct', 'mri', 'ultrasound', 'echocardiogram', 'pet', 'mammogram', 'other'
);

CREATE TYPE imaging_status AS ENUM (
    'ordered', 'completed', 'preliminary', 'final'
);

CREATE TYPE imaging_laterality AS ENUM (
    'left', 'right', 'bilateral', 'not_applicable'
);

CREATE TYPE allergy_severity AS ENUM (
    'mild', 'moderate', 'severe'
);

CREATE TYPE immunization_status AS ENUM (
    'completed', 'not_done', 'refused'
);

CREATE TYPE note_type AS ENUM (
    'progress_note', 'discharge_summary', 'consultation_note',
    'operative_report', 'nursing_note'
);

CREATE TYPE discharge_disposition AS ENUM (
    'home', 'skilled_nursing', 'transferred', 'ama'
);

CREATE TYPE housing_status AS ENUM (
    'stable', 'unstable', 'homeless'
);

CREATE TYPE referral_urgency AS ENUM (
    'routine', 'urgent', 'emergent'
);

CREATE TYPE care_plan_goal_status AS ENUM (
    'in_progress', 'achieved', 'abandoned'
);

CREATE TYPE appointment_status AS ENUM (
    'scheduled', 'completed', 'cancelled', 'no_show'
);


-- =============================================================================
-- SECTION 1: PATIENT IDENTITY
-- Backs: Patient Banner, Demographics panel, Snapshot header
-- Synthea resource: Patient
-- =============================================================================

CREATE TABLE patients (
    -- Identity
    patient_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: Patient.id
    mrn                 TEXT UNIQUE NOT NULL,
    -- Synthea: Patient.identifier[type=MR].value (UUID format, 36 chars)

    -- Banner fields
    full_name           VARCHAR(200) NOT NULL,
    -- Synthea: Patient.name[use=official].given + family
    preferred_name      VARCHAR(100),
    -- Synthea: Patient.name[use=nickname].given
    date_of_birth       DATE NOT NULL,
    -- Synthea: Patient.birthDate
    biological_sex      VARCHAR(20) NOT NULL,
    -- Synthea: Patient.gender
    gender_identity     VARCHAR(50),
    -- Synthea: Patient.extension[us-core-genderIdentity]
    primary_care_physician VARCHAR(200),
    -- Synthea: Patient.generalPractitioner -> Practitioner.name

    -- Demographics panel (slide-over)
    race                VARCHAR(100),
    -- Synthea: Patient.extension[us-core-race]
    ethnicity           VARCHAR(100),
    -- Synthea: Patient.extension[us-core-ethnicity]
    primary_language    VARCHAR(100),
    -- Synthea: Patient.communication[preferred=true].language

    address_line1       VARCHAR(200),
    address_line2       VARCHAR(200),
    city                VARCHAR(100),
    state               VARCHAR(50),
    zip                 VARCHAR(20),
    -- Synthea: Patient.address

    phone               VARCHAR(30),
    -- Synthea: Patient.telecom[system=phone]
    email               VARCHAR(200),
    -- Synthea: Patient.telecom[system=email]

    -- Insurance (Demographics panel)
    insurance_payer_name VARCHAR(200),
    insurance_member_id  VARCHAR(100),
    insurance_group_number VARCHAR(100),
    -- Synthea: Coverage.payor, Coverage.subscriberId, Coverage.grouping.group

    -- Timestamps
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Emergency contacts (1:many — a patient may list multiple)
CREATE TABLE patient_emergency_contacts (
    contact_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    contact_name        VARCHAR(200) NOT NULL,
    relationship        VARCHAR(100),
    phone               VARCHAR(30),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Active alerts / flags shown in the banner
CREATE TABLE patient_alerts (
    alert_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    alert_text          TEXT NOT NULL,
    -- e.g. "Fall risk", "DNR order on file", "Isolation precautions"
    severity            VARCHAR(50),
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    recorded_at         TIMESTAMPTZ,
    recorded_by         VARCHAR(200),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =============================================================================
-- SECTION 2: PRACTITIONERS & ORGANIZATIONS
-- Referenced by encounters, labs, meds, imaging, notes, etc.
-- Synthea resources: Practitioner, Organization
-- =============================================================================

CREATE TABLE practitioners (
    practitioner_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: Practitioner.id
    full_name           VARCHAR(200) NOT NULL,
    -- Synthea: Practitioner.name
    npi                 VARCHAR(20),
    -- Synthea: Practitioner.identifier[system=NPI]
    specialty           VARCHAR(200),
    -- Synthea: Practitioner.qualification
    department          VARCHAR(200),
    -- Home department (e.g. "Cardiology", "Emergency Medicine")
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE organizations (
    organization_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: Organization.id
    name                VARCHAR(300) NOT NULL,
    -- Synthea: Organization.name
    type                VARCHAR(100),
    -- e.g. "hospital", "clinic", "lab", "imaging center"
    address             TEXT,
    phone               VARCHAR(30),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =============================================================================
-- SECTION 3: ENCOUNTERS
-- Backs: Encounters destination (list state + detail state), Snapshot recent encounters card
-- Synthea resource: Encounter
-- =============================================================================

CREATE TABLE encounters (
    encounter_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: Encounter.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,

    -- List state columns
    encounter_date      DATE NOT NULL,
    -- Synthea: Encounter.period.start (date part)
    visit_type          visit_type NOT NULL,
    -- Synthea: Encounter.class (mapped to enum)
    department          VARCHAR(200),
    -- Synthea: Encounter.type[0].text or serviceType
    facility_id         UUID REFERENCES organizations(organization_id),
    -- Synthea: Encounter.serviceProvider
    attending_physician_id UUID REFERENCES practitioners(practitioner_id),
    -- Synthea: Encounter.participant[type=ATND].individual
    chief_complaint     TEXT,
    -- Synthea: Encounter.reasonCode[0].text
    primary_diagnosis   VARCHAR(500),
    -- Synthea: Encounter.diagnosis[rank=1] -> Condition display
    status              encounter_status NOT NULL DEFAULT 'completed',
    -- Synthea: Encounter.status

    -- Detail state header fields
    admission_datetime  TIMESTAMPTZ,
    -- Synthea: Encounter.period.start
    discharge_datetime  TIMESTAMPTZ,
    -- Synthea: Encounter.period.end
    length_of_stay_hours NUMERIC(8,2),
    -- Derived: discharge - admission

    -- Billing (optional section in detail)
    payer_name          VARCHAR(200),
    -- Synthea: ExplanationOfBenefit.insurer.display
    claim_status        VARCHAR(100),
    -- Synthea: ExplanationOfBenefit.outcome
    drg_code            VARCHAR(20),
    -- Synthea: ExplanationOfBenefit.diagnosis[type=drg].diagnosisCodeableConcept
    total_charges       NUMERIC(12,2),
    -- Synthea: ExplanationOfBenefit.total[category=submitted].amount.value

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Care team members for a given encounter (nurses, residents, consultants)
CREATE TABLE encounter_care_team (
    care_team_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id        UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    practitioner_id     UUID NOT NULL REFERENCES practitioners(practitioner_id),
    role                VARCHAR(100) NOT NULL,
    -- e.g. "attending", "nurse", "resident", "consultant"
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Discharge details (shown only for inpatient / ER encounters)
CREATE TABLE encounter_discharge (
    discharge_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id        UUID NOT NULL UNIQUE REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    -- Synthea: Encounter.hospitalization

    disposition         discharge_disposition,
    -- Synthea: Encounter.hospitalization.dischargeDisposition
    instructions_summary TEXT,
    -- Synthea: DocumentReference[type=discharge-summary] note text
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Prescriptions explicitly issued at discharge (subset of medications for that encounter)
-- Fixes: discharge section spec requires "prescriptions given at discharge (list)" as explicit data
-- medication_id FK wired via ALTER TABLE at the end of Section 7, after medications is defined
CREATE TABLE discharge_prescriptions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discharge_id    UUID NOT NULL REFERENCES encounter_discharge(discharge_id) ON DELETE CASCADE,
    medication_id   UUID NOT NULL,
    UNIQUE(discharge_id, medication_id)
);

-- Follow-up appointments scheduled at discharge
CREATE TABLE encounter_followup_appointments (
    followup_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id        UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    scheduled_date      DATE,
    provider_name       VARCHAR(200),
    -- Free-text for external / unknown providers
    provider_id         UUID REFERENCES practitioners(practitioner_id),
    -- FK when the provider is a known practitioner in this system
    department          VARCHAR(200),
    notes               TEXT,
    -- Set when this follow-up is booked into a real scheduled appointment
    appointment_id      UUID,
    -- FK to scheduled_appointments.appointment_id; set via ALTER below after that table exists
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Scheduled / upcoming appointments (from scheduling system or booked follow-ups)
-- Backs: Snapshot → Recent Encounters card "next scheduled appointment" single value
CREATE TABLE scheduled_appointments (
    appointment_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    scheduled_datetime  TIMESTAMPTZ NOT NULL,
    department          VARCHAR(200),
    provider_id         UUID REFERENCES practitioners(practitioner_id),
    reason              TEXT,
    status              appointment_status NOT NULL DEFAULT 'scheduled',
    -- Links back to the discharge follow-up that generated this booking, when applicable
    source_followup_id  UUID REFERENCES encounter_followup_appointments(followup_id),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Now that scheduled_appointments exists, wire the FK on encounter_followup_appointments
ALTER TABLE encounter_followup_appointments
    ADD CONSTRAINT fk_followup_appointment
    FOREIGN KEY (appointment_id)
    REFERENCES scheduled_appointments(appointment_id);

-- medications is defined in Section 7 below; discharge_prescriptions FK added there via ALTER


-- =============================================================================
-- SECTION 4: DIAGNOSES / CONDITIONS
-- Backs: Clinical Chart → Conditions tab, Encounter Detail → Diagnoses section,
--        Snapshot → Active Problems card
-- Synthea resource: Condition
-- =============================================================================

CREATE TABLE conditions (
    condition_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: Condition.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    first_encounter_id  UUID REFERENCES encounters(encounter_id),
    -- Synthea: Condition.encounter (first time this condition appears)

    -- Conditions tab columns
    condition_name      VARCHAR(500) NOT NULL,
    -- Synthea: Condition.code.text
    snomed_code         VARCHAR(50),
    -- Synthea: Condition.code.coding[system=SNOMED-CT].code
    icd10_code          VARCHAR(20),
    -- Synthea: Condition.code.coding[system=ICD-10].code
    category            condition_category NOT NULL,
    -- Synthea: Condition.category[0].coding[0].code (mapped to enum)
    onset_date          DATE,
    -- Synthea: Condition.onsetDateTime
    resolution_date     DATE,
    -- Synthea: Condition.abatementDateTime
    status              condition_status NOT NULL DEFAULT 'active',
    -- Synthea: Condition.clinicalStatus
    treating_physician_id UUID REFERENCES practitioners(practitioner_id),
    -- Synthea: Condition.recorder (mapped to Practitioner)

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Links a condition to a specific encounter (a condition may appear across multiple visits)
CREATE TABLE encounter_conditions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id        UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    condition_id        UUID NOT NULL REFERENCES conditions(condition_id) ON DELETE CASCADE,
    is_primary          BOOLEAN NOT NULL DEFAULT FALSE,
    diagnosis_rank      INTEGER,
    -- Synthea: Encounter.diagnosis.rank
    is_new_this_visit   BOOLEAN NOT NULL DEFAULT FALSE,
    -- TRUE if condition was first diagnosed this encounter
    confirmed_date      DATE,
    -- Date the diagnosis was formally confirmed during this encounter
    confirmed_by_id     UUID REFERENCES practitioners(practitioner_id),
    -- Clinician who confirmed the diagnosis (Encounter Detail → Diagnoses: "confirming clinician")
    UNIQUE(encounter_id, condition_id)
);


-- =============================================================================
-- SECTION 5: VITALS
-- Backs: Snapshot → Latest Vitals card, Vitals Trend panel,
--        Encounter Detail → Vitals section
-- Synthea resource: Observation (category=vital-signs)
-- =============================================================================

CREATE TABLE vitals (
    vital_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: Observation.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id        UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    -- Synthea: Observation.encounter

    recorded_at         TIMESTAMPTZ NOT NULL,
    -- Synthea: Observation.effectiveDateTime
    recorded_by         VARCHAR(200),
    -- Synthea: Observation.performer[0].display

    -- Individual vital sign values (all nullable — not every reading captures every vital)
    systolic_bp         INTEGER,           -- mmHg
    diastolic_bp        INTEGER,           -- mmHg
    heart_rate          INTEGER,           -- bpm
    temperature_c       NUMERIC(4,1),      -- Celsius
    respiratory_rate    INTEGER,           -- breaths/min
    spo2_pct            NUMERIC(4,1),      -- %
    height_cm           NUMERIC(5,1),
    weight_kg           NUMERIC(6,2),
    bmi                 NUMERIC(5,2),
    -- Synthea: each is a separate Observation with LOINC code, pulled and pivoted

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =============================================================================
-- SECTION 6: LAB RESULTS
-- Backs: Clinical Chart → Labs tab (grouped tables + Trend view),
--        Encounter Detail → Lab results section,
--        Snapshot → Recent and Key Labs card
-- Synthea resource: Observation (category=laboratory)
-- =============================================================================

-- Lab panel groups (CBC, Metabolic, Lipid, etc.) — lookup table
CREATE TABLE lab_panels (
    panel_id            SERIAL PRIMARY KEY,
    panel_name          VARCHAR(100) UNIQUE NOT NULL
    -- Values: 'CBC', 'Metabolic', 'Liver Function', 'Lipid Panel',
    --         'Endocrine', 'Coagulation', 'Urinalysis', 'Other'
);

INSERT INTO lab_panels (panel_name) VALUES
    ('CBC'),
    ('Metabolic'),
    ('Liver Function'),
    ('Lipid Panel'),
    ('Endocrine'),
    ('Coagulation'),
    ('Urinalysis'),
    ('Other');

CREATE TABLE lab_results (
    lab_result_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: Observation.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id        UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    -- Synthea: Observation.encounter
    panel_id            INTEGER REFERENCES lab_panels(panel_id),
    ordering_physician_id UUID REFERENCES practitioners(practitioner_id),
    -- Synthea: Observation.performer[0]

    -- Labs tab columns
    test_name           VARCHAR(300) NOT NULL,
    -- Synthea: Observation.code.text
    loinc_code          VARCHAR(20),
    -- Synthea: Observation.code.coding[system=LOINC].code
    result_value        NUMERIC(12,4),
    -- Synthea: Observation.valueQuantity.value
    result_value_text   VARCHAR(100),
    -- Synthea: Observation.valueString or valueCodeableConcept.text (for non-numeric results)
    unit                VARCHAR(50),
    -- Synthea: Observation.valueQuantity.unit
    reference_range_low  NUMERIC(12,4),
    -- Synthea: Observation.referenceRange[0].low.value
    reference_range_high NUMERIC(12,4),
    -- Synthea: Observation.referenceRange[0].high.value
    interpretation_flag  lab_flag,
    -- Synthea: Observation.interpretation[0].coding[0].code (mapped to enum)
    performing_lab      VARCHAR(200),
    -- Synthea: Observation.performer (Organization type)
    collected_at        TIMESTAMPTZ NOT NULL,
    -- Synthea: Observation.effectiveDateTime

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for Trend view queries (all readings of a single test for a patient)
CREATE INDEX idx_lab_results_trend ON lab_results(patient_id, loinc_code, collected_at);


-- =============================================================================
-- SECTION 7: MEDICATIONS
-- Backs: Clinical Chart → Medications tab, Encounter Detail → Medications section,
--        Snapshot → Current Medications card
-- Synthea resource: MedicationRequest
-- =============================================================================

CREATE TABLE medications (
    medication_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: MedicationRequest.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    ordering_encounter_id UUID REFERENCES encounters(encounter_id),
    -- Synthea: MedicationRequest.encounter
    prescribed_by_id    UUID REFERENCES practitioners(practitioner_id),
    -- Synthea: MedicationRequest.requester

    -- Medications tab columns
    drug_name           VARCHAR(300) NOT NULL,
    -- Synthea: MedicationRequest.medicationCodeableConcept.text
    rxnorm_code         VARCHAR(20),
    -- Synthea: MedicationRequest.medicationCodeableConcept.coding[system=RxNorm].code
    brand_name          VARCHAR(200),
    is_generic          BOOLEAN DEFAULT TRUE,
    dose                VARCHAR(100),
    -- Synthea: MedicationRequest.dosageInstruction[0].doseAndRate[0].doseQuantity
    route               VARCHAR(100),
    -- Synthea: MedicationRequest.dosageInstruction[0].route.text
    frequency           VARCHAR(100),
    -- Synthea: MedicationRequest.dosageInstruction[0].timing.code.text
    ordered_at          TIMESTAMPTZ,
    -- Synthea: MedicationRequest.authoredOn (TIMESTAMPTZ to match "ordered at" in Encounter Detail)
    start_date          DATE,
    -- Synthea: MedicationRequest.dosageInstruction[0].timing.repeat.boundsPeriod.start
    end_date            DATE,
    -- Synthea: MedicationRequest.dosageInstruction[0].timing.repeat.boundsPeriod.end
    status              medication_status NOT NULL DEFAULT 'active',
    -- Synthea: MedicationRequest.status
    discontinuation_reason TEXT,
    delivery_type       medication_delivery,
    -- Synthea: MedicationRequest.dispenseRequest vs MedicationAdministration
    refill_count        INTEGER DEFAULT 0,
    -- Synthea: MedicationRequest.dispenseRequest.numberOfRepeatsAllowed
    last_filled_date    DATE,
    renewal_due_date    DATE,
    -- Snapshot renewal-due flag: set when end_date is within 30 days or refills exhausted;
    -- populated by the application layer or a scheduled job

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Links condition to medications prescribed for it (Conditions tab → condition detail panel)
CREATE TABLE condition_medications (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    condition_id        UUID NOT NULL REFERENCES conditions(condition_id) ON DELETE CASCADE,
    medication_id       UUID NOT NULL REFERENCES medications(medication_id) ON DELETE CASCADE,
    UNIQUE(condition_id, medication_id)
);

-- Now that medications exists, wire the FK on discharge_prescriptions
ALTER TABLE discharge_prescriptions
    ADD CONSTRAINT fk_discharge_prescription_medication
    FOREIGN KEY (medication_id)
    REFERENCES medications(medication_id);


-- =============================================================================
-- SECTION 8: ALLERGIES & INTOLERANCES
-- Backs: Patient Banner (short list), Snapshot → Allergies card,
--        Clinical Chart → Medications tab (secondary section)
-- Synthea resource: AllergyIntolerance
-- =============================================================================

CREATE TABLE allergies (
    allergy_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: AllergyIntolerance.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    recorder_id         UUID REFERENCES practitioners(practitioner_id),
    -- Synthea: AllergyIntolerance.recorder

    allergen            VARCHAR(300) NOT NULL,
    -- Synthea: AllergyIntolerance.code.text
    reaction_detail     TEXT,
    -- Synthea: AllergyIntolerance.reaction[0].manifestation[0].text
    severity            allergy_severity,
    -- Synthea: AllergyIntolerance.reaction[0].severity
    recorded_date       DATE,
    -- Synthea: AllergyIntolerance.recordedDate

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =============================================================================
-- SECTION 9: PROCEDURES
-- Backs: Encounter Detail → Procedures section
-- Synthea resource: Procedure
-- =============================================================================

CREATE TABLE procedures (
    procedure_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: Procedure.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id        UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    -- Synthea: Procedure.encounter
    performing_physician_id UUID REFERENCES practitioners(practitioner_id),
    -- Synthea: Procedure.performer[0].actor

    procedure_name      VARCHAR(500) NOT NULL,
    -- Synthea: Procedure.code.text
    cpt_code            VARCHAR(20),
    -- Synthea: Procedure.code.coding[system=CPT]
    snomed_code         VARCHAR(50),
    -- Synthea: Procedure.code.coding[system=SNOMED-CT]
    performed_at        TIMESTAMPTZ,
    -- Synthea: Procedure.performedDateTime
    duration_minutes    INTEGER,
    -- Synthea: not always present; derived from performedPeriod when available
    status              VARCHAR(50) DEFAULT 'completed',
    -- Synthea: Procedure.status
    -- Note: linked_report_id removed; join via diagnostic_reports.procedure_id instead

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =============================================================================
-- SECTION 10: IMAGING STUDIES
-- Backs: Clinical Chart → Imaging tab, Encounter Detail → Imaging section
-- Synthea resource: ImagingStudy
-- =============================================================================

CREATE TABLE imaging_studies (
    imaging_study_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: ImagingStudy.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id        UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    -- Synthea: ImagingStudy.encounter
    ordering_physician_id UUID REFERENCES practitioners(practitioner_id),
    -- Synthea: ImagingStudy.referrer
    performing_facility_id UUID REFERENCES organizations(organization_id),
    -- Synthea: ImagingStudy.location

    modality            imaging_modality NOT NULL,
    -- Synthea: ImagingStudy.series[0].modality.code (mapped to enum)
    body_region         VARCHAR(200),
    -- Synthea: ImagingStudy.series[0].bodySite.display
    laterality          imaging_laterality DEFAULT 'not_applicable',
    -- Synthea: ImagingStudy.series[0].laterality
    clinical_indication TEXT,
    -- Synthea: ImagingStudy.reasonCode[0].text
    date_ordered        DATE,
    date_performed      DATE,
    -- Synthea: ImagingStudy.started
    interpreting_radiologist VARCHAR(200),
    status              imaging_status NOT NULL DEFAULT 'completed',
    -- Synthea: ImagingStudy.status

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =============================================================================
-- SECTION 11: DIAGNOSTIC REPORTS
-- Backs: Encounter Detail → Imaging studies (report narrative panel),
--        Procedures (linked diagnostic report)
-- Synthea resource: DiagnosticReport
-- Note: join to procedures via diagnostic_reports.procedure_id (one direction only,
--       avoids the circular FK that existed when procedures also held linked_report_id)
-- =============================================================================

CREATE TABLE diagnostic_reports (
    report_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: DiagnosticReport.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id        UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    imaging_study_id    UUID REFERENCES imaging_studies(imaging_study_id),
    procedure_id        UUID REFERENCES procedures(procedure_id),
    authored_by         VARCHAR(200),
    -- Synthea: DiagnosticReport.performer[0].display

    report_type         VARCHAR(200),
    -- Synthea: DiagnosticReport.code.text
    report_narrative    TEXT,
    -- Synthea: DiagnosticReport.conclusion or presentedForm[0].data (base64 decoded)
    issued_at           TIMESTAMPTZ,
    -- Synthea: DiagnosticReport.issued
    status              VARCHAR(50),
    -- Synthea: DiagnosticReport.status

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =============================================================================
-- SECTION 12: CLINICAL NOTES
-- Backs: Encounter Detail → Clinical Notes section
-- Synthea resource: DocumentReference (limited text; supplement with generated notes)
-- =============================================================================

CREATE TABLE clinical_notes (
    note_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: DocumentReference.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id        UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    -- Synthea: DocumentReference.context.encounter
    author_id           UUID REFERENCES practitioners(practitioner_id),
    -- Synthea: DocumentReference.author

    note_type           note_type NOT NULL,
    -- Synthea: DocumentReference.type.coding[system=LOINC].display (mapped to enum)
    note_text           TEXT,
    -- Synthea: DocumentReference.content[0].attachment.data (base64 decoded)
    authored_at         TIMESTAMPTZ,
    -- Synthea: DocumentReference.date

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =============================================================================
-- SECTION 13: CARE PLANS
-- Backs: Encounter Detail → Care Plan section
-- Synthea resource: CarePlan
-- =============================================================================

CREATE TABLE care_plans (
    care_plan_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: CarePlan.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id        UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    -- Synthea: CarePlan.encounter

    follow_up_instructions TEXT,
    -- Synthea: CarePlan.note[0].text
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE care_plan_goals (
    goal_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    care_plan_id        UUID NOT NULL REFERENCES care_plans(care_plan_id) ON DELETE CASCADE,
    goal_description    TEXT NOT NULL,
    -- Synthea: CarePlan.goal -> Goal.description.text
    goal_status         care_plan_goal_status DEFAULT 'in_progress',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE care_plan_interventions (
    intervention_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    care_plan_id        UUID NOT NULL REFERENCES care_plans(care_plan_id) ON DELETE CASCADE,
    intervention_description TEXT NOT NULL,
    -- Synthea: CarePlan.activity[0].detail.description
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE care_plan_referrals (
    referral_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    care_plan_id        UUID NOT NULL REFERENCES care_plans(care_plan_id) ON DELETE CASCADE,
    specialist_name     VARCHAR(200),
    department          VARCHAR(200),
    urgency             referral_urgency DEFAULT 'routine',
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =============================================================================
-- SECTION 14: IMMUNIZATIONS
-- Backs: Clinical Chart → Immunizations tab, Snapshot → Immunization Status card
-- Synthea resource: Immunization
-- =============================================================================

CREATE TABLE immunizations (
    immunization_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Synthea: Immunization.id
    patient_id          UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id        UUID REFERENCES encounters(encounter_id),
    -- Synthea: Immunization.encounter (nullable — some recorded outside encounters)
    administered_by     VARCHAR(200),
    -- Synthea: Immunization.performer[0].actor.display
    administered_at_facility_id UUID REFERENCES organizations(organization_id),

    vaccine_name        VARCHAR(300) NOT NULL,
    -- Synthea: Immunization.vaccineCode.text
    cvx_code            VARCHAR(20),
    -- Synthea: Immunization.vaccineCode.coding[system=CVX].code
    date_administered   DATE NOT NULL,
    -- Synthea: Immunization.occurrenceDateTime
    dose_number         INTEGER,
    -- Synthea: Immunization.protocolApplied[0].doseNumberPositiveInt
    series_total        INTEGER,
    route               VARCHAR(100),
    -- Synthea: Immunization.route.text
    site                VARCHAR(100),
    -- Synthea: Immunization.site.text
    lot_number          VARCHAR(100),
    -- Synthea: Immunization.lotNumber
    manufacturer        VARCHAR(200),
    -- Synthea: Immunization.manufacturer.display
    status              immunization_status NOT NULL DEFAULT 'completed',
    -- Synthea: Immunization.status

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Vaccine schedule reference for Snapshot immunization compliance logic (up-to-date / overdue)
-- Maps CVX codes to family names and recommended re-administration intervals
CREATE TABLE vaccine_families (
    cvx_code                    VARCHAR(20) PRIMARY KEY,
    family_name                 VARCHAR(100) NOT NULL,
    -- e.g. 'influenza', 'tetanus_diphtheria', 'covid19'
    recommended_interval_days   INTEGER
    -- NULL = one-time or series-based; > 0 = recurring (365 = annual)
);

INSERT INTO vaccine_families (cvx_code, family_name, recommended_interval_days) VALUES
    ('88',  'influenza',          365),
    ('115', 'tetanus_diphtheria', 3650),
    ('207', 'covid19',            NULL),
    ('208', 'covid19',            NULL),
    ('217', 'covid19',            NULL),
    ('218', 'covid19',            NULL),
    ('219', 'covid19',            NULL),
    ('228', 'covid19',            NULL);


-- =============================================================================
-- SECTION 15: SOCIAL HISTORY & SDOH
-- Backs: Clinical Chart → Social History tab, Snapshot → Social History card
-- Synthea resource: Observation (category=social-history)
-- =============================================================================

CREATE TABLE social_history (
    social_history_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id          UUID NOT NULL UNIQUE REFERENCES patients(patient_id) ON DELETE CASCADE,
    -- One canonical record per patient; updated in place

    -- Smoking
    smoking_status      VARCHAR(100),
    -- Synthea: Observation[LOINC=72166-2].valueCodeableConcept.text
    pack_years          NUMERIC(5,1),
    quit_date           DATE,

    -- Alcohol
    alcohol_frequency   VARCHAR(100),
    -- Synthea: Observation[LOINC=74013-4].valueCodeableConcept.text
    alcohol_drinks_per_week NUMERIC(4,1),

    -- Substance use
    recreational_substance_use VARCHAR(200),
    -- Synthea: Observation[LOINC=74204-9]

    -- Lifestyle
    physical_activity_level VARCHAR(100),
    -- Synthea: Observation[LOINC=68516-4]
    diet_notes          TEXT,

    -- Occupation & housing
    occupation          VARCHAR(200),
    -- Synthea: Observation[LOINC=21843-8]
    employer            VARCHAR(200),
    housing_status      housing_status,
    -- Synthea: Observation[LOINC=71802-3]
    education_level     VARCHAR(100),
    -- Synthea: Observation[LOINC=63504-5]

    -- SDOH flags
    financial_strain    BOOLEAN,
    -- Synthea: Observation[LOINC=76513-1]
    transportation_access BOOLEAN,
    social_support_status VARCHAR(200),

    last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    recorded_by         VARCHAR(200),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- =============================================================================
-- SECTION 16: CONDITION ↔ LAB ASSOCIATIONS
-- Backs: Conditions tab → condition detail panel → "associated lab tests"
-- Application-level mapping (not directly in Synthea; define manually or by LOINC logic)
-- =============================================================================

CREATE TABLE condition_lab_associations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    condition_id        UUID NOT NULL REFERENCES conditions(condition_id) ON DELETE CASCADE,
    lab_result_id       UUID NOT NULL REFERENCES lab_results(lab_result_id) ON DELETE CASCADE,
    UNIQUE(condition_id, lab_result_id)
);


-- =============================================================================
-- INDEXES for common query patterns
-- =============================================================================

-- Patient lookup
CREATE INDEX idx_patients_mrn ON patients(mrn);

-- Encounter queries by patient
CREATE INDEX idx_encounters_patient ON encounters(patient_id, encounter_date DESC);

-- Vitals trend
CREATE INDEX idx_vitals_patient_time ON vitals(patient_id, recorded_at DESC);

-- Lab trend (per test across time)
CREATE INDEX idx_labs_patient_loinc_time ON lab_results(patient_id, loinc_code, collected_at DESC);

-- Active conditions quick fetch
CREATE INDEX idx_conditions_patient_status ON conditions(patient_id, status);

-- Active medications quick fetch
CREATE INDEX idx_medications_patient_status ON medications(patient_id, status);

-- Imaging by patient
CREATE INDEX idx_imaging_patient ON imaging_studies(patient_id, date_performed DESC);

-- Immunizations by patient
CREATE INDEX idx_immunizations_patient ON immunizations(patient_id, date_administered DESC);

-- Upcoming appointments (Snapshot "next scheduled appointment")
CREATE INDEX idx_scheduled_appointments_patient ON scheduled_appointments(patient_id, scheduled_datetime);


-- =============================================================================
-- TABLE SUMMARY (32 tables)
-- =============================================================================
-- patients                       — core identity, banner, demographics panel
-- patient_emergency_contacts     — emergency contacts (1:many)
-- patient_alerts                 — active alerts/flags in banner
-- practitioners                  — doctors, nurses, radiologists (+ department column)
-- organizations                  — hospitals, labs, imaging centers
-- encounters                     — every visit (list + detail header)
-- encounter_care_team            — care team members per encounter
-- encounter_discharge            — discharge details for inpatient/ER
-- discharge_prescriptions        — prescriptions explicitly issued at discharge
-- encounter_followup_appointments — follow-up appointments at discharge (+ provider_id FK)
-- scheduled_appointments         — upcoming/future appointments (Snapshot next-appt value)
-- conditions                     — full longitudinal condition/problem list
-- encounter_conditions           — which conditions appeared in which encounter (+ confirmed_date, confirmed_by_id)
-- vitals                         — all vital sign readings per encounter
-- lab_panels                     — lookup: CBC, Metabolic, Urinalysis, etc.
-- lab_results                    — all lab results with trend support
-- medications                    — all prescriptions (ordered_at TIMESTAMPTZ; renewal_due_date)
-- condition_medications          — condition ↔ medication associations
-- allergies                      — allergy and intolerance records
-- procedures                     — procedures performed per encounter
-- imaging_studies                — imaging orders and metadata
-- diagnostic_reports             — report narratives for imaging and procedures (one-directional FK)
-- clinical_notes                 — progress notes, discharge summaries, etc.
-- care_plans                     — care plan header per encounter
-- care_plan_goals                — goals within a care plan
-- care_plan_interventions        — interventions within a care plan
-- care_plan_referrals            — referrals placed within a care plan
-- immunizations                  — vaccination history
-- vaccine_families               — CVX code → family + recommended interval (compliance logic)
-- social_history                 — one row per patient, all SDOH fields
-- condition_lab_associations     — condition ↔ relevant lab result links
