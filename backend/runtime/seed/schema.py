"""
runtime.seed.schema
-------------------
Idempotent DDL: creates all tables, the app_runtime role, grants, and RLS.

Run order:
  1. apply_schema()  — create tables + role + grants
  2. apply_rls()     — enable RLS + create policies (called by apply_schema)

All statements use IF NOT EXISTS / DO NOTHING so re-running is safe.
The table owner (neondb_owner) bypasses RLS; app_runtime is subject to it.
"""

from __future__ import annotations

import logging

from .db import transaction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL blocks (applied in dependency order)
# ---------------------------------------------------------------------------

_POLICY_TABLES = """
-- ── Policy (slow-changing, global, no session scope) ─────────────────────

CREATE TABLE IF NOT EXISTS roles (
    role_id     text PRIMARY KEY,
    label       text NOT NULL,
    description text
);

CREATE TABLE IF NOT EXISTS permission_categories (
    resource    text PRIMARY KEY,
    description text
);

CREATE TABLE IF NOT EXISTS role_permissions (
    role_id         text REFERENCES roles(role_id),
    resource        text,
    action          text,           -- 'read' | 'write'
    patient_binding text NOT NULL,  -- 'assigned' | 'deidentified' | 'none'
    allowed_fields  text[],         -- {*} = all fields
    PRIMARY KEY (role_id, resource, action)
);
"""

_IDENTITY_TABLES = """
-- ── Identity ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    user_id    text PRIMARY KEY,
    name       text NOT NULL,
    role_id    text REFERENCES roles(role_id),
    department text,
    care_team  text,
    is_persona boolean DEFAULT false
);
"""

_RUNTIME_TABLES = """
-- ── Runtime ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sessions (
    session_id   text PRIMARY KEY,
    user_id      text REFERENCES users(user_id),
    created_at   timestamptz DEFAULT now(),
    last_seen_at timestamptz DEFAULT now(),
    expires_at   timestamptz
);

CREATE TABLE IF NOT EXISTS audit_log (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts             timestamptz DEFAULT now(),
    session_id     text,
    user_id        text,
    role_id        text,
    action         text,
    resource       text,
    patient_id     text,
    effect         text,       -- 'permit' | 'deny'
    reason         text,
    break_glass    boolean DEFAULT false,
    fields_accessed text[]
);

CREATE TABLE IF NOT EXISTS break_glass_grants (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id  text,
    user_id     text,
    patient_id  text,
    reason      text NOT NULL,
    granted_at  timestamptz DEFAULT now(),
    expires_at  timestamptz,
    reviewed    boolean DEFAULT false
);
"""

_EHR_ASSIGNMENT_TABLE = """
-- ── EHR practitioner assignments (non-session-scoped, cross-schema) ───────
-- Links runtime users to Synthea EHR patients by UUID.
-- Not RLS-protected (EHR plane is not session-scoped); app-level enforcement only.

CREATE TABLE IF NOT EXISTS ehr_patient_assignments (
    practitioner_id  text REFERENCES users(user_id) ON DELETE CASCADE,
    patient_id       uuid NOT NULL,   -- references ehr.patients(patient_id)
    PRIMARY KEY (practitioner_id, patient_id)
);
"""

_CLINICAL_TABLES = """
-- ── Clinical (synthetic PHI, session-scoped) ──────────────────────────────

CREATE TABLE IF NOT EXISTS patients (
    session_id   text NOT NULL,
    patient_id   text NOT NULL,
    name         text,
    dob          date,
    sex          text,
    mrn          text,
    address      text,
    insurance_id text,
    department   text,
    care_team    text,
    headline     text,
    PRIMARY KEY (session_id, patient_id)
);

CREATE TABLE IF NOT EXISTS patient_assignments (
    session_id   text NOT NULL,
    user_id      text REFERENCES users(user_id),
    patient_id   text NOT NULL,
    relationship text,
    care_team    text,
    PRIMARY KEY (session_id, user_id, patient_id)
);

CREATE TABLE IF NOT EXISTS conditions (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id  text NOT NULL,
    patient_id  text NOT NULL,
    code        text,
    label       text,
    onset_date  date,
    status      text
);

CREATE TABLE IF NOT EXISTS vitals (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id    text NOT NULL,
    patient_id    text NOT NULL,
    taken_at      timestamptz DEFAULT now(),
    bp_systolic   int,
    bp_diastolic  int,
    heart_rate    int,
    temp_c        numeric(4,1),
    resp_rate     int
);

CREATE TABLE IF NOT EXISTS medications (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id         text NOT NULL,
    patient_id         text NOT NULL,
    drug_name          text,
    dose               text,
    route              text,
    frequency          text,
    prescriber_user_id text,
    status             text    -- 'active' | 'discontinued'
);

CREATE TABLE IF NOT EXISTS lab_results (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id         text NOT NULL,
    patient_id         text NOT NULL,
    test_name          text,
    value              numeric,
    unit               text,
    reference_range    text,
    collected_at       timestamptz,
    status             text,
    entered_by_user_id text
);

CREATE TABLE IF NOT EXISTS genetic_markers (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id     text NOT NULL,
    patient_id     text NOT NULL,
    gene           text,
    variant        text,
    interpretation text,
    source         text
);

CREATE TABLE IF NOT EXISTS clinical_notes (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id     text NOT NULL,
    patient_id     text NOT NULL,
    author_user_id text,
    created_at     timestamptz DEFAULT now(),
    body           text
);
"""

# The app_runtime role: limited Postgres role used for every request path.
# It is granted to neondb_owner so the app can SET ROLE to it per transaction.
# CRITICAL: audit_log gets INSERT + SELECT only — no UPDATE, no DELETE.
_ROLE_AND_GRANTS = """
DO $$ BEGIN
    CREATE ROLE app_runtime NOLOGIN;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Allow neondb_owner (our connection user) to switch to app_runtime
DO $$ BEGIN
    GRANT app_runtime TO neondb_owner;
EXCEPTION WHEN others THEN NULL;
END $$;

-- Policy tables: read-only
GRANT SELECT ON roles, permission_categories, role_permissions, users TO app_runtime;

-- Session/runtime tables: full CRUD except audit_log
GRANT SELECT, INSERT, UPDATE, DELETE ON sessions           TO app_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON break_glass_grants TO app_runtime;

-- EHR assignments: SELECT only — app reads to enforce patient filter
GRANT SELECT ON ehr_patient_assignments TO app_runtime;

-- Clinical + relationship: full CRUD (RLS provides the actual restriction)
GRANT SELECT, INSERT, UPDATE, DELETE ON patients            TO app_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON patient_assignments TO app_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON conditions          TO app_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON vitals              TO app_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON medications         TO app_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON lab_results         TO app_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON genetic_markers     TO app_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON clinical_notes      TO app_runtime;

-- audit_log: INSERT + SELECT ONLY — the append-only invariant.
-- Explicit REVOKE before GRANT so any previously accumulated permissions
-- (UPDATE, DELETE) are stripped on every boot, not just newly added grants.
REVOKE ALL ON audit_log FROM app_runtime;
GRANT SELECT, INSERT ON audit_log TO app_runtime;

-- Identity column sequences
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_runtime;
"""

# RLS: enforce session and patient scope at the DB level (independent backstop).
# The table owner (neondb_owner) bypasses RLS automatically — seed/clone paths
# are unaffected. app_runtime is always subject to these policies.
_RLS_POLICIES = """
-- ── Session-and-patient-scoped clinical tables ────────────────────────────

ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_patients ON patients;
CREATE POLICY rls_patients ON patients FOR ALL TO app_runtime
    USING (
        session_id = current_setting('app.session_id', true)
        AND patient_id = ANY(
            string_to_array(current_setting('app.patient_scope', true), ',')
        )
    );

ALTER TABLE conditions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_conditions ON conditions;
CREATE POLICY rls_conditions ON conditions FOR ALL TO app_runtime
    USING (
        session_id = current_setting('app.session_id', true)
        AND patient_id = ANY(
            string_to_array(current_setting('app.patient_scope', true), ',')
        )
    );

ALTER TABLE vitals ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_vitals ON vitals;
CREATE POLICY rls_vitals ON vitals FOR ALL TO app_runtime
    USING (
        session_id = current_setting('app.session_id', true)
        AND patient_id = ANY(
            string_to_array(current_setting('app.patient_scope', true), ',')
        )
    );

ALTER TABLE medications ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_medications ON medications;
CREATE POLICY rls_medications ON medications FOR ALL TO app_runtime
    USING (
        session_id = current_setting('app.session_id', true)
        AND patient_id = ANY(
            string_to_array(current_setting('app.patient_scope', true), ',')
        )
    );

ALTER TABLE lab_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_lab_results ON lab_results;
CREATE POLICY rls_lab_results ON lab_results FOR ALL TO app_runtime
    USING (
        session_id = current_setting('app.session_id', true)
        AND patient_id = ANY(
            string_to_array(current_setting('app.patient_scope', true), ',')
        )
    );

ALTER TABLE genetic_markers ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_genetic_markers ON genetic_markers;
CREATE POLICY rls_genetic_markers ON genetic_markers FOR ALL TO app_runtime
    USING (
        session_id = current_setting('app.session_id', true)
        AND patient_id = ANY(
            string_to_array(current_setting('app.patient_scope', true), ',')
        )
    );

ALTER TABLE clinical_notes ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_clinical_notes ON clinical_notes;
CREATE POLICY rls_clinical_notes ON clinical_notes FOR ALL TO app_runtime
    USING (
        session_id = current_setting('app.session_id', true)
        AND patient_id = ANY(
            string_to_array(current_setting('app.patient_scope', true), ',')
        )
    );

-- ── Session-only scoped tables (no patient filter needed) ─────────────────

-- patient_assignments: session-scoped only (PDP reads all of a user's
-- assignments to determine if the pinned patient is authorized — cannot
-- restrict by patient_scope here, as the assignment check IS the auth check)
ALTER TABLE patient_assignments ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_assignments ON patient_assignments;
CREATE POLICY rls_assignments ON patient_assignments FOR ALL TO app_runtime
    USING (session_id = current_setting('app.session_id', true));

ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_sessions ON sessions;
CREATE POLICY rls_sessions ON sessions FOR ALL TO app_runtime
    USING (session_id = current_setting('app.session_id', true));

ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_audit_log ON audit_log;
CREATE POLICY rls_audit_log ON audit_log FOR ALL TO app_runtime
    USING (session_id = current_setting('app.session_id', true));

ALTER TABLE break_glass_grants ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rls_break_glass ON break_glass_grants;
CREATE POLICY rls_break_glass ON break_glass_grants FOR ALL TO app_runtime
    USING (session_id = current_setting('app.session_id', true));
"""


_MIGRATIONS = """
-- Column migrations (idempotent via IF NOT EXISTS)
ALTER TABLE patients ADD COLUMN IF NOT EXISTS headline text;
"""


def apply_schema() -> None:
    """
    Create all tables, the app_runtime role, grants, and RLS policies.
    Idempotent — safe to run on every boot.
    """
    blocks = [
        ("policy tables",        _POLICY_TABLES),
        ("identity tables",      _IDENTITY_TABLES),
        ("runtime tables",       _RUNTIME_TABLES),
        ("EHR assignment table", _EHR_ASSIGNMENT_TABLE),
        ("clinical tables",      _CLINICAL_TABLES),
        ("role + grants",        _ROLE_AND_GRANTS),
        ("RLS policies",         _RLS_POLICIES),
        ("migrations",           _MIGRATIONS),
    ]
    with transaction() as conn:
        cur = conn.cursor()
        for label, sql in blocks:
            cur.execute(sql)
            logger.info("Schema applied: %s", label)
    logger.info("Schema complete.")
