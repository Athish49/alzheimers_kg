# Doctor-Facing EHR: Layout and Navigation Spec

## Purpose of this document
This describes the information architecture, page structure, data shapes, and click-through navigation for a doctor-facing patient chart. It defines what each piece of data is (a single value, a list, a table with named columns, a trend series, etc.) and how a user moves between views. It intentionally contains no visual, style, or theme direction so that design decisions stay open.

---

## The consolidation
The original design spanned 9 pages. This collapses them into 3 primary destinations plus one persistent banner:

1. **Patient Banner** (persistent, visible on every screen)
2. **Snapshot** (the landing destination, all summaries with paths into detail)
3. **Encounters** (a master-detail destination: a list of visits that opens into a full visit detail)
4. **Clinical Chart** (a single tabbed destination that holds Conditions, Medications, Labs, Imaging, Immunizations, and Social History)

Original page to new destination mapping:

- Page 1 (Patient Overview) becomes **Snapshot**, with demographics moved into a panel off the banner.
- Page 2 (Encounter History table) and Page 3 (Encounter Detail) merge into **Encounters** (list state and detail state).
- Pages 4 through 9 (Conditions, Medications, Labs, Imaging, Immunizations, Social History) become the six **tabs** of the **Clinical Chart**.

The guiding principle: a new user should never have to hunt. The Snapshot answers "what is going on with this patient right now" and every summary block has one obvious path to its full history.

---

## Global elements (present on all screens)

### Patient Banner (persistent header strip)
Always visible, does not scroll away. It is a row of single values plus one short list.

Contents:
- Patient full name and preferred name (single value)
- MRN and Patient ID (single value)
- DOB and age (single value)
- Biological sex and gender identity (single value)
- Primary care physician (single value)
- Allergies (short list of allergen names, each with a severity indicator shown as text)
- Any active alert or flag (short list, may be empty)

**Navigation from the banner:**
- Clicking the patient name (or an info affordance on the banner) opens the **Demographics and Identity panel** as a slide-over or modal. This is where the fuller identity data lives so it does not consume primary screen space.

**Demographics and Identity panel** (opened from banner). A key-value block containing:
- Race, ethnicity, primary language (single values)
- Address, phone, email (single values)
- Emergency contact: name, relationship, phone (single value group)
- Insurance / payer name, member ID, group number (single value group)

The panel closes back to whatever screen the user was on.

### Primary navigation
A persistent way to reach the 3 destinations (Snapshot, Encounters, Clinical Chart) from anywhere. When the user is inside the Clinical Chart, the tab they last used should remain selected when they return to it.

### Cross-cutting rules
- Every data section shows a **last-updated timestamp**.
- Every individual data point shows **ordering physician or recorded-by attribution**.
- Any data point that originated in a specific visit is a **link back to that Encounter Detail** (see Encounters below). This is the connective tissue of the whole chart.

---

## Destination 1: Snapshot (landing)

A set of summary cards. Each card shows a condensed view and, where a fuller history exists, a single action that opens the relevant Clinical Chart tab or the Encounters destination. This is the "small list here, view all there" pattern applied throughout.

### Card: Active Problems
- Shape: a short list. Each row is one active condition: condition name, ICD-10 or SNOMED code, onset date, status (active / chronic).
- Shows only currently active conditions.
- **Click a condition row:** opens that condition's detail (routes to the Clinical Chart Conditions tab with that condition expanded).
- **"View all conditions" action:** opens the Clinical Chart on the Conditions tab.

### Card: Current Medications
- Shape: a short list. Each row: drug name, dose, frequency, route, and a renewal-due flag when applicable.
- Shows active prescriptions only.
- **Click a medication row:** opens that medication's detail (routes to the Clinical Chart Medications tab with that drug expanded).
- **"View full medication history" action:** opens the Clinical Chart on the Medications tab.

### Card: Allergies and Reactions
- Shape: a short list. Each row: allergen, reaction type, severity (mild / moderate / severe), date recorded.
- Mirrors the allergy summary in the banner but with reaction detail.

### Card: Latest Vitals
- Shape: a key-value block of single values, each paired with its reading date: blood pressure, heart rate, temperature, respiratory rate, SpO2, height, weight, BMI.
- **"View vitals trend" action:** opens a **Vitals Trend panel** showing each vital as a time-ordered series across encounters (a set of readings suitable for plotting or listing chronologically).

### Card: Recent Encounters
- Shape: a short list (last 3 visits). Each row: date, visit type, department, provider, reason. If a future appointment exists, show the next scheduled appointment as a single value below the list.
- **Click an encounter row:** opens that visit in Encounter Detail (see Encounters).
- **"View all encounters" action:** opens the Encounters destination in list state.

### Card: Immunization Status
- Shape: a short summary. A single up-to-date / overdue status, plus a few key vaccines as single values (last flu shot date, last tetanus date, COVID series status).
- **"View all immunizations" action:** opens the Clinical Chart on the Immunizations tab.

### Card: Recent and Key Labs
- Shape: a short list of the most recent or abnormal results. Each row: test name, value, unit, interpretation flag (Normal / High / Low / Critical).
- **Click a lab row:** opens that test's Trend view (see Labs tab).
- **"View all labs" action:** opens the Clinical Chart on the Labs tab.

### Card: Social History Snapshot
- Shape: a key-value block of single values: smoking status, alcohol use, recreational drug use, occupation, housing status.
- **"View full social history" action:** opens the Clinical Chart on the Social History tab.

---

## Destination 2: Encounters (master-detail)

One destination with two states. The user lands in **list state** and clicks a row to move into **detail state** for a single visit. A clear back action returns to the list.

### List state (the encounter table)
- Shape: a table. One row per visit.
- Columns: encounter date, visit type (Inpatient / Outpatient / Emergency / Telehealth / Surgical / Observation), department or specialty, facility, attending provider, chief complaint, primary discharge diagnosis, status (Completed / Active / Cancelled).
- The table is sortable and filterable (by date range, visit type, department, status).
- **Click any row:** opens Encounter Detail for that visit.

### Detail state (Encounter Detail)
Everything about one visit. This is the densest view. It is a single scrollable view broken into labeled sections, with a way to jump to any section. Each section below lists its data shape.

**Encounter Header**
- Shape: a key-value block. Encounter ID, date, visit type, department, facility, attending physician, admission date and time, discharge date and time, length of stay, chief complaint.
- Care team (nurses, consultants, residents): a short list of names with role.

**Diagnoses for this encounter**
- Shape: a table. Columns: diagnosis name, ICD-10 code, primary or secondary, onset (new vs existing condition), status, date confirmed, confirming clinician. Secondary diagnoses listed in order below the primary.

**Vitals recorded during this visit**
- Shape: a table (a timeline of readings). Each row: timestamp, BP, HR, temperature, RR, SpO2, weight. One row per reading taken during the encounter.

**Lab results for this encounter**
- Shape: grouped tables, one group per panel type (CBC, Metabolic Panel, Lipid Panel, Urinalysis, and so on).
- Each row: test name (LOINC), result value, unit, reference range (low and high), interpretation flag (Normal / High / Low / Critical), collection date and time, ordering physician, performing lab.
- **Click a lab row:** opens that test's Trend view showing how this result compares to prior readings of the same test (a time-ordered series).

**Procedures performed**
- Shape: a table. Columns: procedure name, CPT or SNOMED code, date and time performed, duration, performing physician, status (completed / cancelled).
- Any linked diagnostic report is a link on the row.

**Medications ordered or administered**
- Shape: a table. Columns: medication name, dose, route, frequency, ordered by, ordered at, administered-during-visit vs sent-as-prescription, start date, end date if applicable.

**Imaging studies ordered**
- Shape: a table. Columns: study type (X-ray / CT / MRI / Ultrasound / Echo / etc.), body region and laterality, date ordered, date performed, ordering physician, performing facility, status (ordered / completed / preliminary / final).
- **Click a row:** opens a report narrative panel (text only, no DICOM).

**Clinical notes**
- Shape: a list of note entries. Each entry: note type (Progress Note / Discharge Summary / Consultation Note / Operative Report / Nursing Note), author, date and time, and note text.
- Each note's text is expandable in place.

**Care plan** (shown when one exists)
- Shape: a structured block. Goals set during this encounter (list), interventions planned (list), follow-up instructions (text), referrals placed (list, each with specialist name, department, urgency).

**Discharge information** (shown only for inpatient and ER visits)
- Shape: a mixed block. Discharge disposition (single value: home / skilled nursing / transferred / AMA), discharge instructions summary (text), follow-up appointments scheduled (list), prescriptions given at discharge (list).

**Insurance and billing** (optional, shown when present)
- Shape: a key-value block. Payer billed, claim status, DRG code, total charges.

**Navigation out of Encounter Detail:** a back action returns to the encounter list state. Section jump links let the user move within the detail without scrolling.

---

## Destination 3: Clinical Chart (tabbed)

A single destination with six tabs. Only one tab is shown at a time. This is where all longitudinal, cross-encounter history lives. Each tab is described below with its data shape and click behavior.

### Tab: Conditions
- Shape: a table (full longitudinal list of every condition, not just active ones).
- Columns: condition name, SNOMED or ICD-10 code, category (chronic / acute / historical), onset date, resolution date if resolved, status (active / resolved / recurrence), first-diagnosed encounter (a link to Encounter Detail), treating physician.
- Filterable by status and category.
- **Click a condition row:** opens a condition detail panel containing associated medications for this condition (list), associated lab tests relevant to this condition (list, each linking to that test's Trend view), and a link back to the originating encounter.

### Tab: Medications
- Shape: a table (every medication ever prescribed).
- Columns: drug name, RxNorm code, brand or generic, dose, route, frequency, prescribed by, prescribed date, start date, end date, status (active / completed / discontinued / on hold), reason for discontinuation if stopped, ordering encounter (a link to Encounter Detail), refill count, last filled date.
- Filterable by status (active / discontinued / completed / on hold).
- **Click a medication row:** opens medication detail with all fields plus the encounter link.
- **Secondary section within this tab: Allergies and Intolerances.** Shape: a table. Columns: allergen, reaction detail, severity, date recorded, recorder.

### Tab: Labs
- Shape: grouped tables, one group per test family:
  - CBC: WBC, RBC, Hemoglobin, Hematocrit, Platelets, MCV, MCH
  - Metabolic: Sodium, Potassium, Chloride, CO2, BUN, Creatinine, eGFR, Glucose, Calcium
  - Liver function: ALT, AST, ALP, Bilirubin, Albumin
  - Lipid panel: Total Cholesterol, LDL, HDL, Triglycerides
  - Endocrine: HbA1c, TSH, Free T4, Cortisol
  - Coagulation: PT, INR, PTT
  - Other: PSA, Ferritin, Vitamin D, B12, CRP, ESR
- Each row: test name, value, unit, reference range, flag (Normal / High / Low / Critical), collection date and time, ordering encounter (a link to Encounter Detail), ordering physician.
- **Click a test name or row:** opens the **Trend view** for that single test. Shape: a chronological series of every reading of that test across time (for example all HbA1c readings since 2018), suitable for a trend line or a dated list.

### Tab: Imaging
- Shape: a table (every imaging study across all encounters).
- Columns: study type (X-ray / CT / MRI / Ultrasound / Echocardiogram / PET / Mammogram), body region, laterality, date ordered, date performed, ordering physician, performing facility, interpreting radiologist, indication or clinical reason, status, ordering encounter (a link to Encounter Detail).
- **Click a row:** opens the report narrative panel (text only, no DICOM).

### Tab: Immunizations
- Shape: a table (full vaccination history), with an optional summary strip at the top mirroring the Snapshot immunization status (up-to-date vs overdue).
- Columns: vaccine name (CVX code), date administered, dose number in series, route, site, lot number, manufacturer, administering provider or facility, status (completed / not done / refused).

### Tab: Social History and SDOH
- Shape: a key-value block grouped into labeled categories. Most entries are single values.
  - Smoking: status, pack-years, quit date if applicable
  - Alcohol: frequency, drinks per week
  - Recreational substance use (single value)
  - Physical activity level (single value)
  - Diet notes (text)
  - Occupation and employer (single values)
  - Housing status (single value: stable / unstable / homeless)
  - Education level (single value)
  - Financial strain flag (single value)
  - Transportation access (single value)
  - Social support or caregiver status (single value)
  - Last updated date and recorded by (single values)

---

## Navigation map (clicking X opens Y)

- Banner patient name to Demographics and Identity panel.
- Snapshot Active Problems row to Clinical Chart, Conditions tab, condition expanded.
- Snapshot "View all conditions" to Clinical Chart, Conditions tab.
- Snapshot Current Medications row to Clinical Chart, Medications tab, drug expanded.
- Snapshot "View full medication history" to Clinical Chart, Medications tab.
- Snapshot Latest Vitals "View vitals trend" to Vitals Trend panel.
- Snapshot Recent Encounters row to Encounter Detail.
- Snapshot "View all encounters" to Encounters list state.
- Snapshot "View all immunizations" to Clinical Chart, Immunizations tab.
- Snapshot Recent Labs row to that test's Trend view.
- Snapshot "View all labs" to Clinical Chart, Labs tab.
- Snapshot "View full social history" to Clinical Chart, Social History tab.
- Encounters list row to Encounter Detail.
- Encounter Detail lab row to that test's Trend view.
- Encounter Detail imaging row to report narrative panel.
- Encounter Detail back action to Encounters list state.
- Conditions tab row to condition detail panel (with associated meds, associated labs, originating encounter link).
- Medications tab row to medication detail.
- Labs tab test to Trend view.
- Imaging tab row to report narrative panel.
- Any "first-diagnosed encounter," "ordering encounter," or encounter-origin link, anywhere in the Clinical Chart, to the corresponding Encounter Detail.

---

## Data shape legend
- **Single value:** one labeled datum (for example, blood pressure with its reading date).
- **Key-value block:** a group of labeled single values shown together.
- **Short list:** a small number of list items, used on the Snapshot.
- **Table:** rows and named columns, sortable and filterable where noted.
- **Grouped tables:** several tables, one per category or panel.
- **Trend view / series:** a time-ordered set of readings for a single metric.
- **Panel:** content that opens over or beside the current screen and closes back to it.
- **Detail state / view:** a full-screen view reached from a list, with a back action.
