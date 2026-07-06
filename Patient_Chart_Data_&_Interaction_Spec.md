# Patient Chart — Data Model & Interaction Spec

Reference doc for rebuilding this EHR patient-chart prototype. Describes every data entity, where it's shown, and exactly what happens on every click.

---

## 1. App shell (persistent across all views)

**Top bar**: wordmark + 3-way nav (`Snapshot` / `Encounters` / `Clinical Chart`) + "Ask Atlas" chat button.

**Patient banner** (sticky, always visible): patient identity (name, MRN, DOB/age, sex, PCP), allergy chips, alert badges.
- Click patient name/avatar → opens **Demographics panel** (right slide-over).
- Click an allergy chip (or the "+N" overflow chip) → navigates to Clinical Chart → Medications tab, scrolled to the "Allergies & intolerances" section.

**Chart tab bar**: only visible when top nav = Clinical Chart. 6 tabs: Conditions, Medications, Labs, Imaging, Immunizations, Social History.

**Right slide-over panel**: a secondary reading pane that opens over the main content (not a navigation — main view stays put underneath). Used for drill-downs that don't deserve a full page: demographics, trends, reports, condition detail, medication detail. Closing it (X or backdrop click) just removes the overlay; underlying view/scroll position is preserved.

**Ask Atlas chat drawer**: separate right drawer (different z-layer from the panel), a chat interface scoped to this one patient's full record. Not part of the click-through data flows below.

---

## 2. Top-level views (primary nav)

### 2.1 Snapshot (default view)
A dashboard of independent summary cards, arranged in 3 columns:
- **Column 1**: Active Problems, Recent & Key Labs
- **Column 2**: Current Medications, Recent Encounters
- **Column 3**: Social History, Immunization Status, Latest Vitals

Each card has a title, "updated" timestamp, a list or key-value body, and (usually) a "View all…" action link at the bottom.

**Card contents & click behavior:**

| Card | Rows shown | Click a row | Card's "View all" action |
|---|---|---|---|
| Active Problems | 5 chronic/active conditions (name, onset, ICD-10 code, status badge) | → **Condition panel** (right slide-over) for that condition | → Clinical Chart / Conditions tab |
| Current Medications | 5 active meds (name, dose/route/freq, renewal badge if due) | → **Medication panel** for that drug | → Clinical Chart / Medications tab |
| Latest Vitals | key-value: BP, HR, Temp, RR, SpO2, Ht/Wt, BMI (no per-row click) | — | → **Vitals Trend panel** |
| Recent Encounters | 3 most recent encounters (type/dept, chief complaint, date) + a footnote with the next scheduled appt | → **Encounter Detail** (full view navigation, not a panel) | → Encounters view (list) |
| Immunization Status | key-value summary (overall status + 3 recent vaccines) | — | → Clinical Chart / Immunizations tab |
| Recent & Key Labs | 5 key labs (name, result, high/normal/low badge) | → **Lab Trend panel** for that test | → Clinical Chart / Labs tab |
| Social History | key-value: smoking, alcohol, drugs, occupation, housing | — | → Clinical Chart / Social History tab |

Card body types: **list rows** (primary/secondary text + right-aligned meta/badge, whole row clickable) vs **key-value rows** (label left, value+sub right, not clickable).

### 2.2 Encounters
Two states of the *same view* (not two separate routes):

**List state** — one sortable/filterable table of all encounters. Columns: Date, Type, Department, Facility, Attending, Chief complaint, Primary dx, Status.
- Click any row → opens **Encounter Detail** for that row's encounter, same view.

**Detail state** (`encounterId` set) — shows one encounter in full. Chrome specific to this state:
- "← All encounters" back link → returns to list state (clears `encounterId`).
- A row of section filter chips generated from the encounter's own section list (e.g. "Encounter header", "Diagnoses", "Vitals", "Lab results", "Procedures", "Medications", "Imaging", "Clinical notes", "Care plan", "Discharge", "Insurance & billing" — the exact set depends on the encounter, lighter encounters have fewer sections). Plus an "All" chip.
  - Selecting a chip filters the page to show only the title + that one section (client-side, no reload).
  - Selecting "All" (or clicking the active chip again) shows every section.
- Sections render top-to-bottom as a mix of block types (see §4: Block Types). Two encounters (e1 = the inpatient cardiology/NSTEMI stay, e2 = the endocrinology follow-up) have fully hand-authored rich detail; the other four (e3–e6) render from a shared lighter generic template (header kv + one diagnosis row + one note).

**Cross-links found inside encounter detail body:**
- A lab result row's **mono "Ordering enc" cell** (in the Labs tab table) is a link back to the originating encounter (`e1` in this dataset) → opens **Encounter Detail** for that encounter.
- A lab row with a trend defined → click opens **Lab Trend panel** (e.g. Troponin I, Glucose in e1; HbA1c, LDL, eGFR in e2).
- A procedure/imaging row → click opens **Report panel** (e.g. Coronary angiography, Echocardiogram, Chest X-ray).

### 2.3 Clinical Chart
Tab bar with 6 tabs; each tab is an independent dataset/page, switched instantly (no full nav, `chartTab` state only):

1. **Conditions** — "Problem List" table. Columns: Condition, Code, Category (Chronic/Acute/Historical badge), Onset, Resolved, Status (Active/Resolved badge), First dx (link to originating encounter), Physician.
   - Click a row (only for the 5 conditions that have a detail defined: T2DM, HTN, Hyperlipidemia, Hypothyroidism, OA) → opens **Condition panel**.
   - Click the "First dx" cell → opens that **Encounter Detail** (currently all point to `e1` in the sample data — wire this to each condition's real originating encounter).
2. **Medications** — two stacked tables:
   - "Medications" — Drug, RxNorm code, Dose, Route, Freq, Prescribed by, Start, End, Status badge (Active/On hold/Discontinued). Click a row (5 of 8 rows wired) → opens **Medication panel**.
   - "Allergies & intolerances" — Allergen, Reaction, Severity badge, Recorded date, Recorded by. Not clickable; this is the destination the allergy chips in the banner scroll to.
3. **Labs** — grouped tables by panel: Complete Blood Count, Metabolic panel, Lipid panel, Endocrine, Cardiac. Each row: Test, Result, Unit, Ref range, Flag badge, Ordering encounter (link). Rows with a trend defined → click opens **Lab Trend panel**; the "Ordering enc" link cell → opens **Encounter Detail**.
4. **Imaging** — table: Study, Region, Ordered, Performed, Ordering MD, Facility, Status. Every row click → opens **Report panel** (radiology narrative).
5. **Immunizations** — a green "up to date" summary banner + full vaccination table (Vaccine, CVX code, Date, Dose, Route, Site, Lot, Mfr, Status). No row clicks.
6. **Social History** — key-value groups only (Smoking, Alcohol, Substances & lifestyle, Socioeconomic, Support & access). No clicks.

---

## 3. Right slide-over panel — content per kind

The panel is a generic container: eyebrow label + title + width + a list of content blocks (same block vocabulary as main pages, §4). One `panel.kind` is active at a time; opening a new one replaces the old.

- **`demographics`** — identity (race, ethnicity, language, preferred name), contact (address/phone/email), emergency contact, insurance. Opened from: patient name/avatar in banner.
- **`vitalsTrend`** — three trend charts (Systolic BP, Heart rate, Weight) each as a sparkline-style list of dated readings with a colored bar + flag badge. Opened from: Snapshot → Latest Vitals → "View vitals trend".
- **`labTrend`** (keyed by test, e.g. `hba1c`, `ldl`, `glucose`, `egfr`, `tsh`, `troponin`, `hgb`) — one trend chart of every historical reading for that test + a narrative sentence. Opened from: any lab row across Snapshot, Encounter Detail, and the Labs tab.
- **`report`** (keyed by study, e.g. `angio`, `cxr`, `echo`, `knee`, `mammo`, `carotid`) — section header + full radiology/procedure narrative text. Opened from: Imaging tab rows and Encounter Detail procedure/imaging rows.
- **`condition`** (keyed by condition, e.g. `t2dm`, `htn`, `lipid`, `hypothy`, `oa`) — kv summary (code, category, status, first diagnosed) **then** three linked sub-lists:
  - *Associated medications* — meds tied to this condition (plain rows, not currently clickable further — could link to Medication panel).
  - *Associated labs* — labs tied to this condition; click a lab row → opens **Lab Trend panel** for that test (panel replaces panel, stacking is not supported — this is a lateral panel-to-panel jump).
  - *Originating encounter* — one row; click → opens **Encounter Detail** (closes panel implicitly since Encounter Detail is a main-view navigation, panel state is cleared as part of that nav action).
  Opened from: Snapshot → Active Problems rows; Conditions tab rows.
- **`med`** (keyed by drug, e.g. `metformin`, `lisinopril`, `atorvastatin`, `levo`, `clopidogrel`) — kv detail (RxNorm, type, dose, route, frequency, prescribed by, start, status, indication) **then** *Ordering encounter* — one row; click → opens **Encounter Detail**.
  Opened from: Snapshot → Current Medications rows; Medications tab rows.

**Panel behavior rules to preserve:**
- Opening any panel does not change the main view underneath (main view stays exactly as it was).
- Opening a panel from inside another panel replaces panel content in place (no panel stack/breadcrumb).
- Navigating to an Encounter Detail or a Chart tab from *inside* a panel closes the panel and performs the real navigation (changes `view`/`chartTab`), then (for the allergy-chip case) auto-scrolls to a target section anchor.
- Panel width varies by kind (480–560px) — wider for trends/reports/kv-heavy demographics, narrower for condition/med detail.

---

## 4. Reusable content block types (shared by main pages and panels)

Every page/tab/panel is just an ordered array of typed blocks rendered generically. Rebuilding this system means having one block-rendering component with these block "kinds," each independently reusable and composable in any order:

1. **Title** — eyebrow (optional) + H1 + subtitle (optional). Page/tab header.
2. **Section** — a mid-page H2 label + optional right-aligned meta (e.g. encounter ID, timestamp). Used to break an Encounter Detail into anchored sub-sections (`data-block-anchor`) that the section-filter chips target.
3. **Cards** — the Snapshot's 3-column grid of cards. Each card = title + updated-date + either:
   - **List body**: clickable rows (primary text, secondary text, right meta text, right badge).
   - **Key-value body**: non-clickable label/value(+sub-unit) rows.
   - optional footnote strip, optional "view all" action link.
4. **Table** (with optional filter chips and/or a summary banner above it) — column headers + rows of cells. Cell types: plain text, monospace text (codes/dates/numbers), or an arbitrary inline element (badge, link button). Rows are optionally clickable (whole `<tr>`).
5. **Grouped tables** — same as Table but split into multiple titled sub-tables stacked vertically (used for Labs, grouped by panel type; used in Encounter Detail for Labs grouped by panel).
6. **KV block** — a bordered card containing 1-N titled groups of label/value pairs, laid out in a responsive multi-column grid. Used for encounter header info, demographics, condition/med summary, social history, care plan/discharge/billing sections.
7. **Simple list** — a bordered card of primary/secondary/badge rows, each optionally clickable. Used inside panels for "Associated medications," "Associated labs," "Originating encounter," "Ordering encounter."
8. **Notes** — collapsible `<details>` cards, one per clinical note, header shows note type + author/date, body is free-text (whitespace-preserved).
9. **Trend** — a chart-like list: each reading is a date label + a proportional horizontal bar (colored by flag: normal/high/low/critical) + the value and an optional flag badge.
10. **Narrative** — a single block of free-text prose (radiology/procedure report bodies, lab-trend explainer sentence).

**Badges** are a shared, reused primitive across every block type: a colored pill keyed by semantic kind (`normal`, `high`, `low`, `critical`, `active`, `chronic`, `resolved`, `onhold`, `discontinued`, `completed`, `severe`, `moderate`, `mild`, `renewal`, `primary`/`secondary` dx type, `neutral`, `acute`, `historical`) — each kind maps to a fixed background/foreground color pair. Reuse this exact enum + color mapping so status meaning stays visually consistent everywhere (a "High" lab flag looks the same whether it's on Snapshot, in an encounter, or in the Labs tab).

---

## 5. Full click-through map (who opens what)

```
Patient banner
├─ name/avatar click ─────────────────→ Panel: demographics
└─ allergy chip click ────────────────→ Nav: Clinical Chart > Medications tab, scroll→"Allergies & intolerances"

Snapshot
├─ Active Problems row ───────────────→ Panel: condition[key]
├─ Active Problems "view all" ────────→ Nav: Clinical Chart > Conditions
├─ Current Medications row ───────────→ Panel: med[key]
├─ Current Medications "view all" ────→ Nav: Clinical Chart > Medications
├─ Latest Vitals "view trend" ────────→ Panel: vitalsTrend
├─ Recent Encounters row ─────────────→ Nav: Encounters > detail[encId]
├─ Recent Encounters "view all" ──────→ Nav: Encounters (list)
├─ Immunization "view all" ───────────→ Nav: Clinical Chart > Immunizations
├─ Recent Labs row ────────────────────→ Panel: labTrend[key]
├─ Recent Labs "view all" ────────────→ Nav: Clinical Chart > Labs
└─ Social History "view all" ─────────→ Nav: Clinical Chart > Social History

Encounters (list)
└─ any row ────────────────────────────→ Nav: Encounters > detail[encId] (same view, adds encounterId)

Encounters (detail)
├─ back link ───────────────────────────→ Nav: Encounters (list)
├─ section chip ─────────────────────────→ filter in place (no nav)
├─ lab row w/ trend ──────────────────────→ Panel: labTrend[key]
├─ lab "ordering enc" link cell ─────────→ Nav: Encounters > detail[otherEncId]
├─ procedure/imaging row ────────────────→ Panel: report[key]
└─ (dx / notes / careplan / discharge / billing rows — display only)

Clinical Chart > Conditions
├─ condition row (5 wired) ──────────────→ Panel: condition[key]
└─ "First dx" link cell ─────────────────→ Nav: Encounters > detail[encId]

Clinical Chart > Medications
└─ medication row (5 wired) ─────────────→ Panel: med[key]

Clinical Chart > Labs
├─ lab row w/ trend ──────────────────────→ Panel: labTrend[key]
└─ "Ordering enc" link cell ─────────────→ Nav: Encounters > detail[encId]

Clinical Chart > Imaging
└─ any row ───────────────────────────────→ Panel: report[key]

Panel: condition[x]
├─ associated lab row ───────────────────→ Panel: labTrend[key] (replaces current panel)
└─ originating encounter row ────────────→ Nav: Encounters > detail[encId] (closes panel)

Panel: med[x]
└─ ordering encounter row ───────────────→ Nav: Encounters > detail[encId] (closes panel)
```

---

## 6. Data entities checklist (for schema design)

When implementing with real data, model these as distinct entities with stable IDs, each cross-referenced as shown above:

- **Patient** — demographics, contact, insurance, emergency contact, allergies[], alerts[].
- **Condition** (problem list item) — id/key, name, ICD-10 code, category (chronic/acute/historical), onset date, resolved date, status, physician, originating encounter ref, associated medication refs[], associated lab refs[].
- **Medication** — id/key, name, RxNorm code, dose, route, frequency, prescriber, start/end dates, status (active/on hold/discontinued/renewal-due), indication, ordering encounter ref.
- **Allergy** — allergen, reaction, severity (severe/moderate/mild), recorded date/by.
- **Lab result** — test name, value, unit, reference range, flag (normal/high/low/critical), panel group (CBC/metabolic/lipid/endocrine/cardiac), collected date, ordering encounter ref, trend key (groups repeat readings of the same test over time into one trend series).
- **Vital sign reading** — type (BP/HR/Temp/RR/SpO2/weight/etc.), value, timestamp, encounter ref (for trend + encounter-embedded vitals tables).
- **Encounter** — id, type (inpatient/outpatient/telehealth/emergency), department, facility, attending + care team, date(s)/LOS, chief complaint, diagnoses[] (with primary/secondary type), vitals[], labs[], procedures[], imaging[], medications ordered[], notes[], care plan, discharge info, billing info.
- **Imaging/procedure study & report** — study name, region, ordered/performed dates, physician, facility, status, full narrative report text.
- **Immunization** — vaccine, CVX code, date, dose #, route, site, lot, manufacturer, status.
- **Clinical note** — type (progress/discharge summary/etc.), author, date, free text.
- **Social history** — smoking, alcohol, substances, occupation, housing, support/access — as a flat kv record, versioned by "last updated" date/author.

Keep every cross-reference (condition→encounter, lab→trend series, med→encounter, encounter→lab/procedure) as an explicit foreign key so the click-through map above can be driven generically instead of hardcoded per record.
