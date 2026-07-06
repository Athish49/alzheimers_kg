import type { ReactNode } from 'react'
import type { Block, KvItem, KvGroup, ChartTab } from '../types'
import type { BadgeKind } from '../types'
import { tc, mc, ec, row, lrow, b } from './builders'
import {
  fmtDate, fmtDateShort, fmtDateTime, fmtYear, fmtResult,
  visitTypeLabel, labFlagKind, labFlagLabel,
  conditionCategoryKind, conditionStatusKind, medStatusKind,
  imagingStatusLabel, imagingStatusKind,
  allergyKind, refRange, or, celsiusToF, kgToLbs, cmToFtIn,
  titleCase,
} from '../api/format'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Rec = Record<string, any>

// ── Encounter list ────────────────────────────────────────────────────────────

export function adaptEncounterList(encounters: Rec[], openEncounter: (id: string) => void): Block[] {
  const rows = encounters.map(enc => {
    const vt = visitTypeLabel(enc.visit_type)
    const vtKind: BadgeKind = enc.visit_type?.toLowerCase() === 'emergency' ? 'critical' : 'neutral'
    const statusLabel = enc.status === 'finished' ? 'Completed' : enc.status === 'in-progress' ? 'In progress' : titleCase(enc.status)
    const statusKind: BadgeKind = enc.status === 'finished' ? 'completed' : enc.status === 'in-progress' ? 'active' : 'neutral'
    return row(
      [
        mc(fmtDate(enc.encounter_date)),
        ec(b(vt, vtKind)),
        tc(or(enc.department)),
        tc(or(enc.facility)),
        tc(or(enc.attending_physician)),
        tc(or(enc.chief_complaint)),
        tc(or(enc.primary_diagnosis)),
        ec(b(statusLabel, statusKind)),
      ],
      () => openEncounter(enc.encounter_id),
    )
  })
  return [
    { isTitle: true, title: 'Encounters' },
    {
      isTable: true, hasFilters: false,
      columns: ['Date', 'Type', 'Department', 'Facility', 'Attending', 'Chief complaint', 'Primary dx', 'Status'],
      rows,
    },
  ]
}

// ── Encounter detail ──────────────────────────────────────────────────────────

export function adaptEncounterDetail(detail: Rec, openPanel: (p: unknown) => void): Block[] {
  const vt = visitTypeLabel(detail.visit_type)
  const subtitle = [fmtDate(detail.encounter_date), or(detail.facility)].filter(s => s !== '—').join(' · ')
  const blocks: Block[] = []

  blocks.push({ isTitle: true, title: `${vt} · ${or(detail.department)}`, subtitle })

  // Encounter header
  blocks.push({ isSection: true, title: 'Encounter header', anchor: 'header', updated: or(detail.encounter_id, '').slice(0, 8) })
  const visitItems: KvItem[] = [
    { label: 'Visit type', value: vt },
    { label: 'Department', value: or(detail.department) },
    { label: 'Facility', value: or(detail.facility) },
    { label: 'Chief complaint', value: or(detail.chief_complaint) },
    { label: 'Attending', value: or(detail.attending_physician) },
  ]
  if (detail.admission_datetime) {
    visitItems.push({ label: 'Admitted', value: fmtDateTime(detail.admission_datetime) })
    visitItems.push({ label: 'Discharged', value: detail.discharge_datetime ? fmtDateTime(detail.discharge_datetime) : 'Ongoing' })
    if (detail.length_of_stay_hours) {
      const h = Number(detail.length_of_stay_hours)
      visitItems.push({ label: 'Length of stay', value: h >= 24 ? `${Math.floor(h / 24)}d ${h % 24}h` : `${h}h` })
    }
  }
  const headerGroups: KvGroup[] = [{ title: 'Visit', items: visitItems }]

  // Care team
  if (Array.isArray(detail.care_team) && detail.care_team.length > 0) {
    const ctItems: KvItem[] = (detail.care_team as Rec[]).map(m => ({ label: or(m.role, 'Team member'), value: or(m.name) }))
    headerGroups.push({ title: 'Care team', items: ctItems })
  }

  if (detail.payer_name || detail.claim_status) {
    const billItems: KvItem[] = [
      { label: 'Payer', value: or(detail.payer_name) },
      { label: 'Claim status', value: or(detail.claim_status) },
    ]
    if (detail.drg_code) billItems.push({ label: 'DRG code', value: String(detail.drg_code) })
    if (detail.total_charges) billItems.push({ label: 'Total charges', value: `$${Number(detail.total_charges).toLocaleString('en-US', { minimumFractionDigits: 2 })}` })
    headerGroups.push({ title: 'Billing', items: billItems })
  }

  blocks.push({ isKvBlock: true, groups: headerGroups })

  // Diagnoses
  if (Array.isArray(detail.diagnoses) && detail.diagnoses.length > 0) {
    blocks.push({ isSection: true, title: 'Diagnoses', anchor: 'dx' })
    const dxRows = (detail.diagnoses as Rec[]).map(dx => {
      const typeBadge: BadgeKind = dx.is_primary ? 'primary' : 'neutral'
      const typeLabel = dx.is_primary ? 'Primary' : 'Secondary'
      const newBadge: BadgeKind = dx.is_new_this_visit ? 'acute' : 'neutral'
      const newLabel = dx.is_new_this_visit ? 'New' : 'Existing'
      return row([
        tc(or(dx.condition_name)),
        mc(or(dx.icd10_code)),
        ec(b(typeLabel, typeBadge)),
        ec(b(newLabel, newBadge)),
        ec(b('Active', 'active')),
        tc(or(dx.confirmed_by)),
      ])
    })
    blocks.push({ isTable: true, columns: ['Diagnosis', 'ICD-10', 'Type', 'Onset', 'Status', 'Confirmed by'], rows: dxRows })
  }

  // Vitals
  if (Array.isArray(detail.vitals) && detail.vitals.length > 0) {
    blocks.push({ isSection: true, title: 'Vitals during visit', anchor: 'vitals' })
    const vitalRows = (detail.vitals as Rec[]).map(v =>
      row([
        mc(fmtDateTime(v.recorded_at)),
        tc(v.systolic_bp && v.diastolic_bp ? `${v.systolic_bp}/${v.diastolic_bp}` : '—'),
        tc(or(v.heart_rate != null ? String(v.heart_rate) : null)),
        tc(celsiusToF(v.temperature_c) === '—' ? '—' : `${celsiusToF(v.temperature_c)}°F`),
        tc(or(v.respiratory_rate != null ? String(v.respiratory_rate) : null)),
        tc(v.spo2_pct != null ? `${v.spo2_pct}%` : '—'),
        tc(kgToLbs(v.weight_kg)),
      ]),
    )
    blocks.push({ isTable: true, columns: ['Timestamp', 'BP', 'HR', 'Temp', 'RR', 'SpO₂', 'Weight'], rows: vitalRows })
  }

  // Lab results (grouped)
  if (Array.isArray(detail.labs) && detail.labs.length > 0) {
    blocks.push({ isSection: true, title: 'Lab results', anchor: 'labs' })
    const labCols = ['Test', 'Result', 'Unit', 'Ref range', 'Flag', 'Collected']
    const labRows = (detail.labs as Rec[]).map(lab =>
      row([
        tc(or(lab.test_name)),
        mc(fmtResult(lab.result_value)),
        tc(or(lab.unit)),
        tc(refRange(lab.reference_range_low, lab.reference_range_high)),
        ec(b(labFlagLabel(lab.interpretation_flag), labFlagKind(lab.interpretation_flag))),
        mc(fmtDate(lab.collected_at?.split('T')[0] ?? lab.collected_at)),
      ], () => openPanel({ kind: 'labTrend', key: lab.test_name })),
    )
    blocks.push({ isTable: true, columns: labCols, rows: labRows })
  }

  // Procedures
  if (Array.isArray(detail.procedures) && detail.procedures.length > 0) {
    blocks.push({ isSection: true, title: 'Procedures', anchor: 'proc' })
    const procRows = (detail.procedures as Rec[]).map(p =>
      row([
        tc(or(p.procedure_name)),
        mc(or(p.cpt_code)),
        mc(fmtDateTime(p.performed_at)),
        tc(or(p.performing_physician)),
        ec(b('Completed', 'completed')),
      ]),
    )
    blocks.push({ isTable: true, columns: ['Procedure', 'CPT', 'Performed', 'Physician', 'Status'], rows: procRows })
  }

  // Medications ordered
  if (Array.isArray(detail.medications) && detail.medications.length > 0) {
    blocks.push({ isSection: true, title: 'Medications ordered / administered', anchor: 'meds' })
    const medRows = (detail.medications as Rec[]).map(m => {
      const statusLabel = m.status === 'active' ? 'Administered' : titleCase(m.status) || 'Ordered'
      const statusKind: BadgeKind = m.status === 'active' ? 'neutral' : 'neutral'
      return row([
        tc(or(m.drug_name)),
        tc(or(m.dose)),
        tc(or(m.route)),
        tc(or(m.frequency)),
        ec(b(statusLabel, statusKind)),
      ])
    })
    blocks.push({ isTable: true, columns: ['Medication', 'Dose', 'Route', 'Frequency', 'Disposition'], rows: medRows })
  }

  // Clinical notes
  if (Array.isArray(detail.notes) && detail.notes.length > 0) {
    blocks.push({ isSection: true, title: 'Clinical notes', anchor: 'notes' })
    const noteItems = (detail.notes as Rec[]).map(n => ({
      noteType: or(n.note_type, 'Note'),
      meta: `${or(n.author)} · ${fmtDate(n.authored_at?.split('T')[0] ?? n.authored_at)}`,
      text: or(n.note_text, '(no text)'),
    }))
    blocks.push({ isNotes: true, items: noteItems })
  }

  // Discharge
  if (detail.discharge) {
    const d = detail.discharge as Rec
    blocks.push({ isSection: true, title: 'Discharge', anchor: 'discharge' })
    const dispItems: KvItem[] = [
      { label: 'Disposition', value: or(d.disposition) },
      { label: 'Instructions', value: or(d.instructions_summary) },
    ]
    const discGroups: KvGroup[] = [{ title: 'Disposition', items: dispItems }]
    if (Array.isArray(d.discharge_prescriptions) && d.discharge_prescriptions.length > 0) {
      const rxItems: KvItem[] = (d.discharge_prescriptions as string[]).map((name, i) => ({ label: `Rx ${i + 1}`, value: name }))
      discGroups.push({ title: 'Discharge prescriptions', items: rxItems })
    }
    blocks.push({ isKvBlock: true, groups: discGroups })
  }

  return blocks
}

// ── Snapshot ──────────────────────────────────────────────────────────────────

export function adaptSnapshot(
  resources: Rec,
  setTab: (t: ChartTab) => void,
  openEncounter: (id: string) => void,
  openPanel: (p: unknown) => void,
  _openAllergies: () => void,
  openEncountersFn: () => void,
): Block[] {
  const snap = (resources.snapshot as Rec) ?? {}
  const allMeds = ((resources.medications as Rec)?.medications as Rec[]) ?? []
  const allConds = (resources.conditions as Rec[]) ?? []
  const allLabs = (resources.labs as Rec[]) ?? []

  // ── Active Problems
  const activeProblems = {
    title: 'Active Problems',
    updated: '',
    isList: true,
    hasAction: true,
    actionLabel: 'View all conditions',
    onAction: () => setTab('conditions'),
    rows: ((snap.active_problems as Rec[]) ?? []).map(p => {
      const full = allConds.find(c => c.icd10_code === p.icd10_code || c.condition_name === p.condition_name)
      return lrow({
        primary: or(p.condition_name),
        secondary: p.onset_date ? `Onset ${fmtYear(p.onset_date)}` : '',
        meta: or(p.icd10_code),
        badge: b('Active', 'active') as ReactNode,
        onClick: full ? () => openPanel({ kind: 'condition', key: full.condition_id }) : null,
      })
    }),
  }

  // ── Current Medications
  const currentMeds = {
    title: 'Current Medications',
    updated: '',
    isList: true,
    hasAction: true,
    actionLabel: 'View full medication history',
    onAction: () => setTab('medications'),
    rows: ((snap.active_medications as Rec[]) ?? []).map(m => {
      const full = allMeds.find(med => med.drug_name === m.drug_name)
      const parts = [m.dose, m.route, m.frequency].filter((x): x is string => !!(x?.trim()))
      return lrow({
        primary: or(m.drug_name),
        secondary: parts.join(' · '),
        onClick: full ? () => openPanel({ kind: 'med', key: full.medication_id }) : null,
      })
    }),
  }

  // ── Latest Vitals
  const lv = snap.latest_vitals as Rec | null
  const vitalsItems: KvItem[] = lv ? [
    { label: 'Blood pressure', value: lv.systolic_bp && lv.diastolic_bp ? `${lv.systolic_bp}/${lv.diastolic_bp}` : '—', sub: 'mmHg' },
    { label: 'Heart rate', value: lv.heart_rate != null ? String(lv.heart_rate) : '—', sub: 'bpm' },
    { label: 'Temperature', value: celsiusToF(lv.temperature_c), sub: '°F' },
    { label: 'Respiratory rate', value: lv.respiratory_rate != null ? String(lv.respiratory_rate) : '—', sub: '/min' },
    { label: 'SpO₂', value: lv.spo2_pct != null ? String(lv.spo2_pct) : '—', sub: '%' },
    { label: 'Height / Weight', value: lv.height_cm || lv.weight_kg ? `${cmToFtIn(lv.height_cm)} / ${kgToLbs(lv.weight_kg).replace(' lb', '')}` : '—', sub: 'lb' },
    { label: 'BMI', value: lv.bmi != null ? String(Number(lv.bmi).toFixed(1)) : '—', sub: 'kg/m²' },
  ] : [{ label: 'Vitals', value: 'No readings on record', sub: '' }]

  const latestVitals = {
    title: 'Latest Vitals',
    updated: lv ? fmtDate(lv.recorded_at?.split('T')[0] ?? lv.recorded_at).split(', ')[0] : '',
    isKv: true,
    hasAction: true,
    actionLabel: 'View vitals trend',
    onAction: () => openPanel({ kind: 'vitalsTrend' }),
    items: vitalsItems,
  }

  // ── Recent Encounters
  const recentEncounters = {
    title: 'Recent Encounters',
    updated: '',
    isList: true,
    hasAction: true,
    actionLabel: 'View all encounters',
    onAction: openEncountersFn,
    rows: ((snap.recent_encounters as Rec[]) ?? []).map(enc =>
      lrow({
        primary: `${visitTypeLabel(enc.visit_type)} · ${or(enc.department)}`,
        secondary: or(enc.chief_complaint ?? enc.primary_diagnosis),
        meta: fmtDateShort(enc.encounter_date),
        onClick: () => openEncounter(enc.encounter_id),
      }),
    ),
  }

  // ── Recent Labs
  const recentLabs = {
    title: 'Recent & Key Labs',
    updated: '',
    isList: true,
    hasAction: true,
    actionLabel: 'View all labs',
    onAction: () => setTab('labs'),
    rows: ((snap.recent_labs as Rec[]) ?? []).map(lab => {
      const flag = labFlagKind(lab.interpretation_flag)
      const unit = lab.unit?.trim() ?? ''
      const secondary = fmtResult(lab.result_value) + (unit ? ` ${unit}` : '')
      const matchLab = allLabs.find(l => l.test_name === lab.test_name)
      return lrow({
        primary: or(lab.test_name),
        secondary,
        badge: b(labFlagLabel(lab.interpretation_flag), flag) as ReactNode,
        onClick: matchLab ? () => openPanel({ kind: 'labTrend', key: lab.test_name }) : null,
      })
    }),
  }

  // ── Social History
  const social = resources.social_history as Rec | null
  const socialItems: KvItem[] = [
    { label: 'Smoking', value: or(social?.smoking_status), sub: social?.quit_date ? `quit ${fmtYear(social.quit_date)}` : social?.pack_years ? `${social.pack_years} pack-yrs` : '' },
    { label: 'Alcohol', value: or(social?.alcohol_frequency), sub: social?.alcohol_drinks_per_week != null ? `~${social.alcohol_drinks_per_week}/wk` : '' },
    { label: 'Occupation', value: or(social?.occupation) },
    { label: 'Housing', value: or(social?.housing_status) },
  ]

  const socialHistory = {
    title: 'Social History',
    updated: '',
    isKv: true,
    hasAction: true,
    actionLabel: 'View full social history',
    onAction: () => setTab('social'),
    items: socialItems,
  }

  // ── Immunizations
  const immStatus = snap.immunization_status as Rec | null
  const immCompleted = immStatus?.completed ?? 0
  const immTotal = (immStatus?.completed ?? 0) + (immStatus?.not_done ?? 0)
  const immItems: KvItem[] = [
    { label: 'Overall', value: immTotal > 0 ? `${immCompleted} of ${immTotal} completed` : 'No records', sub: '' },
  ]
  const allImm = (resources.immunizations as Rec[]) ?? []
  const latestImm = allImm.slice(0, 3)
  latestImm.forEach(i => {
    if (i.vaccine_name) {
      const name = String(i.vaccine_name).split(/\s+/).slice(0, 3).join(' ')
      immItems.push({ label: name, value: fmtDate(i.date_administered), sub: '' })
    }
  })

  const immunizations = {
    title: 'Immunization Status',
    updated: '',
    isKv: true,
    hasAction: true,
    actionLabel: 'View all immunizations',
    onAction: () => setTab('immunizations'),
    items: immItems,
  }

  return [
    { isTitle: true, title: 'Patient Snapshot' },
    {
      isCards: true,
      columns: [
        { cards: [activeProblems, recentLabs] },
        { cards: [currentMeds, recentEncounters] },
        { cards: [socialHistory, immunizations, latestVitals] },
      ],
    },
  ]
}

// ── Conditions ────────────────────────────────────────────────────────────────

export function adaptConditions(
  conditions: Rec[],
  encounters: Rec[],
  openEncounter: (id: string) => void,
  openPanel: (p: unknown) => void,
): Block[] {
  const encById = Object.fromEntries(encounters.map(e => [e.encounter_id, e]))

  const rows = conditions.map(cond => {
    const catKind = conditionCategoryKind(cond.category)
    const stKind = conditionStatusKind(cond.status)
    const stLabel = cond.status === 'active' ? 'Active' : cond.status === 'resolved' ? 'Resolved' : titleCase(cond.status)
    const catLabel = cond.category ? titleCase(cond.category) : '—'

    let firstDxCell
    if (cond.first_encounter_id && encById[cond.first_encounter_id]) {
      const enc = encById[cond.first_encounter_id]
      const encDate = fmtDate(enc.encounter_date)
      firstDxCell = ec(
        <button
          onClick={(e) => { e.stopPropagation(); openEncounter(cond.first_encounter_id) }}
          style={{ background: 'none', border: 'none', padding: 0, font: "600 12px/1.4 'JetBrains Mono',monospace", color: '#2860D8', cursor: 'pointer', textAlign: 'left' }}
        >
          {encDate}
        </button> as ReactNode,
      )
    } else {
      firstDxCell = tc('—')
    }

    return row(
      [
        tc(or(cond.condition_name)),
        mc(or(cond.icd10_code)),
        ec(b(catLabel, catKind)),
        mc(fmtYear(cond.onset_date)),
        tc(cond.resolution_date ? fmtYear(cond.resolution_date) : '—'),
        ec(b(stLabel, stKind)),
        firstDxCell,
        tc(or(cond.treating_physician)),
      ],
      () => openPanel({ kind: 'condition', key: cond.condition_id }),
    )
  })

  return [
    { isTitle: true, title: 'Problem List' },
    {
      isTable: true, hasFilters: true, filters: ['Status: All', 'Category: All'],
      columns: ['Condition', 'Code', 'Category', 'Onset', 'Resolved', 'Status', 'First dx', 'Physician'],
      rows,
    },
  ]
}

// ── Medications ───────────────────────────────────────────────────────────────

export function adaptMedications(
  medsData: { medications: Rec[]; allergies: Rec[] } | null,
  openPanel: (p: unknown) => void,
): Block[] {
  const meds = medsData?.medications ?? []
  const allergies = medsData?.allergies ?? []

  const medRows = meds.map(m => {
    const stLabel = m.status === 'active' ? 'Active' : m.status === 'stopped' ? 'Discontinued' : titleCase(m.status) || '—'
    const stKind = medStatusKind(m.status)
    return row(
      [
        tc(or(m.drug_name)),
        mc(or(m.rxnorm_code)),
        tc(or(m.dose)),
        tc(or(m.route)),
        tc(or(m.frequency)),
        tc(or(m.prescribing_physician)),
        mc(m.start_date ? fmtDate(m.start_date) : '—'),
        mc(m.end_date ? fmtDate(m.end_date) : '—'),
        ec(b(stLabel, stKind)),
      ],
      () => openPanel({ kind: 'med', key: m.medication_id }),
    )
  })

  const allergyRows = allergies.map(a => {
    const sevKind = allergyKind(a.severity)
    const sevLabel = a.severity ? titleCase(a.severity) : 'Unknown'
    return row([
      tc(or(a.allergen)),
      tc(or(a.reaction_detail)),
      ec(b(sevLabel, sevKind)),
      mc(a.recorded_date ? fmtDate(a.recorded_date) : '—'),
      tc('—'),
    ])
  })

  return [
    { isTitle: true, title: 'Medications' },
    { isSection: true, title: 'Medications', anchor: 'meds' },
    {
      isTable: true, hasFilters: true, filters: ['Status: All'],
      columns: ['Drug', 'RxNorm', 'Dose', 'Route', 'Freq', 'Prescribed by', 'Start', 'End', 'Status'],
      rows: medRows,
    },
    { isSection: true, title: 'Allergies & intolerances', anchor: 'allergies' },
    {
      isTable: true,
      columns: ['Allergen', 'Reaction', 'Severity', 'Recorded', 'Recorded by'],
      rows: allergyRows.length > 0 ? allergyRows : [row([tc('No allergies on record'), tc(''), tc(''), tc(''), tc('')])],
    },
  ]
}

// ── Labs ──────────────────────────────────────────────────────────────────────

export function adaptLabs(
  labs: Rec[],
  encounters: Rec[],
  openEncounter: (id: string) => void,
  openPanel: (p: unknown) => void,
): Block[] {
  const encById = Object.fromEntries(encounters.map(e => [e.encounter_id, e]))
  const cols = ['Test', 'Result', 'Unit', 'Ref range', 'Flag', 'Ordering enc']

  // Group by panel_name; ungrouped go to "Other"
  const grouped = new Map<string, Rec[]>()
  for (const lab of labs) {
    const panel = or(lab.panel_name, 'Other')
    if (!grouped.has(panel)) grouped.set(panel, [])
    grouped.get(panel)!.push(lab)
  }

  const groups = Array.from(grouped.entries()).map(([panelName, panelLabs]) => {
    const rows = panelLabs.map(lab => {
      const enc = lab.encounter_id ? encById[lab.encounter_id] : null
      const encCell = enc
        ? ec(
            <button
              onClick={(e) => { e.stopPropagation(); openEncounter(enc.encounter_id) }}
              style={{ background: 'none', border: 'none', padding: 0, font: "600 12px/1.4 'JetBrains Mono',monospace", color: '#2860D8', cursor: 'pointer', textAlign: 'left' }}
            >
              {fmtDate(enc.encounter_date)}
            </button> as ReactNode,
          )
        : tc('—')

      return row(
        [
          tc(or(lab.test_name)),
          mc(fmtResult(lab.result_value)),
          tc(or(lab.unit)),
          tc(refRange(lab.reference_range_low, lab.reference_range_high)),
          ec(b(labFlagLabel(lab.interpretation_flag), labFlagKind(lab.interpretation_flag))),
          encCell,
        ],
        () => openPanel({ kind: 'labTrend', key: lab.test_name }),
      )
    })
    return { title: panelName, columns: cols, rows }
  })

  return [
    { isTitle: true, title: 'Laboratory Results' },
    { isGrouped: true, groups },
  ]
}

// ── Imaging ───────────────────────────────────────────────────────────────────

export function adaptImaging(imaging: Rec[], openPanel: (p: unknown) => void): Block[] {
  const rows = imaging.map(img => {
    const stLabel = imagingStatusLabel(img.status)
    const stKind = imagingStatusKind(img.status)
    const modality = img.modality ? titleCase(img.modality) : '—'
    const region = [img.body_region, img.laterality].filter((x): x is string => !!(x?.trim())).join(', ') || '—'
    return row(
      [
        tc(modality),
        tc(region),
        mc(img.date_performed ? fmtDate(img.date_performed) : '—'),
        tc(or(img.ordering_physician)),
        tc(or(img.performing_facility)),
        ec(b(stLabel, stKind)),
      ],
      () => openPanel({ kind: 'report', key: img.imaging_study_id }),
    )
  })

  return [
    { isTitle: true, title: 'Imaging' },
    {
      isTable: true, hasFilters: true, filters: ['All modalities', 'All regions'],
      columns: ['Modality', 'Region', 'Performed', 'Ordering MD', 'Facility', 'Status'],
      rows: rows.length > 0 ? rows : [row([tc('No imaging studies on record'), tc(''), tc(''), tc(''), tc(''), tc('')])],
    },
  ]
}

// ── Immunizations ─────────────────────────────────────────────────────────────

export function adaptImmunizations(immunizations: Rec[]): Block[] {
  const completed = immunizations.filter(i => i.status === 'completed').length
  const total = immunizations.length
  const summary = (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, background: '#E9F2EC', border: '1px solid #CFE4D6', borderRadius: 12, padding: '14px 18px', marginBottom: 14 }}>
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#1F7A4D', display: 'inline-block' }} />
      <span style={{ fontWeight: 600, fontSize: 13.5, color: '#1F7A4D' }}>{total > 0 ? `${completed} of ${total} completed` : 'No records'}</span>
    </div>
  ) as ReactNode

  const rows = immunizations.map(i =>
    row([
      tc(or(i.vaccine_name)),
      mc(or(i.cvx_code)),
      mc(i.date_administered ? fmtDate(i.date_administered) : '—'),
      tc(i.dose_number != null ? String(i.dose_number) : '—'),
      tc(or(i.route)),
      tc(or(i.site)),
      mc(or(i.lot_number)),
      tc(or(i.manufacturer)),
      ec(b('Completed', 'completed')),
    ]),
  )

  return [
    { isTitle: true, title: 'Immunizations' },
    {
      isTable: true, summaryEl: summary,
      columns: ['Vaccine', 'CVX', 'Date', 'Dose', 'Route', 'Site', 'Lot', 'Mfr', 'Status'],
      rows: rows.length > 0 ? rows : [row([tc('No immunization records'), tc(''), tc(''), tc(''), tc(''), tc(''), tc(''), tc(''), tc('')])],
    },
  ]
}

// ── Social History ────────────────────────────────────────────────────────────

export function adaptSocialHistory(social: Rec | null): Block[] {
  const subtitle = social?.last_updated_at ? `Last updated ${fmtDate(social.last_updated_at.split('T')[0])}` : ''
  const s = social ?? {}

  const groups: KvGroup[] = [
    {
      title: 'Smoking',
      items: [
        { label: 'Status', value: or(s.smoking_status) },
        { label: 'Pack-years', value: s.pack_years != null ? String(s.pack_years) : '—' },
        { label: 'Quit date', value: s.quit_date ? fmtYear(s.quit_date) : '—' },
      ],
    },
    {
      title: 'Alcohol',
      items: [
        { label: 'Frequency', value: or(s.alcohol_frequency) },
        { label: 'Drinks / week', value: s.alcohol_drinks_per_week != null ? String(s.alcohol_drinks_per_week) : '—' },
      ],
    },
    {
      title: 'Substances & lifestyle',
      items: [
        { label: 'Recreational drugs', value: '—' },
        { label: 'Physical activity', value: '—' },
        { label: 'Diet', value: '—' },
      ],
    },
    {
      title: 'Socioeconomic',
      items: [
        { label: 'Occupation', value: or(s.occupation) },
        { label: 'Employer', value: '—' },
        { label: 'Education', value: or(s.education_level) },
        { label: 'Housing', value: or(s.housing_status) },
      ],
    },
    {
      title: 'Support & access',
      items: [
        { label: 'Financial strain', value: '—' },
        { label: 'Transportation', value: '—' },
        { label: 'Social support', value: '—' },
      ],
    },
  ]

  return [
    { isTitle: true, title: 'Social History & SDOH', subtitle },
    { isKvBlock: true, groups },
  ]
}

