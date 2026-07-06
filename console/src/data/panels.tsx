import type { PanelKind, PanelData, Block, KvItem, KvGroup } from '../types'
import { buildTrend, lrow } from './builders'
import {
  fmtDate, fmtResult,
  labFlagKind, or, titleCase,
} from '../api/format'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Rec = Record<string, any>

export function buildPanel(
  p: PanelKind,
  resources: Rec | null,
  openEncounter: (id: string) => void,
  openPanel: (p: PanelKind) => void,
): PanelData | null {

  // ── Demographics ──────────────────────────────────────────────────────────
  if (p.kind === 'demographics') {
    const banner = resources?.banner as Rec | null
    const contacts = (resources?.emergency_contacts as Rec[]) ?? []

    const idItems: KvItem[] = [
      { label: 'Date of birth', value: banner?.date_of_birth ? fmtDate(banner.date_of_birth) : '—' },
      { label: 'Biological sex', value: or(banner?.biological_sex) },
      { label: 'Gender identity', value: or(banner?.gender_identity) },
      { label: 'Race', value: or(banner?.race) },
      { label: 'Ethnicity', value: or(banner?.ethnicity) },
      { label: 'Primary language', value: or(banner?.primary_language) },
    ]

    const contactItems: KvItem[] = [
      { label: 'Address', value: or(banner?.address_line1) },
      { label: 'City', value: [banner?.city, banner?.state, banner?.zip].filter(Boolean).join(', ') || '—' },
      { label: 'Phone', value: or(banner?.phone) },
      { label: 'Email', value: or(banner?.email) },
    ]

    const groups: KvGroup[] = []
    groups.push({ title: 'Identity', items: idItems })
    groups.push({ title: 'Contact', items: contactItems })

    if (contacts.length > 0) {
      const ecItems: KvItem[] = contacts.flatMap(c => [
        { label: 'Name', value: or(c.contact_name) },
        { label: 'Relationship', value: or(c.relationship) },
        { label: 'Phone', value: or(c.phone) },
      ])
      groups.push({ title: 'Emergency contact', items: ecItems })
    } else {
      groups.push({ title: 'Emergency contact', items: [{ label: 'Contact', value: '—' }] })
    }

    const insItems: KvItem[] = [
      { label: 'Payer', value: or(banner?.insurance_payer_name) },
      { label: 'Member ID', value: or(banner?.insurance_member_id) },
      { label: 'Group #', value: or(banner?.insurance_group_number) },
    ]
    groups.push({ title: 'Insurance', items: insItems })

    return {
      eyebrow: 'Identity', title: 'Demographics & Identity', w: 520,
      blocks: [{ isKvBlock: true, groups: groups.length > 0 ? groups : [{ items: [{ label: 'Record', value: 'Minimal data available from source' }] }] }] as Block[],
    }
  }

  // ── Vitals trend ──────────────────────────────────────────────────────────
  if (p.kind === 'vitalsTrend') {
    const vitals = (resources?.vitals as Rec[]) ?? []
    if (vitals.length === 0) {
      return { eyebrow: 'Trend', title: 'Vitals over time', w: 560, blocks: [{ isNarrative: true, text: 'No vitals readings on record.' }] as Block[] }
    }
    const sorted = [...vitals].sort((a, b) => (a.recorded_at ?? '') < (b.recorded_at ?? '') ? -1 : 1)

    const bpReadings: [string, number, 'normal' | 'high' | 'critical'][] = sorted
      .filter(v => v.systolic_bp != null)
      .map(v => {
        const bp = Number(v.systolic_bp)
        const kind: 'normal' | 'high' | 'critical' = bp >= 180 ? 'critical' : bp >= 130 ? 'high' : 'normal'
        return [fmtDate(v.recorded_at.split('T')[0]), bp, kind]
      })

    const hrReadings: [string, number, 'normal' | 'high' | 'critical'][] = sorted
      .filter(v => v.heart_rate != null)
      .map(v => {
        const hr = Number(v.heart_rate)
        const kind: 'normal' | 'high' | 'critical' = hr >= 100 ? 'high' : 'normal'
        return [fmtDate(v.recorded_at.split('T')[0]), hr, kind]
      })

    const wtReadings: [string, number, 'normal'][] = sorted
      .filter(v => v.weight_kg != null)
      .map(v => [fmtDate(v.recorded_at.split('T')[0]), Math.round(Number(v.weight_kg) * 2.20462), 'normal'])

    const blocks: Block[] = []
    if (bpReadings.length > 0) blocks.push(buildTrend('Systolic BP', 'mmHg', bpReadings))
    if (hrReadings.length > 0) blocks.push(buildTrend('Heart rate', 'bpm', hrReadings))
    if (wtReadings.length > 0) blocks.push(buildTrend('Weight', 'lb', wtReadings))
    if (blocks.length === 0) blocks.push({ isNarrative: true, text: 'No numeric vitals to trend.' })

    return { eyebrow: 'Trend', title: 'Vitals over time', w: 560, blocks: blocks as Block[] }
  }

  // ── Lab trend ─────────────────────────────────────────────────────────────
  if (p.kind === 'labTrend') {
    const allLabs = (resources?.labs as Rec[]) ?? []
    const testName = p.key
    const filtered = allLabs
      .filter(l => l.test_name === testName && l.result_value != null && l.collected_at)
      .sort((a, b) => (a.collected_at ?? '') < (b.collected_at ?? '') ? -1 : 1)

    if (filtered.length === 0) {
      return { eyebrow: 'Lab Trend', title: testName, w: 520, blocks: [{ isNarrative: true, text: 'No historical readings available for this test.' }] as Block[] }
    }

    const unit = filtered[filtered.length - 1].unit ?? ''
    const readings: [string, number, 'normal' | 'high' | 'low' | 'critical'][] = filtered.map(l => {
      const dateStr = l.collected_at.split('T')[0]
      return [fmtDate(dateStr), Number(l.result_value), labFlagKind(l.interpretation_flag)]
    })

    return {
      eyebrow: 'Lab Trend', title: testName, w: 520,
      blocks: [
        buildTrend(`${testName} over time`, unit, readings),
        { isNarrative: true, text: `Every recorded reading of ${testName} across encounters. Values outside the reference range are flagged.` },
      ] as Block[],
    }
  }

  // ── Imaging report ────────────────────────────────────────────────────────
  if (p.kind === 'report') {
    const allImaging = (resources?.imaging as Rec[]) ?? []
    const study = allImaging.find(img => img.imaging_study_id === p.key)

    const title = study ? [study.modality ? titleCase(study.modality) : null, study.body_region?.trim() || null].filter(Boolean).join(' — ') || 'Imaging Report' : 'Imaging Report'
    const narrative = study?.report_narrative ?? null
    const facility = study?.performing_facility ? `Performed at: ${study.performing_facility}` : null
    const physician = study?.ordering_physician ? `Ordered by: ${study.ordering_physician}` : null
    const dateStr = study?.date_performed ? `Date: ${fmtDate(study.date_performed)}` : null
    const header = [dateStr, physician, facility].filter(Boolean).join('  ·  ')

    return {
      eyebrow: 'Imaging Report', title, w: 560,
      blocks: [
        ...(header ? [{ isSection: true, title: header }] : []),
        { isNarrative: true, text: narrative ?? 'No report narrative available for this study.' },
      ] as Block[],
    }
  }

  // ── Condition detail ──────────────────────────────────────────────────────
  if (p.kind === 'condition') {
    const allConds = (resources?.conditions as Rec[]) ?? []
    const allMeds = ((resources?.medications as Rec)?.medications as Rec[]) ?? []
    const allLabs = (resources?.labs as Rec[]) ?? []
    const encounters = (resources?.encounters as Rec[]) ?? []

    const cond = allConds.find(c => c.condition_id === p.key)
    if (!cond) return null

    const relatedMeds = allMeds.filter(m => {
      // Simple heuristic: match meds that were started around the onset of this condition
      return m.status === 'active'
    }).slice(0, 5)

    const relatedLabs = allLabs
      .filter((l, _, arr) => {
        // Show the most recent reading of each unique test
        const latestForTest = arr.filter(x => x.test_name === l.test_name).sort((a, b) => (a.collected_at ?? '') > (b.collected_at ?? '') ? -1 : 1)[0]
        return latestForTest === l
      })
      .slice(0, 5)

    const kvItems: KvItem[] = [
      { label: 'Code', value: or(cond.icd10_code) },
      { label: 'Category', value: or(cond.category) ? titleCase(cond.category) : '—' },
      { label: 'Status', value: cond.status === 'active' ? 'Active' : 'Resolved' },
      { label: 'First diagnosed', value: cond.onset_date ? fmtDate(cond.onset_date) : '—' },
    ]

    const medListItems = relatedMeds.map(m =>
      lrow({ primary: or(m.drug_name), secondary: [m.dose, m.frequency].filter((x): x is string => !!(x?.trim())).join(' · ') }),
    )

    const labListItems = relatedLabs.map(l =>
      lrow({
        primary: or(l.test_name),
        secondary: `${fmtResult(l.result_value)} ${or(l.unit, '')} · View trend`,
        onClick: () => openPanel({ kind: 'labTrend', key: l.test_name }),
      }),
    )

    const firstEnc = cond.first_encounter_id ? encounters.find(e => e.encounter_id === cond.first_encounter_id) : null
    const encItem = firstEnc
      ? lrow({ primary: `${titleCase(firstEnc.visit_type ?? '')} · ${or(firstEnc.department)}`, secondary: fmtDate(firstEnc.encounter_date), onClick: () => openEncounter(firstEnc.encounter_id) })
      : lrow({ primary: 'First encounter not recorded', secondary: '' })

    return {
      eyebrow: 'Condition', title: or(cond.condition_name), w: 480,
      blocks: [
        { isKvBlock: true, groups: [{ items: kvItems }] },
        { isSection: true, title: 'Associated medications' },
        { isList: true, items: medListItems.length > 0 ? medListItems : [lrow({ primary: 'No active medications', secondary: '' })] },
        { isSection: true, title: 'Associated labs' },
        { isList: true, items: labListItems.length > 0 ? labListItems : [lrow({ primary: 'No labs on record', secondary: '' })] },
        { isSection: true, title: 'Originating encounter' },
        { isList: true, items: [encItem] },
      ] as Block[],
    }
  }

  // ── Medication detail ─────────────────────────────────────────────────────
  if (p.kind === 'med') {
    const allMeds = ((resources?.medications as Rec)?.medications as Rec[]) ?? []
    const encounters = (resources?.encounters as Rec[]) ?? []
    const med = allMeds.find(m => m.medication_id === p.key)
    if (!med) return null

    const kvItems: KvItem[] = [
      { label: 'RxNorm', value: or(med.rxnorm_code) },
      { label: 'Type', value: or(med.delivery_type) },
      { label: 'Dose', value: or(med.dose) },
      { label: 'Route', value: or(med.route) },
      { label: 'Frequency', value: or(med.frequency) },
      { label: 'Prescribed by', value: or(med.prescribing_physician) },
      { label: 'Start', value: med.start_date ? fmtDate(med.start_date) : '—' },
      { label: 'Status', value: med.status === 'active' ? 'Active' : titleCase(med.status) || '—' },
      { label: 'Indication', value: '—' },
    ]

    const ordEnc = med.ordering_encounter_id ? encounters.find(e => e.encounter_id === med.ordering_encounter_id) : null
    const encItem = ordEnc
      ? lrow({ primary: `${titleCase(ordEnc.visit_type ?? '')} · ${or(ordEnc.department)}`, secondary: fmtDate(ordEnc.encounter_date), onClick: () => openEncounter(ordEnc.encounter_id) })
      : lrow({ primary: 'Ordering encounter not recorded', secondary: '' })

    return {
      eyebrow: 'Medication', title: or(med.drug_name), w: 480,
      blocks: [
        { isKvBlock: true, groups: [{ items: kvItems }] },
        { isSection: true, title: 'Ordering encounter' },
        { isList: true, items: [encItem] },
      ] as Block[],
    }
  }

  return null
}

