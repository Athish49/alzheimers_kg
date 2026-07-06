const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (isNaN(d.getTime())) return iso
  return `${MON[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()}`
}

export function fmtDateShort(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (isNaN(d.getTime())) return iso
  return `${MON[d.getUTCMonth()]} ${d.getUTCDate()}`
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const normalized = iso.replace(' ', 'T')
  const d = new Date(normalized)
  if (isNaN(d.getTime())) return iso
  const h = d.getUTCHours()
  const min = String(d.getUTCMinutes()).padStart(2, '0')
  const ampm = h >= 12 ? 'PM' : 'AM'
  const h12 = h % 12 || 12
  return `${MON[d.getUTCMonth()]} ${d.getUTCDate()} ${h12}:${min} ${ampm}`
}

export function fmtYear(iso: string | null | undefined): string {
  if (!iso) return '—'
  return iso.substring(0, 4)
}

export function ageFromDob(dob: string | null | undefined): string {
  if (!dob) return '?'
  const today = new Date('2026-07-05')
  const birth = new Date(dob)
  let age = today.getFullYear() - birth.getFullYear()
  const m = today.getMonth() - birth.getMonth()
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--
  return `${age}y`
}

export function celsiusToF(c: number | null | undefined): string {
  if (c == null) return '—'
  return `${((c * 9 / 5) + 32).toFixed(1)}`
}

export function cmToFtIn(cm: number | null | undefined): string {
  if (cm == null) return '—'
  const totalIn = cm / 2.54
  const ft = Math.floor(totalIn / 12)
  const inch = Math.round(totalIn % 12)
  return `${ft}′${inch}″`
}

export function kgToLbs(kg: number | null | undefined): string {
  if (kg == null) return '—'
  return `${Math.round(kg * 2.20462)} lb`
}

export function titleCase(s: string | null | undefined): string {
  if (!s) return '—'
  return s.replace(/\b\w/g, c => c.toUpperCase())
}

export function visitTypeLabel(t: string | null | undefined): string {
  if (!t) return '—'
  const map: Record<string, string> = {
    ambulatory: 'Outpatient',
    outpatient: 'Outpatient',
    inpatient: 'Inpatient',
    emergency: 'Emergency',
    wellness: 'Wellness',
    telehealth: 'Telehealth',
    'urgent care': 'Urgent Care',
    urgentcare: 'Urgent Care',
  }
  return map[t.toLowerCase()] ?? titleCase(t)
}

export function labFlagKind(flag: string | null | undefined): 'normal' | 'high' | 'low' | 'critical' {
  if (!flag) return 'normal'
  const f = flag.toLowerCase()
  if (f === 'critical' || f === 'c') return 'critical'
  if (f === 'high' || f === 'h' || f === 'abnormal' || f === 'a') return 'high'
  if (f === 'low' || f === 'l') return 'low'
  return 'normal'
}

export function labFlagLabel(flag: string | null | undefined): string {
  const k = labFlagKind(flag)
  return { normal: 'Normal', high: 'High', low: 'Low', critical: 'Critical' }[k]
}

export function conditionCategoryKind(cat: string | null | undefined): 'chronic' | 'acute' | 'historical' | 'neutral' {
  if (!cat) return 'neutral'
  switch (cat.toLowerCase()) {
    case 'chronic': return 'chronic'
    case 'acute': return 'acute'
    case 'historical': return 'historical'
    default: return 'neutral'
  }
}

export function conditionStatusKind(status: string | null | undefined): 'active' | 'resolved' | 'neutral' {
  if (!status) return 'neutral'
  switch (status.toLowerCase()) {
    case 'active': return 'active'
    case 'resolved': return 'resolved'
    default: return 'neutral'
  }
}

export function medStatusKind(status: string | null | undefined): 'active' | 'discontinued' | 'onhold' | 'neutral' {
  if (!status) return 'neutral'
  switch (status.toLowerCase()) {
    case 'active': return 'active'
    case 'stopped':
    case 'discontinued': return 'discontinued'
    case 'on-hold':
    case 'on_hold': return 'onhold'
    default: return 'neutral'
  }
}

export function imagingStatusLabel(status: string | null | undefined): string {
  if (!status) return '—'
  const s = status.toLowerCase()
  if (s === 'final' || s === 'completed') return 'Final'
  if (s === 'preliminary') return 'Preliminary'
  if (s === 'registered' || s === 'pending') return 'Pending'
  return titleCase(status)
}

export function imagingStatusKind(status: string | null | undefined): 'completed' | 'neutral' | 'onhold' {
  if (!status) return 'neutral'
  const s = status.toLowerCase()
  if (s === 'final' || s === 'completed') return 'completed'
  if (s === 'registered' || s === 'pending') return 'onhold'
  return 'neutral'
}

export function allergyKind(severity: string | null | undefined): 'severe' | 'moderate' | 'mild' {
  if (!severity) return 'mild'
  const s = severity.toLowerCase()
  if (s === 'severe') return 'severe'
  if (s === 'moderate') return 'moderate'
  return 'mild'
}

export function initials(name: string | null | undefined): string {
  if (!name) return '?'
  const parts = name.trim().split(/\s+/)
  if (parts.length === 1) return parts[0].charAt(0).toUpperCase()
  return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase()
}

export function refRange(low: number | null | undefined, high: number | null | undefined): string {
  if (low == null && high == null) return '—'
  if (low == null) return `< ${high}`
  if (high == null) return `> ${low}`
  return `${low}–${high}`
}

export function or(v: string | null | undefined, fallback = '—'): string {
  return (v != null && v.toString().trim()) ? v.toString() : fallback
}

export function fmtResult(v: number | null | undefined): string {
  if (v == null) return '—'
  return Number.isInteger(v) ? String(v) : v.toPrecision(4).replace(/\.?0+$/, '')
}

export function nowTime(): string {
  const now = new Date()
  const h = now.getHours()
  const m = String(now.getMinutes()).padStart(2, '0')
  const ampm = h >= 12 ? 'PM' : 'AM'
  const h12 = h % 12 || 12
  return `${h12}:${m} ${ampm}`
}
