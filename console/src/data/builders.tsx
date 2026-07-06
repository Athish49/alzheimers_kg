import type { ReactNode } from 'react'
import Badge from '../components/Badge'
import type { Block, ListItem, TableRow, TableCell } from '../types'
import type { BadgeKind } from '../types'

// ── Primitive cell helpers ────────────────────────────────────────────────────
export const tc = (t: string | number | null | undefined): TableCell => ({ plain: true, text: t == null ? '' : String(t) })
export const mc = (t: string | number | null | undefined): TableCell => ({ mono: true,  text: t == null ? '' : String(t) })
export const ec = (el: ReactNode): TableCell => ({ el })
export const row = (cells: TableCell[], onClick?: (() => void) | null): TableRow => ({ cells, onClick: onClick ?? null })
export const lrow = (o: { primary: string; secondary?: string; meta?: string; badge?: ReactNode; onClick?: (() => void) | null }): ListItem => ({
  primary: o.primary, secondary: o.secondary ?? '', meta: o.meta ?? '', badgeEl: o.badge ?? null, onClick: o.onClick ?? null,
})

export function b(text: string, kind: BadgeKind): ReactNode {
  return <Badge text={text} kind={kind} />
}

export function encLink(label: string, openEncounter: (id: string) => void, encId: string): ReactNode {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); openEncounter(encId) }}
      style={{ background: 'none', border: 'none', padding: 0, font: "600 12px/1.4 'JetBrains Mono',monospace", color: '#2860D8', cursor: 'pointer', textAlign: 'left' }}
    >
      {label}
    </button>
  )
}

export function buildTrend(title: string, unit: string, readings: [string, number, BadgeKind][]): Extract<Block, { isTrend: true }> {
  const nums = readings.map(r => r[1])
  const min = Math.min(...nums)
  const max = Math.max(...nums)
  const span = (max - min) || 1
  const col: Record<BadgeKind, string> = {
    normal: '#1F7A4D', high: '#C89A2B', low: '#264F9E', critical: '#B23B2E',
    active: '#1F7A4D', chronic: '#5B584F', resolved: '#8A867C', onhold: '#8A5A06', discontinued: '#8A867C',
    completed: '#5B584F', severe: '#B23B2E', moderate: '#8A5A06', mild: '#264F9E',
    renewal: '#8A5A06', primary: '#1C1A17', neutral: '#5B584F', acute: '#8A5A06', historical: '#8A867C',
  }
  const flabel: Partial<Record<BadgeKind, string>> = { high: 'H', low: 'L', critical: 'C' }
  return {
    isTrend: true, title, unit,
    readings: readings.map(r => {
      const pct = Math.round(14 + ((r[1] - min) / span) * 86)
      return {
        date: r[0], value: String(r[1]), pct: `${pct}%`,
        barColor: col[r[2]] ?? '#8A867C',
        badgeEl: r[2] === 'normal' ? null : <Badge text={flabel[r[2]] ?? ''} kind={r[2]} />,
      }
    }),
  }
}
