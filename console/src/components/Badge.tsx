import type { BadgeKind } from '../types'

const BADGE_MAP: Record<BadgeKind, [string, string]> = {
  normal:       ['#E9F2EC', '#1F7A4D'],
  high:         ['#FBF0DA', '#8A5A06'],
  low:          ['#E6EDFB', '#264F9E'],
  critical:     ['#FBE6E2', '#B23B2E'],
  active:       ['#E9F2EC', '#1F7A4D'],
  chronic:      ['#EDEBE5', '#5B584F'],
  resolved:     ['#EDEBE5', '#8A867C'],
  onhold:       ['#FBF0DA', '#8A5A06'],
  discontinued: ['#EDEBE5', '#8A867C'],
  completed:    ['#EDEBE5', '#5B584F'],
  severe:       ['#FBE6E2', '#B23B2E'],
  moderate:     ['#FBF0DA', '#8A5A06'],
  mild:         ['#E6EDFB', '#264F9E'],
  renewal:      ['#FBF0DA', '#8A5A06'],
  primary:      ['#1C1A17', '#fff'],
  neutral:      ['#EDEBE5', '#5B584F'],
  acute:        ['#F3E9DF', '#8A5A06'],
  historical:   ['#EDEBE5', '#8A867C'],
}

interface Props { text: string; kind: BadgeKind }

export default function Badge({ text, kind }: Props) {
  const [bg, fg] = BADGE_MAP[kind] ?? BADGE_MAP.neutral
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center',
      background: bg, color: fg,
      font: "600 10.5px/1 'Hanken Grotesk',sans-serif",
      letterSpacing: '.02em', padding: '4px 8px',
      borderRadius: '999px', whiteSpace: 'nowrap',
    }}>
      {text}
    </span>
  )
}
