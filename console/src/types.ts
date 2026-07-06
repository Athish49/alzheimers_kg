import type { ReactNode } from 'react'

export type BadgeKind =
  | 'normal' | 'high' | 'low' | 'critical'
  | 'active' | 'chronic' | 'resolved' | 'onhold' | 'discontinued'
  | 'completed' | 'severe' | 'moderate' | 'mild'
  | 'renewal' | 'primary' | 'neutral' | 'acute' | 'historical'

export type View = 'snapshot' | 'encounters' | 'chart'
export type ChartTab = 'conditions' | 'medications' | 'labs' | 'imaging' | 'immunizations' | 'social'

export type PanelKind =
  | { kind: 'demographics' }
  | { kind: 'vitalsTrend' }
  | { kind: 'labTrend'; key: string }
  | { kind: 'report'; key: string }
  | { kind: 'condition'; key: string }
  | { kind: 'med'; key: string }

// Block types for the block renderer
export interface TitleBlock   { isTitle: true; title: string; subtitle?: string; eyebrow?: string; anchor?: string }
export interface SectionBlock { isSection: true; title: string; anchor?: string; updated?: string }
export interface CardsBlock   { isCards: true; columns: CardColumn[]; anchor?: string }
export interface TableBlock   { isTable: true; columns: string[]; rows: TableRow[]; hasFilters?: boolean; filters?: string[]; summaryEl?: ReactNode; anchor?: string }
export interface GroupedBlock { isGrouped: true; groups: TableGroup[]; anchor?: string }
export interface KvBlock      { isKvBlock: true; groups: KvGroup[]; anchor?: string }
export interface ListBlock    { isList: true; items: ListItem[]; anchor?: string }
export interface NotesBlock   { isNotes: true; items: NoteItem[]; anchor?: string }
export interface TrendBlock   { isTrend: true; title: string; unit: string; readings: TrendReading[]; anchor?: string }
export interface NarrativeBlock { isNarrative: true; text: string; anchor?: string }

export type Block =
  | TitleBlock | SectionBlock | CardsBlock | TableBlock
  | GroupedBlock | KvBlock | ListBlock | NotesBlock | TrendBlock | NarrativeBlock

export interface CardColumn { cards: SnapshotCard[] }

export interface SnapshotCard {
  title: string
  updated?: string
  isList?: boolean
  isKv?: boolean
  rows?: ListItem[]
  items?: KvItem[]
  hasAction?: boolean
  actionLabel?: string
  onAction?: () => void
  footNote?: string
}

export interface ListItem {
  primary: string
  secondary?: string
  meta?: string
  badgeEl?: ReactNode
  onClick?: (() => void) | null
  hoverStyle?: Record<string, string>
}

export interface KvItem { label: string; value: string; sub?: string }
export interface KvGroup { title?: string; items: KvItem[] }

export interface TableRow  { cells: TableCell[]; onClick?: (() => void) | null }
export interface TableGroup { title: string; columns: string[]; rows: TableRow[] }
export interface TableCell { plain?: boolean; mono?: boolean; el?: ReactNode; text?: string }

export interface NoteItem { noteType: string; meta: string; text: string }

export interface TrendReading {
  date: string
  value: string
  pct: string
  barColor: string
  badgeEl: ReactNode
}

export interface PanelData {
  eyebrow: string
  title: string
  w: number
  blocks: Block[]
}

export interface AllergyEntry { name: string; kind: 'severe' | 'moderate' | 'mild' }
