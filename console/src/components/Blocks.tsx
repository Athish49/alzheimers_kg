import { useState } from 'react'
import { C } from '../tokens'
import type {
  Block, SnapshotCard, ListItem, TableRow as TRow,
} from '../types'

// ── Title ─────────────────────────────────────────────────────────────────────
function TitleBlock({ block }: { block: Extract<Block, { isTitle: true }> }) {
  return (
    <div style={{ borderBottom: `1px solid ${C.border}`, paddingBottom: 16 }}>
      {block.eyebrow && (
        <div style={{ font: "600 11px/1 'Hanken Grotesk',sans-serif", letterSpacing: '.12em', textTransform: 'uppercase', color: C.textFaint, marginBottom: 7 }}>
          {block.eyebrow}
        </div>
      )}
      <h1 style={{ fontFamily: "'Hanken Grotesk',sans-serif", fontWeight: 700, fontSize: 22, lineHeight: 1.2, letterSpacing: '-.01em', color: C.text, margin: 0 }}>
        {block.title}
      </h1>
      {block.subtitle && (
        <p style={{ fontSize: 13, lineHeight: 1.5, color: C.textSec, margin: '7px 0 0' }}>{block.subtitle}</p>
      )}
    </div>
  )
}

// ── Section label ─────────────────────────────────────────────────────────────
function SectionBlock({ block }: { block: Extract<Block, { isSection: true }> }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 2 }}>
      <h2 style={{ fontFamily: "'Newsreader',serif", fontWeight: 500, fontSize: 21, color: C.text, margin: 0 }}>
        {block.title}
      </h2>
      {block.updated && (
        <span style={{ font: "500 11.5px/1 'JetBrains Mono',monospace", color: C.textFaint }}>{block.updated}</span>
      )}
    </div>
  )
}

// ── Snapshot card ─────────────────────────────────────────────────────────────
function ListRow({ row }: { row: ListItem }) {
  const [hov, setHov] = useState(false)
  return (
    <div
      onClick={row.onClick ?? undefined}
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12,
        padding: '10px 8px', margin: '0 -8px', borderTop: `1px solid ${C.borderLight}`,
        borderRadius: 8, cursor: row.onClick ? 'pointer' : 'default',
        background: hov && row.onClick ? C.bg : 'transparent', transition: 'background .12s',
      }}
    >
      <div style={{ minWidth: 0 }}>
        <div style={{ fontWeight: row.onClick ? 700 : 600, fontSize: 13, color: C.text, lineHeight: 1.3 }}>{row.primary}</div>
        {row.secondary && <div style={{ fontSize: 12, color: C.textSec, marginTop: 2, lineHeight: 1.35 }}>{row.secondary}</div>}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0, paddingTop: 1 }}>
        {row.meta && <span style={{ font: "500 11px/1 'JetBrains Mono',monospace", color: C.textFaint, whiteSpace: 'nowrap' }}>{row.meta}</span>}
        {row.badgeEl}
      </div>
    </div>
  )
}

function SnapshotCardComp({ card }: { card: SnapshotCard }) {
  const [hov, setHov] = useState(false)
  return (
    <div style={{
      background: C.white, border: `1px solid ${C.border}`, borderRadius: 16,
      padding: '20px 20px 15px',
      boxShadow: '0 1px 2px rgba(28,26,23,.04),0 10px 30px -18px rgba(28,26,23,.14)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
        <h3 style={{ fontWeight: 600, fontSize: 14.5, color: C.text, margin: 0, letterSpacing: '-.01em' }}>{card.title}</h3>
        {card.updated && <span style={{ font: "500 10.5px/1 'JetBrains Mono',monospace", color: C.textFaintest }}>{card.updated}</span>}
      </div>
      {card.isList && card.rows && (
        <div style={{ marginTop: 8 }}>
          {card.rows.map((row, i) => <ListRow key={i} row={row} />)}
        </div>
      )}
      {card.isKv && card.items && (
        <div style={{ marginTop: 8 }}>
          {card.items.map((it, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12, padding: '8px 0', borderTop: `1px solid ${C.borderLight}` }}>
              <span style={{ fontSize: 12.5, color: C.textSec }}>{it.label}</span>
              <span style={{ fontSize: 13, color: C.text, fontWeight: 600, textAlign: 'right' }}>
                {it.value}{it.sub && <span style={{ font: "400 11px/1 'JetBrains Mono',monospace", color: C.textFaintest, fontWeight: 400 }}> {it.sub}</span>}
              </span>
            </div>
          ))}
        </div>
      )}
      {card.footNote && (
        <div style={{ marginTop: 12, padding: '10px 12px', background: '#F5F3EF', borderRadius: 10, fontSize: 12, color: C.textSec }}>{card.footNote}</div>
      )}
      {card.hasAction && card.onAction && (
        <button
          onClick={card.onAction}
          onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
          style={{ marginTop: 14, background: 'none', border: 'none', padding: '4px 0', font: "600 12.5px/1 'Hanken Grotesk',sans-serif", color: hov ? '#1a4fa8' : C.blue, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 5 }}
        >
          {card.actionLabel} →
        </button>
      )}
    </div>
  )
}

function CardsBlock({ block }: { block: Extract<Block, { isCards: true }> }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16, alignItems: 'start' }}>
      {block.columns.map((col, ci) => (
        <div key={ci} style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
          {col.cards.map((card, ci2) => <SnapshotCardComp key={ci2} card={card} />)}
        </div>
      ))}
    </div>
  )
}

// ── Table ─────────────────────────────────────────────────────────────────────
function TableRowComp({ row }: { row: TRow }) {
  const [hov, setHov] = useState(false)
  return (
    <tr
      onClick={row.onClick ?? undefined}
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{ cursor: row.onClick ? 'pointer' : 'default', background: hov && row.onClick ? C.bg : 'transparent', transition: 'background .1s' }}
    >
      {row.cells.map((cell, ci) => (
        <td key={ci} style={{ padding: '13px 16px', borderTop: `1px solid ${C.borderLight}`, fontSize: 13, color: C.text, verticalAlign: 'top', lineHeight: 1.4 }}>
          {cell.el
            ? cell.el
            : cell.mono
              ? <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12, lineHeight: 1.4, fontWeight: row.onClick && ci === 0 ? 700 : 500, color: C.textMid }}>{cell.text}</span>
              : <span style={{ fontWeight: row.onClick && ci === 0 ? 600 : 400 }}>{cell.text}</span>}
        </td>
      ))}
    </tr>
  )
}

function TableBlock({ block }: { block: Extract<Block, { isTable: true }> }) {
  return (
    <div>
      {block.summaryEl && <div style={{ marginBottom: 14 }}>{block.summaryEl}</div>}
      {block.hasFilters && block.filters && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}>
          {block.filters.map((f, i) => (
            <span key={i} style={{ font: "500 12px/1 'Hanken Grotesk',sans-serif", color: C.textSec, background: C.white, border: `1px solid ${C.border}`, borderRadius: '999px', padding: '7px 13px' }}>{f}</span>
          ))}
        </div>
      )}
      <div style={{ overflow: 'hidden', border: `1px solid ${C.border}`, borderRadius: 14, background: C.white, boxShadow: '0 1px 2px rgba(28,26,23,.04)' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 640 }}>
            <thead>
              <tr>
                {block.columns.map((col, i) => (
                  <th key={i} style={{ font: "600 10.5px/1 'Hanken Grotesk',sans-serif", letterSpacing: '.07em', textTransform: 'uppercase', color: C.textFaint, textAlign: 'left', padding: '12px 16px', borderBottom: `1px solid ${C.border}`, background: C.tableBg, whiteSpace: 'nowrap' }}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, ri) => <TableRowComp key={ri} row={row} />)}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ── Grouped tables ────────────────────────────────────────────────────────────
function GroupedBlock({ block }: { block: Extract<Block, { isGrouped: true }> }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {block.groups.map((g, gi) => (
        <div key={gi}>
          <div style={{ font: "600 12px/1 'Hanken Grotesk',sans-serif", letterSpacing: '.06em', textTransform: 'uppercase', color: C.textMuted, marginBottom: 9 }}>{g.title}</div>
          <div style={{ overflow: 'hidden', border: `1px solid ${C.border}`, borderRadius: 14, background: C.white, boxShadow: '0 1px 2px rgba(28,26,23,.04)' }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 560 }}>
                <thead>
                  <tr>
                    {g.columns.map((col, i) => (
                      <th key={i} style={{ font: "600 10.5px/1 'Hanken Grotesk',sans-serif", letterSpacing: '.07em', textTransform: 'uppercase', color: C.textFaint, textAlign: 'left', padding: '11px 16px', borderBottom: `1px solid ${C.border}`, background: C.tableBg, whiteSpace: 'nowrap' }}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {g.rows.map((row, ri) => <TableRowComp key={ri} row={row} />)}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

// ── KV block ─────────────────────────────────────────────────────────────────
function KvBlock({ block }: { block: Extract<Block, { isKvBlock: true }> }) {
  return (
    <div style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 16, padding: '6px 24px 20px', boxShadow: '0 1px 2px rgba(28,26,23,.04)' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(240px,1fr))', gap: '0 40px' }}>
        {block.groups.map((g, gi) => (
          <div key={gi} style={{ paddingTop: 18 }}>
            {g.title && <div style={{ font: "600 11px/1 'Hanken Grotesk',sans-serif", letterSpacing: '.08em', textTransform: 'uppercase', color: C.textFaint, marginBottom: 4 }}>{g.title}</div>}
            {g.items.map((it, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 14, padding: '9px 0', borderTop: `1px solid ${C.borderLight}` }}>
                <span style={{ fontSize: 12.5, color: C.textSec, flexShrink: 0 }}>{it.label}</span>
                <span style={{ fontSize: 13, color: C.text, fontWeight: 500, textAlign: 'right' }}>{it.value}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Simple list ───────────────────────────────────────────────────────────────
function SimpleListItem({ it }: { it: ListItem }) {
  const [hov, setHov] = useState(false)
  return (
    <div
      onClick={it.onClick ?? undefined}
      onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, padding: '12px 18px', borderTop: `1px solid ${C.borderLight}`, margin: '0 -18px', cursor: it.onClick ? 'pointer' : 'default', background: hov && it.onClick ? C.surfaceHover : 'transparent', transition: 'background .1s' }}
    >
      <div>
        <div style={{ fontWeight: it.onClick ? 700 : 600, fontSize: 13, color: C.text }}>{it.primary}</div>
        {it.secondary && <div style={{ fontSize: 12, color: C.textSec, marginTop: 2 }}>{it.secondary}</div>}
      </div>
      {it.badgeEl}
    </div>
  )
}

function SimpleListBlock({ block }: { block: Extract<Block, { isList: true }> }) {
  return (
    <div style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 14, padding: '4px 18px 12px', boxShadow: '0 1px 2px rgba(28,26,23,.04)' }}>
      {block.items.map((it, i) => <SimpleListItem key={i} it={it} />)}
    </div>
  )
}

// ── Notes ─────────────────────────────────────────────────────────────────────
function NotesBlock({ block }: { block: Extract<Block, { isNotes: true }> }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {block.items.map((n, i) => (
        <details key={i} style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 12, overflow: 'hidden', boxShadow: '0 1px 2px rgba(28,26,23,.04)' }}>
          <summary style={{ cursor: 'pointer', listStyle: 'none', padding: '14px 18px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
            <span style={{ fontWeight: 600, fontSize: 13, color: C.text }}>{n.noteType}</span>
            <span style={{ font: "500 11px/1 'JetBrains Mono',monospace", color: C.textFaint }}>{n.meta}</span>
          </summary>
          <div style={{ padding: '14px 18px 18px', fontSize: 13, lineHeight: 1.6, color: C.textMid, whiteSpace: 'pre-wrap', borderTop: `1px solid ${C.borderLight}` }}>
            {n.text}
          </div>
        </details>
      ))}
    </div>
  )
}

// ── Trend ─────────────────────────────────────────────────────────────────────
function TrendBlockComp({ block }: { block: Extract<Block, { isTrend: true }> }) {
  return (
    <div style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 16, padding: '20px 22px', boxShadow: '0 1px 2px rgba(28,26,23,.04)' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 14 }}>
        <h3 style={{ fontWeight: 600, fontSize: 14, color: C.text, margin: 0 }}>{block.title}</h3>
        <span style={{ font: "500 11px/1 'JetBrains Mono',monospace", color: C.textFaint }}>{block.unit}</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
        {block.readings.map((r, i) => (
          <div key={i} style={{ display: 'grid', gridTemplateColumns: '78px 1fr auto', alignItems: 'center', gap: 12 }}>
            <span style={{ font: "500 11px/1 'JetBrains Mono',monospace", color: C.textMuted }}>{r.date}</span>
            <div style={{ height: 8, background: '#F1EFEA', borderRadius: 999, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: r.pct, borderRadius: 999, background: r.barColor }} />
            </div>
            <span style={{ font: "600 12px/1 'JetBrains Mono',monospace", color: C.text, minWidth: 64, textAlign: 'right', display: 'inline-flex', justifyContent: 'flex-end', alignItems: 'center', gap: 6 }}>
              {r.value} {r.badgeEl}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Narrative ─────────────────────────────────────────────────────────────────
function NarrativeBlock({ block }: { block: Extract<Block, { isNarrative: true }> }) {
  return (
    <div style={{ background: C.white, border: `1px solid ${C.border}`, borderRadius: 16, padding: '22px 24px', boxShadow: '0 1px 2px rgba(28,26,23,.04)', fontSize: 13.5, lineHeight: 1.65, color: C.textMid, whiteSpace: 'pre-wrap' }}>
      {block.text}
    </div>
  )
}

// ── Renderer ──────────────────────────────────────────────────────────────────
interface Props { blocks: Block[] }

export default function Blocks({ blocks }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 26 }}>
      {blocks.map((block, i) => (
        <div key={i} id={block.anchor} style={{ scrollMarginTop: 20 }}>
          {'isTitle'    in block && <TitleBlock    block={block} />}
          {'isSection'  in block && <SectionBlock  block={block} />}
          {'isCards'    in block && <CardsBlock     block={block} />}
          {'isTable'    in block && <TableBlock     block={block} />}
          {'isGrouped'  in block && <GroupedBlock   block={block} />}
          {'isKvBlock'  in block && <KvBlock        block={block} />}
          {'isList'     in block && <SimpleListBlock block={block} />}
          {'isNotes'    in block && <NotesBlock     block={block} />}
          {'isTrend'    in block && <TrendBlockComp block={block} />}
          {'isNarrative' in block && <NarrativeBlock block={block} />}
        </div>
      ))}
    </div>
  )
}
