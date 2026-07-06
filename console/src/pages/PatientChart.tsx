import { useState, useRef, useEffect } from 'react'
import { C } from '../tokens'
import type { View, ChartTab, PanelKind, AllergyEntry } from '../types'
import Blocks from '../components/Blocks'
import {
  adaptSnapshot, adaptEncounterList, adaptEncounterDetail,
  adaptConditions, adaptMedications, adaptLabs, adaptImaging,
  adaptImmunizations, adaptSocialHistory,
} from '../data/adapters'
import { buildPanel } from '../data/panels'
import { useChart } from '../hooks/useChart'
import { useEncounterDetail } from '../hooks/useEncounterDetail'
import { fmtDate, ageFromDob, initials, allergyKind, or, nowTime } from '../api/format'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Rec = Record<string, any>

const ALLERGY_COLOR: Record<AllergyEntry['kind'], [string, string]> = {
  severe:   ['#FBE6E2', '#B23B2E'],
  moderate: ['#FBF0DA', '#8A5A06'],
  mild:     ['#E6EDFB', '#264F9E'],
}

interface Props {
  patientId: string
  onBack: () => void
}

export default function PatientChart({ patientId, onBack }: Props) {
  const chart           = useChart(patientId)
  const [view,          setView]          = useState<View>('snapshot')
  const [chartTab,      setChartTab]      = useState<ChartTab>('conditions')
  const [encounterId,   setEncounterId]   = useState<string | null>(null)
  const [panel,         setPanel]         = useState<PanelKind | null>(null)
  const [chatOpen,      setChatOpen]      = useState(false)
  const [chatMsgs,      setChatMsgs]      = useState<{ role: 'user' | 'assistant'; content: string }[]>([])
  const [chatDraft,     setChatDraft]     = useState('')
  const [chatThink,     setChatThink]     = useState(false)
  const [selSection,    setSelSection]    = useState<string | null>(null)
  const [syncedTime,    setSyncedTime]    = useState<string>('—')

  const encDetail     = useEncounterDetail(patientId, encounterId)
  const scrollerRef   = useRef<HTMLDivElement>(null)
  const chatScrollRef = useRef<HTMLDivElement>(null)

  // Capture sync time once chart loads
  useEffect(() => {
    if (!chart.loading && chart.resources) setSyncedTime(nowTime())
  }, [chart.loading, chart.resources])

  const goSnapshot   = () => { setView('snapshot');   setEncounterId(null) }
  const goEncounters = () => { setView('encounters'); setEncounterId(null) }
  const goChart      = () => { setView('chart');      setEncounterId(null) }

  const openEncounter = (id: string) => {
    setView('encounters'); setEncounterId(id); setPanel(null); setSelSection(null)
    scrollerRef.current?.scrollTo({ top: 0 })
  }

  const backToList = () => { setEncounterId(null); setSelSection(null); scrollerRef.current?.scrollTo({ top: 0 }) }

  const doSetTab = (t: ChartTab) => { setView('chart'); setChartTab(t); setEncounterId(null) }

  const openPanel = (p: unknown) => setPanel(p as PanelKind)
  const openAllergies = () => { doSetTab('medications'); setTimeout(() => jumpTo('allergies'), 80) }

  const jumpTo = (id: string) => {
    const sc = scrollerRef.current
    if (!sc) return
    const el = sc.querySelector<HTMLElement>(`[id="${id}"]`)
    if (el) { const r = el.getBoundingClientRect(), sr = sc.getBoundingClientRect(); sc.scrollTo({ top: sc.scrollTop + (r.top - sr.top) - 20, behavior: 'smooth' }) }
  }

  // ── Derive data from chart ─────────────────────────────────────────────────
  const res = chart.resources as Rec | null
  const banner        = res?.banner as Rec | null
  const alerts        = (res?.alerts as Rec[]) ?? []
  const allergiesRaw  = (res?.allergies as Rec[] | null) ?? ((res?.medications as Rec)?.allergies as Rec[]) ?? []
  const encounters    = (res?.encounters as Rec[]) ?? []
  const conditions    = (res?.conditions as Rec[]) ?? []
  const labs          = (res?.labs as Rec[]) ?? []
  const imaging       = (res?.imaging as Rec[]) ?? []
  const immunizations = (res?.immunizations as Rec[]) ?? []
  const socialHistory = (res?.social_history as Rec | null) ?? null

  const allergies: AllergyEntry[] = allergiesRaw.map(a => ({
    name: or(a.allergen, 'Unknown'),
    kind: allergyKind(a.severity),
  }))

  // ── Build main content ─────────────────────────────────────────────────────
  const isDetail = view === 'encounters' && !!encounterId

  let rawBlocks = (() => {
    if (!res) return []

    if (view === 'snapshot') return adaptSnapshot(res, doSetTab, openEncounter, openPanel, openAllergies, goEncounters)
    if (view === 'encounters') {
      if (isDetail) {
        if (encDetail.loading) return []
        if (encDetail.detail) return adaptEncounterDetail(encDetail.detail, openPanel)
        return []
      }
      return adaptEncounterList(encounters, openEncounter)
    }
    if (chartTab === 'conditions')    return adaptConditions(conditions, encounters, openEncounter, openPanel)
    if (chartTab === 'medications')   return adaptMedications(res.medications as { medications: Rec[]; allergies: Rec[] } | null ?? null, openPanel)
    if (chartTab === 'labs')          return adaptLabs(labs, encounters, openEncounter, openPanel)
    if (chartTab === 'imaging')       return adaptImaging(imaging, openPanel)
    if (chartTab === 'immunizations') return adaptImmunizations(immunizations)
    return adaptSocialHistory(socialHistory)
  })()

  // Section filter for encounter detail
  if (isDetail && selSection) {
    const out = []
    let cur: string | null = null
    for (const blk of rawBlocks) {
      if ('isTitle'   in blk) { out.push(blk); continue }
      if ('isSection' in blk) { cur = blk.title; if (cur === selSection) out.push(blk); continue }
      if (cur === selSection) out.push(blk)
    }
    rawBlocks = out
  }

  const allSections = isDetail && encDetail.detail
    ? adaptEncounterDetail(encDetail.detail, openPanel).filter(b => 'isSection' in b).map(b => ('isSection' in b ? b.title : ''))
    : []

  const panelData = panel ? buildPanel(panel, res, openEncounter, (p) => setPanel(p as PanelKind)) : null

  // ── Banner fields ──────────────────────────────────────────────────────────
  const patientName    = or(banner?.full_name, 'Loading…')
  const preferredName  = banner?.preferred_name ? `(${banner.preferred_name})` : ''
  const patientInitials = initials(banner?.full_name)
  const mrnRaw         = banner?.mrn ?? '—'
  const mrn            = mrnRaw.length > 16 ? mrnRaw.slice(0, 8) : mrnRaw
  const dobAge         = banner?.date_of_birth ? `${fmtDate(banner.date_of_birth)} · ${ageFromDob(banner.date_of_birth)}` : '—'
  const sexGender      = banner?.biological_sex ?? '—'
  const pcp            = or(banner?.primary_care_physician)

  // Nav & tab styles
  const navStyle = (key: View) => ({
    border: 'none', borderRadius: '999px', padding: '8px 17px',
    font: "600 12.5px/1 'Hanken Grotesk',sans-serif", cursor: 'pointer', transition: 'all .15s', whiteSpace: 'nowrap' as const,
    background: view === key ? C.text : 'transparent',
    color:      view === key ? '#fff'  : C.textSec,
    boxShadow:  view === key ? '0 1px 2px rgba(0,0,0,.15)' : '',
  })

  const tabStyle = (key: ChartTab) => {
    const active = chartTab === key && view === 'chart'
    return {
      border: 'none', background: 'none', padding: '14px 4px', marginRight: 20,
      font: "600 13px/1 'Hanken Grotesk',sans-serif", cursor: 'pointer', whiteSpace: 'nowrap' as const,
      borderBottom: `2px solid ${active ? C.text : 'transparent'}`,
      color: active ? C.text : C.textMuted, transition: 'color .12s',
    }
  }

  const chipStyle = (active: boolean) => ({
    background: active ? C.text : C.white, border: `1px solid ${active ? C.text : C.border}`,
    color: active ? '#fff' : '#4A473F', borderRadius: '999px', padding: '7px 13px',
    font: "500 12px/1 'Hanken Grotesk',sans-serif", cursor: 'pointer', transition: 'all .12s',
  })

  // Chat (visual stub — real orchestration not wired in this view)
  useEffect(() => { chatScrollRef.current && (chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight) }, [chatMsgs, chatThink])

  const sendDisabled = !chatDraft.trim() || chatThink
  const sendChat = (text: string) => {
    const t = text.trim()
    if (!t) return
    const msgs = [...chatMsgs, { role: 'user' as const, content: t }]
    setChatMsgs(msgs); setChatDraft(''); setChatThink(true)
    setTimeout(() => {
      setChatMsgs([...msgs, { role: 'assistant' as const, content: "Ask Atlas isn't connected in this environment — this is a visual demo. In a deployed instance with valid LLM credentials, this would call /runtime/orchestrate and return a grounded answer from the patient record." }])
      setChatThink(false)
    }, 800)
  }

  return (
    <div style={{ minHeight: '100vh', background: C.bg, fontFamily: "'Hanken Grotesk',sans-serif", color: C.text, display: 'flex', flexDirection: 'column' }}>

      {/* ── TOP BAR ─────────────────────────────────────────────────────── */}
      <div style={{ position: 'sticky', top: 0, zIndex: 30, background: 'rgba(250,249,247,.86)', backdropFilter: 'saturate(140%) blur(10px)', borderBottom: `1px solid ${C.border}` }}>
        <div style={{ maxWidth: 1180, margin: '0 auto', padding: '0 32px', height: 56, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexShrink: 0 }}>
            <button onClick={onBack} style={{ border: 'none', background: 'none', padding: 0, font: "500 12px/1 'Hanken Grotesk'", color: C.textSec, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5 }}>← Patients</button>
            <div style={{ width: 1, height: 18, background: C.border }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
              <div style={{ width: 20, height: 20, borderRadius: 6, background: C.text, transform: 'rotate(45deg)' }} />
              <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: '-.02em' }}>Atlas<span style={{ color: C.textFaint, fontWeight: 500 }}> Healthcare</span></span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, background: '#F0EEE8', border: `1px solid ${C.border}`, borderRadius: '999px', padding: 4 }}>
            <button onClick={goSnapshot}   style={navStyle('snapshot')}>Snapshot</button>
            <button onClick={goEncounters} style={navStyle('encounters')}>Encounters</button>
            <button onClick={goChart}      style={navStyle('chart')}>Clinical&nbsp;Chart</button>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexShrink: 0 }}>
            <span style={{ font: "500 11.5px/1 'JetBrains Mono',monospace", color: C.textFaintest }}>Synced {syncedTime}</span>
            <AskAtlasBtn onClick={() => setChatOpen(true)} />
          </div>
        </div>
      </div>

      {/* ── PATIENT BANNER ──────────────────────────────────────────────── */}
      <div style={{ position: 'sticky', top: 56, zIndex: 29, background: C.white, borderBottom: `1px solid ${C.border}`, boxShadow: '0 1px 2px rgba(28,26,23,.03)' }}>
        <div style={{ maxWidth: 1180, margin: '0 auto', padding: '14px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 28, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 26, flexWrap: 'wrap' }}>
            <PatientNameBtn
              name={patientName}
              preferredName={preferredName}
              initials={patientInitials}
              mrn={mrn}
              onClick={() => setPanel({ kind: 'demographics' })}
            />
            <div style={{ display: 'flex', gap: 22, flexWrap: 'wrap' }}>
              <MetaField label="DOB / Age"    value={dobAge} />
              <MetaField label="Sex"          value={sexGender} />
              <MetaField label="Primary Care" value={pcp} />
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
            <AllergyChips allergies={allergies} onOpen={openAllergies} />
            {alerts.length > 0 && (
              <>
                <div style={{ width: 1, height: 34, background: C.border }} />
                <div>
                  <div style={{ font: "600 9.5px/1 'Hanken Grotesk'", letterSpacing: '.1em', textTransform: 'uppercase', color: C.textFaintest, marginBottom: 6 }}>Alerts</div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {alerts.slice(0, 2).map((al: Rec, i: number) => (
                      <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, background: C.coralLight, color: C.coral, font: "600 11px/1 'Hanken Grotesk'", padding: '5px 10px', borderRadius: '999px' }}>
                        ▲ {or(al.alert_text, 'Alert')}
                      </span>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* ── CHART TAB BAR ───────────────────────────────────────────────── */}
      {view === 'chart' && (
        <div style={{ position: 'sticky', top: 129, zIndex: 28, background: 'rgba(250,249,247,.9)', backdropFilter: 'blur(8px)', borderBottom: `1px solid ${C.border}` }}>
          <div style={{ maxWidth: 1180, margin: '0 auto', padding: '0 32px', display: 'flex', gap: 4, overflowX: 'auto' }}>
            {(['conditions','medications','labs','imaging','immunizations','social'] as ChartTab[]).map(k => (
              <button key={k} onClick={() => doSetTab(k)} style={tabStyle(k)}>
                {{ conditions:'Conditions', medications:'Medications', labs:'Labs', imaging:'Imaging', immunizations:'Immunizations', social:'Social History' }[k]}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── MAIN SCROLLER ───────────────────────────────────────────────── */}
      <div ref={scrollerRef} style={{ flex: 1, overflowY: 'auto' }}>
        <div style={{ maxWidth: 1180, margin: '0 auto', padding: '34px 32px 90px' }}>

          {/* Loading state */}
          {chart.loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, color: C.textSec, fontSize: 13.5, padding: '60px 0' }}>
              <div style={{ width: 20, height: 20, borderRadius: '50%', border: `2px solid ${C.border}`, borderTopColor: C.text, animation: 'spin 0.8s linear infinite' }} />
              Loading patient record…
            </div>
          )}

          {/* Error state */}
          {chart.error && (
            <div style={{ background: '#FBE6E2', border: '1px solid #E8B4AE', borderRadius: 12, padding: '20px 24px', color: '#8A2020', maxWidth: 480 }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Failed to load chart</div>
              <div style={{ fontSize: 13 }}>{chart.error}</div>
            </div>
          )}

          {/* Encounter detail loading */}
          {isDetail && encDetail.loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, color: C.textSec, fontSize: 13.5, padding: '60px 0' }}>
              <div style={{ width: 20, height: 20, borderRadius: '50%', border: `2px solid ${C.border}`, borderTopColor: C.text, animation: 'spin 0.8s linear infinite' }} />
              Loading encounter…
            </div>
          )}

          {!chart.loading && !chart.error && (
            <>
              {isDetail && !encDetail.loading && (
                <div style={{ marginBottom: 24 }}>
                  <BackBtn onClick={backToList} />
                  {allSections.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7, marginTop: 18 }}>
                      <button onClick={() => setSelSection(null)} style={chipStyle(!selSection)}>All</button>
                      {allSections.map(t => (
                        <button key={t} onClick={() => setSelSection(s => s === t ? null : t)} style={chipStyle(selSection === t)}>{t}</button>
                      ))}
                    </div>
                  )}
                </div>
              )}
              {!(isDetail && encDetail.loading) && <Blocks blocks={rawBlocks} />}
            </>
          )}
        </div>
      </div>

      {/* ── SLIDE-OVER PANEL ────────────────────────────────────────────── */}
      {panel && panelData && (
        <>
          <div onClick={() => setPanel(null)} style={{ position: 'fixed', inset: 0, background: 'rgba(28,26,23,.28)', zIndex: 40, animation: 'fadeIn .18s ease' }} />
          <div style={{ position: 'fixed', top: 0, right: 0, bottom: 0, width: panelData.w, maxWidth: '94vw', background: C.bg, zIndex: 41, boxShadow: '-24px 0 70px -30px rgba(0,0,0,.45)', display: 'flex', flexDirection: 'column', animation: 'slideIn .22s cubic-bezier(.2,.7,.3,1)' }}>
            <div style={{ flexShrink: 0, padding: '20px 26px', borderBottom: `1px solid ${C.border}`, background: C.white, display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}>
              <div>
                <div style={{ font: "600 11px/1 'Hanken Grotesk'", letterSpacing: '.13em', textTransform: 'uppercase', color: C.textFaint }}>{panelData.eyebrow}</div>
                <h2 style={{ fontFamily: "'Newsreader',serif", fontWeight: 500, fontSize: 23, margin: '8px 0 0', letterSpacing: '-.01em' }}>{panelData.title}</h2>
              </div>
              <CloseBtn onClick={() => setPanel(null)} />
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '24px 26px 60px' }}>
              <Blocks blocks={panelData.blocks} />
            </div>
          </div>
        </>
      )}

      {/* ── ASK ATLAS CHAT DRAWER (visual stub) ─────────────────────────── */}
      {chatOpen && (
        <div style={{ position: 'fixed', top: 0, right: 0, bottom: 0, width: 400, maxWidth: '94vw', background: C.bg, zIndex: 45, boxShadow: '-24px 0 70px -30px rgba(0,0,0,.45)', display: 'flex', flexDirection: 'column', animation: 'slideIn .22s cubic-bezier(.2,.7,.3,1)', borderLeft: `1px solid ${C.border}` }}>
          <div style={{ flexShrink: 0, padding: '16px 20px', borderBottom: `1px solid ${C.border}`, background: C.white, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 30, height: 30, borderRadius: 8, background: C.text, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: C.blue, boxShadow: '0 0 0 3px rgba(40,96,216,.25)', display: 'inline-block' }} />
              </div>
              <div>
                <div style={{ fontWeight: 700, fontSize: 14, letterSpacing: '-.01em' }}>Ask Atlas</div>
                <div style={{ font: "500 10.5px/1 'JetBrains Mono',monospace", color: C.textFaint, marginTop: 2 }}>{patientName} · MRN {mrn}</div>
              </div>
            </div>
            <CloseBtn onClick={() => setChatOpen(false)} size={30} />
          </div>

          <div ref={chatScrollRef} style={{ flex: 1, overflowY: 'auto', padding: 20, display: 'flex', flexDirection: 'column', gap: 14 }}>
            {chatMsgs.length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div style={{ background: '#FBF0DA', border: '1px solid #E8D09A', borderRadius: 10, padding: '10px 14px', fontSize: 12, color: '#6B4B08' }}>
                  <strong>Demo mode</strong> — Ask Atlas is not connected to an LLM in this environment. Replies show what the system would return when /runtime/orchestrate is wired up.
                </div>
                <p style={{ fontSize: 13, lineHeight: 1.55, color: C.textSec, margin: 0 }}>Ask about this patient's chart — conditions, medications, labs, recent encounters, or anything in the record.</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {["Summarize this patient in three lines.", "What changed at the last encounter?", "Any medication conflicts with recorded allergies?"].map((t, i) => (
                    <SuggestionBtn key={i} text={t} onClick={() => sendChat(t)} />
                  ))}
                </div>
              </div>
            )}
            {chatMsgs.map((m, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
                <div style={m.role === 'user'
                  ? { maxWidth: '85%', background: C.text, color: '#fff',   font: "400 13px/1.5 'Hanken Grotesk',sans-serif",  padding: '10px 14px', borderRadius: '14px 14px 4px 14px',  whiteSpace: 'pre-wrap' }
                  : { maxWidth: '88%', background: C.white,color: C.text,   font: "400 13px/1.55 'Hanken Grotesk',sans-serif", padding: '11px 14px', borderRadius: '14px 14px 14px 4px', whiteSpace: 'pre-wrap', border: `1px solid ${C.border}` }}>
                  {m.content}
                </div>
              </div>
            ))}
            {chatThink && (
              <div style={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: 6, padding: '11px 14px', background: C.white, border: `1px solid ${C.border}`, borderRadius: '14px 14px 14px 4px' }}>
                {[0, 0.2, 0.4].map((d, i) => <span key={i} style={{ width: 6, height: 6, borderRadius: '50%', background: C.textFaintest, display: 'inline-block', animation: `blink 1s infinite ${d}s` }} />)}
              </div>
            )}
          </div>

          <div style={{ flexShrink: 0, padding: '14px 16px 16px', borderTop: `1px solid ${C.border}`, background: C.white }}>
            <form onSubmit={e => { e.preventDefault(); sendChat(chatDraft) }} style={{ display: 'flex', alignItems: 'flex-end', gap: 8, background: '#F7F5F1', border: `1px solid ${C.border}`, borderRadius: 14, padding: '8px 8px 8px 14px' }}>
              <input value={chatDraft} onChange={e => setChatDraft(e.target.value)} placeholder={`Ask about ${banner?.preferred_name ?? patientName.split(' ')[0]}'s chart…`} style={{ flex: 1, border: 'none', background: 'none', outline: 'none', font: "400 13px/1.4 'Hanken Grotesk',sans-serif", color: C.text, padding: '5px 0' }} />
              <button type="submit" disabled={sendDisabled} style={{ flexShrink: 0, width: 32, height: 32, borderRadius: 10, border: 'none', cursor: sendDisabled ? 'default' : 'pointer', color: '#fff', fontSize: 15, display: 'flex', alignItems: 'center', justifyContent: 'center', background: sendDisabled ? C.textFaintest : C.text, transition: 'background .15s' }}>↑</button>
            </form>
            <div style={{ fontSize: 10.5, color: C.textFaintest, marginTop: 8, textAlign: 'center' }}>Demo assistant · verify against the source record</div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin     { to { transform: rotate(360deg) } }
        @keyframes fadeIn   { from { opacity: 0 } to { opacity: 1 } }
        @keyframes slideIn  { from { transform: translateX(100%) } to { transform: translateX(0) } }
        @keyframes blink    { 0%,80%,100% { opacity: .25 } 40% { opacity: 1 } }
      `}</style>
    </div>
  )
}

// ── Small sub-components ──────────────────────────────────────────────────────

function AskAtlasBtn({ onClick }: { onClick: () => void }) {
  const [hov, setHov] = useState(false)
  return (
    <button onClick={onClick} onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{ border: '1px solid #D8D4CB', background: hov ? '#F7F5F1' : '#fff', borderRadius: '999px', padding: '8px 15px', font: "600 12.5px/1 'Hanken Grotesk',sans-serif", color: C.text, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6, transition: 'background .15s' }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: C.blue, display: 'inline-block' }} />Ask Atlas
    </button>
  )
}

function PatientNameBtn({ name, preferredName, initials: ini, mrn, onClick }: { name: string; preferredName: string; initials: string; mrn: string; onClick: () => void }) {
  const [hov, setHov] = useState(false)
  return (
    <button onClick={onClick} onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{ background: 'none', border: 'none', padding: 0, textAlign: 'left', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 12, opacity: hov ? 0.75 : 1, transition: 'opacity .15s' }}>
      <div style={{ width: 42, height: 42, borderRadius: '50%', background: 'linear-gradient(135deg,#2860D8,#5B84E8)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600, fontSize: 15, flexShrink: 0 }}>{ini}</div>
      <div>
        <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: '-.01em', display: 'flex', alignItems: 'center', gap: 7 }}>
          {name} {preferredName && <span style={{ color: C.textFaint, fontWeight: 400, fontSize: 13 }}>{preferredName}</span>}
          <span style={{ width: 15, height: 15, border: '1.5px solid #C4C0B6', borderRadius: '50%', color: '#8A867C', fontSize: 9, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700 }}>i</span>
        </div>
        <div style={{ font: "500 11px/1 'JetBrains Mono',monospace", color: C.textFaint, marginTop: 3 }}>MRN {mrn}</div>
      </div>
    </button>
  )
}

function MetaField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ font: "600 9.5px/1 'Hanken Grotesk'", letterSpacing: '.1em', textTransform: 'uppercase', color: C.textFaintest }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 500, marginTop: 4 }}>{value}</div>
    </div>
  )
}

function AllergyChips({ allergies, onOpen }: { allergies: AllergyEntry[]; onOpen: () => void }) {
  const show = allergies.slice(0, 3)
  const overflow = allergies.length - show.length
  return (
    <div>
      <div style={{ font: "600 9.5px/1 'Hanken Grotesk'", letterSpacing: '.1em', textTransform: 'uppercase', color: C.textFaintest, marginBottom: 6 }}>Allergies</div>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
        {allergies.length === 0 ? (
          <span style={{ flex: '0 0 auto', display: 'inline-flex', alignItems: 'center', gap: 5, font: "600 11px/1 'Hanken Grotesk',sans-serif", padding: '5px 9px', borderRadius: '999px', whiteSpace: 'nowrap', background: '#E9F2EC', color: '#1F7A4D' }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#1F7A4D', flexShrink: 0, display: 'inline-block' }} />NKDA
          </span>
        ) : (
          <>
            {show.map((a, i) => {
              const [bg, fg] = ALLERGY_COLOR[a.kind]
              return (
                <button key={i} onClick={onOpen} style={{ flex: '0 0 auto', display: 'inline-flex', alignItems: 'center', gap: 5, border: 'none', cursor: 'pointer', font: "600 11px/1 'Hanken Grotesk',sans-serif", padding: '5px 9px', borderRadius: '999px', whiteSpace: 'nowrap', background: bg, color: fg }}>
                  <span style={{ width: 5, height: 5, borderRadius: '50%', background: fg, flexShrink: 0, display: 'inline-block' }} />{a.name}
                </button>
              )
            })}
            {overflow > 0 && (
              <button onClick={onOpen} style={{ flex: '0 0 auto', display: 'inline-flex', alignItems: 'center', background: C.chip, color: C.chipText, font: "600 11px/1 'Hanken Grotesk',sans-serif", padding: '5px 9px', border: 'none', borderRadius: '999px', cursor: 'pointer' }}>+{overflow}</button>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function BackBtn({ onClick }: { onClick: () => void }) {
  const [hov, setHov] = useState(false)
  return (
    <button onClick={onClick} onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{ background: 'none', border: 'none', padding: 0, font: "600 12.5px/1 'Hanken Grotesk'", color: hov ? C.text : C.textSec, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 7 }}>
      ← All encounters
    </button>
  )
}

function CloseBtn({ onClick, size = 32 }: { onClick: () => void; size?: number }) {
  const [hov, setHov] = useState(false)
  return (
    <button onClick={onClick} onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{ flexShrink: 0, width: size, height: size, borderRadius: '50%', border: `1px solid ${C.border}`, background: hov ? '#F0EEE8' : C.white, color: C.textSec, cursor: 'pointer', fontSize: 14, display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'background .15s' }}>
      ✕
    </button>
  )
}

function SuggestionBtn({ text, onClick }: { text: string; onClick: () => void }) {
  const [hov, setHov] = useState(false)
  return (
    <button onClick={onClick} onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{ textAlign: 'left', background: hov ? '#F0EEE8' : C.white, border: `1px solid ${C.border}`, borderRadius: 12, padding: '11px 14px', font: "500 12.5px/1.4 'Hanken Grotesk',sans-serif", color: C.text, cursor: 'pointer', transition: 'background .15s' }}>
      {text}
    </button>
  )
}
