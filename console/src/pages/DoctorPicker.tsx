import { useState, useEffect } from 'react'
import { C } from '../tokens'
import { fetchPersonas, selectPersona } from '../api/client'
import type { Persona } from '../api/client'

interface Props {
  onSelect: (userId: string, roleLabel: string, userName: string) => void
}

// Human-readable specialty labels that appear as group headers
const SPECIALTY_ORDER = [
  'Primary Care Physician',
  'Psychiatrist',
  'Endocrinologist',
  'Cardiologist',
  'Heart Failure Specialist',
  'Electrophysiologist',
  'Pulmonologist',
  'Nephrologist',
  'Gastroenterologist',
  'Rheumatologist',
  'Hematologist',
  'Orthopedic Surgeon',
  'Urologist',
  'Allergist / Immunologist',
  'Bariatrician',
]

function groupByRole(personas: Persona[]): Map<string, Persona[]> {
  const map = new Map<string, Persona[]>()
  for (const p of personas) {
    const label = p.role_label || p.role_id
    if (!map.has(label)) map.set(label, [])
    map.get(label)!.push(p)
  }
  // Sort by SPECIALTY_ORDER, unknown roles go at the end
  const sorted = new Map<string, Persona[]>()
  for (const spec of SPECIALTY_ORDER) {
    if (map.has(spec)) sorted.set(spec, map.get(spec)!)
  }
  for (const [k, v] of map) {
    if (!sorted.has(k)) sorted.set(k, v)
  }
  return sorted
}

// Consistent avatar color per role
const ROLE_COLORS: Record<string, string> = {
  'Primary Care Physician':  'linear-gradient(135deg,#2860D8,#5B84E8)',
  'Psychiatrist':            'linear-gradient(135deg,#283593,#3949AB)',
  'Endocrinologist':         'linear-gradient(135deg,#E65100,#F57C00)',
  'Cardiologist':            'linear-gradient(135deg,#C2185B,#E91E63)',
  'Heart Failure Specialist':'linear-gradient(135deg,#AD1457,#D81B60)',
  'Electrophysiologist':     'linear-gradient(135deg,#7B1FA2,#9C27B0)',
  'Pulmonologist':           'linear-gradient(135deg,#0288D1,#039BE5)',
  'Nephrologist':            'linear-gradient(135deg,#00838F,#00ACC1)',
  'Gastroenterologist':      'linear-gradient(135deg,#558B2F,#7CB342)',
  'Rheumatologist':          'linear-gradient(135deg,#4527A0,#5E35B1)',
  'Hematologist':            'linear-gradient(135deg,#B71C1C,#E53935)',
  'Orthopedic Surgeon':      'linear-gradient(135deg,#37474F,#546E7A)',
  'Urologist':               'linear-gradient(135deg,#1B5E20,#43A047)',
  'Allergist / Immunologist':'linear-gradient(135deg,#F57F17,#F9A825)',
  'Bariatrician':            'linear-gradient(135deg,#880E4F,#AD1457)',
}

function initials(name: string): string {
  return name
    .replace(/^Dr\.\s*/i, '')
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map(w => w[0].toUpperCase())
    .join('')
}

export default function DoctorPicker({ onSelect }: Props) {
  const [personas, setPersonas] = useState<Persona[]>([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState<string | null>(null)
  const [selecting, setSelecting] = useState<string | null>(null)

  useEffect(() => {
    fetchPersonas()
      .then(data => { setPersonas(data); setLoading(false) })
      .catch(err  => { setError(String(err.message ?? err)); setLoading(false) })
  }, [])

  const handleSelect = async (persona: Persona) => {
    if (selecting) return
    setSelecting(persona.user_id)
    try {
      await selectPersona(persona.user_id, persona.role_label, persona.name)
      onSelect(persona.user_id, persona.role_label, persona.name)
    } catch (err) {
      setError(String((err as Error).message ?? err))
      setSelecting(null)
    }
  }

  const grouped = groupByRole(personas)

  return (
    <div style={{ minHeight: '100vh', background: C.bg, fontFamily: "'Hanken Grotesk',sans-serif", color: C.text }}>
      {/* Top bar */}
      <div style={{ position: 'sticky', top: 0, zIndex: 30, background: 'rgba(250,249,247,.92)', backdropFilter: 'saturate(140%) blur(10px)', borderBottom: `1px solid ${C.border}` }}>
        <div style={{ maxWidth: 1280, margin: '0 auto', padding: '0 32px', height: 56, display: 'flex', alignItems: 'center', gap: 10 }}>
          <a
            href={import.meta.env.VITE_LANDING_URL ?? 'http://localhost:5173'}
            style={{ display: 'flex', alignItems: 'center', gap: 8, textDecoration: 'none', color: C.text }}
          >
            <div style={{ width: 20, height: 20, borderRadius: 6, background: C.text, transform: 'rotate(45deg)' }} />
            <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: '-.02em' }}>
              Atlas<span style={{ color: C.textFaint, fontWeight: 500 }}> Healthcare</span>
            </span>
          </a>
          <div style={{ flex: 1 }} />
          <a
            href={import.meta.env.VITE_LANDING_URL ?? 'http://localhost:5173'}
            style={{ font: "500 11.5px/1 'JetBrains Mono',monospace", color: C.textFaintest, textDecoration: 'none' }}
          >
            ← Back to Atlas
          </a>
        </div>
      </div>

      <div style={{ maxWidth: 1280, margin: '0 auto', padding: '40px 32px 90px' }}>
        <div style={{ marginBottom: 36 }}>
          <h1 style={{ fontFamily: "'Newsreader',serif", fontWeight: 500, fontSize: 30, letterSpacing: '-.02em', margin: '0 0 8px' }}>
            Provider Directory
          </h1>
          <p style={{ fontSize: 13.5, color: C.textSec, margin: 0, maxWidth: 560, lineHeight: 1.6 }}>
            Click any provider to log in as them. Each specialist sees only their assigned patients
            and the chart sections their role permits — demonstrating enterprise-grade, role-based access control.
          </p>
        </div>

        {error && (
          <div style={{ background: '#FBE6E2', border: '1px solid #E8B4AE', borderRadius: 12, padding: '14px 18px', marginBottom: 28, color: '#8A2020', fontSize: 13 }}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {loading ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: C.textSec, fontSize: 13.5, padding: '40px 0' }}>
            <div style={{ width: 18, height: 18, borderRadius: '50%', border: `2px solid ${C.border}`, borderTopColor: C.text, animation: 'spin 0.8s linear infinite' }} />
            Loading providers…
          </div>
        ) : (
          <div>
            {Array.from(grouped.entries()).map(([roleLabel, group]) => (
              <SpecialtySection
                key={roleLabel}
                roleLabel={roleLabel}
                personas={group}
                selecting={selecting}
                onSelect={handleSelect}
              />
            ))}
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg) } }
      `}</style>
    </div>
  )
}

function SpecialtySection({
  roleLabel, personas, selecting, onSelect,
}: {
  roleLabel: string
  personas: Persona[]
  selecting: string | null
  onSelect: (p: Persona) => void
}) {
  return (
    <div style={{ marginBottom: 32 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        <span style={{ font: "600 11px/1 'Hanken Grotesk'", letterSpacing: '.1em', textTransform: 'uppercase', color: C.textSec }}>
          {roleLabel}
        </span>
        <div style={{ flex: 1, height: 1, background: C.border }} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 12 }}>
        {personas.map(p => (
          <DoctorCard
            key={p.user_id}
            persona={p}
            isSelecting={selecting === p.user_id}
            disabled={selecting !== null && selecting !== p.user_id}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  )
}

function DoctorCard({
  persona, isSelecting, disabled, onSelect,
}: {
  persona: Persona
  isSelecting: boolean
  disabled: boolean
  onSelect: (p: Persona) => void
}) {
  const [hov, setHov] = useState(false)
  const avatarBg = ROLE_COLORS[persona.role_label] ?? 'linear-gradient(135deg,#2860D8,#5B84E8)'
  const ini = initials(persona.name)

  return (
    <button
      onClick={() => !disabled && !isSelecting && onSelect(persona)}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      disabled={disabled || isSelecting}
      style={{
        display: 'flex', alignItems: 'center', gap: 14, width: '100%', textAlign: 'left',
        background: isSelecting ? '#EDF2FF' : hov ? '#F7F5F1' : C.white,
        border: `1px solid ${isSelecting ? '#AABBF0' : hov ? '#C4C0B6' : C.border}`,
        borderRadius: 14, padding: '14px 16px', cursor: disabled ? 'default' : 'pointer',
        transition: 'background .12s, border-color .12s',
        boxShadow: hov && !disabled ? '0 2px 10px rgba(28,26,23,.07)' : 'none',
        opacity: disabled ? 0.45 : 1,
      }}
    >
      <div style={{
        width: 44, height: 44, borderRadius: '50%', background: avatarBg,
        color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontWeight: 700, fontSize: 15, flexShrink: 0,
      }}>
        {isSelecting
          ? <div style={{ width: 18, height: 18, borderRadius: '50%', border: '2.5px solid rgba(255,255,255,.4)', borderTopColor: '#fff', animation: 'spin 0.7s linear infinite' }} />
          : ini
        }
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontWeight: 700, fontSize: 14, letterSpacing: '-.01em', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {persona.name}
        </div>
        <div style={{ font: "500 11.5px/1 'Hanken Grotesk'", color: C.textSec, marginTop: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {persona.department.replace(/_/g, ' ')}
        </div>
        <div style={{ font: "500 10px/1 'JetBrains Mono',monospace", color: C.textFaintest, marginTop: 4 }}>
          {persona.user_id}
        </div>
      </div>
    </button>
  )
}
