import { useState, useEffect, useRef } from 'react'
import { C } from '../tokens'
import { fetchPatients } from '../api/client'
import type { PatientSummary } from '../api/client'
import { fmtDate, ageFromDob, initials } from '../api/format'

interface Props {
  onSelect: (patientId: string) => void
  onBack?: () => void
  doctorInfo?: { userId: string; roleLabel: string; userName: string } | null
}

export default function PatientList({ onSelect, onBack, doctorInfo }: Props) {
  const [search,   setSearch]   = useState('')
  const [page,     setPage]     = useState(1)
  const [patients, setPatients] = useState<PatientSummary[]>([])
  const [total,    setTotal]    = useState(0)
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const limit = 18

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetchPatients(page, limit, search)
      .then(data => {
        if (!cancelled) {
          setPatients(data.patients)
          setTotal(data.total)
          setLoading(false)
        }
      })
      .catch(err => {
        if (!cancelled) {
          setError(String(err.message ?? err))
          setLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [page, search])

  const totalPages = Math.ceil(total / limit)

  const handleSearch = (v: string) => {
    setSearch(v)
    setPage(1)
  }

  return (
    <div style={{ minHeight: '100vh', background: C.bg, fontFamily: "'Hanken Grotesk',sans-serif", color: C.text }}>
      {/* Top bar */}
      <div style={{ position: 'sticky', top: 0, zIndex: 30, background: 'rgba(250,249,247,.86)', backdropFilter: 'saturate(140%) blur(10px)', borderBottom: `1px solid ${C.border}` }}>
        <div style={{ maxWidth: 1180, margin: '0 auto', padding: '0 32px', height: 56, display: 'flex', alignItems: 'center', gap: 16 }}>
          {onBack && (
            <button
              onClick={onBack}
              style={{ background: 'none', border: `1px solid ${C.border}`, borderRadius: 8, padding: '5px 12px', font: "600 12px/1 'Hanken Grotesk',sans-serif", color: C.textSec, cursor: 'pointer' }}
            >
              ← Providers
            </button>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
            <div style={{ width: 20, height: 20, borderRadius: 6, background: C.text, transform: 'rotate(45deg)' }} />
            <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: '-.02em' }}>Atlas<span style={{ color: C.textFaint, fontWeight: 500 }}> Healthcare</span></span>
          </div>
          <div style={{ flex: 1 }} />
          {doctorInfo && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ font: "500 12px/1 'Hanken Grotesk'", color: C.textSec }}>{doctorInfo.userName}</span>
              <span style={{ font: "600 10.5px/1 'JetBrains Mono',monospace", color: C.textFaintest, background: '#F0EEE8', border: `1px solid ${C.border}`, borderRadius: 6, padding: '3px 7px' }}>{doctorInfo.roleLabel}</span>
            </div>
          )}
        </div>
      </div>

      <div style={{ maxWidth: 1180, margin: '0 auto', padding: '40px 32px 90px' }}>
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ fontFamily: "'Newsreader',serif", fontWeight: 500, fontSize: 30, letterSpacing: '-.02em', margin: '0 0 6px' }}>Patient Records</h1>
          <p style={{ fontSize: 13.5, color: C.textSec, margin: 0 }}>
            {total > 0 ? `${total} patient${total === 1 ? '' : 's'}` : 'Loading…'}
            {doctorInfo ? ` assigned to ${doctorInfo.userName}` : ' · Synthetic Synthea dataset'}
            {' · No real PHI'}
          </p>
        </div>

        {/* Search */}
        <div style={{ position: 'relative', marginBottom: 28, maxWidth: 420 }}>
          <span style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: C.textFaint, fontSize: 14, pointerEvents: 'none' }}>⌕</span>
          <input
            ref={inputRef}
            value={search}
            onChange={e => handleSearch(e.target.value)}
            placeholder="Search by name or MRN…"
            style={{
              width: '100%', boxSizing: 'border-box',
              padding: '11px 16px 11px 38px',
              border: `1px solid ${C.border}`, borderRadius: 12,
              font: "400 13.5px/1 'Hanken Grotesk',sans-serif",
              background: C.white, color: C.text, outline: 'none',
            }}
          />
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: '#FBE6E2', border: '1px solid #E8B4AE', borderRadius: 12, padding: '14px 18px', marginBottom: 24, color: '#8A2020', fontSize: 13 }}>
            <strong>Could not connect to backend:</strong> {error}
            <br /><span style={{ color: '#B23B2E', fontSize: 12 }}>Make sure the server is running at http://localhost:8000</span>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: C.textSec, fontSize: 13.5, padding: '40px 0' }}>
            <div style={{ width: 18, height: 18, borderRadius: '50%', border: `2px solid ${C.border}`, borderTopColor: C.text, animation: 'spin 0.8s linear infinite' }} />
            Fetching patients…
          </div>
        )}

        {/* Patient grid */}
        {!loading && !error && (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
              {patients.map(pt => (
                <PatientCard key={pt.patient_id} patient={pt} onSelect={onSelect} />
              ))}
            </div>

            {patients.length === 0 && (
              <div style={{ textAlign: 'center', color: C.textSec, padding: '60px 0', fontSize: 13.5 }}>
                No patients match your search.
              </div>
            )}

            {/* Pagination */}
            {totalPages > 1 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 32, justifyContent: 'center' }}>
                <PageBtn label="← Prev" disabled={page === 1} onClick={() => setPage(p => p - 1)} />
                <span style={{ fontSize: 13, color: C.textSec }}>Page {page} of {totalPages}</span>
                <PageBtn label="Next →" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)} />
              </div>
            )}
          </>
        )}
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg) } }
      `}</style>
    </div>
  )
}

function PatientCard({ patient, onSelect }: { patient: PatientSummary; onSelect: (id: string) => void }) {
  const [hov, setHov] = useState(false)
  const age = patient.dob ? ageFromDob(patient.dob) : '?'
  const dob = patient.dob ? fmtDate(patient.dob) : '—'
  const ini = initials(patient.name)

  return (
    <button
      onClick={() => onSelect(patient.patient_id)}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'block', width: '100%', textAlign: 'left',
        background: hov ? '#F7F5F1' : C.white,
        border: `1px solid ${hov ? '#C4C0B6' : C.border}`,
        borderRadius: 14, padding: '18px 20px', cursor: 'pointer',
        transition: 'background .12s, border-color .12s',
        boxShadow: hov ? '0 2px 12px rgba(28,26,23,.07)' : 'none',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 12 }}>
        <div style={{ width: 40, height: 40, borderRadius: '50%', background: 'linear-gradient(135deg,#2860D8,#5B84E8)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600, fontSize: 14, flexShrink: 0 }}>
          {ini}
        </div>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontWeight: 700, fontSize: 14.5, letterSpacing: '-.01em', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{patient.name}</div>
          <div style={{ font: "500 11px/1 'JetBrains Mono',monospace", color: C.textFaint, marginTop: 3 }}>MRN {patient.mrn.length > 16 ? patient.mrn.slice(0, 8) : (patient.mrn || '—')}</div>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
        <MetaField label="DOB" value={dob} />
        <MetaField label="Age" value={age} />
        <MetaField label="Sex" value={patient.sex || '—'} />
      </div>
      {patient.headline && (
        <div style={{ marginTop: 10, fontSize: 12, color: C.textSec, lineHeight: 1.45, overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' as const }}>
          {patient.headline}
        </div>
      )}
    </button>
  )
}

function MetaField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ font: "600 9px/1 'Hanken Grotesk'", letterSpacing: '.1em', textTransform: 'uppercase', color: C.textFaintest }}>{label}</div>
      <div style={{ fontSize: 12.5, fontWeight: 500, marginTop: 3, color: C.text }}>{value}</div>
    </div>
  )
}

function PageBtn({ label, disabled, onClick }: { label: string; disabled: boolean; onClick: () => void }) {
  const [hov, setHov] = useState(false)
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        border: `1px solid ${C.border}`, borderRadius: 8, padding: '8px 14px',
        font: "600 12px/1 'Hanken Grotesk',sans-serif",
        background: hov && !disabled ? '#F0EEE8' : C.white,
        color: disabled ? C.textFaintest : C.text,
        cursor: disabled ? 'default' : 'pointer', transition: 'background .12s',
      }}
    >
      {label}
    </button>
  )
}
