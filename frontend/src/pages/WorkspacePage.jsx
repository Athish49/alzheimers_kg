import { useState, useEffect, useRef, useCallback } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  healthCheck, listPatients, getChart, orchestrate,
  getAuditLog, requestBreakGlass,
} from '../api';
import { Icon } from '../components/Icons';
import { Prose } from '../components/Prose';
import '../styles/enterprise.css';

const RESOURCE_LABELS = {
  demographics:    'Demographics',
  conditions:      'Conditions',
  vitals:          'Vitals',
  medications:     'Medications',
  lab_results:     'Lab Results',
  genetic_markers: 'Genetic Markers',
  clinical_notes:  'Clinical Notes',
};

function useAuth() {
  const raw = sessionStorage.getItem('atlas_auth');
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

function formatAuditTs(ts) {
  try { return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
  catch { return ts; }
}

/* ── Resource row in chart ── */
function ResourceSection({ name, entry }) {
  const label = RESOURCE_LABELS[name] || name;
  if (!entry.granted) {
    return (
      <div className="ws-resource-section">
        <div className="ws-resource-label">{label}</div>
        <div className="ws-resource-row denied">
          <span className="ws-resource-icon"><Icon.Lock size={12} /></span>
          <span>{entry.reason || 'Access denied'}</span>
        </div>
      </div>
    );
  }
  if (entry.deidentified) {
    return (
      <div className="ws-resource-section">
        <div className="ws-resource-label">{label}</div>
        <div className="ws-resource-row">
          <span className="ws-resource-icon"><Icon.Shield size={12} /></span>
          <span>De-identified aggregate (individual data withheld)</span>
        </div>
      </div>
    );
  }
  if (entry.fields) {
    const fields = Object.entries(entry.fields);
    if (!fields.length) return null;
    return (
      <div className="ws-resource-section">
        <div className="ws-resource-label">{label}</div>
        <div className="ws-resource-items">
          {fields.map(([k, v]) => (
            <div key={k} className="ws-resource-row">
              <span style={{ color: 'var(--fg-muted)', minWidth: 90, fontSize: 11.5, fontFamily: 'var(--font-mono)' }}>{k}</span>
              <span>{String(v)}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  if (entry.items) {
    if (!entry.items.length) return null;
    return (
      <div className="ws-resource-section">
        <div className="ws-resource-label">{label} ({entry.items.length})</div>
        <div className="ws-resource-items">
          {entry.items.map((item, i) => (
            <div key={i} className="ws-resource-row">
              <span>{Object.entries(item).map(([k, v]) => `${k}: ${v}`).join(' · ')}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return null;
}

/* ── Break-glass form ── */
function BreakGlassPanel({ token, patientId, onGranted }) {
  const [reason, setReason] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [granted, setGranted] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!reason.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      await requestBreakGlass(token, patientId, reason.trim());
      setGranted(true);
      setTimeout(onGranted, 800);
    } catch (err) {
      setError(err.message || 'Failed to request break-glass access.');
      setSubmitting(false);
    }
  }

  return (
    <div className="ws-breakglass">
      <div className="ws-breakglass-head">
        <Icon.Shield size={14} />
        Emergency access available
      </div>
      <div className="ws-breakglass-body">
        You are not assigned to this patient. Provide a clinical reason to request
        time-limited (15-minute) break-glass access. This action is logged and flagged.
      </div>
      {granted ? (
        <div className="ws-breakglass-granted">
          <Icon.Shield size={13} /> Access granted — 15-minute window active. Refreshing chart…
        </div>
      ) : (
        <form className="ws-breakglass-form" onSubmit={handleSubmit}>
          <textarea
            className="ws-breakglass-input"
            rows={2}
            placeholder="Clinical reason (required)…"
            value={reason}
            onChange={e => setReason(e.target.value)}
            disabled={submitting}
          />
          {error && <div style={{ fontSize: 12, color: 'var(--up)' }}>{error}</div>}
          <button className="ws-breakglass-submit" type="submit" disabled={submitting || !reason.trim()}>
            {submitting ? <><div className="spin" style={{ width: 12, height: 12 }} /> Requesting…</> : 'Request emergency access'}
          </button>
        </form>
      )}
    </div>
  );
}

/* ── Main page ── */
export function WorkspacePage() {
  const navigate = useNavigate();
  const auth = useAuth();

  const [warming, setWarming] = useState(true);
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [chart, setChart] = useState(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [asking, setAsking] = useState(false);
  const [cooldown, setCooldown] = useState(false);
  const [auditRows, setAuditRows] = useState([]);
  const abortRef = useRef(null);
  const threadRef = useRef(null);

  /* redirect if no auth */
  useEffect(() => {
    if (!auth) navigate('/demo');
  }, [auth, navigate]);

  /* health check + initial data load */
  useEffect(() => {
    if (!auth) return;
    let alive = true;
    async function init() {
      for (let attempt = 0; attempt < 20; attempt++) {
        try {
          await healthCheck();
          break;
        } catch {
          if (!alive) return;
          await new Promise(r => setTimeout(r, 3000));
        }
      }
      if (!alive) return;
      setWarming(false);
      try {
        const pts = await listPatients(auth.token);
        setPatients(pts);
        if (pts.length) selectPatient(pts[0]);
      } catch { /* patients stay empty */ }
    }
    init();
    return () => { alive = false; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /* poll audit log */
  useEffect(() => {
    if (!auth || warming) return;
    let alive = true;
    async function pollAudit() {
      while (alive) {
        try {
          const rows = await getAuditLog(auth.token);
          if (alive) setAuditRows(rows);
        } catch { /* silent */ }
        await new Promise(r => setTimeout(r, 5000));
      }
    }
    pollAudit();
    return () => { alive = false; };
  }, [warming]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectPatient = useCallback(async (patient) => {
    setSelectedPatient(patient);
    setChart(null);
    setMessages([]);
    setChartLoading(true);
    try {
      const data = await getChart(auth.token, patient.patient_id);
      setChart(data);
    } catch { setChart(null); }
    setChartLoading(false);
  }, [auth]); // eslint-disable-line react-hooks/exhaustive-deps

  /* detect if all resources denied (not assigned) */
  const allDenied = chart && Object.values(chart.resources).length > 0
    && Object.values(chart.resources).every(e => !e.granted);
  const hasDeniedNotAssigned = allDenied
    && Object.values(chart.resources).some(e => e.reason && e.reason.toLowerCase().includes('assign'));

  async function handleBreakGlassGranted() {
    if (!selectedPatient) return;
    const data = await getChart(auth.token, selectedPatient.patient_id).catch(() => null);
    if (data) setChart(data);
  }

  async function handleAsk(e) {
    e?.preventDefault();
    if (!input.trim() || asking || !auth) return;
    const question = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: question }]);
    setAsking(true);
    setCooldown(false);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const result = await orchestrate(
        auth.token, question,
        selectedPatient?.patient_id ?? null,
        ctrl.signal,
      );
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: result.answer,
        patient_evidence: result.patient_evidence || [],
        knowledge_evidence: result.knowledge_evidence || [],
        abstained_on: result.abstained_on || [],
      }]);
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (err.status === 429) {
        setCooldown(true);
        setMessages(prev => [...prev, { role: 'error', content: err.message }]);
      } else {
        setMessages(prev => [...prev, { role: 'error', content: err.message || 'Request failed.' }]);
      }
    } finally {
      setAsking(false);
      abortRef.current = null;
      setTimeout(() => threadRef.current?.scrollTo({ top: 999999, behavior: 'smooth' }), 50);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  }

  function handleLogout() {
    sessionStorage.removeItem('atlas_auth');
    navigate('/demo');
  }

  if (!auth) return null;

  if (warming) {
    return (
      <div className="ws-warming">
        <div className="spin" style={{ width: 24, height: 24 }} />
        <div className="ws-warming-title">Waking the demo…</div>
        <div className="ws-warming-sub">
          The backend is starting up (free-tier cold start). This takes 20–40 s.
        </div>
      </div>
    );
  }

  return (
    <div className="ws-root">
      {/* Header */}
      <header className="ws-header">
        <Link className="ws-brand" to="/">
          <span className="brand-mark">A</span>
          Atlas
        </Link>
        <div className="ws-role-chip">
          <Icon.User size={11} />
          {auth.name} · {auth.role_id}
        </div>
        <div style={{ flex: 1 }} />
        <div className="ws-header-right">
          <button className="ws-btn" onClick={handleLogout}>
            <Icon.Logout size={13} />
            Switch role
          </button>
        </div>
      </header>

      {/* Body */}
      <div className="ws-body">
        {/* Left: patient list */}
        <aside className="ws-patient-col">
          <div className="ws-col-head">
            <div className="ws-col-kicker">Assigned patients</div>
          </div>
          <div className="ws-patient-list">
            {patients.length === 0 ? (
              <div className="ws-patient-empty">
                No patients assigned to this role.
              </div>
            ) : patients.map(p => (
              <div
                key={p.patient_id}
                className={`ws-patient-row${selectedPatient?.patient_id === p.patient_id ? ' active' : ''}`}
                onClick={() => selectPatient(p)}
              >
                <div className="ws-patient-name">{p.name}</div>
                <div className="ws-patient-meta">{p.mrn} · {p.sex} · {p.dob}</div>
                {p.headline && <div className="ws-patient-hl">{p.headline}</div>}
              </div>
            ))}
          </div>
        </aside>

        {/* Center: chart + chat */}
        <main className="ws-center">
          {/* Chart */}
          <div className="ws-chart">
            {!selectedPatient ? (
              <div className="ws-chart-empty">Select a patient to view their chart.</div>
            ) : chartLoading ? (
              <div className="ws-chart-inner">
                <div className="skeleton-answer">
                  <div className="skeleton-line" style={{ width: '40%' }} />
                  <div className="skeleton-line" style={{ width: '60%' }} />
                  <div className="skeleton-line" style={{ width: '30%' }} />
                </div>
              </div>
            ) : chart ? (
              <div className="ws-chart-inner">
                <div className="ws-chart-patient">
                  <div className="ws-chart-name">{selectedPatient.name}</div>
                  <div className="ws-chart-sub">
                    {selectedPatient.mrn} · {selectedPatient.sex} · DOB {selectedPatient.dob}
                  </div>
                </div>
                {Object.entries(chart.resources).map(([name, entry]) => (
                  <ResourceSection key={name} name={name} entry={entry} />
                ))}
                {hasDeniedNotAssigned && (
                  <BreakGlassPanel
                    token={auth.token}
                    patientId={selectedPatient.patient_id}
                    onGranted={handleBreakGlassGranted}
                  />
                )}
              </div>
            ) : (
              <div className="ws-chart-empty">Could not load chart.</div>
            )}
          </div>

          {/* Chat */}
          <div className="ws-chat">
            <div className="ws-thread" ref={threadRef}>
              {messages.length === 0 ? (
                <div className="ws-thread-empty">
                  <div className="ws-thread-empty-inner">
                    <Icon.Shield size={24} />
                    <div className="ws-thread-empty-title">Ask a clinical question</div>
                    <div className="ws-thread-empty-sub">
                      {selectedPatient
                        ? `Ask about ${selectedPatient.name.split(' ')[0]}'s chart or general AD knowledge.`
                        : 'Select a patient first, then ask questions grounded in their chart.'}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                  {messages.map((m, i) => {
                    if (m.role === 'user') {
                      return <div key={i} className="ws-msg-user">{m.content}</div>;
                    }
                    if (m.role === 'error') {
                      return (
                        <div key={i} className="ws-cooldown">
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, fontWeight: 500 }}>
                            <Icon.Alert size={14} />
                            {cooldown ? 'Demo cooling down' : 'Error'}
                          </div>
                          {m.content}
                        </div>
                      );
                    }
                    return (
                      <div key={i} className="ws-msg-assistant">
                        <Prose text={m.content} />
                        {(m.patient_evidence?.length > 0 || m.knowledge_evidence?.length > 0 || m.abstained_on?.length > 0) && (
                          <div className="ws-msg-evidence">
                            {m.patient_evidence?.map(r => (
                              <span key={r} className="ws-evidence-chip">
                                <Icon.User size={10} /> {r}
                              </span>
                            ))}
                            {m.knowledge_evidence?.map(ke => (
                              <span key={ke.key} className="ws-evidence-chip">
                                <Icon.Shield size={10} /> {ke.key}
                              </span>
                            ))}
                            {m.abstained_on?.map(r => (
                              <span key={r} className="ws-abstained-chip">
                                <Icon.Lock size={10} /> {r}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                  {asking && (
                    <div className="ws-msg-assistant">
                      <div className="cls-bar">
                        <span className="phase active"><span className="phase-dot" /> Reasoning…</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="ws-composer-wrap">
              <form className="ws-composer" onSubmit={handleAsk}>
                <textarea
                  rows={1}
                  placeholder={
                    selectedPatient
                      ? `Ask about ${selectedPatient.name.split(' ')[0]} or general AD knowledge…`
                      : 'Ask a clinical question…'
                  }
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={asking}
                />
                <div className="ws-composer-row">
                  <span className="ws-composer-hint">
                    {selectedPatient ? selectedPatient.name : 'No patient selected'}
                  </span>
                  <button className="ws-send" type="submit" disabled={!input.trim() || asking}>
                    {asking
                      ? <><Icon.Stop size={12} /> Stop</>
                      : <><Icon.Send size={12} /> Ask</>}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </main>

        {/* Right: audit panel */}
        <aside className="ws-audit-col">
          <div className="ws-col-head">
            <div className="ws-col-kicker">
              <Icon.Log size={11} /> Audit log
            </div>
          </div>
          <div className="ws-audit-body">
            {auditRows.length === 0 ? (
              <div className="ws-audit-empty">No audit events yet. Ask a question or view a chart.</div>
            ) : auditRows.map(row => (
              <div key={row.id} className="ws-audit-row">
                <div className="ws-audit-action">
                  <span className={`ws-audit-effect-${row.effect}`}>
                    {row.effect.toUpperCase()} · {row.action}
                  </span>
                  {row.break_glass && <span className="ws-audit-bg-badge">BG</span>}
                </div>
                <div className="ws-audit-meta">
                  {row.resource}
                  {row.patient_id ? ` · ${row.patient_id}` : ''}
                  {' · '}{formatAuditTs(row.ts)}
                </div>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}
