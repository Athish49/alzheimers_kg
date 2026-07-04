import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { listPersonas, selectPersona } from '../api';
import { Icon } from '../components/Icons';
import '../styles/enterprise.css';

const PERSONA_META = {
  u_014: {
    initials: 'SC',
    subtitle: 'attending_physician',
    can: ['Full chart for assigned patients', 'Genetic markers & clinical notes', 'Ask clinical questions (LLM)'],
    cannot: ['Break-glass without reason', 'Unassigned patients (without break-glass)'],
  },
  u_027: {
    initials: 'RP',
    subtitle: 'nurse',
    can: ['Demographics, vitals, meds, conditions', 'Assigned patients only'],
    cannot: ['Genetic markers', 'Clinical notes', 'Lab entry'],
  },
  u_033: {
    initials: 'ER',
    subtitle: 'pharmacist',
    can: ['Medication review for assigned patients', 'Cross-department access (p_5501)'],
    cannot: ['Genetic markers', 'Lab results', 'Clinical notes'],
  },
  u_041: {
    initials: 'ML',
    subtitle: 'lab_technician',
    can: ['Enter & view lab results', 'Assigned patients including p_3310'],
    cannot: ['Genetic markers', 'Medications', 'Demographics address/insurance'],
  },
  u_059: {
    initials: 'TB',
    subtitle: 'research_analyst',
    can: ['De-identified aggregate data', 'Knowledge graph queries (no LLM)'],
    cannot: ['Any individual patient record', 'All patient-scoped resources'],
  },
};

export function DemoPage() {
  const navigate = useNavigate();
  const [personas, setPersonas] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selecting, setSelecting] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    listPersonas()
      .then(setPersonas)
      .catch(() => setError('Could not load personas — the backend may be warming up. Refresh in a moment.'))
      .finally(() => setLoading(false));
  }, []);

  async function handleSelect(userId) {
    setSelecting(userId);
    try {
      const data = await selectPersona(userId);
      sessionStorage.setItem('atlas_auth', JSON.stringify({
        token: data.token,
        session_id: data.session_id,
        user_id: data.user_id,
        role_id: data.role_id,
        name: data.name,
      }));
      navigate('/workspace');
    } catch (e) {
      setError(e.message || 'Failed to start session. Try again.');
      setSelecting(null);
    }
  }

  return (
    <div className="demo-page" style={{ height: '100vh', overflowY: 'auto', overflowX: 'hidden' }}>
      <div className="demo-page-inner">
        <nav className="demo-page-nav">
          <Link className="demo-page-brand" to="/">
            <span className="brand-mark">A</span>
            <span>Atlas</span>
          </Link>
          <Link className="demo-page-back" to="/">
            <Icon.Arrow size={11} />
            Back to overview
          </Link>
        </nav>

        <div className="demo-page-head">
          <div className="demo-page-kicker">Enterprise demo · Role-based access</div>
          <h1 className="demo-page-title">Pick a role to <em>step inside</em>.</h1>
          <p className="demo-page-sub">
            Each persona has a different permission set. The same question will return different data
            depending on who is asking — or be denied entirely for an unassigned patient.
          </p>
        </div>

        {loading && (
          <div className="persona-loading">
            <div className="spin" />
            Loading personas…
          </div>
        )}

        {error && (
          <div className="demo-notice" style={{ borderColor: 'oklch(0.88 0.02 30)', background: 'var(--up-bg)', color: 'var(--up)' }}>
            {error}
          </div>
        )}

        {!loading && !error && (
          <div className="persona-grid">
            {personas.map((p) => {
              const meta = PERSONA_META[p.user_id] || {};
              return (
                <button
                  key={p.user_id}
                  className="persona-card"
                  onClick={() => handleSelect(p.user_id)}
                  disabled={!!selecting}
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                    <div className="persona-avatar">
                      {selecting === p.user_id
                        ? <div className="spin" style={{ borderColor: 'rgba(255,255,255,0.3)', borderTopColor: '#fff' }} />
                        : meta.initials}
                    </div>
                    <div>
                      <div className="persona-name">{p.name}</div>
                      <div className="persona-role">{p.role_id}</div>
                    </div>
                  </div>

                  <div className="persona-acl">
                    {(meta.can || []).map((item, i) => (
                      <div key={i} className="persona-acl-row can">
                        <span className="persona-acl-dot" />
                        {item}
                      </div>
                    ))}
                    {(meta.cannot || []).map((item, i) => (
                      <div key={i} className="persona-acl-row cannot">
                        <span className="persona-acl-dot" />
                        {item}
                      </div>
                    ))}
                  </div>

                  <div className="persona-card-foot">
                    <span>Enter as {p.name.split(' ')[0]}</span>
                    <Icon.Arrow size={12} />
                  </div>
                </button>
              );
            })}
          </div>
        )}

        <div className="demo-notice" style={{ marginTop: 28 }}>
          <strong>Synthetic data only.</strong> All patient names, MRNs, and clinical values are fictional.
          No real health information is stored or processed. Sessions are ephemeral and reset on server restart.
        </div>
      </div>
    </div>
  );
}
