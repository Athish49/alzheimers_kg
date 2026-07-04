const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function queryAnswer(question, { temperature, maxTokens, history, signal } = {}) {
  const body = {
    question,
    return_context: true,
    history: history || [],
  };
  if (temperature !== undefined) body.temperature = temperature;
  if (maxTokens !== undefined) body.max_tokens = maxTokens;

  const res = await fetch(`${API_BASE}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `HTTP ${res.status}`);
  }

  return res.json();
}

// ---------------------------------------------------------------------------
// Runtime plane helpers
// ---------------------------------------------------------------------------

function authHeaders(token) {
  return { "Content-Type": "application/json", Authorization: `Bearer ${token}` };
}

async function runtimeFetch(path, opts = {}) {
  const res = await fetch(`${API_BASE}/runtime${path}`, opts);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      detail = j.detail || detail;
    } catch (_) {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export async function healthCheck() {
  return runtimeFetch("/health");
}

export async function listPersonas() {
  return runtimeFetch("/personas");
}

export async function selectPersona(userId) {
  return runtimeFetch(`/personas/${userId}/select`, { method: "POST" });
}

export async function listPatients(token) {
  return runtimeFetch("/patients", { headers: authHeaders(token) });
}

export async function getChart(token, patientId) {
  return runtimeFetch(`/chart/${patientId}`, { headers: authHeaders(token) });
}

export async function orchestrate(token, question, patientId, signal) {
  return runtimeFetch("/orchestrate", {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ question, patient_id: patientId }),
    signal,
  });
}

export async function getAuditLog(token) {
  return runtimeFetch("/audit", { headers: authHeaders(token) });
}

export async function requestBreakGlass(token, patientId, reason) {
  return runtimeFetch("/break-glass", {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ patient_id: patientId, reason }),
  });
}
