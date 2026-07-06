const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

// ---------------------------------------------------------------------------
// Token + user session state
// ---------------------------------------------------------------------------

let _token: string | null = null
let _userId: string | null = null
let _roleName: string | null = null
let _userName: string | null = null
let _authInFlight: Promise<string> | null = null

/** Auto-select u_100 if no persona has been explicitly chosen. */
async function _ensureToken(): Promise<string> {
  if (_token) return _token
  if (_authInFlight) return _authInFlight
  _authInFlight = _doSelect('u_100')
  return _authInFlight
}

async function _doSelect(userId: string): Promise<string> {
  const r = await fetch(`${BASE}/runtime/personas/${userId}/select`, { method: 'POST' })
  if (!r.ok) throw new Error(`auth: ${r.status}`)
  const d = await r.json() as { token: string; user_id: string; role_id: string; name: string }
  _token = d.token
  _userId = d.user_id
  _authInFlight = null
  return _token
}

export async function selectPersona(
  userId: string,
  roleLabel: string,
  name: string,
): Promise<{ token: string; session_id: string; role_id: string }> {
  clearToken()
  const r = await fetch(`${BASE}/runtime/personas/${userId}/select`, { method: 'POST' })
  if (!r.ok) throw new Error(`select: ${r.status}`)
  const d = await r.json()
  _token    = d.token
  _userId   = d.user_id
  _roleName = roleLabel
  _userName = name
  _authInFlight = null
  return d
}

export function clearToken() {
  _token = null
  _userId = null
  _roleName = null
  _userName = null
  _authInFlight = null
}

export function getCurrentUser() {
  return { userId: _userId, roleName: _roleName, userName: _userName }
}

// ---------------------------------------------------------------------------
// Persona list
// ---------------------------------------------------------------------------

export interface Persona {
  user_id:    string
  name:       string
  role_id:    string
  role_label: string
  department: string
  care_team:  string
}

export async function fetchPersonas(): Promise<Persona[]> {
  const r = await fetch(`${BASE}/runtime/personas`)
  if (!r.ok) throw new Error(`personas: ${r.status}`)
  return r.json()
}

// ---------------------------------------------------------------------------
// Patient list + chart
// ---------------------------------------------------------------------------

export interface PatientSummary {
  patient_id: string
  name: string
  dob: string
  sex: string
  mrn: string
  headline: string
}

export interface PatientListResponse {
  total: number
  page: number
  limit: number
  patients: PatientSummary[]
}

export async function fetchPatients(page: number, limit: number, search: string): Promise<PatientListResponse> {
  const token = await _ensureToken()
  const params = new URLSearchParams({ page: String(page), limit: String(limit), search })
  const r = await fetch(`${BASE}/runtime/patients?${params}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!r.ok) throw new Error(`patients: ${r.status}`)
  return r.json()
}

export async function fetchChart(patientId: string): Promise<{ patient_id: string; resources: Record<string, unknown> }> {
  const token = await _ensureToken()
  const r = await fetch(`${BASE}/runtime/chart/${patientId}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!r.ok) throw new Error(`chart: ${r.status}`)
  return r.json()
}

export async function fetchEncounterDetail(patientId: string, encounterId: string): Promise<Record<string, unknown>> {
  const token = await _ensureToken()
  const r = await fetch(`${BASE}/runtime/chart/${patientId}/encounter/${encounterId}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!r.ok) throw new Error(`encounter: ${r.status}`)
  return r.json()
}
