import { useState, useEffect } from 'react'
import { fetchEncounterDetail } from '../api/client'

export interface EncounterDetailState {
  detail: Record<string, unknown> | null
  loading: boolean
  error: string | null
}

export function useEncounterDetail(patientId: string | null, encounterId: string | null): EncounterDetailState {
  const [state, setState] = useState<EncounterDetailState>({ detail: null, loading: false, error: null })

  useEffect(() => {
    if (!patientId || !encounterId) { setState({ detail: null, loading: false, error: null }); return }
    let cancelled = false
    setState({ detail: null, loading: true, error: null })
    fetchEncounterDetail(patientId, encounterId)
      .then(data => {
        if (!cancelled) setState({ detail: data, loading: false, error: null })
      })
      .catch(err => {
        if (!cancelled) setState({ detail: null, loading: false, error: String(err.message ?? err) })
      })
    return () => { cancelled = true }
  }, [patientId, encounterId])

  return state
}
