import { useState, useEffect } from 'react'
import { fetchChart } from '../api/client'

export interface ChartState {
  resources: Record<string, unknown> | null
  loading: boolean
  error: string | null
}

export function useChart(patientId: string | null): ChartState {
  const [state, setState] = useState<ChartState>({ resources: null, loading: false, error: null })

  useEffect(() => {
    if (!patientId) { setState({ resources: null, loading: false, error: null }); return }
    let cancelled = false
    setState({ resources: null, loading: true, error: null })
    fetchChart(patientId)
      .then(data => {
        if (!cancelled) setState({ resources: data.resources, loading: false, error: null })
      })
      .catch(err => {
        if (!cancelled) setState({ resources: null, loading: false, error: String(err.message ?? err) })
      })
    return () => { cancelled = true }
  }, [patientId])

  return state
}
