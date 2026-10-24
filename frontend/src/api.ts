export type Prediction = 'phishing' | 'legitimate'
export type ScanResult = {
  prediction: Prediction
  legitimate_probability?: number
  phishing_probability?: number
  reasons?: string[]
}

const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

export async function analyzeUrl(url: string): Promise<ScanResult> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), 60_000)
  try {
    let response: Response
    try {
      response = await fetch(`${baseUrl}/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }), signal: controller.signal,
      })
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') throw new Error('The request timed out. Please try again.')
      throw new Error('Could not reach the prediction service. Check that the backend is running and try again.')
    }
    if (!response.ok) throw new Error(`The prediction service returned an error (${response.status}). Please try again.`)
    let data: unknown
    try { data = await response.json() } catch { throw new Error('The prediction service returned an unreadable response.') }
    if (!data || typeof data !== 'object') throw new Error('The prediction service returned an invalid response.')
    const raw = data as Record<string, unknown>
    if (raw.prediction !== 'phishing' && raw.prediction !== 'legitimate') throw new Error('The prediction service returned an unexpected classification.')
    const probability = (key: string): number | undefined => {
      const value = raw[key]
      if (value === undefined || value === null) return undefined
      if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) throw new Error('The prediction service returned invalid probability values.')
      return value
    }
    const reasons = raw.reasons === undefined ? undefined :
      Array.isArray(raw.reasons) && raw.reasons.every((reason) => typeof reason === 'string') ? raw.reasons as string[] :
        (() => { throw new Error('The prediction service returned invalid explanation data.') })()
    return { prediction: raw.prediction, legitimate_probability: probability('legitimate_probability'), phishing_probability: probability('phishing_probability'), reasons }
  } finally { window.clearTimeout(timer) }
}
