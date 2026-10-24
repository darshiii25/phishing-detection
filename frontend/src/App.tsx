import { useState, type FormEvent } from 'react'
import { AlertCircle, ArrowUpRight, Check, CheckCircle2, Clipboard, Globe2, LoaderCircle, LockKeyhole, ScanSearch, Shield, ShieldAlert, ShieldCheck, Sparkles, X } from 'lucide-react'
import { analyzeUrl, type ScanResult } from './api'

const sample: ScanResult = {
  prediction: 'phishing', legitimate_probability: 0.08, phishing_probability: 0.92,
  reasons: ['The hostname uses an IP address instead of a registered domain.', 'The URL contains an unusually long path.', 'Multiple nested subdomains obscure the destination.'],
}

function validUrl(value: string): boolean {
  try { const parsed = new URL(value); return ['http:', 'https:'].includes(parsed.protocol) && Boolean(parsed.hostname.includes('.')) }
  catch { return false }
}

function App() {
  const [url, setUrl] = useState('')
  const [result, setResult] = useState<ScanResult | null>(null)
  const [demo, setDemo] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  async function scan(event?: FormEvent) {
    event?.preventDefault()
    setError(''); setNotice(''); setResult(null); setDemo(false)
    const candidate = url.trim()
    if (!candidate) { setError('Enter a URL to analyze.'); return }
    if (!validUrl(candidate)) { setError('Enter a complete URL, including https:// or http://.'); return }
    setLoading(true)
    try { setResult(await analyzeUrl(candidate)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Something went wrong. Please try again.') }
    finally { setLoading(false) }
  }

  async function paste() {
    setNotice(''); setError('')
    if (!navigator.clipboard?.readText) { setNotice('Clipboard access is unavailable. Paste your link manually with Ctrl+V.'); return }
    try { const text = await navigator.clipboard.readText(); if (!text) setNotice('Your clipboard is empty.'); else { setUrl(text.trim()); setResult(null); setNotice('URL pasted from clipboard.') } }
    catch { setNotice('Clipboard permission was denied. Paste your link manually with Ctrl+V.') }
  }

  function showDemo() { setResult(sample); setDemo(true); setError(''); setNotice(''); setUrl('http://192.0.2.42/account/verify-your-information?continue=secure-login') }
  const isPhishing = result?.prediction === 'phishing'
  const safePercent = (value?: number) => value === undefined ? 'Unavailable' : `${Math.round(value * 100)}%`

  return <div className="app-shell">
    <div className="ambient ambient-one"/><div className="ambient ambient-two"/>
    <header className="topbar"><a className="brand" href="#top" aria-label="PhishGuard home"><span className="brand-mark"><Shield size={19} strokeWidth={2.2}/></span><span className="brand-copy"><strong>PhishGuard</strong><small>AI-powered phishing URL detection</small></span></a><div className="top-tag"><span className="status-dot"/>URL SECURITY</div></header>

    <main id="top">
      <section className="hero"><div className="eyebrow"><span className="eyebrow-line"/>LINK INTELLIGENCE, MADE SIMPLE</div><h1>Is this URL <span>safe?</span></h1><p className="hero-subtitle">Check a suspicious link before you open it.</p>
        <div className="scanner-card"><div className="field-label-row"><label htmlFor="url-input">URL to inspect</label><span><LockKeyhole size={12}/> PRIVATE BY DESIGN</span></div>
          <form onSubmit={scan} noValidate><div className={`url-field ${error ? 'field-error' : ''}`}><Globe2 className="input-globe" size={18}/><input id="url-input" type="url" value={url} onChange={(event) => { setUrl(event.target.value); setResult(null); setDemo(false); setError('') }} placeholder="https://example.com" autoComplete="url" aria-describedby={error ? 'form-message' : undefined}/>{url && <button className="icon-button clear-button" type="button" aria-label="Clear URL" onClick={() => { setUrl(''); setResult(null); setDemo(false); setError('') }}><X size={16}/></button>}<button type="button" className="paste-button" onClick={paste}><Clipboard size={15}/> <span>Paste</span></button></div>
            <button className="analyze-button" type="submit" disabled={loading}>{loading ? <><LoaderCircle className="spin" size={17}/> Analyzing URL…</> : <><ScanSearch size={17}/> Analyze URL <ArrowUpRight size={15} className="button-arrow"/></>}</button></form>
          {error && <p id="form-message" className="message error-message" role="alert"><AlertCircle size={15}/>{error}</p>}{notice && <p className="message notice-message" role="status"><Clipboard size={14}/>{notice}</p>}
          <div className="privacy-note"><LockKeyhole size={13}/><span>Your link is analyzed as text. We never open or visit it.</span></div>
        </div>
        <div className="demo-row"><span>Just exploring?</span><button onClick={showDemo} className="demo-button"><Sparkles size={14}/> View sample result</button><span className="demo-caption">Clearly labeled demo data</span></div>
      </section>

      {result && <section className={`result-card reveal ${isPhishing ? 'result-danger' : 'result-safe'}`} aria-live="polite">
        <div className="result-top"><div className={`result-icon ${isPhishing ? 'danger-icon' : 'safe-icon'}`}>{isPhishing ? <ShieldAlert size={21}/> : <ShieldCheck size={21}/>}</div><div className="result-heading"><span className="result-kicker">{demo ? 'ILLUSTRATIVE SAMPLE · DEMO ONLY' : 'ANALYSIS COMPLETE'}</span><h2>{isPhishing ? 'Potential phishing' : 'Likely legitimate'}</h2></div><span className={`result-badge ${isPhishing ? 'badge-danger' : 'badge-safe'}`}><span/> {result.prediction.toUpperCase()}</span></div>
        {demo && <div className="demo-warning"><Sparkles size={14}/> Sample result for interface preview. This is not a model prediction.</div>}
        <div className="result-content"><div className="probabilities"><div className="prob-heading"><span>Model probabilities</span><span className="prob-note">{result.legitimate_probability !== undefined && result.phishing_probability !== undefined ? 'CLASS CONFIDENCE' : 'NOT FULLY PROVIDED'}</span></div>
            <Probability label="Legitimate probability" value={result.legitimate_probability} tone="green" format={safePercent}/><Probability label="Phishing probability" value={result.phishing_probability} tone="red" format={safePercent}/>
          </div><div className="recommendation"><div className="recommendation-label"><CheckCircle2 size={15}/> RECOMMENDATION</div><p>{isPhishing ? 'Avoid opening this link or entering personal information.' : 'Still check the domain carefully before sharing sensitive information.'}</p></div></div>
        <div className="reasons"><div className="reasons-title"><span>Why this result?</span><span className="source-label">{result.reasons?.length ? 'MODEL EXPLANATION' : 'EXPLANATION'}</span></div>{result.reasons?.length ? <ul>{result.reasons.slice(0, 5).map((reason, index) => <li key={`${reason}-${index}`}><span className="reason-bullet"><Check size={12}/></span>{reason}</li>)}</ul> : <p className="no-reasons">Detailed reasons are not available for this prediction yet.</p>}</div>
        <p className="analyzed-url"><span>ANALYZED URL</span><code>{url}</code></p>
      </section>}
      <div className="trust-strip"><div><span className="trust-icon"><Shield size={15}/></span><span><strong>Stay one step ahead</strong><small>Pause before you click.</small></span></div><span className="trust-divider"/><div><span className="trust-icon"><Globe2 size={15}/></span><span><strong>URL-only analysis</strong><small>No site visits or redirects.</small></span></div><span className="trust-divider"/><div><span className="trust-icon"><LockKeyhole size={15}/></span><span><strong>Private by design</strong><small>Your browsing stays yours.</small></span></div></div>
    </main>
    <footer><span>Predictions are model-based estimates and do not guarantee website safety.</span><span className="footer-mark">PHISHGUARD <span>·</span> URL SECURITY</span></footer>
  </div>
}

function Probability({ label, value, tone, format }: { label: string; value?: number; tone: 'green' | 'red'; format: (value?: number) => string }) {
  return <div className="prob-row"><div className="prob-label"><span>{label}</span><strong>{format(value)}</strong></div><div className="progress-track" role="img" aria-label={`${label}: ${format(value)}`}><div className={`progress-fill ${tone}`} style={{ width: value === undefined ? '0%' : `${value * 100}%` }}/></div></div>
}

export default App
