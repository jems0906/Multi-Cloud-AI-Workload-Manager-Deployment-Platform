import { FormEvent, startTransition, useEffect, useRef, useState } from 'react'
import type { AuthAuditEvent, AuthIdentity, BudgetStatus, DeploymentDetail, DeploymentStatusSummary, Overview } from './types'

const API_URL = (globalThis as typeof globalThis & { FLEXAI_API_URL?: string }).FLEXAI_API_URL ?? 'http://127.0.0.1:8000'
const DEFAULT_TOKEN =
  (globalThis as typeof globalThis & { FLEXAI_TOKEN?: string }).FLEXAI_TOKEN ??
  window.localStorage.getItem('FLEXAI_TOKEN') ??
  'local-dev-token'

const initialForm = {
  model_name: 'vision-prod',
  artifact_path: '/models/vision-prod/model.ckpt',
  gpu: 'A100',
  region: 'us-west',
  cloud: 'aws',
  replicas: 4,
}

const roleRank: Record<string, number> = {
  unknown: 0,
  viewer: 1,
  operator: 2,
  admin: 3,
}

export default function App() {
  const [overview, setOverview] = useState<Overview | null>(null)
  const [selectedDeployment, setSelectedDeployment] = useState<DeploymentDetail | null>(null)
  const [authEvents, setAuthEvents] = useState<AuthAuditEvent[]>([])
  const [authIdentity, setAuthIdentity] = useState<AuthIdentity | null>(null)
  const [auditDecision, setAuditDecision] = useState<'all' | 'allow' | 'deny'>('all')
  const [auditPathFilter, setAuditPathFilter] = useState('')
  const [auditError, setAuditError] = useState<string | null>(null)
  const [apiToken, setApiToken] = useState(DEFAULT_TOKEN)
  const [tokenDraft, setTokenDraft] = useState(DEFAULT_TOKEN)
  const [form, setForm] = useState(initialForm)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [activePolicyTip, setActivePolicyTip] = useState<'deploy' | 'compliance' | null>(null)
  const [liveLog, setLiveLog] = useState<string[]>([])
  const logEndRef = useRef<HTMLDivElement>(null)
  const [lastChecked, setLastChecked] = useState<DeploymentStatusSummary | null>(null)
  const [scaleReplicas, setScaleReplicas] = useState<number>(1)
  const [scaling, setScaling] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [budgetStatus, setBudgetStatus] = useState<BudgetStatus | null>(null)
  const [canaryDraft, setCanaryDraft] = useState<number>(0)
  const [updatingCanary, setUpdatingCanary] = useState(false)
  const [updatingMesh, setUpdatingMesh] = useState(false)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState<{ artifact_id: string; location: string } | null>(null)

  const currentRole = authIdentity?.role ?? 'unknown'
  const canDeploy = (roleRank[currentRole] ?? 0) >= roleRank.operator
  const canViewAudit = (roleRank[currentRole] ?? 0) >= roleRank.admin

  function authHeaders(): HeadersInit {
    return {
      Authorization: `Bearer ${apiToken}`,
      'Content-Type': 'application/json',
    }
  }

  function applyToken() {
    const normalized = tokenDraft.trim() || 'local-dev-token'
    setApiToken(normalized)
    window.localStorage.setItem('FLEXAI_TOKEN', normalized)
    setAuditError(null)
    setError(null)
  }

  function clearToken() {
    setTokenDraft('local-dev-token')
    setApiToken('local-dev-token')
    window.localStorage.removeItem('FLEXAI_TOKEN')
    setAuditError(null)
    setError(null)
  }

  function requestElevatedToken() {
    setTokenDraft('local-dev-token')
  }

  function togglePolicyTip(target: 'deploy' | 'compliance') {
    setActivePolicyTip((prev) => (prev === target ? null : target))
  }

  // WebSocket live log tail
  useEffect(() => {
    const depId = selectedDeployment?.deployment_id
    if (!depId) {
      setLiveLog([])
      return
    }
    setLiveLog([])

    const wsBase = API_URL.replace(/^http/, 'ws')
    const url = `${wsBase}/ws/deployments/${depId}/logs?token=${encodeURIComponent(apiToken)}`
    const ws = new WebSocket(url)

    ws.onmessage = (event: MessageEvent<string>) => {
      setLiveLog((prev) => {
        const next = [...prev, event.data]
        return next.length > 200 ? next.slice(-200) : next
      })
    }
    ws.onerror = () => {
      // Connection error — silently degrade; static logs still visible
    }

    return () => {
      ws.close()
    }
  }, [selectedDeployment?.deployment_id, apiToken])

  // Auto-scroll log pane when new lines arrive
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [liveLog])

  // Poll /api/deployments/{id}/status every 15 s for the last-checked badge
  useEffect(() => {
    const depId = selectedDeployment?.deployment_id
    if (!depId) {
      setLastChecked(null)
      return
    }
    let mounted = true
    const poll = async () => {
      try {
        const res = await apiFetch(`/api/deployments/${depId}/status`)
        if (!mounted || !res.ok) return
        const data: DeploymentStatusSummary = await res.json()
        if (mounted) setLastChecked(data)
      } catch {
        // Silently degrade
      }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 15_000)
    return () => {
      mounted = false
      window.clearInterval(timer)
    }
  }, [selectedDeployment?.deployment_id, apiToken])

  useEffect(() => {
    if (!activePolicyTip) return
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setActivePolicyTip(null)
    }
    function handleOutside(e: MouseEvent | TouchEvent) {
      const target = e.target as Element
      if (!target.closest('.policy-tip')) setActivePolicyTip(null)
    }
    document.addEventListener('keydown', handleKey)
    document.addEventListener('mousedown', handleOutside)
    document.addEventListener('touchstart', handleOutside)
    return () => {
      document.removeEventListener('keydown', handleKey)
      document.removeEventListener('mousedown', handleOutside)
      document.removeEventListener('touchstart', handleOutside)
    }
  }, [activePolicyTip])

  async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
    const headers = {
      ...authHeaders(),
      ...(init?.headers ?? {}),
    }
    return fetch(`${API_URL}${path}`, { ...init, headers })
  }

  async function loadAuthEvents() {
    try {
      const params = new URLSearchParams()
      params.set('limit', '20')
      if (auditDecision !== 'all') {
        params.set('decision', auditDecision)
      }
      if (auditPathFilter.trim()) {
        params.set('path_contains', auditPathFilter.trim())
      }

      const response = await apiFetch(`/api/auth/audit?${params.toString()}`)
      if (response.status === 403) {
        setAuditError('Audit feed is admin-only. Use an admin token to view compliance events.')
        setAuthEvents([])
        return
      }
      if (!response.ok) {
        throw new Error(`Unable to load auth audit feed (${response.status})`)
      }
      const data: AuthAuditEvent[] = await response.json()
      setAuthEvents(data)
      setAuditError(null)
    } catch (auditLoadError) {
      setAuditError(auditLoadError instanceof Error ? auditLoadError.message : 'Unknown audit error')
      setAuthEvents([])
    }
  }

  async function loadIdentity() {
    try {
      const response = await apiFetch('/api/auth/whoami')
      if (!response.ok) {
        setAuthIdentity(null)
        return
      }
      const data: AuthIdentity = await response.json()
      setAuthIdentity(data)
    } catch {
      setAuthIdentity(null)
    }
  }

  useEffect(() => {
    let mounted = true

    const load = async () => {
      try {
        const response = await apiFetch('/api/overview')
        if (!response.ok) {
          throw new Error('Unable to load overview')
        }
        const data: Overview = await response.json()
        if (!mounted) {
          return
        }
        startTransition(() => {
          setOverview(data)
          setLoading(false)
        })
        void loadIdentity()
        void loadAuthEvents()
        // Fetch budget status (operator+ only; silently skip if 403/not configured)
        apiFetch('/api/budget').then((br) => {
          if (br.ok) br.json().then((bd: BudgetStatus) => setBudgetStatus(bd)).catch(() => null)
        }).catch(() => null)
        if (!selectedDeployment && data.deployments[0]) {
          void selectDeployment(data.deployments[0].deployment_id)
        }
      } catch (loadError) {
        if (!mounted) {
          return
        }
        setError(loadError instanceof Error ? loadError.message : 'Unknown error')
        setLoading(false)
      }
    }

    void load()
    const interval = window.setInterval(() => {
      void load()
      if (selectedDeployment) {
        void selectDeployment(selectedDeployment.deployment_id)
      }
    }, 5000)

    return () => {
      mounted = false
      window.clearInterval(interval)
    }
  }, [selectedDeployment?.deployment_id, auditDecision, auditPathFilter, apiToken])

  async function selectDeployment(deploymentId: string) {
    const response = await apiFetch(`/api/deployments/${deploymentId}`)
    if (!response.ok) {
      throw new Error('Unable to load deployment detail')
    }
    const data: DeploymentDetail = await response.json()
    startTransition(() => {
      setSelectedDeployment(data)
      setScaleReplicas(data.replicas)
    })
  }

  async function handleScale() {
    if (!selectedDeployment) return
    setScaling(true)
    setError(null)
    try {
      const res = await apiFetch(`/api/deployments/${selectedDeployment.deployment_id}/scale`, {
        method: 'PATCH',
        body: JSON.stringify({ replicas: scaleReplicas }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `Scale failed (${res.status})`)
      }
      const data: DeploymentDetail = await res.json()
      setSelectedDeployment(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scale failed')
    } finally {
      setScaling(false)
    }
  }

  async function handleDelete() {
    if (!selectedDeployment) return
    if (!window.confirm(`Delete deployment "${selectedDeployment.model_name}"? This tears down all cloud resources.`)) return
    setDeleting(true)
    setError(null)
    try {
      const res = await apiFetch(`/api/deployments/${selectedDeployment.deployment_id}`, { method: 'DELETE' })
      if (res.status !== 204) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `Delete failed (${res.status})`)
      }
      setSelectedDeployment(null)
      setLastChecked(null)
      setLiveLog([])
      // Refresh overview
      const ov = await apiFetch('/api/overview')
      if (ov.ok) setOverview(await ov.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed')
    } finally {
      setDeleting(false)
    }
  }

  async function handleCanaryUpdate() {
    if (!selectedDeployment) return
    setUpdatingCanary(true)
    setError(null)
    try {
      const res = await apiFetch(`/api/deployments/${selectedDeployment.deployment_id}/canary`, {
        method: 'PATCH',
        body: JSON.stringify({ canary_percent: canaryDraft }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail ?? `Canary update failed (${res.status})`)
      }
      const data: DeploymentDetail = await res.json()
      setSelectedDeployment(data)
      setCanaryDraft(data.canary_percent)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Canary update failed')
    } finally {
      setUpdatingCanary(false)
    }
  }

  async function handleMeshToggle() {
    if (!selectedDeployment) return
    setUpdatingMesh(true)
    setError(null)
    try {
      const res = await apiFetch(`/api/deployments/${selectedDeployment.deployment_id}/mesh`, {
        method: 'PATCH',
        body: JSON.stringify({ enabled: !selectedDeployment.mesh_enabled }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail ?? `Mesh update failed (${res.status})`)
      }
      const data: DeploymentDetail = await res.json()
      setSelectedDeployment(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Mesh update failed')
    } finally {
      setUpdatingMesh(false)
    }
  }

  async function handleUpload() {
    if (!uploadFile) return
    setUploading(true)
    setError(null)
    setUploadResult(null)
    try {
      const formData = new FormData()
      formData.append('file', uploadFile)
      const res = await fetch(`${API_URL}/api/artifacts/upload`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${apiToken}` },
        body: formData,
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error((body as { detail?: string }).detail ?? `Upload failed (${res.status})`)
      }
      const data = await res.json() as { artifact_id: string; location: string }
      setUploadResult(data)
      setForm((prev) => ({ ...prev, artifact_path: data.location }))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!canDeploy) {
      setError('Deploy requires operator or admin role.')
      return
    }
    setCreating(true)
    setError(null)

    try {
      const response = await apiFetch('/api/deployments', {
        method: 'POST',
        body: JSON.stringify({
          ...form,
          runtime: 'custom',
          min_replicas: 1,
          max_replicas: Math.max(form.replicas * 4, 4),
          canary_percent: 0,
        }),
      })
      if (!response.ok) {
        throw new Error('Unable to create deployment')
      }
      const data: DeploymentDetail = await response.json()
      setSelectedDeployment(data)
      setForm(initialForm)
      const overviewResponse = await apiFetch('/api/overview')
      const overviewData: Overview = await overviewResponse.json()
      setOverview(overviewData)
      void loadAuthEvents()
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Unknown error')
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Multi-Cloud AI Workload Manager & Deployment Platform</p>
          <h1>Ship GPU-backed models in under five minutes.</h1>
          <p className="lede">
            One control plane for artifact packaging, Kubernetes provisioning, traffic shaping,
            observability, and compliance across AWS, Azure, GCP, and hybrid clusters.
          </p>
        </div>
        <div className="hero-panel">
          <span>Live policy status</span>
          <strong>Encryption, RBAC, audit trail, and SLA alerts enabled</strong>
          <div className="role-badge-wrap">
            <span className={`role-badge role-${authIdentity?.role ?? 'unknown'}`}>
              Role: {(authIdentity?.role ?? 'unknown').toUpperCase()}
            </span>
            {authIdentity?.subject ? <small>Subject: {authIdentity.subject}</small> : null}
          </div>
          <div className="token-controls">
            <label>
              API Token
              <input
                type="password"
                value={tokenDraft}
                onChange={(event) => setTokenDraft(event.target.value)}
                placeholder="Bearer token"
              />
            </label>
            <div className="token-actions">
              <button type="button" className="secondary" onClick={applyToken}>Apply</button>
              <button type="button" className="secondary" onClick={clearToken}>Reset</button>
            </div>
          </div>
          {!canViewAudit ? (
            <div className="elevation-hint">
              <p>Need admin-only actions? Request an elevated token from platform security.</p>
              <button type="button" className="secondary" onClick={requestElevatedToken}>Use Local Admin Token Template</button>
            </div>
          ) : null}
          <div className="permission-matrix">
            <h4>Permissions</h4>
            <table>
              <tbody>
                <tr>
                  <td>View Fleet</td>
                  <td>Viewer+</td>
                </tr>
                <tr>
                  <td>Deploy Models</td>
                  <td>Operator+</td>
                </tr>
                <tr>
                  <td>Rollback / A/B Admin Ops</td>
                  <td>Admin</td>
                </tr>
                <tr>
                  <td>Compliance Audit Feed</td>
                  <td>Admin</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="stats-grid">
        <MetricCard label="Fleet GPU" value={`${overview?.total_gpu_utilization ?? 0}%`} accent="amber" />
        <MetricCard label="Throughput" value={`${overview?.total_throughput_rps ?? 0} rps`} accent="mint" />
        <MetricCard label="Monthly Cost" value={`$${overview?.monthly_cost_estimate ?? 0}`} accent="blue" />
        <MetricCard label="Active Deployments" value={`${overview?.deployments.length ?? 0}`} accent="rose" />
      </section>

      {budgetStatus && budgetStatus.global_limit !== null ? (
        <section className="budget-bar-section">
          <div className="budget-bar-header">
            <span className="budget-bar-label">Hourly Budget</span>
            <span className="budget-bar-numbers">
              ${budgetStatus.global_current.toFixed(2)} / ${budgetStatus.global_limit.toFixed(2)}
              {budgetStatus.global_pct !== null
                ? <span className={`budget-pct ${budgetStatus.global_pct >= 90 ? 'budget-pct-critical' : budgetStatus.global_pct >= 70 ? 'budget-pct-warn' : ''}`}>{budgetStatus.global_pct}%</span>
                : null}
            </span>
          </div>
          <div className="budget-track">
            <progress
              className={`budget-meter ${(budgetStatus.global_pct ?? 0) >= 90 ? 'budget-fill-critical' : (budgetStatus.global_pct ?? 0) >= 70 ? 'budget-fill-warn' : 'budget-fill-ok'}`}
              max={100}
              value={Math.min(budgetStatus.global_pct ?? 0, 100)}
            />
          </div>
          {Object.keys(budgetStatus.cloud_limits).length > 0 ? (
            <div className="budget-cloud-row">
              {Object.entries(budgetStatus.cloud_limits).map(([cloud, limit]) => {
                const current = budgetStatus.cloud_current[cloud] ?? 0
                const pct = limit > 0 ? Math.min((current / limit) * 100, 100) : 0
                return (
                  <div key={cloud} className="budget-cloud-chip">
                    <span className="budget-cloud-name">{cloud.toUpperCase()}</span>
                    <span className="budget-cloud-val">${current.toFixed(2)}/${limit.toFixed(2)}</span>
                    <div className="budget-mini-track">
                      <progress
                        className={`budget-meter ${pct >= 90 ? 'budget-fill-critical' : pct >= 70 ? 'budget-fill-warn' : 'budget-fill-ok'}`}
                        max={100}
                        value={pct}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          ) : null}
        </section>
      ) : null}

      <main className="grid">
        <section className="panel form-panel">
          <div className="panel-header">
            <h2>Deploy from UI</h2>
            <p>Upload path, pick GPU, and target the cheapest viable region.</p>
          </div>
          <form onSubmit={handleSubmit} className="deploy-form">
            <label>
              Model Name
              <input value={form.model_name} onChange={(event) => setForm({ ...form, model_name: event.target.value })} />
            </label>
            <label>
              Artifact Path
              <input value={form.artifact_path} onChange={(event) => setForm({ ...form, artifact_path: event.target.value })} />
            </label>
            <div className="split">
              <label>
                GPU
                <select value={form.gpu} onChange={(event) => setForm({ ...form, gpu: event.target.value })}>
                  <option>A100</option>
                  <option>H100</option>
                  <option>L4</option>
                  <option>T4</option>
                </select>
              </label>
              <label>
                Region
                <select value={form.region} onChange={(event) => setForm({ ...form, region: event.target.value })}>
                  <option>us-west</option>
                  <option>us-east</option>
                  <option>europe-west</option>
                  <option>asia-south</option>
                </select>
              </label>
            </div>
            <div className="split">
              <label>
                Cloud
                <select value={form.cloud} onChange={(event) => setForm({ ...form, cloud: event.target.value })}>
                  <option value="aws">AWS</option>
                  <option value="azure">Azure</option>
                  <option value="gcp">GCP</option>
                  <option value="onprem">On-Prem</option>
                </select>
              </label>
              <label>
                Replicas
                <input type="number" min={1} max={1000} value={form.replicas} onChange={(event) => setForm({ ...form, replicas: Number(event.target.value) })} />
              </label>
            </div>

            {canDeploy ? (
              <div className="upload-section">
                <label>
                  Upload Model Artifact
                  <input
                    type="file"
                    accept=".ckpt,.pt,.bin,.onnx,.safetensors,.pkl,.tar,.tar.gz,.zip"
                    onChange={(e) => {
                      setUploadFile(e.target.files?.[0] ?? null)
                      setUploadResult(null)
                    }}
                  />
                </label>
                <button
                  type="button"
                  className="secondary"
                  disabled={uploading || !uploadFile}
                  onClick={() => void handleUpload()}
                >
                  {uploading ? 'Uploading…' : 'Upload Artifact'}
                </button>
                {uploadResult ? (
                  <p className="upload-success">
                    ✓ Uploaded — artifact <code>{uploadResult.artifact_id.slice(0, 8)}</code> stored at{' '}
                    <code>{uploadResult.location}</code>. Artifact path auto-filled above.
                  </p>
                ) : null}
              </div>
            ) : null}

            <button type="submit" disabled={creating || !canDeploy}>
              {creating ? 'Deploying...' : canDeploy ? 'Launch Deployment' : 'Deploy Locked'}
            </button>
            {!canDeploy ? (
              <p className="access-note">
                Deploy requires operator or admin role.
                <button
                  type="button"
                  className={`policy-tip ${activePolicyTip === 'deploy' ? 'active' : ''}`}
                  data-tip="Policy: role >= operator. Endpoint: POST /api/deployments"
                  aria-label="Deploy policy details"
                  onClick={() => togglePolicyTip('deploy')}
                  onBlur={() => setActivePolicyTip(null)}
                >
                  i
                </button>
              </p>
            ) : null}
          </form>
        </section>

        <section className="panel fleet-panel">
          <div className="panel-header">
            <h2>Multi-Cloud Fleet</h2>
            <p>Current workload distribution and health.</p>
          </div>

          {overview && overview.deployments.length > 0 ? (
            <div className="cloud-distribution">
              <h3>Cloud Distribution</h3>
              {(['aws', 'azure', 'gcp', 'onprem'] as const).map((cloud) => {
                const count = overview.deployments.filter((d) => d.cloud === cloud).length
                const pct = overview.deployments.length > 0 ? Math.round((count / overview.deployments.length) * 100) : 0
                if (count === 0) return null
                return (
                  <div key={cloud} className="cloud-dist-row">
                    <span className="cloud-dist-label">{cloud.toUpperCase()}</span>
                    <div className="cloud-dist-track">
                      <progress className={`cloud-dist-meter cloud-dist-${cloud}`} max={100} value={pct} />
                    </div>
                    <span className="cloud-dist-count">{count} ({pct}%)</span>
                  </div>
                )
              })}
            </div>
          ) : null}

          {overview && overview.deployments.length > 0 ? (
            <div className="cost-breakdown">
              <h3>Per-Model Cost (est. hourly)</h3>
              <table className="cost-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Cloud</th>
                    <th>Region</th>
                    <th>$/hr</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.deployments.map((d) => {
                    const hourly = (overview.monthly_cost_estimate / Math.max(overview.deployments.length, 1) / 730)
                    return (
                      <tr key={d.deployment_id}>
                        <td>{d.model_name}</td>
                        <td>{d.cloud.toUpperCase()}</td>
                        <td>{d.region}</td>
                        <td>${hourly.toFixed(2)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          ) : null}

          <div className="deployment-list">
            {loading ? <p>Loading deployments...</p> : null}
            {overview?.deployments.map((deployment) => (
              <button
                key={deployment.deployment_id}
                className={`deployment-card ${selectedDeployment?.deployment_id === deployment.deployment_id ? 'selected' : ''}`}
                onClick={() => void selectDeployment(deployment.deployment_id)}
              >
                <div>
                  <strong>{deployment.model_name}</strong>
                  <span>{deployment.cloud.toUpperCase()} · {deployment.region}</span>
                </div>
                <div>
                  <span className={`pill pill-${deployment.status}`}>{deployment.status}</span>
                  <small>{deployment.gpu}</small>
                </div>
              </button>
            ))}
          </div>
          <div className="alerts">
            <h3>Alerts</h3>
            {(overview?.alerts.length ?? 0) === 0 ? <p>No active alerts.</p> : null}
            {overview?.alerts.map((alert) => (
              <div key={`${alert.message}-${alert.created_at}`} className={`alert alert-${alert.severity}`}>
                <strong>{alert.severity.toUpperCase()}</strong>
                <span>{alert.message}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="panel detail-panel">
          <div className="panel-header">
            <h2>Inference Monitor</h2>
            <p>Metrics, rollout state, logs, and audit history.</p>
          </div>
          <div className="detail-body">
          {selectedDeployment ? (
            <>
              <div className="detail-topline">
                <div>
                  <p className="eyebrow">Endpoint</p>
                  <strong>{selectedDeployment.endpoint}</strong>
                </div>
                <div>
                  <p className="eyebrow">Auto-scaling</p>
                  <strong>{selectedDeployment.min_replicas} to {selectedDeployment.max_replicas} replicas</strong>
                </div>
                <div>
                  <p className="eyebrow">Hourly Cost</p>
                  <strong>${selectedDeployment.metrics.estimated_hourly_cost}</strong>
                </div>
              </div>

              {lastChecked ? (
                <div className="health-badge">
                  <span className={`pill pill-${lastChecked.status}`}>{lastChecked.status}</span>
                  <span className="health-checked-at">
                    Provider status last reconciled at {new Date(lastChecked.updated_at).toLocaleTimeString()}
                  </span>
                </div>
              ) : null}

              {canDeploy ? (
                <div className="canary-mesh-bar">
                  <div className="canary-control">
                    <label>
                      Canary Traffic
                      <input
                        type="range"
                        min={0}
                        max={100}
                        step={5}
                        value={canaryDraft}
                        onChange={(e) => setCanaryDraft(Number(e.target.value))}
                      />
                      <span className="canary-pct">{canaryDraft}%</span>
                    </label>
                    <button
                      type="button"
                      className="secondary"
                      disabled={updatingCanary || canaryDraft === selectedDeployment.canary_percent}
                      onClick={() => void handleCanaryUpdate()}
                    >
                      {updatingCanary ? 'Updating…' : 'Set Canary'}
                    </button>
                  </div>
                  <div className="mesh-control">
                    <span>Service Mesh</span>
                    <button
                      type="button"
                      className={`mesh-toggle ${selectedDeployment.mesh_enabled ? 'mesh-on' : 'mesh-off'}`}
                      disabled={updatingMesh}
                      onClick={() => void handleMeshToggle()}
                    >
                      {updatingMesh ? '…' : selectedDeployment.mesh_enabled ? 'Enabled' : 'Disabled'}
                    </button>
                  </div>
                </div>
              ) : null}

              {canDeploy ? (
                <div className="scale-delete-bar">
                  <div className="scale-control">
                    <label>
                      Replicas
                      <input
                        type="number"
                        min={selectedDeployment.min_replicas}
                        max={selectedDeployment.max_replicas}
                        value={scaleReplicas}
                        onChange={(e) => setScaleReplicas(Number(e.target.value))}
                      />
                      <span className="scale-range">
                        ({selectedDeployment.min_replicas}–{selectedDeployment.max_replicas})
                      </span>
                    </label>
                    <button
                      type="button"
                      className="secondary"
                      disabled={scaling || scaleReplicas === selectedDeployment.replicas}
                      onClick={() => void handleScale()}
                    >
                      {scaling ? 'Scaling…' : 'Apply Scale'}
                    </button>
                  </div>
                  {(roleRank[currentRole] ?? 0) >= roleRank.admin ? (
                    <button
                      type="button"
                      className="danger"
                      disabled={deleting}
                      onClick={() => void handleDelete()}
                    >
                      {deleting ? 'Deleting…' : 'Teardown'}
                    </button>
                  ) : null}
                </div>
              ) : null}

              <div className="bars">
                <Bar label="GPU Utilization" value={selectedDeployment.metrics.gpu_utilization} suffix="%" />
                <Bar label="Inference Latency" value={selectedDeployment.metrics.inference_latency_ms} suffix=" ms" max={300} />
                <Bar label="Throughput" value={selectedDeployment.metrics.throughput_rps} suffix=" rps" max={120} />
              </div>

              <div className="columns">
                <div>
                  <h3>Model Versions</h3>
                  {selectedDeployment.versions.map((version) => (
                    <div key={version.version} className="version-row">
                      <div>
                        <strong>{version.version}</strong>
                        <span>{version.image_uri}</span>
                      </div>
                      <span>{version.traffic_weight}% traffic</span>
                    </div>
                  ))}
                </div>
                <div>
                  <h3>Audit Trail</h3>
                  <ul>
                    {selectedDeployment.audit_trail.map((entry) => (
                      <li key={entry}>{entry}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div>
                <h3>Live Logs</h3>
                <div className="log-pane">
                  {liveLog.length === 0
                    ? (selectedDeployment.logs.length > 0
                        ? selectedDeployment.logs.map((l, i) => <div key={i} className="log-line">{l}</div>)
                        : <span className="log-connecting">Connecting to log stream…</span>)
                    : liveLog.map((l, i) => <div key={i} className="log-line">{l}</div>)
                  }
                  <div ref={logEndRef} />
                </div>
              </div>
            </>
          ) : (
            <p>Select a deployment to inspect live metrics.</p>
          )}
          </div>
        </section>

        <section className="panel compliance-panel">
          <div className="panel-header">
            <h2>Compliance Audit Feed</h2>
            <p>Admin-only auth decisions with role and endpoint context.</p>
          </div>

          {canViewAudit ? (
            <>
              <div className="audit-controls">
                <label>
                  Decision
                  <select
                    value={auditDecision}
                    onChange={(event) => setAuditDecision(event.target.value as 'all' | 'allow' | 'deny')}
                  >
                    <option value="all">All</option>
                    <option value="allow">Allow</option>
                    <option value="deny">Deny</option>
                  </select>
                </label>
                <label>
                  Path Filter
                  <input
                    value={auditPathFilter}
                    onChange={(event) => setAuditPathFilter(event.target.value)}
                    placeholder="e.g. rollback"
                  />
                </label>
                <button type="button" className="secondary" onClick={() => void loadAuthEvents()}>
                  Refresh Feed
                </button>
              </div>

              {auditError ? <div className="error-banner">{auditError}</div> : null}

              <div className="audit-table-wrap">
                <table className="audit-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Decision</th>
                      <th>Role</th>
                      <th>Method</th>
                      <th>Path</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {authEvents.map((event) => (
                      <tr key={`${event.timestamp}-${event.path}-${event.status_code}`}>
                        <td>{new Date(event.timestamp).toLocaleTimeString()}</td>
                        <td>
                          <span className={`pill pill-${event.decision}`}>{event.decision}</span>
                        </td>
                        <td>{event.resolved_role ?? 'n/a'} → {event.required_role}</td>
                        <td>{event.method}</td>
                        <td>{event.path}</td>
                        <td>{event.status_code}</td>
                      </tr>
                    ))}
                    {authEvents.length === 0 ? (
                      <tr>
                        <td colSpan={6}>No audit events for current filter.</td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="audit-locked">
              <p>
                This panel is visible only to admin role tokens.
                <button
                  type="button"
                  className={`policy-tip ${activePolicyTip === 'compliance' ? 'active' : ''}`}
                  data-tip="Policy: role = admin. Endpoint: GET /api/auth/audit"
                  aria-label="Compliance policy details"
                  onClick={() => togglePolicyTip('compliance')}
                  onBlur={() => setActivePolicyTip(null)}
                >
                  i
                </button>
              </p>
              <p className="access-note">
                Apply an admin token above to unlock compliance analytics.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

function MetricCard({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className={`metric-card accent-${accent}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function Bar({ label, value, suffix, max = 100 }: { label: string; value: number; suffix: string; max?: number }) {
  const boundedValue = Math.min(value, max)
  return (
    <div className="bar-card">
      <div className="bar-meta">
        <span>{label}</span>
        <strong>{value}{suffix}</strong>
      </div>
      <progress className="track" value={boundedValue} max={max} />
    </div>
  )
}
