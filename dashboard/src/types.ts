export type Cloud = 'aws' | 'azure' | 'gcp' | 'onprem'

export interface DeploymentStatusSummary {
  deployment_id: string
  status: string
  updated_at: string
  model_name: string
  cloud: Cloud
  region: string
}

export interface DeploymentSummary {
  deployment_id: string
  model_name: string
  status: string
  region: string
  cloud: Cloud
  gpu: string
  endpoint: string
  created_at: string
  updated_at: string
}

export interface ModelVersion {
  version: string
  image_uri: string
  traffic_weight: number
  created_at: string
}

export interface DeploymentDetail extends DeploymentSummary {
  runtime: string
  replicas: number
  min_replicas: number
  max_replicas: number
  routing_strategy: string
  versions: ModelVersion[]
  active_version: string | null
  canary_percent: number
  mesh_enabled: boolean
  failover_group_id: string | null
  is_primary: boolean
  audit_trail: string[]
  logs: string[]
  metrics: {
    gpu_utilization: number
    inference_latency_ms: number
    throughput_rps: number
    estimated_hourly_cost: number
  }
}

export interface Alert {
  severity: 'info' | 'warning' | 'critical'
  message: string
  created_at: string
}

export interface Overview {
  deployments: DeploymentSummary[]
  total_gpu_utilization: number
  total_throughput_rps: number
  monthly_cost_estimate: number
  alerts: Alert[]
}

export interface BudgetStatus {
  global_limit: number | null
  global_current: number
  global_pct: number | null
  global_headroom: number | null
  cloud_limits: Record<string, number>
  cloud_current: Record<string, number>
}

export interface AuthAuditEvent {
  timestamp: string
  method: string
  path: string
  decision: 'allow' | 'deny'
  detail: string
  required_role: string
  resolved_role: string | null
  status_code: number
  token_fingerprint: string | null
  client: string | null
}

export interface AuthIdentity {
  subject: string | null
  role: 'viewer' | 'operator' | 'admin' | string
}
