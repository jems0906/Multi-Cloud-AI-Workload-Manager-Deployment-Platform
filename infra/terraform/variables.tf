variable "model_name" {
  type        = string
  description = "Model service name"
}

variable "image_uri" {
  type        = string
  description = "Container image for the model"
}

variable "cloud" {
  type        = string
  description = "Target cloud or onprem"
  default     = "aws"
}

variable "region" {
  type        = string
  description = "Deployment region"
  default     = "us-west"
}

variable "gpu_type" {
  type        = string
  description = "GPU SKU"
  default     = "A100"
}

variable "node_count" {
  type        = number
  default     = 3
}

variable "desired_replicas" {
  type        = number
  default     = 2
}

variable "min_replicas" {
  type        = number
  default     = 1
}

variable "max_replicas" {
  type        = number
  default     = 20
}

variable "canary_percent" {
  type        = number
  default     = 0
}

variable "hybrid_mode" {
  type    = bool
  default = true
}

# ---------------------------------------------------------------------------
# AWS credentials (optional — falls back to env vars / instance role)
# ---------------------------------------------------------------------------
variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "aws_access_key" {
  type      = string
  default   = ""
  sensitive = true
}

variable "aws_secret_key" {
  type      = string
  default   = ""
  sensitive = true
}

# ---------------------------------------------------------------------------
# Azure credentials (optional — falls back to env vars / managed identity)
# ---------------------------------------------------------------------------
variable "azure_subscription_id" {
  type      = string
  default   = ""
  sensitive = true
}

variable "azure_client_id" {
  type      = string
  default   = ""
  sensitive = true
}

variable "azure_client_secret" {
  type      = string
  default   = ""
  sensitive = true
}

variable "azure_tenant_id" {
  type      = string
  default   = ""
  sensitive = true
}

# ---------------------------------------------------------------------------
# GCP credentials (optional — falls back to ADC / Workload Identity)
# ---------------------------------------------------------------------------
variable "gcp_project" {
  type    = string
  default = ""
}

variable "gcp_region" {
  type    = string
  default = ""
}

variable "gcp_credentials_json" {
  type      = string
  default   = ""
  sensitive = true
  description = "Path to a GCP service account JSON key file, or inline JSON."
}

# ---------------------------------------------------------------------------
# Kubernetes (on-prem / hybrid)
# ---------------------------------------------------------------------------
variable "kubeconfig_path" {
  type    = string
  default = ""
}

variable "kubeconfig_context" {
  type    = string
  default = ""
}
