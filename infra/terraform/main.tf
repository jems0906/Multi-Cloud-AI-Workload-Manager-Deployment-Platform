terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# ---------------------------------------------------------------------------
# AWS provider — enabled when var.cloud == "aws"
# Credentials via env vars (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY),
# ~/.aws/credentials, or an IAM instance role.
# ---------------------------------------------------------------------------
provider "aws" {
  region     = var.aws_region
  access_key = var.aws_access_key != "" ? var.aws_access_key : null
  secret_key = var.aws_secret_key != "" ? var.aws_secret_key : null

  default_tags {
    tags = {
      ManagedBy = "flexai-terraform"
      Model     = var.model_name
    }
  }
}

# ---------------------------------------------------------------------------
# Azure provider — enabled when var.cloud == "azure"
# Credentials via AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID,
# a managed identity, or `az login`.
# ---------------------------------------------------------------------------
provider "azurerm" {
  features {}

  subscription_id = var.azure_subscription_id != "" ? var.azure_subscription_id : null
  client_id       = var.azure_client_id != "" ? var.azure_client_id : null
  client_secret   = var.azure_client_secret != "" ? var.azure_client_secret : null
  tenant_id       = var.azure_tenant_id != "" ? var.azure_tenant_id : null
}

# ---------------------------------------------------------------------------
# GCP provider — enabled when var.cloud == "gcp"
# Credentials via GOOGLE_APPLICATION_CREDENTIALS, Workload Identity, or
# `gcloud auth application-default login`.
# ---------------------------------------------------------------------------
provider "google" {
  project     = var.gcp_project != "" ? var.gcp_project : null
  region      = var.gcp_region != "" ? var.gcp_region : var.region
  credentials = var.gcp_credentials_json != "" ? var.gcp_credentials_json : null
}

# ---------------------------------------------------------------------------
# Kubernetes provider — used for on-prem / hybrid deployments
# Reads kubeconfig from KUBE_CONFIG_PATH or in-cluster service account.
# ---------------------------------------------------------------------------
provider "kubernetes" {
  config_path    = var.kubeconfig_path != "" ? var.kubeconfig_path : null
  config_context = var.kubeconfig_context != "" ? var.kubeconfig_context : null
}

locals {
  deployment_id = "${var.model_name}-${var.cloud}-${var.region}"
  security_profile = {
    encryption_in_transit = true
    encryption_at_rest    = true
    rbac_enabled          = true
    audit_logging         = true
  }
  observability = {
    metrics   = ["gpu_utilization", "inference_latency_ms", "throughput_rps"]
    logs      = ["application", "audit", "kubernetes"]
    tracing   = true
    alerting  = ["sla_violation", "resource_exhaustion"]
  }
}

resource "terraform_data" "container_registry" {
  input = {
    name          = "flexai-registry-${var.cloud}-${var.region}"
    artifact_store = "enabled"
    retention_days = 30
  }
}

resource "terraform_data" "cluster" {
  input = {
    cloud         = var.cloud
    region        = var.region
    gpu_type      = var.gpu_type
    node_count    = var.node_count
    autoscaling   = true
    hybrid_mode   = var.hybrid_mode
    service_mesh  = true
  }
}

module "kubernetes_service" {
  source = "./modules/kubernetes-service"

  model_name      = var.model_name
  image_uri       = var.image_uri
  desired_replicas = var.desired_replicas
  min_replicas    = var.min_replicas
  max_replicas    = var.max_replicas
  canary_percent  = var.canary_percent
}

resource "terraform_data" "security_controls" {
  input = local.security_profile
}

resource "terraform_data" "observability_stack" {
  input = local.observability
}
