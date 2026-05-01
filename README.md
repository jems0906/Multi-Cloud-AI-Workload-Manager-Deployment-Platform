# Multi-Cloud AI Workload Manager & Deployment Platform

FlexAI is an end-to-end control plane for deploying AI inference services across AWS, Azure, GCP, and hybrid/on-prem environments. The repository ships a runnable MVP with a Python CLI/SDK, a FastAPI deployment engine, Terraform-based infrastructure definitions, Kubernetes manifests, and a React dashboard.

## Included Components

- `backend/`: FastAPI deployment engine with deployment lifecycle APIs, model versioning, rollback, A/B routing, autoscaling metadata, health policies, logs, metrics, alerts, and audit trail storage.
- `cli/`: Python CLI/SDK exposing `flexai deploy`, `list`, `status`, `logs`, `rollback`, `ab-test`, and `overview`.
- `dashboard/`: React + TypeScript control plane for UI-driven deployment, live fleet monitoring, cost tracking, and multi-cloud health views.
- `infra/terraform/`: Cloud-agnostic Terraform blueprint for registry, cluster, mesh, observability, and security controls.
- `k8s/`: Example Kubernetes deployment, HPA, and service mesh routing manifest.
- `examples/vision-service/`: Sample model artifact and lightweight serving app for local demos.

## Architecture

1. The CLI authenticates the user, checks the artifact path, and generates a container Dockerfile inline if needed.
2. The backend accepts deployment requests and records provisioning, rollout, metrics, logs, and audit events.
3. Terraform describes multi-cloud control-plane resources including registry, GPU cluster capacity, mesh routing, observability, and security controls.
4. The dashboard polls the backend to present deployment health, cost, alerts, traffic distribution, and aggregated logs.

## Quick Start

### 1. Start the backend

```powershell
Set-Location "backend"
py -3 -m venv ..\.venv
..\.venv\Scripts\pip install -r requirements.txt
..\.venv\Scripts\uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the dashboard

```powershell
Set-Location "dashboard"
npm install
npm run dev
```

### 3. Use the CLI

```powershell
Set-Location "cli"
py -3 -m pip install -e .
flexai deploy ..\examples\vision-service\model.ckpt --model-name vision-prod --gpu A100 --region us-west --cloud aws --replicas 4
```

### 4. Inspect the platform

- Dashboard: `http://127.0.0.1:5173`
- Backend API docs: `http://127.0.0.1:8000/docs`

## Example CLI Commands

```powershell
flexai list
flexai status <deployment-id>
flexai logs <deployment-id>
flexai rollback <deployment-id> --target-version v1
flexai ab-test <deployment-id> --version v2 --image registry.flexai.local/aws/vision-prod:v2 --weight 15
flexai overview
```

## Terraform Usage

```powershell
Set-Location "infra/terraform"
terraform init
terraform plan -var="model_name=vision-prod" -var="image_uri=registry.flexai.local/aws/vision-prod:latest"
```

## Security & Compliance Coverage

- Encryption in transit and at rest modeled in Terraform security controls.
- RBAC and audit logging represented in platform policy outputs and backend audit trail.
- Hybrid/on-prem support via `cloud=onprem` and `hybrid_mode=true` Terraform variable.
- Full deployment audit trail stored per deployment in the backend state.

## Validation Notes

This repository was scaffolded for local execution. The backend and CLI are directly runnable. The Terraform layer is provider-agnostic and uses `terraform_data` resources to model the orchestration flow without requiring live cloud credentials.
