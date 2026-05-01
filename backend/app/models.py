from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DeploymentStatus(str, Enum):
    pending = "pending"
    provisioning = "provisioning"
    running = "running"
    degraded = "degraded"
    failed = "failed"
    rolled_back = "rolled_back"


class CloudProvider(str, Enum):
    aws = "aws"
    azure = "azure"
    gcp = "gcp"
    onprem = "onprem"


class RoutingStrategy(str, Enum):
    single = "single"
    ab_test = "ab_test"
    failover = "failover"


class ModelVersion(BaseModel):
    version: str
    image_uri: str
    traffic_weight: int = Field(default=100, ge=0, le=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeploymentRequest(BaseModel):
    model_name: str = Field(min_length=2)
    artifact_path: str
    runtime: Literal["pytorch", "tensorflow", "onnx", "custom"] = "custom"
    gpu: str = Field(default="A100")
    region: str = Field(default="us-west")
    cloud: CloudProvider = CloudProvider.aws
    replicas: int = Field(default=1, ge=1, le=1000)
    min_replicas: int = Field(default=1, ge=1, le=1000)
    max_replicas: int = Field(default=10, ge=1, le=1000)
    service_mesh: bool = True
    canary_percent: int = Field(default=0, ge=0, le=100)
    labels: Dict[str, str] = Field(default_factory=dict)
    failover_regions: List[str] = Field(default_factory=list)


class DeploymentSummary(BaseModel):
    deployment_id: str = Field(default_factory=lambda: str(uuid4()))
    model_name: str
    status: DeploymentStatus = DeploymentStatus.pending
    region: str
    cloud: CloudProvider
    gpu: str
    endpoint: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeploymentDetail(DeploymentSummary):
    runtime: str
    replicas: int
    min_replicas: int
    max_replicas: int
    routing_strategy: RoutingStrategy = RoutingStrategy.single
    versions: List[ModelVersion] = Field(default_factory=list)
    active_version: Optional[str] = None
    failover_group_id: Optional[str] = None
    is_primary: bool = True
    canary_percent: int = Field(default=0, ge=0, le=100)
    mesh_enabled: bool = False
    health_checks: Dict[str, str] = Field(
        default_factory=lambda: {
            "readiness": "/healthz/ready",
            "liveness": "/healthz/live",
        }
    )
    restart_policy: str = "Always"
    audit_trail: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(
        default_factory=lambda: {
            "gpu_utilization": 0.0,
            "inference_latency_ms": 0.0,
            "throughput_rps": 0.0,
            "estimated_hourly_cost": 0.0,
        }
    )
    logs: List[str] = Field(default_factory=list)


class RollbackRequest(BaseModel):
    target_version: str


class CanaryUpdateRequest(BaseModel):
    canary_percent: int = Field(ge=0, le=100)


class MeshUpdateRequest(BaseModel):
    enabled: bool


class ABTestRequest(BaseModel):
    challenger_version: str
    challenger_image_uri: str
    challenger_weight: int = Field(ge=1, le=99)


class Alert(BaseModel):
    severity: Literal["info", "warning", "critical"]
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlatformOverview(BaseModel):
    deployments: List[DeploymentSummary]
    total_gpu_utilization: float
    total_throughput_rps: float
    monthly_cost_estimate: float
    alerts: List[Alert]
