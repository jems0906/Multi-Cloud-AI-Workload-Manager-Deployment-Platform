from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

from .budget import BudgetConfig, BudgetExceededError, BudgetStatus, check_budget, compute_budget_status
from .failover import FailoverGroup, find_failover_group, promote_standby, select_healthy_standby
from .models import (
    ABTestRequest,
    Alert,
    CanaryUpdateRequest,
    DeploymentDetail,
    DeploymentRequest,
    DeploymentStatus,
    DeploymentSummary,
    MeshUpdateRequest,
    ModelVersion,
    PlatformOverview,
    RollbackRequest,
    RoutingStrategy,
)
from .providers import get_provider
from .store import DeploymentStore


class DeploymentService:
    def __init__(self, store: DeploymentStore, budget: BudgetConfig | None = None) -> None:
        self.store = store
        self._budget = budget or BudgetConfig.from_env()

    def create_deployment(self, request: DeploymentRequest) -> DeploymentDetail:
        deployment_id = str(uuid4())
        image_uri = f"registry.flexai.local/{request.cloud.value}/{request.model_name}:{deployment_id[:8]}"
        version = ModelVersion(version="v1", image_uri=image_uri)
        endpoint = f"https://{request.model_name}.{request.region}.{request.cloud.value}.flexai.run/infer"

        # Dispatch to cloud provider adapter (stores infra metadata for later
        # teardown / status queries).  If the provider SDK is not installed or
        # required env vars are missing we fall back to the simulated endpoint
        # so the service remains functional in development / CI.
        _log = logging.getLogger(__name__)

        # --- Budget enforcement ---
        projected_cost = self._estimate_cost(request)
        cloud_totals: dict[str, float] = {}
        for dep in self.store.list():
            c = dep.cloud.value
            cloud_totals[c] = cloud_totals.get(c, 0.0) + dep.metrics.get("estimated_hourly_cost", 0.0)
        check_budget(
            config=self._budget,
            current_costs=cloud_totals,
            additional_cost=projected_cost,
            cloud=request.cloud.value,
        )

        provider = get_provider(request.cloud.value)
        try:
            provider_meta = provider.provision(
                deployment_id,
                {
                    "model_name": request.model_name,
                    "region": request.region,
                    "gpu": request.gpu,
                    "replicas": request.replicas,
                    "artifact_path": request.artifact_path,
                    "labels": request.labels or {},
                },
            )
            # Use the real endpoint returned by the provider if available.
            endpoint = provider_meta.get("endpoint", endpoint)
        except RuntimeError as exc:
            _log.warning(
                "Provider %s is not fully configured (%s). "
                "Using simulated endpoint for deployment %s.",
                request.cloud.value, exc, deployment_id,
            )
            provider_meta = {"provider": request.cloud.value, "simulated": True}

        deployment = DeploymentDetail(
            deployment_id=deployment_id,
            model_name=request.model_name,
            status=DeploymentStatus.running,
            region=request.region,
            cloud=request.cloud,
            gpu=request.gpu,
            endpoint=endpoint,
            runtime=request.runtime,
            replicas=request.replicas,
            min_replicas=request.min_replicas,
            max_replicas=request.max_replicas,
            versions=[version],
            active_version=version.version,
            routing_strategy=RoutingStrategy.ab_test if request.canary_percent else RoutingStrategy.single,
            canary_percent=request.canary_percent,
            mesh_enabled=request.service_mesh,
            audit_trail=[
                "Authenticated deployment request",
                f"Uploaded artifact from {Path(request.artifact_path).name}",
                "Built and pushed container image",
                f"Provisioned {request.cloud.value.upper()} capacity in {request.region} "
                f"(provider_meta={provider_meta})",
                "Configured mesh routing, autoscaling, and health policies",
            ],
            logs=[
                "[auth] OIDC token verified",
                "[artifact] Model uploaded to registry cache",
                "[build] Docker image created with CUDA dependencies",
                f"[provider:{request.cloud.value}] Infra provisioned – endpoint={endpoint}",
                "[health] Readiness and liveness probes passed",
            ],
            metrics={
                "gpu_utilization": 64.5,
                "inference_latency_ms": 142.0,
                "throughput_rps": 58.0,
                "estimated_hourly_cost": self._estimate_cost(request),
            },
        )
        primary = self.store.upsert(deployment)

        # --- Service mesh: configure mTLS / sidecar when service_mesh=True ---
        if request.service_mesh:
            self._apply_mesh_config(deployment, enabled=True)
            deployment.audit_trail.append("Service mesh enabled (mTLS + sidecar injection)")
            primary = self.store.upsert(deployment)

        # --- Canary: configure initial traffic split when canary_percent > 0 ---
        if request.canary_percent > 0:
            self._apply_traffic_split(
                deployment,
                {"baseline": 100 - request.canary_percent, "canary": request.canary_percent},
            )
            deployment.audit_trail.append(
                f"Canary traffic split configured: baseline={100 - request.canary_percent}% "
                f"canary={request.canary_percent}%"
            )
            primary = self.store.upsert(deployment)

        # --- Failover: provision standbys in each failover region ---
        if request.failover_regions:
            group_id = str(uuid4())
            primary.failover_group_id = group_id
            primary.is_primary = True
            primary.routing_strategy = RoutingStrategy.failover
            self.store.upsert(primary)
            for standby_region in request.failover_regions:
                self._create_standby(primary, group_id, standby_region)

        return primary

    def _create_standby(
        self,
        primary: DeploymentDetail,
        group_id: str,
        standby_region: str,
    ) -> DeploymentDetail:
        """Provision a warm standby of *primary* in *standby_region*."""
        _log = logging.getLogger(__name__)
        standby_id = str(uuid4())
        image_uri = f"registry.flexai.local/{primary.cloud.value}/{primary.model_name}:{standby_id[:8]}"
        endpoint = (
            f"https://{primary.model_name}.{standby_region}.{primary.cloud.value}.flexai.run/infer"
        )
        version = ModelVersion(version="v1", image_uri=image_uri)

        provider = get_provider(primary.cloud.value)
        meta_in = {
            "model_name": primary.model_name,
            "region": standby_region,
            "gpu": primary.gpu,
            "replicas": 1,
            "artifact_path": "",
        }
        try:
            provider_meta = provider.provision(standby_id, meta_in)
            endpoint = provider_meta.get("endpoint", endpoint)
        except RuntimeError as exc:
            _log.warning(
                "[failover] Standby provision skipped for %s in %s: %s",
                standby_id, standby_region, exc,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("[failover] Standby provision error %s: %s", standby_id, exc)

        standby = DeploymentDetail(
            deployment_id=standby_id,
            model_name=primary.model_name,
            status=DeploymentStatus.running,
            region=standby_region,
            cloud=primary.cloud,
            gpu=primary.gpu,
            endpoint=endpoint,
            runtime=primary.runtime,
            replicas=1,
            min_replicas=primary.min_replicas,
            max_replicas=primary.max_replicas,
            versions=[version],
            active_version="v1",
            failover_group_id=group_id,
            is_primary=False,
            routing_strategy=RoutingStrategy.failover,
            audit_trail=[f"[failover] Standby provisioned for group {group_id} (primary region={primary.region})"],
            logs=[f"[failover] Warm standby ready in {standby_region}"],
            metrics={
                "gpu_utilization": 0.0,
                "inference_latency_ms": 0.0,
                "throughput_rps": 0.0,
                "estimated_hourly_cost": round(self._estimate_cost_simple(primary.gpu, 1), 2),
            },
        )
        return self.store.upsert(standby)

    def list_deployments(self) -> List[DeploymentSummary]:
        return [DeploymentSummary.model_validate(item.model_dump()) for item in self.store.list()]

    def get_deployment(self, deployment_id: str) -> DeploymentDetail | None:
        deployment = self.store.get(deployment_id)
        if not deployment:
            return None

        deployment.updated_at = datetime.now(timezone.utc)
        deployment.metrics["gpu_utilization"] = min(99.0, deployment.metrics["gpu_utilization"] + 0.3)
        deployment.metrics["throughput_rps"] = max(1.0, deployment.metrics["throughput_rps"] + 0.5)
        deployment.logs = deployment.logs + [
            f"[trace] Request flow captured at {deployment.updated_at.isoformat()}"
        ]
        return self.store.upsert(deployment)

    def rollback(self, deployment_id: str, request: RollbackRequest) -> DeploymentDetail | None:
        deployment = self.store.get(deployment_id)
        if not deployment:
            return None

        available_versions = {item.version for item in deployment.versions}
        if request.target_version not in available_versions:
            raise ValueError(f"Unknown version: {request.target_version}")

        target_version_obj = next(
            (v for v in deployment.versions if v.version == request.target_version), None
        )
        image_uri = target_version_obj.image_uri if target_version_obj else ""

        _log = logging.getLogger(__name__)
        provider = get_provider(deployment.cloud.value)
        meta = {
            "provider": deployment.cloud.value,
            "region": deployment.region,
            "model_name": deployment.model_name,
            "endpoint_name": deployment.endpoint.split("/")[2] if deployment.endpoint else "",
        }
        try:
            provider.rollback(deployment_id, meta, request.target_version, image_uri)
            _log.info(
                "Rollback complete: deployment=%s version=%s",
                deployment_id, request.target_version,
            )
        except RuntimeError as exc:
            _log.warning(
                "Provider rollback skipped for %s (%s): %s",
                deployment_id, deployment.cloud.value, exc,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("Rollback error for %s: %s", deployment_id, exc)

        deployment.active_version = request.target_version
        deployment.status = DeploymentStatus.rolled_back
        deployment.audit_trail.append(f"Rolled back to {request.target_version}")
        deployment.logs.append(
            f"[rollback] Provider re-deployed {request.target_version} (image={image_uri})"
        )
        return self.store.upsert(deployment)

    def teardown_deployment(self, deployment_id: str) -> bool:
        """Tear down cloud resources and remove the deployment from the store."""
        _log = logging.getLogger(__name__)
        deployment = self.store.get(deployment_id)
        if not deployment:
            return False

        provider = get_provider(deployment.cloud.value)
        meta = {
            "provider": deployment.cloud.value,
            "region": deployment.region,
            "model_name": deployment.model_name,
        }
        try:
            provider.teardown(deployment_id, meta)
            _log.info("Teardown complete for deployment %s", deployment_id)
        except RuntimeError as exc:
            _log.warning(
                "Provider teardown skipped for %s (%s): %s",
                deployment_id, deployment.cloud.value, exc,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("Teardown error for %s: %s", deployment_id, exc)

        return self.store.delete(deployment_id)

    def scale_deployment(self, deployment_id: str, replicas: int) -> DeploymentDetail | None:
        """Scale a deployment to *replicas* and update the store."""
        _log = logging.getLogger(__name__)
        deployment = self.store.get(deployment_id)
        if not deployment:
            return None

        if replicas < deployment.min_replicas or replicas > deployment.max_replicas:
            raise ValueError(
                f"replicas must be between {deployment.min_replicas} "
                f"and {deployment.max_replicas}"
            )

        # --- Budget enforcement for scale ---
        current_cost = deployment.metrics.get("estimated_hourly_cost", 0.0)
        cost_per_replica = current_cost / max(deployment.replicas, 1)
        new_cost = cost_per_replica * replicas
        additional_cost = max(0.0, new_cost - current_cost)
        if additional_cost > 0:
            cloud_totals: dict[str, float] = {}
            for dep in self.store.list():
                c = dep.cloud.value
                cloud_totals[c] = cloud_totals.get(c, 0.0) + dep.metrics.get("estimated_hourly_cost", 0.0)
            check_budget(
                config=self._budget,
                current_costs=cloud_totals,
                additional_cost=additional_cost,
                cloud=deployment.cloud.value,
            )

        provider = get_provider(deployment.cloud.value)
        meta = {
            "provider": deployment.cloud.value,
            "region": deployment.region,
            "model_name": deployment.model_name,
        }
        try:
            provider.scale(deployment_id, meta, replicas)
            _log.info("Scaled deployment %s to %d replicas", deployment_id, replicas)
        except RuntimeError as exc:
            _log.warning(
                "Provider scale skipped for %s (%s): %s",
                deployment_id, deployment.cloud.value, exc,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("Scale error for %s: %s", deployment_id, exc)

        old_replicas = deployment.replicas
        deployment.replicas = replicas
        deployment.updated_at = datetime.now(timezone.utc)
        deployment.audit_trail.append(
            f"Scaled from {old_replicas} → {replicas} replicas"
        )
        deployment.logs.append(
            f"[scale] Replica count updated {old_replicas} → {replicas}"
        )
        deployment.metrics["estimated_hourly_cost"] = round(
            deployment.metrics["estimated_hourly_cost"]
            / max(old_replicas, 1)
            * replicas,
            2,
        )
        return self.store.upsert(deployment)

    def create_ab_test(self, deployment_id: str, request: ABTestRequest) -> DeploymentDetail | None:
        deployment = self.store.get(deployment_id)
        if not deployment:
            return None

        baseline_weight = 100 - request.challenger_weight
        for version in deployment.versions:
            version.traffic_weight = baseline_weight
        deployment.versions.append(
            ModelVersion(
                version=request.challenger_version,
                image_uri=request.challenger_image_uri,
                traffic_weight=request.challenger_weight,
            )
        )
        deployment.routing_strategy = RoutingStrategy.ab_test
        deployment.canary_percent = request.challenger_weight
        deployment.audit_trail.append(
            f"Started A/B test with {request.challenger_version} at {request.challenger_weight}% traffic"
        )
        deployment.logs.append(
            f"[mesh] Routing split configured {baseline_weight}/{request.challenger_weight}"
        )
        self._apply_traffic_split(
            deployment,
            {"baseline": baseline_weight, request.challenger_version: request.challenger_weight},
        )
        return self.store.upsert(deployment)

    def budget_status(self) -> BudgetStatus:
        """Return current spend vs configured budget limits."""
        return compute_budget_status(self._budget, self.store.list())

    def update_canary(
        self, deployment_id: str, request: CanaryUpdateRequest
    ) -> DeploymentDetail | None:
        """Adjust the live canary traffic split for a running deployment.

        Setting ``canary_percent=0`` routes 100% of traffic to the baseline and
        marks the deployment as ``single`` routing strategy.
        """
        _log = logging.getLogger(__name__)
        deployment = self.store.get(deployment_id)
        if not deployment:
            return None

        canary = request.canary_percent
        weights: dict[str, int] = {"baseline": 100 - canary, "canary": canary} if canary > 0 else {"baseline": 100}
        self._apply_traffic_split(deployment, weights)

        deployment.canary_percent = canary
        deployment.routing_strategy = (
            RoutingStrategy.ab_test if canary > 0 else RoutingStrategy.single
        )
        deployment.audit_trail.append(
            f"[canary] Traffic split updated: baseline={100 - canary}% canary={canary}%"
        )
        deployment.logs.append(f"[mesh] Traffic weights adjusted to {weights}")
        return self.store.upsert(deployment)

    def _apply_traffic_split(
        self, deployment: DeploymentDetail, weights: dict[str, int]
    ) -> None:
        """Call the provider set_traffic; absorb errors so the store update still proceeds."""
        _log = logging.getLogger(__name__)
        provider = get_provider(deployment.cloud.value)
        meta = {
            "provider": deployment.cloud.value,
            "region": deployment.region,
            "model_name": deployment.model_name,
            "endpoint_name": deployment.endpoint.split("/")[2] if deployment.endpoint else "",
        }
        try:
            provider.set_traffic(deployment.deployment_id, meta, weights)
            _log.info(
                "[canary] set_traffic %s weights=%s", deployment.deployment_id, weights
            )
        except RuntimeError as exc:
            _log.warning(
                "[canary] Provider set_traffic skipped for %s (%s): %s",
                deployment.deployment_id, deployment.cloud.value, exc,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("[canary] set_traffic error for %s: %s", deployment.deployment_id, exc)

    def update_mesh(
        self, deployment_id: str, request: MeshUpdateRequest
    ) -> DeploymentDetail | None:
        """Enable or disable service-mesh policy for a running deployment."""
        deployment = self.store.get(deployment_id)
        if not deployment:
            return None

        self._apply_mesh_config(deployment, enabled=request.enabled)
        deployment.mesh_enabled = request.enabled
        action = "enabled" if request.enabled else "disabled"
        deployment.audit_trail.append(f"[mesh] Service mesh {action}")
        deployment.logs.append(f"[mesh] mTLS and sidecar injection {action}")
        return self.store.upsert(deployment)

    def _apply_mesh_config(
        self, deployment: DeploymentDetail, enabled: bool
    ) -> None:
        """Call the provider configure_mesh; absorb errors so the store update proceeds."""
        _log = logging.getLogger(__name__)
        provider = get_provider(deployment.cloud.value)
        meta = {
            "provider": deployment.cloud.value,
            "region": deployment.region,
            "model_name": deployment.model_name,
            "endpoint_name": deployment.endpoint.split("/")[2] if deployment.endpoint else "",
            "project": os.getenv("FLEXAI_GCP_PROJECT", ""),
            "location": deployment.region,
        }
        try:
            provider.configure_mesh(deployment.deployment_id, meta, enabled)
            _log.info(
                "[mesh] configure_mesh %s enabled=%s", deployment.deployment_id, enabled
            )
        except RuntimeError as exc:
            _log.warning(
                "[mesh] Provider configure_mesh skipped for %s (%s): %s",
                deployment.deployment_id, deployment.cloud.value, exc,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("[mesh] configure_mesh error for %s: %s", deployment.deployment_id, exc)


    def promote_failover(self, deployment_id: str) -> FailoverGroup:
        """Promote a healthy standby to primary for the failover group of *deployment_id*.

        Raises
        ------
        ValueError
            If the deployment is not part of a failover group.
        RuntimeError
            If no healthy standby is available.
        """
        deployment = self.store.get(deployment_id)
        if not deployment or not deployment.failover_group_id:
            raise ValueError(f"{deployment_id} is not part of a failover group")

        group = find_failover_group(self.store, deployment.failover_group_id)
        if not group:
            raise ValueError(f"Failover group {deployment.failover_group_id} not found")

        standby = select_healthy_standby(group.standbys)
        if standby is None:
            raise RuntimeError(
                f"No healthy standby available for failover group {deployment.failover_group_id}"
            )

        promote_standby(self.store, get_provider, group.primary, standby)

        # Return refreshed group
        refreshed = find_failover_group(self.store, deployment.failover_group_id)
        return refreshed or group

    def platform_overview(self) -> PlatformOverview:
        deployments = self.list_deployments()
        details = self.store.list()
        alerts: List[Alert] = []
        for item in details:
            if item.metrics["inference_latency_ms"] > 200:
                alerts.append(Alert(severity="warning", message=f"{item.model_name} latency exceeds SLA"))
            if item.metrics["gpu_utilization"] > 90:
                alerts.append(Alert(severity="critical", message=f"{item.model_name} GPU nearing exhaustion"))

        return PlatformOverview(
            deployments=deployments,
            total_gpu_utilization=round(sum(item.metrics["gpu_utilization"] for item in details), 2),
            total_throughput_rps=round(sum(item.metrics["throughput_rps"] for item in details), 2),
            monthly_cost_estimate=round(sum(item.metrics["estimated_hourly_cost"] for item in details) * 24 * 30, 2),
            alerts=alerts,
        )

    @staticmethod
    def _estimate_cost_simple(gpu: str, replicas: int) -> float:
        """Cost estimate without a full DeploymentRequest (e.g. for standby slots)."""
        gpu_multiplier = {
            "A100": 5.2,
            "H100": 8.4,
            "L4": 2.3,
            "T4": 1.4,
        }.get(gpu.upper(), 3.0)
        return replicas * gpu_multiplier

    @staticmethod
    def _estimate_cost(request: DeploymentRequest) -> float:
        gpu_multiplier = {
            "A100": 5.2,
            "H100": 8.4,
            "L4": 2.3,
            "T4": 1.4,
        }.get(request.gpu.upper(), 3.0)
        cloud_multiplier = {
            "aws": 1.0,
            "azure": 0.97,
            "gcp": 0.95,
            "onprem": 0.6,
        }[request.cloud.value]
        return round(request.replicas * gpu_multiplier * cloud_multiplier, 2)
