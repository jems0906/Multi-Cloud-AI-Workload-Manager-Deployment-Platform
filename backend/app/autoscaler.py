"""autoscaler.py — Metrics-driven horizontal autoscaling for deployments.

The autoscaler evaluates each *running* deployment after every health-poll
cycle and decides whether to scale up, scale down, or do nothing.

Scaling policy (configurable via environment variables)
-------------------------------------------------------
FLEXAI_AS_GPU_SCALE_UP      GPU utilisation % above which to scale up  (default 75)
FLEXAI_AS_GPU_SCALE_DOWN    GPU utilisation % below which to scale down (default 20)
FLEXAI_AS_RPS_SCALE_UP      Throughput RPS above which to scale up      (default 100)
FLEXAI_AS_RPS_SCALE_DOWN    Throughput RPS below which to scale down     (default 5)
FLEXAI_AS_COOLDOWN          Seconds between successive scale decisions   (default 120)

Decision logic (either metric can trigger scale-up; both must be below
threshold to trigger scale-down):

1. If replicas < max_replicas AND (gpu_util > scale_up_gpu OR rps > scale_up_rps)
   → scale up by 1 replica.
2. Elif replicas > min_replicas AND gpu_util < scale_down_gpu AND rps < scale_down_rps
   → scale down by 1 replica.
3. Else → no change.

A per-deployment cooldown is tracked in ``_cooldown_tracker`` (deployment_id →
datetime of last scale action) so we don't flap.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .models import DeploymentDetail
    from .store import DeploymentStore

_logger = logging.getLogger(__name__)


@dataclass
class AutoscalerConfig:
    gpu_scale_up: float = 75.0
    gpu_scale_down: float = 20.0
    rps_scale_up: float = 100.0
    rps_scale_down: float = 5.0
    cooldown_seconds: int = 120

    @classmethod
    def from_env(cls) -> "AutoscalerConfig":
        return cls(
            gpu_scale_up=float(os.getenv("FLEXAI_AS_GPU_SCALE_UP", "75")),
            gpu_scale_down=float(os.getenv("FLEXAI_AS_GPU_SCALE_DOWN", "20")),
            rps_scale_up=float(os.getenv("FLEXAI_AS_RPS_SCALE_UP", "100")),
            rps_scale_down=float(os.getenv("FLEXAI_AS_RPS_SCALE_DOWN", "5")),
            cooldown_seconds=int(os.getenv("FLEXAI_AS_COOLDOWN", "120")),
        )


# Module-level cooldown tracker; keyed by deployment_id.
_cooldown_tracker: dict[str, datetime] = {}


def _is_on_cooldown(deployment_id: str, cooldown_seconds: int) -> bool:
    last = _cooldown_tracker.get(deployment_id)
    if last is None:
        return False
    return (datetime.now(timezone.utc) - last).total_seconds() < cooldown_seconds


def _record_scale_action(deployment_id: str) -> None:
    _cooldown_tracker[deployment_id] = datetime.now(timezone.utc)


def evaluate_autoscale(
    deployment: DeploymentDetail,
    config: AutoscalerConfig,
) -> int | None:
    """Return the desired replica count, or ``None`` if no change is needed.

    Parameters
    ----------
    deployment:
        The deployment to evaluate.
    config:
        Autoscaling thresholds and cooldown configuration.

    Returns
    -------
    int | None
        New replica count if a scale action is warranted, ``None`` otherwise.
    """
    from .models import DeploymentStatus

    if deployment.status != DeploymentStatus.running:
        return None

    if _is_on_cooldown(deployment.deployment_id, config.cooldown_seconds):
        return None

    gpu = deployment.metrics.get("gpu_utilization", 0.0)
    rps = deployment.metrics.get("throughput_rps", 0.0)
    current = deployment.replicas

    should_up = gpu > config.gpu_scale_up or rps > config.rps_scale_up
    should_down = gpu < config.gpu_scale_down and rps < config.rps_scale_down

    if should_up and current < deployment.max_replicas:
        return current + 1
    if should_down and current > deployment.min_replicas:
        return current - 1
    return None


def apply_autoscale(
    deployment: DeploymentDetail,
    new_replicas: int,
    store: DeploymentStore,
    get_provider: Callable,
) -> None:
    """Execute the scale action: call the provider and persist the updated record.

    Provider errors are caught and logged so the store update completes
    regardless of SDK availability.
    """
    from datetime import datetime, timezone

    provider = get_provider(deployment.cloud.value)
    meta = {
        "provider": deployment.cloud.value,
        "region": deployment.region,
        "model_name": deployment.model_name,
    }
    direction = "up" if new_replicas > deployment.replicas else "down"

    try:
        provider.scale(deployment.deployment_id, meta, new_replicas)
        _logger.info(
            "[autoscaler] Scale-%s %s %d → %d replicas",
            direction, deployment.deployment_id, deployment.replicas, new_replicas,
        )
    except RuntimeError as exc:
        _logger.warning(
            "[autoscaler] Provider scale skipped for %s (%s): %s",
            deployment.deployment_id, deployment.cloud.value, exc,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.error("[autoscaler] Scale error for %s: %s", deployment.deployment_id, exc)

    old_replicas = deployment.replicas
    cost_per_replica = deployment.metrics.get("estimated_hourly_cost", 0.0) / max(old_replicas, 1)

    deployment.replicas = new_replicas
    deployment.metrics["estimated_hourly_cost"] = round(cost_per_replica * new_replicas, 2)
    deployment.updated_at = datetime.now(timezone.utc)
    deployment.audit_trail.append(
        f"[autoscaler] Scale-{direction}: {old_replicas} → {new_replicas} replicas "
        f"(gpu={deployment.metrics.get('gpu_utilization', 0):.1f}%, "
        f"rps={deployment.metrics.get('throughput_rps', 0):.1f})"
    )
    deployment.logs.append(
        f"[autoscaler] Replica count adjusted {old_replicas} → {new_replicas}"
    )

    _record_scale_action(deployment.deployment_id)
    store.upsert(deployment)
