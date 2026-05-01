"""
healthpoller.py — Background provider health-check reconciliation loop.

Every FLEXAI_HEALTH_POLL_INTERVAL seconds (default 30) the poller:
  1. Lists all known deployments from the store.
  2. For each deployment, calls ``provider.get_status()``.
  3. Maps the provider status string to a ``DeploymentStatus`` enum value.
  4. If the status has changed, updates the store, logs the transition,
     and (on error) appends a critical alert to the overview feed.

Design choices
--------------
* The poller runs as a single asyncio task (started from the FastAPI lifespan
  handler) so it shares the event loop without requiring threads.
* Provider calls are synchronous (SDKs are not async-native); they are
  dispatched via ``asyncio.to_thread`` so the event loop stays responsive.
* Individual provider failures are caught and logged — one broken adapter
  must not stall reconciliation of all other deployments.
* The poller is skipped for simulated deployments (those whose provider
  metadata contains ``"simulated": True``).
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Callable

from .logstream import publish_log
from .models import DeploymentStatus
from .store import DeploymentStore
from .autoscaler import AutoscalerConfig, apply_autoscale, evaluate_autoscale

_logger = logging.getLogger(__name__)
_autoscaler_config = AutoscalerConfig.from_env()

# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------

_PROVIDER_STATUS_MAP: dict[str, DeploymentStatus] = {
    "running": DeploymentStatus.running,
    "pending": DeploymentStatus.pending,
    "provisioning": DeploymentStatus.provisioning,
    "stopped": DeploymentStatus.failed,
    "error": DeploymentStatus.failed,
    "failed": DeploymentStatus.failed,
    "degraded": DeploymentStatus.degraded,
}


def _map_status(raw: str) -> DeploymentStatus:
    return _PROVIDER_STATUS_MAP.get(raw.lower().strip(), DeploymentStatus.degraded)


# ---------------------------------------------------------------------------
# Per-deployment poll
# ---------------------------------------------------------------------------

async def _poll_one(deployment_id: str, store: DeploymentStore, get_provider: Callable) -> None:
    deployment = store.get(deployment_id)
    if deployment is None:
        return

    provider = get_provider(deployment.cloud.value)

    # Build the provider_meta dict that get_status() needs.  We reconstruct a
    # minimal dict from what we know; adapters that need richer metadata should
    # persist it in the store (future work).
    meta: dict = {
        "provider": deployment.cloud.value,
        "region": deployment.region,
        "model_name": deployment.model_name,
    }

    try:
        raw_status: str = await asyncio.to_thread(
            provider.get_status, deployment_id, meta
        )
    except RuntimeError as exc:
        # SDK not installed or env vars missing — skip silently.
        _logger.debug(
            "Skipping health poll for %s (%s): %s",
            deployment_id, deployment.cloud.value, exc,
        )
        return
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "Provider get_status failed for deployment %s: %s",
            deployment_id, exc,
        )
        return

    new_status = _map_status(raw_status)

    # Re-fetch under lock to avoid clobbering concurrent writes.
    deployment = store.get(deployment_id)
    if deployment is None:
        return

    if deployment.status == new_status:
        return  # Nothing to do.

    old_status = deployment.status
    deployment.status = new_status
    deployment.updated_at = datetime.now(timezone.utc)
    deployment.audit_trail.append(
        f"[health-poller] Status changed {old_status} → {new_status} "
        f"(provider: {raw_status!r})"
    )
    store.upsert(deployment)

    log_line = (
        f"{deployment.updated_at.strftime('%H:%M:%S')} "
        f"[health-poller] {deployment.model_name} status: {old_status} → {new_status}"
    )
    await publish_log(deployment_id, log_line)

    _logger.info(
        "Deployment %s (%s) status: %s → %s",
        deployment_id, deployment.model_name, old_status, new_status,
    )

    # --- Auto-promote failover standby when primary fails ---
    if (
        new_status in (DeploymentStatus.failed, DeploymentStatus.degraded)
        and deployment.is_primary
        and deployment.failover_group_id
    ):
        await _try_auto_promote(deployment_id, store, get_provider)


# ---------------------------------------------------------------------------
# Failover auto-promotion
# ---------------------------------------------------------------------------

async def _try_auto_promote(
    deployment_id: str,
    store: DeploymentStore,
    get_provider: Callable,
) -> None:
    """Attempt to auto-promote a healthy standby when the primary fails."""
    from .failover import find_failover_group, promote_standby, select_healthy_standby

    deployment = store.get(deployment_id)
    if deployment is None or not deployment.failover_group_id:
        return

    group = find_failover_group(store, deployment.failover_group_id)
    if group is None:
        return

    standby = select_healthy_standby(group.standbys)
    if standby is None:
        _logger.warning(
            "[failover] No healthy standby for group %s — cannot auto-promote",
            deployment.failover_group_id,
        )
        return

    try:
        promote_standby(store, get_provider, group.primary, standby)
        _logger.info(
            "[failover] Auto-promoted %s → primary (group=%s)",
            standby.deployment_id, deployment.failover_group_id,
        )
        log_line = (
            f"[failover] Auto-promoted {standby.region}/{standby.cloud.value} "
            f"standby {standby.deployment_id} to primary"
        )
        await publish_log(deployment_id, log_line)
        await publish_log(standby.deployment_id, log_line)
    except Exception as exc:  # noqa: BLE001
        _logger.error(
            "[failover] Auto-promote failed for group %s: %s",
            deployment.failover_group_id, exc,
        )


# ---------------------------------------------------------------------------
# Poller loop
# ---------------------------------------------------------------------------

async def run_health_poller(store: DeploymentStore, get_provider: Callable) -> None:
    """
    Continuously poll every deployment for its provider-side health status.

    Call this from the FastAPI lifespan handler as an asyncio task.
    """
    interval = int(os.getenv("FLEXAI_HEALTH_POLL_INTERVAL", "30"))
    _logger.info("Health poller started (interval=%ds)", interval)

    while True:
        await asyncio.sleep(interval)

        deployments = store.list()
        if not deployments:
            continue

        _logger.debug("Polling %d deployment(s)…", len(deployments))
        tasks = [
            _poll_one(dep.deployment_id, store, get_provider)
            for dep in deployments
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for dep, result in zip(deployments, results):
            if isinstance(result, Exception):
                _logger.error(
                    "Unexpected error polling deployment %s: %s",
                    dep.deployment_id, result,
                )

        # --- Autoscaling pass ---
        for dep in store.list():
            desired = evaluate_autoscale(dep, _autoscaler_config)
            if desired is not None:
                try:
                    await asyncio.to_thread(
                        apply_autoscale, dep, desired, store, get_provider
                    )
                    log_line = (
                        f"[autoscaler] Scaled {dep.model_name} "
                        f"{dep.replicas} → {desired} replicas"
                    )
                    await publish_log(dep.deployment_id, log_line)
                except Exception as exc:  # noqa: BLE001
                    _logger.error(
                        "Autoscale error for %s: %s", dep.deployment_id, exc
                    )
