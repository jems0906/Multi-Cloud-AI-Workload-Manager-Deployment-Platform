"""failover.py — Multi-region failover group logic.

A *failover group* is a set of deployments of the same model that share a
``failover_group_id``.  Exactly one member is designated *primary* (its
``is_primary`` flag is ``True``); the rest are *standbys*.

When the primary enters a ``failed`` or ``degraded`` state the caller can
invoke ``promote_standby`` to atomically:
  1. Elect the healthiest standby as the new primary.
  2. Scale the new primary to match the old primary's replica count via the
     cloud provider adapter.
  3. Mark the old primary as non-primary and retain its ``failed`` status.

Public API
----------
find_failover_group(store, group_id) -> FailoverGroup | None
select_healthy_standby(standbys)     -> DeploymentDetail | None
promote_standby(store, get_provider, primary, standby) -> None
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .models import DeploymentDetail
    from .store import DeploymentStore

_logger = logging.getLogger(__name__)

# Statuses that make a deployment eligible for failover promotion.
_UNHEALTHY_STATUSES = {"failed", "degraded"}


@dataclass
class FailoverGroup:
    primary: DeploymentDetail
    standbys: list[DeploymentDetail] = field(default_factory=list)


def find_failover_group(store: DeploymentStore, group_id: str) -> FailoverGroup | None:
    """Return the :class:`FailoverGroup` for *group_id*, or ``None`` if absent."""
    members = [d for d in store.list() if d.failover_group_id == group_id]
    if not members:
        return None
    primaries = [d for d in members if d.is_primary]
    if not primaries:
        return None
    primary = primaries[0]
    standbys = [d for d in members if not d.is_primary]
    return FailoverGroup(primary=primary, standbys=standbys)


def select_healthy_standby(standbys: list[DeploymentDetail]) -> DeploymentDetail | None:
    """Return the first standby with status *running*, or ``None``."""
    from .models import DeploymentStatus

    for standby in standbys:
        if standby.status == DeploymentStatus.running:
            return standby
    return None


def promote_standby(
    store: DeploymentStore,
    get_provider: Callable,
    primary: DeploymentDetail,
    standby: DeploymentDetail,
) -> None:
    """Promote *standby* to primary and demote *primary*.

    The standby is scaled to match the old primary's replica count via the
    cloud provider adapter.  Provider errors are caught and logged so that
    the store update always completes.
    """
    from datetime import datetime, timezone

    from .models import DeploymentStatus

    target_replicas = primary.replicas

    # --- Scale the new primary via its cloud provider ---
    provider = get_provider(standby.cloud.value)
    meta = {
        "provider": standby.cloud.value,
        "region": standby.region,
        "model_name": standby.model_name,
    }
    try:
        provider.scale(standby.deployment_id, meta, target_replicas)
        _logger.info(
            "[failover] Scaled standby %s to %d replicas",
            standby.deployment_id, target_replicas,
        )
    except RuntimeError as exc:
        _logger.warning(
            "[failover] Provider scale skipped for %s (%s): %s",
            standby.deployment_id, standby.cloud.value, exc,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.error("[failover] Scale error for %s: %s", standby.deployment_id, exc)

    now = datetime.now(timezone.utc)

    # --- Promote standby ---
    standby.is_primary = True
    standby.replicas = target_replicas
    standby.status = DeploymentStatus.running
    standby.updated_at = now
    standby.audit_trail.append(
        f"[failover] Promoted to primary (replaced {primary.deployment_id} in {primary.region})"
    )
    standby.logs.append(
        f"[failover] Traffic shifted from {primary.region}/{primary.cloud.value} "
        f"→ {standby.region}/{standby.cloud.value}"
    )
    store.upsert(standby)

    # --- Demote old primary ---
    primary.is_primary = False
    primary.updated_at = now
    primary.audit_trail.append(
        f"[failover] Demoted — standby {standby.deployment_id} "
        f"({standby.region}/{standby.cloud.value}) is now primary"
    )
    store.upsert(primary)
