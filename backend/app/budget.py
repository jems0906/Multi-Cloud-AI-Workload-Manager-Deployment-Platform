"""
Budget enforcement for the FlexAI deployment platform.

Limits are configured via environment variables:

  FLEXAI_BUDGET_HOURLY_GLOBAL   – hard cap on total hourly cost across ALL
                                   deployments (default: unlimited)
  FLEXAI_BUDGET_HOURLY_AWS      – per-cloud cap for AWS
  FLEXAI_BUDGET_HOURLY_AZURE    – per-cloud cap for Azure
  FLEXAI_BUDGET_HOURLY_GCP      – per-cloud cap for GCP
  FLEXAI_BUDGET_HOURLY_ONPREM   – per-cloud cap for on-prem

If a variable is absent or empty the corresponding limit is not enforced.
All values are in USD / hour.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _float_env(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float for {name}: {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0, got {value}")
    return value


@dataclass(frozen=True)
class BudgetConfig:
    global_hourly: float | None
    cloud_hourly: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "BudgetConfig":
        cloud_hourly: dict[str, float] = {}
        for provider in ("aws", "azure", "gcp", "onprem"):
            limit = _float_env(f"FLEXAI_BUDGET_HOURLY_{provider.upper()}")
            if limit is not None:
                cloud_hourly[provider] = limit
        return cls(
            global_hourly=_float_env("FLEXAI_BUDGET_HOURLY_GLOBAL"),
            cloud_hourly=cloud_hourly,
        )


@dataclass
class BudgetStatus:
    global_limit: float | None
    global_current: float
    cloud_limits: dict[str, float]
    cloud_current: dict[str, float]

    @property
    def global_headroom(self) -> float | None:
        if self.global_limit is None:
            return None
        return max(0.0, self.global_limit - self.global_current)

    @property
    def global_pct(self) -> float | None:
        if self.global_limit is None or self.global_limit == 0:
            return None
        return round(self.global_current / self.global_limit * 100, 1)

    def cloud_headroom(self, cloud: str) -> float | None:
        if cloud not in self.cloud_limits:
            return None
        return max(0.0, self.cloud_limits[cloud] - self.cloud_current.get(cloud, 0.0))


def check_budget(
    *,
    config: BudgetConfig,
    current_costs: dict[str, float],
    additional_cost: float,
    cloud: str,
) -> None:
    """
    Raise ``BudgetExceededError`` if adding *additional_cost* would push the
    total global spend or the per-cloud spend over their configured limits.

    :param current_costs: mapping of cloud → current total hourly cost
    :param additional_cost: cost the proposed new operation would add
    :param cloud: cloud provider of the proposed operation
    """
    total_current = sum(current_costs.values())
    cloud_current = current_costs.get(cloud, 0.0)

    if config.global_hourly is not None:
        if total_current + additional_cost > config.global_hourly:
            raise BudgetExceededError(
                f"Global hourly budget of ${config.global_hourly:.2f} would be exceeded "
                f"(current=${total_current:.2f}, adding=${additional_cost:.2f})"
            )

    if cloud in config.cloud_hourly:
        limit = config.cloud_hourly[cloud]
        if cloud_current + additional_cost > limit:
            raise BudgetExceededError(
                f"{cloud.upper()} hourly budget of ${limit:.2f} would be exceeded "
                f"(current=${cloud_current:.2f}, adding=${additional_cost:.2f})"
            )


def compute_budget_status(
    config: BudgetConfig,
    deployments: list,  # List[DeploymentDetail]
) -> BudgetStatus:
    """Compute current spend totals and return a ``BudgetStatus``."""
    cloud_current: dict[str, float] = {}
    for dep in deployments:
        c = dep.cloud.value if hasattr(dep.cloud, "value") else str(dep.cloud)
        cost = dep.metrics.get("estimated_hourly_cost", 0.0)
        cloud_current[c] = cloud_current.get(c, 0.0) + cost

    return BudgetStatus(
        global_limit=config.global_hourly,
        global_current=round(sum(cloud_current.values()), 4),
        cloud_limits=dict(config.cloud_hourly),
        cloud_current={k: round(v, 4) for k, v in cloud_current.items()},
    )


class BudgetExceededError(Exception):
    """Raised when an operation would push spend past a configured budget cap."""
