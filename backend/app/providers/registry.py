"""Provider registry – resolves cloud name → ProviderAdapter instance."""
from __future__ import annotations

from .aws import AWSAdapter
from .azure import AzureAdapter
from .base import ProviderAdapter
from .gcp import GCPAdapter
from .onprem import OnPremAdapter

_REGISTRY: dict[str, ProviderAdapter] = {
    adapter.name: adapter
    for adapter in [
        AWSAdapter(),
        AzureAdapter(),
        GCPAdapter(),
        OnPremAdapter(),
    ]
}


def get_provider(cloud: str) -> ProviderAdapter:
    """
    Return the adapter for *cloud*.

    Raises ValueError for unknown providers so callers can surface a clean
    HTTP 400 rather than a raw KeyError.
    """
    key = cloud.lower().strip()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown cloud provider {cloud!r}. Available: {available}")
    return _REGISTRY[key]


__all__ = ["get_provider", "ProviderAdapter"]
