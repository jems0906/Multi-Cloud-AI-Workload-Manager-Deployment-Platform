"""Cloud provider adapters package."""
from .registry import get_provider, ProviderAdapter

__all__ = ["get_provider", "ProviderAdapter"]
