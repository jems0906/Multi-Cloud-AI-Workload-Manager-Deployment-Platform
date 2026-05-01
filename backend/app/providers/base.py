"""Cloud provider adapter base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ProviderAdapter(ABC):
    """Abstract base for all cloud provider adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical provider name (e.g. 'aws', 'azure', 'gcp', 'onprem')."""

    @abstractmethod
    def provision(self, deployment_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Trigger infrastructure provisioning for the given deployment.

        Returns a dict of provider-specific metadata (e.g. resource ARNs,
        cluster endpoint, job ID) that is stored alongside the deployment record.
        """

    @abstractmethod
    def teardown(self, deployment_id: str, metadata: dict[str, Any]) -> None:
        """Release all resources associated with *deployment_id*."""

    @abstractmethod
    def get_status(self, deployment_id: str, metadata: dict[str, Any]) -> str:
        """
        Return the current provider-side status string.

        Suggested values: 'running', 'stopped', 'error', 'pending'.
        """

    @abstractmethod
    def scale(self, deployment_id: str, metadata: dict[str, Any], replicas: int) -> None:
        """Adjust the number of running replicas for the deployment."""

    @abstractmethod
    def rollback(self, deployment_id: str, metadata: dict[str, Any], target_version: str, image_uri: str) -> None:
        """
        Re-deploy the deployment using *image_uri* for *target_version*.

        Implementations must atomically swap the running container/model to the
        target image without downtime where the provider supports it.
        """

    @abstractmethod
    def set_traffic(
        self,
        deployment_id: str,
        metadata: dict[str, Any],
        weights: dict[str, int],
    ) -> None:
        """
        Configure the service-mesh / load-balancer traffic split.

        ``weights`` maps variant name → integer percentage (values must sum to 100).
        Example: ``{"baseline": 90, "canary": 10}``
        """

    @abstractmethod
    def configure_mesh(
        self,
        deployment_id: str,
        metadata: dict[str, Any],
        enabled: bool,
    ) -> None:
        """
        Enable or disable service-mesh policy for the deployment.

        When ``enabled=True`` the implementation should enforce mTLS, inject a
        sidecar (where applicable), and apply any default mesh-level network
        policies.  When ``enabled=False`` the mesh sidecar / policy is removed.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.name!r}>"
