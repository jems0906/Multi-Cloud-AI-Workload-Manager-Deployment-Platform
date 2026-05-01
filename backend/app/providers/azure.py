"""Azure provider adapter — Azure Machine Learning online endpoints.

Required packages
-----------------
    pip install azure-identity azure-ai-ml

Required environment variables
-------------------------------
FLEXAI_AZURE_SUBSCRIPTION_ID   Azure subscription ID
FLEXAI_AZURE_RESOURCE_GROUP    Resource group name (default: flexai-rg)
FLEXAI_AZURE_WORKSPACE         Azure ML workspace name
FLEXAI_AZURE_ACR_IMAGE         Docker image URI from Azure Container Registry
FLEXAI_AZURE_REGION            Azure region (default: eastus)

Authentication: uses DefaultAzureCredential (managed identity, service principal
via AZURE_CLIENT_ID/AZURE_CLIENT_SECRET/AZURE_TENANT_ID, or `az login`).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .base import ProviderAdapter

logger = logging.getLogger(__name__)


def _ml_client():
    """Return an azure-ai-ml MLClient, raising RuntimeError if SDK is missing."""
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore[import]
        from azure.ai.ml import MLClient  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "azure-identity and azure-ai-ml are required for the Azure adapter. "
            "Install them with: pip install azure-identity azure-ai-ml"
        ) from exc

    sub = _cfg("FLEXAI_AZURE_SUBSCRIPTION_ID")
    rg = os.getenv("FLEXAI_AZURE_RESOURCE_GROUP", "flexai-rg")
    ws = _cfg("FLEXAI_AZURE_WORKSPACE")
    return MLClient(DefaultAzureCredential(), sub, rg, ws)


def _aml_entities():
    """Return the azure.ai.ml.entities module, raising RuntimeError if SDK is missing."""
    try:
        from azure.ai import ml as _aml  # type: ignore[import]
        return _aml.entities
    except ImportError as exc:
        raise RuntimeError(
            "azure-ai-ml is required for the Azure adapter. "
            "Install it with: pip install azure-ai-ml"
        ) from exc


def _cfg(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        raise RuntimeError(f"Environment variable {key} is required for the Azure adapter")
    return val


class AzureAdapter(ProviderAdapter):
    """Azure adapter — provisions Azure ML managed online endpoints."""

    @property
    def name(self) -> str:
        return "azure"

    def provision(self, deployment_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """Create an Azure ML ManagedOnlineEndpoint + ManagedOnlineDeployment."""
        ml = _ml_client()
        entities = _aml_entities()
        region = os.getenv("FLEXAI_AZURE_REGION", "eastus")
        image = _cfg("FLEXAI_AZURE_ACR_IMAGE")
        endpoint_name = f"flexai-{deployment_id[:20]}"  # AML name limit
        instance_type = config.get("instance_type", "Standard_NC6s_v3")
        replicas = int(config.get("replicas", 1))
        user_labels: dict[str, str] = config.get("labels", {})

        logger.info("[Azure] creating online endpoint %s", endpoint_name)
        endpoint = entities.ManagedOnlineEndpoint(
            name=endpoint_name,
            description=f"FlexAI deployment {deployment_id}",
            auth_mode="key",
            tags=user_labels,
        )
        ml.online_endpoints.begin_create_or_update(endpoint).result()

        logger.info("[Azure] creating online deployment for %s", endpoint_name)
        deployment = entities.ManagedOnlineDeployment(
            name="blue",
            endpoint_name=endpoint_name,
            environment=entities.Environment(image=image),
            instance_type=instance_type,
            instance_count=replicas,
            tags=user_labels,
        )
        ml.online_deployments.begin_create_or_update(deployment).result()

        # Route all traffic to the 'blue' deployment
        endpoint.traffic = {"blue": 100}
        ml.online_endpoints.begin_create_or_update(endpoint).result()

        ep_detail = ml.online_endpoints.get(endpoint_name)
        scoring_uri = ep_detail.scoring_uri or f"https://{endpoint_name}.{region}.inference.ml.azure.com/score"

        return {
            "provider": "azure",
            "region": region,
            "endpoint_name": endpoint_name,
            "endpoint": scoring_uri,
            "resource_group": os.getenv("FLEXAI_AZURE_RESOURCE_GROUP", "flexai-rg"),
            "workspace": os.getenv("FLEXAI_AZURE_WORKSPACE", ""),
        }

    def teardown(self, deployment_id: str, metadata: dict[str, Any]) -> None:
        """Delete the Azure ML online endpoint (and all its deployments)."""
        ml = _ml_client()
        endpoint_name = metadata.get("endpoint_name", f"flexai-{deployment_id[:20]}")
        logger.info("[Azure] deleting online endpoint %s", endpoint_name)
        try:
            ml.online_endpoints.begin_delete(name=endpoint_name).result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Azure] teardown %s: %s", endpoint_name, exc)

    def get_status(self, deployment_id: str, metadata: dict[str, Any]) -> str:
        """Map AML provisioning state to FlexAI status string."""
        ml = _ml_client()
        endpoint_name = metadata.get("endpoint_name", f"flexai-{deployment_id[:20]}")
        try:
            ep = ml.online_endpoints.get(endpoint_name)
            state = (ep.provisioning_state or "").lower()
            mapping = {
                "succeeded": "running",
                "creating": "pending",
                "updating": "pending",
                "deleting": "stopped",
                "failed": "error",
            }
            return mapping.get(state, state or "pending")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Azure] get_status %s: %s", endpoint_name, exc)
            return "error"

    def scale(self, deployment_id: str, metadata: dict[str, Any], replicas: int) -> None:
        """Update the 'blue' deployment instance count."""
        ml = _ml_client()
        endpoint_name = metadata.get("endpoint_name", f"flexai-{deployment_id[:20]}")
        logger.info("[Azure] scaling %s to %d replicas", endpoint_name, replicas)
        try:
            deployment = ml.online_deployments.get(endpoint_name, "blue")
            deployment.instance_count = replicas
            ml.online_deployments.begin_create_or_update(deployment).result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Azure] scale %s: %s", endpoint_name, exc)

    def rollback(self, deployment_id: str, metadata: dict[str, Any], target_version: str, image_uri: str) -> None:
        """Swap the Azure ML online deployment to the *target_version* container image."""
        ml = _ml_client()
        endpoint_name = metadata.get("endpoint_name", f"flexai-{deployment_id[:20]}")
        logger.info("[Azure] rolling back %s to version %s (image=%s)", endpoint_name, target_version, image_uri)
        try:
            from azure.ai.ml.entities import OnlineDeployment, CodeConfiguration  # type: ignore[import]
            dep = ml.online_deployments.get(endpoint_name, "blue")
            dep.environment_variables = {"FLEXAI_TARGET_VERSION": target_version}
            # Swap to the target image by setting the environment image reference.
            # Azure ML uses environment objects; fall back to patching a tag if the
            # deployment's environment supports it.
            if hasattr(dep, "environment") and dep.environment:
                env = dep.environment
                if hasattr(env, "image"):
                    env.image = image_uri
                dep.environment = env
            ml.online_deployments.begin_create_or_update(dep).result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Azure] rollback %s: %s", endpoint_name, exc)

    def set_traffic(
        self,
        deployment_id: str,
        metadata: dict[str, Any],
        weights: dict[str, int],
    ) -> None:
        """Update Azure ML online endpoint traffic allocation for canary/A-B testing."""
        ml = _ml_client()
        endpoint_name = metadata.get("endpoint_name", f"flexai-{deployment_id[:20]}")
        logger.info("[Azure] set_traffic %s weights=%s", endpoint_name, weights)
        try:
            from azure.ai.ml.entities import ManagedOnlineEndpoint  # type: ignore[import]
            endpoint = ml.online_endpoints.get(endpoint_name)
            # Azure ML traffic allocation maps deployment name → integer percentage
            endpoint.traffic = {k: v for k, v in weights.items()}
            ml.online_endpoints.begin_create_or_update(endpoint).result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Azure] set_traffic %s: %s", endpoint_name, exc)

    def configure_mesh(
        self,
        deployment_id: str,
        metadata: dict[str, Any],
        enabled: bool,
    ) -> None:
        """Toggle Azure Service Mesh (Istio add-on) mTLS policy on the endpoint's subnet."""
        endpoint_name = metadata.get("endpoint_name", f"flexai-{deployment_id[:20]}")
        logger.info("[Azure] configure_mesh %s enabled=%s", endpoint_name, enabled)
        try:
            from azure.identity import DefaultAzureCredential  # type: ignore[import]
            from azure.mgmt.containerservice import ContainerServiceClient  # type: ignore[import]
            sub = _cfg("FLEXAI_AZURE_SUBSCRIPTION_ID")
            rg = os.getenv("FLEXAI_AZURE_RESOURCE_GROUP", "flexai-rg")
            cluster_name = os.getenv("FLEXAI_AZURE_AKS_CLUSTER", "flexai-aks")
            aks = ContainerServiceClient(DefaultAzureCredential(), sub)
            cluster = aks.managed_clusters.get(rg, cluster_name)
            mesh_profile = cluster.service_mesh_profile or {}
            if enabled:
                mesh_profile["mode"] = "Istio"
                mesh_profile["istio"] = {
                    "components": {"ingressGateways": [{"enabled": True, "mode": "External"}]}
                }
            else:
                mesh_profile["mode"] = "Disabled"
            cluster.service_mesh_profile = mesh_profile
            aks.managed_clusters.begin_create_or_update(rg, cluster_name, cluster).result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Azure] configure_mesh %s: %s", endpoint_name, exc)

