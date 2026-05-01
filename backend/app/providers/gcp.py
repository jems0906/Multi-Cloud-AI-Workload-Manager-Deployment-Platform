"""GCP provider adapter — Vertex AI managed online endpoints.

Required packages
-----------------
    pip install google-cloud-aiplatform

Required environment variables
-------------------------------
FLEXAI_GCP_PROJECT      GCP project ID
FLEXAI_GCP_REGION       GCP region (default: us-central1)
FLEXAI_GCR_IMAGE        Docker image URI (Artifact Registry or GCR)

Authentication: uses Application Default Credentials (ADC). Run
`gcloud auth application-default login` locally, or attach a service account
with roles/aiplatform.user in production.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .base import ProviderAdapter

logger = logging.getLogger(__name__)


def _aiplatform():
    """Import and initialise google-cloud-aiplatform, raising RuntimeError if missing."""
    try:
        from google.cloud import aiplatform  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-aiplatform is required for the GCP adapter. "
            "Install it with: pip install google-cloud-aiplatform"
        ) from exc
    project = _cfg("FLEXAI_GCP_PROJECT")
    region = os.getenv("FLEXAI_GCP_REGION", "us-central1")
    aiplatform.init(project=project, location=region)
    return aiplatform


def _cfg(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        raise RuntimeError(f"Environment variable {key} is required for the GCP adapter")
    return val


class GCPAdapter(ProviderAdapter):
    """GCP adapter — provisions Vertex AI online endpoints."""

    @property
    def name(self) -> str:
        return "gcp"

    def provision(self, deployment_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """Upload model to Vertex AI Model Registry, deploy to an online endpoint."""
        aip = _aiplatform()
        region = os.getenv("FLEXAI_GCP_REGION", "us-central1")
        project = _cfg("FLEXAI_GCP_PROJECT")
        image = _cfg("FLEXAI_GCR_IMAGE")
        machine_type = config.get("instance_type", "n1-standard-4")
        accelerator = config.get("gpu", "NVIDIA_TESLA_T4")
        replicas = int(config.get("replicas", 1))
        artifact_uri = config.get("artifact_path", f"gs://flexai-models/{deployment_id}/")
        user_labels: dict[str, str] = config.get("labels", {})

        logger.info("[GCP] uploading model for deployment %s", deployment_id)
        model = aip.Model.upload(
            display_name=f"flexai-{deployment_id}",
            artifact_uri=artifact_uri,
            serving_container_image_uri=image,
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            labels=user_labels,
        )

        logger.info("[GCP] creating Vertex AI endpoint for deployment %s", deployment_id)
        endpoint = aip.Endpoint.create(display_name=f"flexai-ep-{deployment_id}", labels=user_labels)

        logger.info("[GCP] deploying model to endpoint %s", endpoint.name)
        endpoint.deploy(
            model=model,
            deployed_model_display_name=f"flexai-{deployment_id}",
            machine_type=machine_type,
            accelerator_type=accelerator,
            accelerator_count=1,
            min_replica_count=replicas,
            max_replica_count=replicas,
            traffic_percentage=100,
        )

        return {
            "provider": "gcp",
            "project": project,
            "region": region,
            "endpoint_name": endpoint.name,
            "endpoint_id": endpoint.resource_name,
            "model_resource": model.resource_name,
            "endpoint": endpoint.resource_name,
        }

    def teardown(self, deployment_id: str, metadata: dict[str, Any]) -> None:
        """Undeploy all models then delete the Vertex AI endpoint."""
        aip = _aiplatform()
        endpoint_name = metadata.get("endpoint_name", "")
        if not endpoint_name:
            logger.warning("[GCP] teardown: no endpoint_name in metadata for %s", deployment_id)
            return
        try:
            endpoint = aip.Endpoint(endpoint_name=endpoint_name)
            logger.info("[GCP] undeploying all models from endpoint %s", endpoint_name)
            endpoint.undeploy_all()
            endpoint.delete()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[GCP] teardown %s: %s", endpoint_name, exc)

    def get_status(self, deployment_id: str, metadata: dict[str, Any]) -> str:
        """Check if the Vertex AI endpoint has active deployed models."""
        aip = _aiplatform()
        endpoint_name = metadata.get("endpoint_name", "")
        if not endpoint_name:
            return "error"
        try:
            endpoint = aip.Endpoint(endpoint_name=endpoint_name)
            deployed = endpoint.list_models()
            if not deployed:
                return "stopped"
            # All models have a create_time if deployed
            return "running"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[GCP] get_status %s: %s", endpoint_name, exc)
            return "error"

    def scale(self, deployment_id: str, metadata: dict[str, Any], replicas: int) -> None:
        """Update min/max replica count for the deployed model."""
        aip = _aiplatform()
        endpoint_name = metadata.get("endpoint_name", "")
        if not endpoint_name:
            logger.warning("[GCP] scale: no endpoint_name for %s", deployment_id)
            return
        try:
            endpoint = aip.Endpoint(endpoint_name=endpoint_name)
            models = endpoint.list_models()
            if not models:
                logger.warning("[GCP] scale: no deployed models on %s", endpoint_name)
                return
            deployed_model_id = models[0].id
            logger.info("[GCP] scaling %s to %d replicas", endpoint_name, replicas)
            endpoint.update(
                deployed_model=aip.DeployedModel(
                    id=deployed_model_id,
                    automatic_resources=aip.gapic.AutomaticResources(
                        min_replica_count=replicas,
                        max_replica_count=replicas,
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[GCP] scale %s: %s", endpoint_name, exc)

    def rollback(self, deployment_id: str, metadata: dict[str, Any], target_version: str, image_uri: str) -> None:
        """Deploy the target container image to the Vertex AI endpoint and shift traffic."""
        aip = _aiplatform()
        endpoint_name = metadata.get("endpoint_name", "")
        project = metadata.get("project", os.getenv("FLEXAI_GCP_PROJECT", ""))
        location = metadata.get("location", os.getenv("FLEXAI_GCP_REGION", "us-central1"))
        if not endpoint_name:
            logger.warning("[GCP] rollback: no endpoint_name for %s", deployment_id)
            return
        logger.info("[GCP] rolling back %s to version %s (image=%s)", endpoint_name, target_version, image_uri)
        try:
            endpoint = aip.Endpoint(endpoint_name=endpoint_name)
            # Upload the target model version
            rollback_model = aip.Model.upload(
                display_name=f"flexai-{deployment_id[:28]}-{target_version[:8]}",
                artifact_uri=metadata.get("artifact_uri", ""),
                serving_container_image_uri=image_uri,
                project=project,
                location=location,
            )
            # Deploy with 100% traffic; existing deployed model traffic will be
            # shifted by specifying traffic_split.
            existing = endpoint.list_models()
            existing_ids = {m.id: 0 for m in existing}
            endpoint.deploy(
                model=rollback_model,
                deployed_model_display_name=f"rb-{target_version[:8]}",
                traffic_split={"0": 100, **existing_ids},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[GCP] rollback %s: %s", endpoint_name, exc)

    def set_traffic(
        self,
        deployment_id: str,
        metadata: dict[str, Any],
        weights: dict[str, int],
    ) -> None:
        """Shift Vertex AI endpoint traffic split for canary or A/B deployments."""
        aip = _aiplatform()
        endpoint_name = metadata.get("endpoint_name", "")
        if not endpoint_name:
            logger.warning("[GCP] set_traffic: no endpoint_name for %s", deployment_id)
            return
        logger.info("[GCP] set_traffic %s weights=%s", endpoint_name, weights)
        try:
            endpoint = aip.Endpoint(endpoint_name=endpoint_name)
            # Vertex AI traffic_split maps deployed model ID → integer percentage.
            # Variant names here map to deployed model IDs stored in metadata.
            deployed_ids = metadata.get("deployed_model_ids", {})
            traffic_split = {
                deployed_ids.get(variant, variant): pct
                for variant, pct in weights.items()
            }
            endpoint.update(traffic_split=traffic_split)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[GCP] set_traffic %s: %s", endpoint_name, exc)

    def configure_mesh(
        self,
        deployment_id: str,
        metadata: dict[str, Any],
        enabled: bool,
    ) -> None:
        """Enable/disable Anthos Service Mesh (Traffic Director) for the Vertex AI endpoint."""
        endpoint_name = metadata.get("endpoint_name", "")
        project = metadata.get("project", os.getenv("FLEXAI_GCP_PROJECT", ""))
        location = metadata.get("location", os.getenv("FLEXAI_GCP_REGION", "us-central1"))
        logger.info("[GCP] configure_mesh %s enabled=%s", endpoint_name, enabled)
        try:
            from google.cloud import networkservices_v1  # type: ignore[import]
            client = networkservices_v1.NetworkServicesClient()
            mesh_name = f"projects/{project}/locations/{location}/meshes/flexai-{deployment_id[:40]}"
            if enabled:
                mesh = networkservices_v1.Mesh(
                    name=mesh_name,
                    interception_port=15001,
                    labels={"flexai-deployment": deployment_id[:63]},
                )
                request = networkservices_v1.CreateMeshRequest(
                    parent=f"projects/{project}/locations/{location}",
                    mesh_id=f"flexai-{deployment_id[:40]}",
                    mesh=mesh,
                )
                client.create_mesh(request=request)
                logger.info("[GCP] Anthos mesh created for %s", endpoint_name)
            else:
                client.delete_mesh(name=mesh_name)
                logger.info("[GCP] Anthos mesh deleted for %s", endpoint_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[GCP] configure_mesh %s: %s", endpoint_name, exc)

