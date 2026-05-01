"""AWS provider adapter — Amazon SageMaker managed inference endpoints.

Required packages
-----------------
    pip install boto3

Required environment variables
-------------------------------
FLEXAI_AWS_REGION          AWS region (default: us-east-1)
FLEXAI_AWS_EXECUTION_ROLE  SageMaker execution role ARN
FLEXAI_AWS_ECR_IMAGE       Docker image URI for the inference container
FLEXAI_AWS_S3_BUCKET       S3 bucket for model artefacts

AWS credentials must be available via IAM instance role, environment variables
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY), or ~/.aws/credentials.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .base import ProviderAdapter

logger = logging.getLogger(__name__)


def _sagemaker():
    """Return a boto3 SageMaker client, raising RuntimeError if boto3 is missing."""
    try:
        import boto3  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for the AWS adapter. Install it with: pip install boto3"
        ) from exc
    region = os.getenv("FLEXAI_AWS_REGION", "us-east-1")
    return boto3.client("sagemaker", region_name=region)


def _cfg(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        raise RuntimeError(f"Environment variable {key} is required for the AWS adapter")
    return val


class AWSAdapter(ProviderAdapter):
    """AWS adapter — provisions SageMaker real-time inference endpoints."""

    @property
    def name(self) -> str:
        return "aws"

    def provision(self, deployment_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """Create a SageMaker Model → EndpointConfig → Endpoint."""
        sm = _sagemaker()
        region = os.getenv("FLEXAI_AWS_REGION", "us-east-1")
        role = _cfg("FLEXAI_AWS_EXECUTION_ROLE")
        image = _cfg("FLEXAI_AWS_ECR_IMAGE")
        bucket = _cfg("FLEXAI_AWS_S3_BUCKET")

        model_name = f"flexai-{deployment_id}"
        config_name = f"flexai-cfg-{deployment_id}"
        endpoint_name = f"flexai-ep-{deployment_id}"
        instance_type = config.get("instance_type", "ml.g4dn.xlarge")
        replicas = int(config.get("replicas", 1))
        artifact_path = config.get("artifact_path", f"{deployment_id}/model.tar.gz")
        model_data_url = f"s3://{bucket}/{artifact_path}"
        user_labels: dict[str, str] = config.get("labels", {})
        aws_tags = [{"Key": k, "Value": v} for k, v in user_labels.items()]

        logger.info("[AWS] creating SageMaker model %s", model_name)
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={"Image": image, "ModelDataUrl": model_data_url},
            ExecutionRoleArn=role,
            Tags=aws_tags,
        )

        logger.info("[AWS] creating endpoint config %s", config_name)
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": replicas,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],
            Tags=aws_tags,
        )

        logger.info("[AWS] creating endpoint %s", endpoint_name)
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name, Tags=aws_tags)

        return {
            "provider": "aws",
            "region": region,
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "config_name": config_name,
            "endpoint": f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations",
            "s3_artifact": model_data_url,
        }

    def teardown(self, deployment_id: str, metadata: dict[str, Any]) -> None:
        """Delete the SageMaker Endpoint, EndpointConfig and Model."""
        sm = _sagemaker()
        endpoint_name = metadata.get("endpoint_name", f"flexai-ep-{deployment_id}")
        config_name = metadata.get("config_name", f"flexai-cfg-{deployment_id}")
        model_name = metadata.get("model_name", f"flexai-{deployment_id}")

        for fn, name, label in [
            (sm.delete_endpoint, endpoint_name, "endpoint"),
            (sm.delete_endpoint_config, config_name, "endpoint config"),
            (sm.delete_model, model_name, "model"),
        ]:
            try:
                logger.info("[AWS] deleting %s %s", label, name)
                if label == "endpoint":
                    sm.delete_endpoint(EndpointName=name)
                elif label == "endpoint config":
                    sm.delete_endpoint_config(EndpointConfigName=name)
                else:
                    sm.delete_model(ModelName=name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[AWS] teardown %s %s: %s", label, name, exc)

    def get_status(self, deployment_id: str, metadata: dict[str, Any]) -> str:
        """Map SageMaker endpoint status to FlexAI status string."""
        sm = _sagemaker()
        endpoint_name = metadata.get("endpoint_name", f"flexai-ep-{deployment_id}")
        try:
            resp = sm.describe_endpoint(EndpointName=endpoint_name)
            status = resp["EndpointStatus"]  # InService | Creating | Updating | Failed | Deleting
            mapping = {
                "InService": "running",
                "Creating": "pending",
                "Updating": "pending",
                "Failed": "error",
                "Deleting": "stopped",
                "OutOfService": "stopped",
            }
            return mapping.get(status, status.lower())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[AWS] get_status %s: %s", endpoint_name, exc)
            return "error"

    def scale(self, deployment_id: str, metadata: dict[str, Any], replicas: int) -> None:
        """Update the endpoint to use a new EndpointConfig with the desired replica count."""
        sm = _sagemaker()
        endpoint_name = metadata.get("endpoint_name", f"flexai-ep-{deployment_id}")
        model_name = metadata.get("model_name", f"flexai-{deployment_id}")
        new_config = f"flexai-cfg-{deployment_id}-r{replicas}"

        # Fetch current instance type from the existing config
        old_config = metadata.get("config_name", f"flexai-cfg-{deployment_id}")
        try:
            desc = sm.describe_endpoint_config(EndpointConfigName=old_config)
            instance_type = desc["ProductionVariants"][0]["InstanceType"]
        except Exception:  # noqa: BLE001
            instance_type = "ml.g4dn.xlarge"

        logger.info("[AWS] scaling %s to %d replicas", endpoint_name, replicas)
        sm.create_endpoint_config(
            EndpointConfigName=new_config,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": replicas,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],
        )
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=new_config)

    def rollback(self, deployment_id: str, metadata: dict[str, Any], target_version: str, image_uri: str) -> None:
        """Re-deploy the SageMaker endpoint using the *image_uri* for *target_version*."""
        sm = _sagemaker()
        endpoint_name = metadata.get("endpoint_name", f"flexai-ep-{deployment_id}")
        role = _cfg("FLEXAI_AWS_EXECUTION_ROLE")
        region = os.getenv("FLEXAI_AWS_REGION", "us-east-1")
        rollback_model = f"flexai-{deployment_id[:28]}-rb-{target_version[:8]}"
        rollback_config = f"flexai-cfg-{deployment_id[:24]}-rb-{target_version[:8]}"

        old_config = metadata.get("config_name", f"flexai-cfg-{deployment_id}")
        try:
            desc = sm.describe_endpoint_config(EndpointConfigName=old_config)
            instance_type = desc["ProductionVariants"][0]["InstanceType"]
            replicas = desc["ProductionVariants"][0]["InitialInstanceCount"]
        except Exception:  # noqa: BLE001
            instance_type = "ml.g4dn.xlarge"
            replicas = 1

        logger.info("[AWS] rolling back %s to version %s (image=%s)", endpoint_name, target_version, image_uri)
        sm.create_model(
            ModelName=rollback_model,
            ExecutionRoleArn=role,
            PrimaryContainer={"Image": image_uri, "Mode": "SingleModel"},
        )
        sm.create_endpoint_config(
            EndpointConfigName=rollback_config,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": rollback_model,
                    "InitialInstanceCount": replicas,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],
        )
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=rollback_config)

    def set_traffic(
        self,
        deployment_id: str,
        metadata: dict[str, Any],
        weights: dict[str, int],
    ) -> None:
        """Update SageMaker endpoint variant weights to realise a canary/A-B split."""
        sm = _sagemaker()
        endpoint_name = metadata.get("endpoint_name", f"flexai-ep-{deployment_id}")
        # SageMaker expresses variant weight as a relative float; normalise to %
        # so a {"baseline": 90, "canary": 10} mapping becomes two variants.
        desired_variants = [
            {"VariantName": variant_name, "DesiredWeight": float(weight)}
            for variant_name, weight in weights.items()
        ]
        logger.info("[AWS] set_traffic %s weights=%s", endpoint_name, weights)
        try:
            sm.update_endpoint_weights_and_capacities(
                EndpointName=endpoint_name,
                DesiredWeightsAndCapacities=desired_variants,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[AWS] set_traffic %s: %s", endpoint_name, exc)

    def configure_mesh(
        self,
        deployment_id: str,
        metadata: dict[str, Any],
        enabled: bool,
    ) -> None:
        """Enable/disable AWS App Mesh virtual node and network isolation for the endpoint."""
        sm = _sagemaker()
        endpoint_name = metadata.get("endpoint_name", f"flexai-ep-{deployment_id}")
        logger.info("[AWS] configure_mesh %s enabled=%s", endpoint_name, enabled)
        try:
            import boto3  # type: ignore[import]
            region = metadata.get("region", os.getenv("FLEXAI_AWS_REGION", "us-east-1"))
            appmesh = boto3.client("appmesh", region_name=region)
            mesh_name = os.getenv("FLEXAI_AWS_MESH_NAME", "flexai-mesh")
            virtual_node_name = f"flexai-vn-{deployment_id[:40]}"
            if enabled:
                appmesh.create_virtual_node(
                    meshName=mesh_name,
                    virtualNodeName=virtual_node_name,
                    spec={
                        "listeners": [{"portMapping": {"port": 8080, "protocol": "http"}}],
                        "serviceDiscovery": {
                            "dns": {"hostname": f"{endpoint_name}.flexai.local"}
                        },
                    },
                )
                logger.info("[AWS] App Mesh virtual node %s created", virtual_node_name)
            else:
                appmesh.delete_virtual_node(
                    meshName=mesh_name,
                    virtualNodeName=virtual_node_name,
                )
                logger.info("[AWS] App Mesh virtual node %s deleted", virtual_node_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[AWS] configure_mesh %s: %s", endpoint_name, exc)

