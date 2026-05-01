"""On-premises provider adapter — Kubernetes via the official Python client.

Required packages
-----------------
    pip install kubernetes

Required environment variables
-------------------------------
FLEXAI_K8S_NAMESPACE        Kubernetes namespace (default: flexai)
FLEXAI_K8S_IMAGE            Docker image URI for the inference container
FLEXAI_K8S_SERVICE_ACCOUNT  ServiceAccount to attach to the inference Pod (default: default)

Kubeconfig: uses ~/.kube/config or the in-cluster service account token
(KUBERNETES_SERVICE_HOST env var present → in-cluster mode is auto-detected).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .base import ProviderAdapter

logger = logging.getLogger(__name__)

_LABEL_KEY = "flexai-deployment-id"


def _k8s_clients():
    """Return (apps_v1, core_v1) Kubernetes API clients."""
    try:
        from kubernetes import client, config as k8s_config  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "kubernetes is required for the OnPrem adapter. "
            "Install it with: pip install kubernetes"
        ) from exc

    if os.getenv("KUBERNETES_SERVICE_HOST"):
        k8s_config.load_incluster_config()
    else:
        k8s_config.load_kube_config()

    return client.AppsV1Api(), client.CoreV1Api()


def _k8s_types():
    """Return the kubernetes.client module for building V1 objects."""
    try:
        from kubernetes import client  # type: ignore[import]
        return client
    except ImportError as exc:
        raise RuntimeError(
            "kubernetes is required for the OnPrem adapter. "
            "Install it with: pip install kubernetes"
        ) from exc


def _cfg(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _ns() -> str:
    return _cfg("FLEXAI_K8S_NAMESPACE", "flexai")


class OnPremAdapter(ProviderAdapter):
    """OnPrem adapter — deploys inference pods to a Kubernetes cluster."""

    @property
    def name(self) -> str:
        return "onprem"

    def provision(self, deployment_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """Create a Kubernetes Deployment + ClusterIP Service for the inference workload."""
        k8s = _k8s_types()
        apps_v1, core_v1 = _k8s_clients()
        ns = _ns()
        image = _cfg("FLEXAI_K8S_IMAGE") or config.get("image", "")
        if not image:
            raise RuntimeError("FLEXAI_K8S_IMAGE must be set for the OnPrem adapter")

        replicas = int(config.get("replicas", 1))
        gpu = config.get("gpu", "")
        name = f"flexai-{deployment_id[:40]}"
        user_labels: dict[str, str] = config.get("labels", {})
        labels: dict[str, str] = {_LABEL_KEY: deployment_id, "app": "flexai-inference", **user_labels}

        # Build resource requirements
        resources: dict[str, Any] = {}
        if gpu:
            resources = {
                "limits": {"nvidia.com/gpu": "1"},
                "requests": {"memory": "4Gi", "cpu": "2"},
            }
        else:
            resources = {"requests": {"memory": "2Gi", "cpu": "1"}}

        container = k8s.V1Container(
            name="inference",
            image=image,
            ports=[k8s.V1ContainerPort(container_port=8080)],
            resources=k8s.V1ResourceRequirements(**resources),
            env=[
                k8s.V1EnvVar(name="DEPLOYMENT_ID", value=deployment_id),
                k8s.V1EnvVar(name="MODEL_PATH", value=config.get("artifact_path", "")),
            ],
        )

        pod_spec = k8s.V1PodSpec(
            containers=[container],
            service_account_name=_cfg("FLEXAI_K8S_SERVICE_ACCOUNT", "default"),
        )

        deployment_body = k8s.V1Deployment(
            metadata=k8s.V1ObjectMeta(name=name, namespace=ns, labels=labels),
            spec=k8s.V1DeploymentSpec(
                replicas=replicas,
                selector=k8s.V1LabelSelector(match_labels=labels),
                template=k8s.V1PodTemplateSpec(
                    metadata=k8s.V1ObjectMeta(labels=labels),
                    spec=pod_spec,
                ),
            ),
        )

        service_body = k8s.V1Service(
            metadata=k8s.V1ObjectMeta(name=name, namespace=ns, labels=labels),
            spec=k8s.V1ServiceSpec(
                selector=labels,
                ports=[k8s.V1ServicePort(port=80, target_port=8080)],
                type="ClusterIP",
            ),
        )

        logger.info("[OnPrem] creating deployment %s in namespace %s", name, ns)
        apps_v1.create_namespaced_deployment(namespace=ns, body=deployment_body)

        logger.info("[OnPrem] creating service %s", name)
        core_v1.create_namespaced_service(namespace=ns, body=service_body)

        return {
            "provider": "onprem",
            "namespace": ns,
            "deployment_name": name,
            "service_name": name,
            "endpoint": f"http://{name}.{ns}.svc.cluster.local",
            "replicas": replicas,
        }

    def teardown(self, deployment_id: str, metadata: dict[str, Any]) -> None:
        """Delete the Kubernetes Deployment and Service."""
        from kubernetes.client.rest import ApiException  # type: ignore[import]

        apps_v1, core_v1 = _k8s_clients()
        ns = metadata.get("namespace", _ns())
        name = metadata.get("deployment_name", f"flexai-{deployment_id[:40]}")

        for fn, label in [
            (lambda: apps_v1.delete_namespaced_deployment(name=name, namespace=ns), "deployment"),
            (lambda: core_v1.delete_namespaced_service(name=name, namespace=ns), "service"),
        ]:
            try:
                logger.info("[OnPrem] deleting %s %s", label, name)
                fn()
            except ApiException as exc:
                if exc.status != 404:
                    logger.warning("[OnPrem] teardown %s %s: %s", label, name, exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[OnPrem] teardown %s %s: %s", label, name, exc)

    def get_status(self, deployment_id: str, metadata: dict[str, Any]) -> str:
        """Return 'running' when at least one replica is ready."""
        apps_v1, _ = _k8s_clients()
        ns = metadata.get("namespace", _ns())
        name = metadata.get("deployment_name", f"flexai-{deployment_id[:40]}")
        try:
            dep = apps_v1.read_namespaced_deployment(name=name, namespace=ns)
            ready = dep.status.ready_replicas or 0
            desired = dep.spec.replicas or 0
            if ready == 0:
                return "pending"
            if ready < desired:
                return "pending"
            return "running"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[OnPrem] get_status %s: %s", name, exc)
            return "error"

    def scale(self, deployment_id: str, metadata: dict[str, Any], replicas: int) -> None:
        """Patch the Deployment replica count."""
        apps_v1, _ = _k8s_clients()
        ns = metadata.get("namespace", _ns())
        name = metadata.get("deployment_name", f"flexai-{deployment_id[:40]}")
        logger.info("[OnPrem] scaling %s to %d replicas", name, replicas)
        try:
            apps_v1.patch_namespaced_deployment_scale(
                name=name,
                namespace=ns,
                body={"spec": {"replicas": replicas}},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[OnPrem] scale %s: %s", name, exc)

    def rollback(self, deployment_id: str, metadata: dict[str, Any], target_version: str, image_uri: str) -> None:
        """Patch the Kubernetes Deployment to run the target container image."""
        apps_v1, _ = _k8s_clients()
        ns = metadata.get("namespace", _ns())
        name = metadata.get("deployment_name", f"flexai-{deployment_id[:40]}")
        container_name = metadata.get("container_name", "inference")
        logger.info("[OnPrem] rolling back %s to version %s (image=%s)", name, target_version, image_uri)
        try:
            apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=ns,
                body={
                    "spec": {
                        "template": {
                            "metadata": {"annotations": {"flexai/rollback-version": target_version}},
                            "spec": {
                                "containers": [{"name": container_name, "image": image_uri}]
                            },
                        }
                    }
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[OnPrem] rollback %s: %s", name, exc)

    def set_traffic(
        self,
        deployment_id: str,
        metadata: dict[str, Any],
        weights: dict[str, int],
    ) -> None:
        """Configure Istio VirtualService traffic weights for canary/A-B testing."""
        ns = metadata.get("namespace", _ns())
        name = metadata.get("deployment_name", f"flexai-{deployment_id[:40]}")
        logger.info("[OnPrem] set_traffic %s/%s weights=%s", ns, name, weights)
        try:
            from kubernetes import client as k8s, config as k8s_config  # type: ignore[import]
            k8s_config.load_incluster_config()
        except Exception:
            try:
                from kubernetes import client as k8s, config as k8s_config  # type: ignore[import]
                k8s_config.load_kube_config()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[OnPrem] set_traffic: k8s config unavailable: %s", exc)
                return
        try:
            custom_api = k8s.CustomObjectsApi()
            # Build an Istio VirtualService with HTTPRoute weight entries.
            http_routes = [
                {
                    "destination": {"host": name, "subset": variant},
                    "weight": weight,
                }
                for variant, weight in weights.items()
            ]
            vs_body = {
                "apiVersion": "networking.istio.io/v1beta1",
                "kind": "VirtualService",
                "metadata": {"name": name, "namespace": ns},
                "spec": {
                    "hosts": [name],
                    "http": [{"route": http_routes}],
                },
            }
            try:
                custom_api.replace_namespaced_custom_object(
                    group="networking.istio.io", version="v1beta1",
                    namespace=ns, plural="virtualservices", name=name, body=vs_body,
                )
            except Exception:
                custom_api.create_namespaced_custom_object(
                    group="networking.istio.io", version="v1beta1",
                    namespace=ns, plural="virtualservices", body=vs_body,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[OnPrem] set_traffic %s: %s", name, exc)

    def configure_mesh(
        self,
        deployment_id: str,
        metadata: dict[str, Any],
        enabled: bool,
    ) -> None:
        """Toggle Istio sidecar injection on the Kubernetes Deployment via annotation."""
        ns = metadata.get("namespace", _ns())
        dep_name = metadata.get("deployment_name", f"flexai-{deployment_id[:40]}")
        logger.info("[OnPrem] configure_mesh %s/%s enabled=%s", ns, dep_name, enabled)
        try:
            from kubernetes import client as k8s, config as k8s_config  # type: ignore[import]
            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()
            apps_v1 = k8s.AppsV1Api()
            core_v1 = k8s.CoreV1Api()
            sidecar_annotation = "sidecar.istio.io/inject"
            inject_value = "true" if enabled else "false"
            # Patch Deployment pod template annotations for sidecar injection.
            apps_v1.patch_namespaced_deployment(
                name=dep_name,
                namespace=ns,
                body={
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {sidecar_annotation: inject_value}
                            }
                        }
                    }
                },
            )
            if enabled:
                # Also label the namespace so the mutating webhook fires.
                core_v1.patch_namespace(
                    name=ns,
                    body={"metadata": {"labels": {"istio-injection": "enabled"}}},
                )
            logger.info(
                "[OnPrem] Istio sidecar injection %s for %s/%s",
                "enabled" if enabled else "disabled", ns, dep_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[OnPrem] configure_mesh %s: %s", dep_name, exc)

