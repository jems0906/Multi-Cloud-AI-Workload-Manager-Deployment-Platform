"""Tests for cloud provider adapters.

All SDK calls are replaced with unittest.mock stubs so no real cloud
credentials or packages are needed at test time.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# AWS / SageMaker
# ---------------------------------------------------------------------------

class TestAWSAdapter:
    def _make_adapter(self):
        from app.providers.aws import AWSAdapter
        return AWSAdapter()

    def _sm_mock(self, status="InService"):
        sm = MagicMock()
        sm.describe_endpoint.return_value = {"EndpointStatus": status}
        sm.describe_endpoint_config.return_value = {
            "ProductionVariants": [{"InstanceType": "ml.g4dn.xlarge"}]
        }
        return sm

    def test_provision_calls_create_model_config_endpoint(self, monkeypatch):
        sm = self._sm_mock()
        monkeypatch.setenv("FLEXAI_AWS_EXECUTION_ROLE", "arn:aws:iam::123:role/SageMaker")
        monkeypatch.setenv("FLEXAI_AWS_ECR_IMAGE", "123.dkr.ecr.us-east-1.amazonaws.com/flexai:latest")
        monkeypatch.setenv("FLEXAI_AWS_S3_BUCKET", "flexai-models")

        with patch("app.providers.aws._sagemaker", return_value=sm):
            adapter = self._make_adapter()
            result = adapter.provision("dep-001", {"replicas": 2, "artifact_path": "dep-001/model.tar.gz"})

        sm.create_model.assert_called_once()
        sm.create_endpoint_config.assert_called_once()
        sm.create_endpoint.assert_called_once()
        assert result["provider"] == "aws"
        assert "endpoint" in result

    def test_get_status_maps_inservice_to_running(self, monkeypatch):
        sm = self._sm_mock(status="InService")
        with patch("app.providers.aws._sagemaker", return_value=sm):
            adapter = self._make_adapter()
            status = adapter.get_status("dep-001", {"endpoint_name": "flexai-ep-dep-001"})
        assert status == "running"

    def test_get_status_maps_creating_to_pending(self, monkeypatch):
        sm = self._sm_mock(status="Creating")
        with patch("app.providers.aws._sagemaker", return_value=sm):
            status = self._make_adapter().get_status("dep-001", {"endpoint_name": "flexai-ep-dep-001"})
        assert status == "pending"

    def test_get_status_returns_error_on_exception(self, monkeypatch):
        sm = MagicMock()
        sm.describe_endpoint.side_effect = Exception("network error")
        with patch("app.providers.aws._sagemaker", return_value=sm):
            status = self._make_adapter().get_status("dep-001", {"endpoint_name": "flexai-ep-dep-001"})
        assert status == "error"

    def test_teardown_calls_delete_methods(self, monkeypatch):
        sm = self._sm_mock()
        with patch("app.providers.aws._sagemaker", return_value=sm):
            self._make_adapter().teardown("dep-001", {
                "endpoint_name": "flexai-ep-dep-001",
                "config_name": "flexai-cfg-dep-001",
                "model_name": "flexai-dep-001",
            })
        sm.delete_endpoint.assert_called_once()
        sm.delete_endpoint_config.assert_called_once()
        sm.delete_model.assert_called_once()

    def test_scale_creates_new_config_and_updates_endpoint(self, monkeypatch):
        sm = self._sm_mock()
        with patch("app.providers.aws._sagemaker", return_value=sm):
            self._make_adapter().scale("dep-001", {
                "endpoint_name": "flexai-ep-dep-001",
                "config_name": "flexai-cfg-dep-001",
                "model_name": "flexai-dep-001",
            }, replicas=4)
        sm.create_endpoint_config.assert_called_once()
        call_kwargs = sm.create_endpoint_config.call_args[1]
        assert call_kwargs["ProductionVariants"][0]["InitialInstanceCount"] == 4
        sm.update_endpoint.assert_called_once()

    def test_missing_env_var_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("FLEXAI_AWS_EXECUTION_ROLE", raising=False)
        monkeypatch.delenv("FLEXAI_AWS_ECR_IMAGE", raising=False)
        monkeypatch.delenv("FLEXAI_AWS_S3_BUCKET", raising=False)

        sm = MagicMock()
        with patch("app.providers.aws._sagemaker", return_value=sm):
            with pytest.raises(RuntimeError, match="FLEXAI_AWS_EXECUTION_ROLE"):
                self._make_adapter().provision("dep-001", {})


# ---------------------------------------------------------------------------
# Azure ML
# ---------------------------------------------------------------------------

class TestAzureAdapter:
    def _make_adapter(self):
        from app.providers.azure import AzureAdapter
        return AzureAdapter()

    def _ml_mock(self):
        ml = MagicMock()
        ep = MagicMock()
        ep.scoring_uri = "https://flexai-ep-dep001.eastus.inference.ml.azure.com/score"
        ep.provisioning_state = "Succeeded"
        ml.online_endpoints.get.return_value = ep
        ml.online_endpoints.begin_create_or_update.return_value = MagicMock(result=MagicMock(return_value=None))
        ml.online_deployments.begin_create_or_update.return_value = MagicMock(result=MagicMock(return_value=None))
        ml.online_deployments.get.return_value = MagicMock(instance_count=1)
        return ml

    def _entities_mock(self):
        ent = MagicMock()
        ent.ManagedOnlineEndpoint = MagicMock(return_value=MagicMock())
        ent.ManagedOnlineDeployment = MagicMock(return_value=MagicMock())
        ent.Environment = MagicMock(return_value=MagicMock())
        return ent

    def test_provision_creates_endpoint_and_deployment(self, monkeypatch):
        monkeypatch.setenv("FLEXAI_AZURE_SUBSCRIPTION_ID", "sub-123")
        monkeypatch.setenv("FLEXAI_AZURE_WORKSPACE", "flexai-ws")
        monkeypatch.setenv("FLEXAI_AZURE_ACR_IMAGE", "flexaiacr.azurecr.io/flexai:latest")
        ml = self._ml_mock()
        ent = self._entities_mock()

        with patch("app.providers.azure._ml_client", return_value=ml), \
             patch("app.providers.azure._aml_entities", return_value=ent):
            result = self._make_adapter().provision("dep001", {"replicas": 2})

        ml.online_endpoints.begin_create_or_update.assert_called()
        ml.online_deployments.begin_create_or_update.assert_called_once()
        assert result["provider"] == "azure"
        assert "endpoint" in result

    def test_get_status_maps_succeeded_to_running(self, monkeypatch):
        monkeypatch.setenv("FLEXAI_AZURE_SUBSCRIPTION_ID", "sub-123")
        monkeypatch.setenv("FLEXAI_AZURE_WORKSPACE", "flexai-ws")
        ml = self._ml_mock()

        with patch("app.providers.azure._ml_client", return_value=ml):
            status = self._make_adapter().get_status("dep001", {"endpoint_name": "flexai-dep001"})
        assert status == "running"

    def test_get_status_returns_error_on_exception(self, monkeypatch):
        monkeypatch.setenv("FLEXAI_AZURE_SUBSCRIPTION_ID", "sub-123")
        monkeypatch.setenv("FLEXAI_AZURE_WORKSPACE", "flexai-ws")
        ml = MagicMock()
        ml.online_endpoints.get.side_effect = Exception("auth error")

        with patch("app.providers.azure._ml_client", return_value=ml):
            status = self._make_adapter().get_status("dep001", {"endpoint_name": "flexai-dep001"})
        assert status == "error"

    def test_missing_subscription_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("FLEXAI_AZURE_SUBSCRIPTION_ID", raising=False)
        monkeypatch.delenv("FLEXAI_AZURE_WORKSPACE", raising=False)
        monkeypatch.delenv("FLEXAI_AZURE_ACR_IMAGE", raising=False)

        ent = self._entities_mock()
        with patch("app.providers.azure._ml_client", side_effect=RuntimeError("FLEXAI_AZURE_SUBSCRIPTION_ID")), \
             patch("app.providers.azure._aml_entities", return_value=ent):
            with pytest.raises(RuntimeError, match="FLEXAI_AZURE_SUBSCRIPTION_ID"):
                self._make_adapter().provision("dep001", {})


# ---------------------------------------------------------------------------
# GCP / Vertex AI
# ---------------------------------------------------------------------------

class TestGCPAdapter:
    def _make_adapter(self):
        from app.providers.gcp import GCPAdapter
        return GCPAdapter()

    def _aip_mock(self):
        aip = MagicMock()
        model = MagicMock()
        model.resource_name = "projects/proj/locations/us-central1/models/123"
        endpoint = MagicMock()
        endpoint.name = "projects/proj/locations/us-central1/endpoints/456"
        endpoint.resource_name = endpoint.name
        endpoint.list_models.return_value = [MagicMock(id="deployed-1")]
        aip.Model.upload.return_value = model
        aip.Endpoint.create.return_value = endpoint
        aip.Endpoint.return_value = endpoint
        return aip

    def test_provision_uploads_model_and_deploys(self, monkeypatch):
        monkeypatch.setenv("FLEXAI_GCP_PROJECT", "my-project")
        monkeypatch.setenv("FLEXAI_GCR_IMAGE", "gcr.io/my-project/flexai:latest")
        aip = self._aip_mock()

        with patch("app.providers.gcp._aiplatform", return_value=aip):
            result = self._make_adapter().provision("dep-gcp-01", {"replicas": 1})

        aip.Model.upload.assert_called_once()
        aip.Endpoint.create.assert_called_once()
        aip.Endpoint.return_value.deploy.assert_called_once()
        assert result["provider"] == "gcp"

    def test_get_status_returns_running_when_models_deployed(self, monkeypatch):
        monkeypatch.setenv("FLEXAI_GCP_PROJECT", "my-project")
        aip = self._aip_mock()

        with patch("app.providers.gcp._aiplatform", return_value=aip):
            status = self._make_adapter().get_status("dep-gcp-01", {
                "endpoint_name": "projects/proj/locations/us-central1/endpoints/456"
            })
        assert status == "running"

    def test_get_status_returns_stopped_when_no_models(self, monkeypatch):
        monkeypatch.setenv("FLEXAI_GCP_PROJECT", "my-project")
        aip = self._aip_mock()
        aip.Endpoint.return_value.list_models.return_value = []

        with patch("app.providers.gcp._aiplatform", return_value=aip):
            status = self._make_adapter().get_status("dep-gcp-01", {
                "endpoint_name": "projects/proj/locations/us-central1/endpoints/456"
            })
        assert status == "stopped"

    def test_missing_project_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("FLEXAI_GCP_PROJECT", raising=False)
        monkeypatch.delenv("FLEXAI_GCR_IMAGE", raising=False)

        with patch("app.providers.gcp._aiplatform", side_effect=RuntimeError("FLEXAI_GCP_PROJECT")):
            with pytest.raises(RuntimeError, match="FLEXAI_GCP_PROJECT"):
                self._make_adapter().provision("dep-gcp-01", {})


# ---------------------------------------------------------------------------
# OnPrem / Kubernetes
# ---------------------------------------------------------------------------

class TestOnPremAdapter:
    def _make_adapter(self):
        from app.providers.onprem import OnPremAdapter
        return OnPremAdapter()

    def _k8s_mock(self, ready=1, desired=1):
        apps_v1 = MagicMock()
        core_v1 = MagicMock()
        dep = MagicMock()
        dep.status.ready_replicas = ready
        dep.spec.replicas = desired
        apps_v1.read_namespaced_deployment.return_value = dep
        return apps_v1, core_v1

    def _k8s_types_mock(self):
        """Return a mock kubernetes.client module with V1* constructors."""
        k8s = MagicMock()
        # Make V1* constructors return simple MagicMocks
        for cls in [
            "V1Container", "V1ContainerPort", "V1ResourceRequirements",
            "V1EnvVar", "V1PodSpec", "V1Deployment", "V1ObjectMeta",
            "V1DeploymentSpec", "V1LabelSelector", "V1PodTemplateSpec",
            "V1Service", "V1ServiceSpec", "V1ServicePort",
        ]:
            setattr(k8s, cls, MagicMock(return_value=MagicMock()))
        return k8s

    def test_provision_creates_deployment_and_service(self, monkeypatch):
        monkeypatch.setenv("FLEXAI_K8S_IMAGE", "registry.local/flexai:latest")
        monkeypatch.setenv("FLEXAI_K8S_NAMESPACE", "flexai")
        apps_v1, core_v1 = self._k8s_mock()
        k8s_types = self._k8s_types_mock()

        with patch("app.providers.onprem._k8s_clients", return_value=(apps_v1, core_v1)), \
             patch("app.providers.onprem._k8s_types", return_value=k8s_types):
            result = self._make_adapter().provision("dep-k8s-01", {"replicas": 2})

        apps_v1.create_namespaced_deployment.assert_called_once()
        core_v1.create_namespaced_service.assert_called_once()
        assert result["provider"] == "onprem"
        assert result["replicas"] == 2

    def test_get_status_running_when_all_replicas_ready(self, monkeypatch):
        monkeypatch.setenv("FLEXAI_K8S_NAMESPACE", "flexai")
        apps_v1, core_v1 = self._k8s_mock(ready=2, desired=2)

        with patch("app.providers.onprem._k8s_clients", return_value=(apps_v1, core_v1)):
            status = self._make_adapter().get_status("dep-k8s-01", {"deployment_name": "flexai-dep-k8s-01"})
        assert status == "running"

    def test_get_status_pending_when_none_ready(self, monkeypatch):
        monkeypatch.setenv("FLEXAI_K8S_NAMESPACE", "flexai")
        apps_v1, core_v1 = self._k8s_mock(ready=0, desired=2)

        with patch("app.providers.onprem._k8s_clients", return_value=(apps_v1, core_v1)):
            status = self._make_adapter().get_status("dep-k8s-01", {"deployment_name": "flexai-dep-k8s-01"})
        assert status == "pending"

    def test_scale_patches_replica_count(self, monkeypatch):
        monkeypatch.setenv("FLEXAI_K8S_NAMESPACE", "flexai")
        apps_v1, core_v1 = self._k8s_mock()

        with patch("app.providers.onprem._k8s_clients", return_value=(apps_v1, core_v1)):
            self._make_adapter().scale("dep-k8s-01", {"deployment_name": "flexai-dep-k8s-01"}, replicas=5)

        apps_v1.patch_namespaced_deployment_scale.assert_called_once()
        call_body = apps_v1.patch_namespaced_deployment_scale.call_args[1]["body"]
        assert call_body["spec"]["replicas"] == 5

    def test_missing_image_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("FLEXAI_K8S_IMAGE", raising=False)
        apps_v1, core_v1 = self._k8s_mock()
        k8s_types = self._k8s_types_mock()

        with patch("app.providers.onprem._k8s_clients", return_value=(apps_v1, core_v1)), \
             patch("app.providers.onprem._k8s_types", return_value=k8s_types):
            with pytest.raises(RuntimeError, match="FLEXAI_K8S_IMAGE"):
                self._make_adapter().provision("dep-k8s-01", {})
