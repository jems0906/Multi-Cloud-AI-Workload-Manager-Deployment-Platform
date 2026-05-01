from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx


class FlexAIClient:
    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = base_url or os.getenv("FLEXAI_API_URL", "http://127.0.0.1:8000")
        self.token = token or os.getenv("FLEXAI_TOKEN", "local-dev-token")

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def deploy(self, artifact_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        body = {**payload, "artifact_path": str(artifact_path)}
        response = httpx.post(f"{self.base_url}/api/deployments", json=body, headers=self.headers, timeout=60)
        response.raise_for_status()
        return response.json()

    def list_deployments(self) -> list[dict[str, Any]]:
        response = httpx.get(f"{self.base_url}/api/deployments", headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/api/deployments/{deployment_id}", headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def rollback(self, deployment_id: str, target_version: str) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/api/deployments/{deployment_id}/rollback",
            json={"target_version": target_version},
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def ab_test(self, deployment_id: str, version: str, image_uri: str, weight: int) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/api/deployments/{deployment_id}/ab-test",
            json={
                "challenger_version": version,
                "challenger_image_uri": image_uri,
                "challenger_weight": weight,
            },
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def scale(self, deployment_id: str, replicas: int) -> dict[str, Any]:
        response = httpx.patch(
            f"{self.base_url}/api/deployments/{deployment_id}/scale",
            json={"replicas": replicas},
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def delete(self, deployment_id: str) -> None:
        response = httpx.delete(
            f"{self.base_url}/api/deployments/{deployment_id}",
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()

    def update_canary(self, deployment_id: str, canary_percent: int) -> dict[str, Any]:
        response = httpx.patch(
            f"{self.base_url}/api/deployments/{deployment_id}/canary",
            json={"canary_percent": canary_percent},
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def update_mesh(self, deployment_id: str, enabled: bool) -> dict[str, Any]:
        response = httpx.patch(
            f"{self.base_url}/api/deployments/{deployment_id}/mesh",
            json={"enabled": enabled},
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def budget_status(self) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/api/budget", headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def promote_failover(self, deployment_id: str) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/api/deployments/{deployment_id}/failover/promote",
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def overview(self) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/api/overview", headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()
