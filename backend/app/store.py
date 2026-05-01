from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Dict, List

from .models import DeploymentDetail


class DeploymentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def list(self) -> List[DeploymentDetail]:
        with self._lock:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [DeploymentDetail.model_validate(item) for item in payload.values()]

    def get(self, deployment_id: str) -> DeploymentDetail | None:
        with self._lock:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        item = payload.get(deployment_id)
        return DeploymentDetail.model_validate(item) if item else None

    def upsert(self, deployment: DeploymentDetail) -> DeploymentDetail:
        with self._lock:
            payload: Dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8"))
            payload[deployment.deployment_id] = deployment.model_dump(mode="json")
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return deployment

    def delete(self, deployment_id: str) -> bool:
        """Remove a deployment from the store. Returns True if it existed."""
        with self._lock:
            payload: Dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8"))
            if deployment_id not in payload:
                return False
            del payload[deployment_id]
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
