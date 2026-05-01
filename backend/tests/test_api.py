import json
from datetime import datetime, timedelta, timezone

import pytest
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app
from app import security

client = TestClient(app)
AUTH_HEADERS = {"Authorization": "Bearer local-dev-token"}


def _jwt_token(payload: dict[str, object], secret: str = "test-secret") -> str:
    return jwt.encode(payload, secret, algorithm="HS256")


def _rs256_keypair() -> tuple[str, str]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return private_pem, public_pem


def test_create_deployment_and_overview() -> None:
    response = client.post(
        "/api/deployments",
        headers=AUTH_HEADERS,
        json={
            "model_name": "vision-prod",
            "artifact_path": "model.ckpt",
            "gpu": "A100",
            "region": "us-west",
            "cloud": "aws",
            "replicas": 2,
            "min_replicas": 1,
            "max_replicas": 8,
        },
    )
    assert response.status_code == 200
    deployment_id = response.json()["deployment_id"]

    detail = client.get(f"/api/deployments/{deployment_id}", headers=AUTH_HEADERS)
    assert detail.status_code == 200
    assert detail.json()["status"] in {"running", "rolled_back"}

    overview = client.get("/api/overview", headers=AUTH_HEADERS)
    assert overview.status_code == 200
    assert overview.json()["deployments"]


def test_auth_required() -> None:
    response = client.get("/api/overview")
    assert response.status_code == 401


def test_rbac_blocks_insufficient_role() -> None:
    operator_headers = {"Authorization": "Bearer local-operator-token"}

    created = client.post(
        "/api/deployments",
        headers=operator_headers,
        json={
            "model_name": "rbac-model",
            "artifact_path": "model.ckpt",
            "gpu": "A100",
            "region": "us-west",
            "cloud": "aws",
            "replicas": 1,
            "min_replicas": 1,
            "max_replicas": 4,
        },
    )
    assert created.status_code == 200
    deployment_id = created.json()["deployment_id"]

    forbidden = client.post(
        f"/api/deployments/{deployment_id}/rollback",
        headers=operator_headers,
        json={"target_version": "v1"},
    )
    assert forbidden.status_code == 403


def test_expired_jwt_token_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("FLEXAI_JWT_HS256_SECRET", "test-secret")
    monkeypatch.setenv("FLEXAI_JWT_VERIFY_SIGNATURE", "true")
    monkeypatch.delenv("FLEXAI_JWT_ISSUER", raising=False)
    monkeypatch.delenv("FLEXAI_JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("FLEXAI_JWT_JWKS_URL", raising=False)

    expired_payload = {
        "sub": "demo-user",
        "role": "viewer",
        "exp": int((datetime.now(timezone.utc) - timedelta(minutes=1)).timestamp()),
    }
    headers = {"Authorization": f"Bearer {_jwt_token(expired_payload)}"}

    response = client.get("/api/overview", headers=headers)
    assert response.status_code == 401


def test_invalid_jwt_signature_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("FLEXAI_JWT_HS256_SECRET", "correct-secret")
    monkeypatch.setenv("FLEXAI_JWT_VERIFY_SIGNATURE", "true")
    monkeypatch.delenv("FLEXAI_JWT_ISSUER", raising=False)
    monkeypatch.delenv("FLEXAI_JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("FLEXAI_JWT_JWKS_URL", raising=False)

    payload = {
        "sub": "demo-user",
        "role": "viewer",
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()),
    }
    token = _jwt_token(payload, secret="wrong-secret")
    response = client.get("/api/overview", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_jwt_with_valid_issuer_and_audience_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("FLEXAI_JWT_HS256_SECRET", "test-secret")
    monkeypatch.setenv("FLEXAI_JWT_VERIFY_SIGNATURE", "true")
    monkeypatch.setenv("FLEXAI_JWT_ISSUER", "https://issuer.flexai.local")
    monkeypatch.setenv("FLEXAI_JWT_AUDIENCE", "flexai-api")
    monkeypatch.delenv("FLEXAI_JWT_JWKS_URL", raising=False)

    payload = {
        "sub": "demo-user",
        "role": "viewer",
        "iss": "https://issuer.flexai.local",
        "aud": "flexai-api",
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()),
    }
    token = _jwt_token(payload)

    response = client.get("/api/overview", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_jwt_with_invalid_issuer_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("FLEXAI_JWT_HS256_SECRET", "test-secret")
    monkeypatch.setenv("FLEXAI_JWT_VERIFY_SIGNATURE", "true")
    monkeypatch.setenv("FLEXAI_JWT_ISSUER", "https://issuer.flexai.local")
    monkeypatch.delenv("FLEXAI_JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("FLEXAI_JWT_JWKS_URL", raising=False)

    payload = {
        "sub": "demo-user",
        "role": "viewer",
        "iss": "https://evil-issuer.local",
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()),
    }
    token = _jwt_token(payload)

    response = client.get("/api/overview", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_jwt_with_invalid_audience_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("FLEXAI_JWT_HS256_SECRET", "test-secret")
    monkeypatch.setenv("FLEXAI_JWT_VERIFY_SIGNATURE", "true")
    monkeypatch.delenv("FLEXAI_JWT_ISSUER", raising=False)
    monkeypatch.setenv("FLEXAI_JWT_AUDIENCE", "flexai-api")
    monkeypatch.delenv("FLEXAI_JWT_JWKS_URL", raising=False)

    payload = {
        "sub": "demo-user",
        "role": "viewer",
        "aud": "some-other-api",
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()),
    }
    token = _jwt_token(payload)

    response = client.get("/api/overview", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_rs256_jwks_token_is_accepted(monkeypatch) -> None:
    private_key, public_key = _rs256_keypair()

    class FakeSigningKey:
        def __init__(self, key: str) -> None:
            self.key = key

    class FakeJWKClient:
        def __init__(self, key: str) -> None:
            self.key = key

        def get_signing_key_from_jwt(self, token: str) -> FakeSigningKey:
            return FakeSigningKey(self.key)

    monkeypatch.setenv("FLEXAI_JWT_VERIFY_SIGNATURE", "true")
    monkeypatch.setenv("FLEXAI_JWT_JWKS_URL", "https://idp.example.local/.well-known/jwks.json")
    monkeypatch.delenv("FLEXAI_JWT_HS256_SECRET", raising=False)
    monkeypatch.delenv("FLEXAI_JWT_ISSUER", raising=False)
    monkeypatch.delenv("FLEXAI_JWT_AUDIENCE", raising=False)
    monkeypatch.setattr(security, "_jwks_client", lambda _: FakeJWKClient(public_key))

    payload = {
        "sub": "demo-user",
        "role": "viewer",
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()),
    }
    token = jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": "test-key"})

    response = client.get("/api/overview", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_rs256_jwks_token_with_wrong_key_is_rejected(monkeypatch) -> None:
    private_key, _ = _rs256_keypair()
    _, wrong_public_key = _rs256_keypair()

    class FakeSigningKey:
        def __init__(self, key: str) -> None:
            self.key = key

    class FakeJWKClient:
        def __init__(self, key: str) -> None:
            self.key = key

        def get_signing_key_from_jwt(self, token: str) -> FakeSigningKey:
            return FakeSigningKey(self.key)

    monkeypatch.setenv("FLEXAI_JWT_VERIFY_SIGNATURE", "true")
    monkeypatch.setenv("FLEXAI_JWT_JWKS_URL", "https://idp.example.local/.well-known/jwks.json")
    monkeypatch.delenv("FLEXAI_JWT_HS256_SECRET", raising=False)
    monkeypatch.delenv("FLEXAI_JWT_ISSUER", raising=False)
    monkeypatch.delenv("FLEXAI_JWT_AUDIENCE", raising=False)
    monkeypatch.setattr(security, "_jwks_client", lambda _: FakeJWKClient(wrong_public_key))

    payload = {
        "sub": "demo-user",
        "role": "viewer",
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()),
    }
    token = jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": "test-key"})

    response = client.get("/api/overview", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_auth_audit_log_records_allow_event(monkeypatch, tmp_path) -> None:
    audit_file = tmp_path / "auth_audit.log"
    monkeypatch.setattr(security, "_AUDIT_LOG_PATH", audit_file)

    response = client.get("/api/overview", headers={"Authorization": "Bearer local-viewer-token"})
    assert response.status_code == 200
    assert audit_file.exists()

    events = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()]
    last_event = events[-1]
    assert last_event["decision"] == "allow"
    assert last_event["required_role"] == "viewer"
    assert last_event["resolved_role"] == "viewer"
    assert last_event["path"] == "/api/overview"


def test_auth_audit_log_records_deny_event(monkeypatch, tmp_path) -> None:
    audit_file = tmp_path / "auth_audit.log"
    monkeypatch.setattr(security, "_AUDIT_LOG_PATH", audit_file)

    response = client.post(
        "/api/deployments/non-existent/rollback",
        headers={"Authorization": "Bearer local-operator-token"},
        json={"target_version": "v1"},
    )
    assert response.status_code == 403
    assert audit_file.exists()

    events = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()]
    last_event = events[-1]
    assert last_event["decision"] == "deny"
    assert last_event["required_role"] == "admin"
    assert last_event["resolved_role"] == "operator"
    assert last_event["detail"] == "Insufficient role"
    assert last_event["path"] == "/api/deployments/non-existent/rollback"


def test_admin_can_query_auth_audit_endpoint(monkeypatch, tmp_path) -> None:
    audit_file = tmp_path / "auth_audit.log"
    monkeypatch.setattr(security, "_AUDIT_LOG_PATH", audit_file)

    # Generate both allow and deny entries before querying.
    ok = client.get("/api/overview", headers={"Authorization": "Bearer local-viewer-token"})
    assert ok.status_code == 200
    denied = client.post(
        "/api/deployments/non-existent/rollback",
        headers={"Authorization": "Bearer local-operator-token"},
        json={"target_version": "v1"},
    )
    assert denied.status_code == 403

    response = client.get(
        "/api/auth/audit?limit=5&decision=deny&path_contains=rollback",
        headers={"Authorization": "Bearer local-dev-token"},
    )
    assert response.status_code == 200
    events = response.json()
    assert len(events) >= 1
    assert all(item["decision"] == "deny" for item in events)
    assert all("rollback" in item["path"] for item in events)


def test_non_admin_cannot_query_auth_audit_endpoint() -> None:
    response = client.get(
        "/api/auth/audit",
        headers={"Authorization": "Bearer local-operator-token"},
    )
    assert response.status_code == 403


def test_whoami_returns_role_for_viewer() -> None:
    response = client.get(
        "/api/auth/whoami",
        headers={"Authorization": "Bearer local-viewer-token"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["role"] == "viewer"
    assert payload["subject"] is None


def test_whoami_requires_auth() -> None:
    response = client.get("/api/auth/whoami")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

def test_metrics_endpoint_returns_prometheus_text() -> None:
    # Warm up at least one instrumented route so counters exist
    client.get("/healthz")
    response = client.get("/metrics")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type
    body = response.text
    # prometheus_fastapi_instrumentator emits these standard metric families
    assert "http_requests_total" in body or "http_request_duration" in body


def test_otel_tracer_provider_is_configured() -> None:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider), (
        "OTel TracerProvider should be an SDK TracerProvider, not a no-op"
    )


# ---------------------------------------------------------------------------
# Artifact ingestion
# ---------------------------------------------------------------------------

def test_upload_artifact_stores_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FLEXAI_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    content = b"fake model weights" * 100
    response = client.post(
        "/api/artifacts/upload",
        headers={"Authorization": "Bearer local-dev-token"},
        files={"file": ("model.ckpt", content, "application/octet-stream")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["storage"] == "local"
    assert body["filename"] == "model.ckpt"
    assert body["size_bytes"] == len(content)
    assert len(body["sha256"]) == 64
    # Static bearer tokens carry no JWT subject; the server records "anonymous"
    assert body["uploaded_by"] == "anonymous"
    # Verify file actually landed on disk
    assert (tmp_path / "artifacts").exists()


def test_upload_artifact_requires_operator_role() -> None:
    content = b"weights"
    response = client.post(
        "/api/artifacts/upload",
        headers={"Authorization": "Bearer local-viewer-token"},
        files={"file": ("model.ckpt", content, "application/octet-stream")},
    )
    assert response.status_code == 403


def test_upload_artifact_rejects_empty_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FLEXAI_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    response = client.post(
        "/api/artifacts/upload",
        headers={"Authorization": "Bearer local-dev-token"},
        files={"file": ("empty.ckpt", b"", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_presign_returns_501_when_s3_bucket_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("FLEXAI_S3_BUCKET", raising=False)

    response = client.post(
        "/api/artifacts/presign",
        headers={"Authorization": "Bearer local-dev-token"},
        json={"filename": "model.ckpt"},
    )
    assert response.status_code == 501
    assert "FLEXAI_S3_BUCKET" in response.json()["detail"]


# ---------------------------------------------------------------------------
# WebSocket live log streaming
# ---------------------------------------------------------------------------

def _make_deployment() -> str:
    """Create a deployment and return its ID."""
    resp = client.post(
        "/api/deployments",
        headers=AUTH_HEADERS,
        json={
            "model_name": "log-test-model",
            "artifact_path": "model.ckpt",
            "gpu": "A100",
            "region": "us-west",
            "cloud": "onprem",
            "replicas": 1,
            "min_replicas": 1,
            "max_replicas": 2,
        },
    )
    assert resp.status_code == 200
    return resp.json()["deployment_id"]


def test_ws_log_stream_delivers_lines() -> None:
    """WS endpoint sends at least the history batch (may be empty) then live lines."""
    dep_id = _make_deployment()

    from app.logstream import publish_log
    import asyncio

    # Pre-seed two log lines so the history snapshot is non-empty.
    asyncio.get_event_loop().run_until_complete(publish_log(dep_id, "line-alpha"))
    asyncio.get_event_loop().run_until_complete(publish_log(dep_id, "line-beta"))

    received: list[str] = []
    with client.websocket_connect(
        f"/ws/deployments/{dep_id}/logs?token=local-dev-token"
    ) as ws:
        # Read up to 5 messages (history + any live) with a short timeout.
        for _ in range(5):
            try:
                msg = ws.receive_text()
                received.append(msg)
                if len(received) >= 2:
                    break
            except Exception:
                break

    assert "line-alpha" in received
    assert "line-beta" in received


def test_ws_log_stream_rejects_missing_token() -> None:
    dep_id = _make_deployment()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            f"/ws/deployments/{dep_id}/logs?token="
        ) as ws:
            ws.receive_text()


def test_ws_log_stream_rejects_unknown_deployment() -> None:
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ws/deployments/does-not-exist/logs?token=local-dev-token"
        ) as ws:
            ws.receive_text()


# ---------------------------------------------------------------------------
# Health poller + status endpoint
# ---------------------------------------------------------------------------

def test_status_endpoint_returns_deployment_status() -> None:
    dep = client.post(
        "/api/deployments",
        headers=AUTH_HEADERS,
        json={
            "model_name": "status-model",
            "artifact_path": "model.ckpt",
            "gpu": "T4",
            "region": "us-east",
            "cloud": "gcp",
            "replicas": 1,
            "min_replicas": 1,
            "max_replicas": 2,
        },
    )
    assert dep.status_code == 200
    dep_id = dep.json()["deployment_id"]

    resp = client.get(f"/api/deployments/{dep_id}/status", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["deployment_id"] == dep_id
    assert "status" in body
    assert "updated_at" in body
    assert body["cloud"] == "gcp"


def test_status_endpoint_returns_404_for_unknown() -> None:
    resp = client.get("/api/deployments/no-such-id/status", headers=AUTH_HEADERS)
    assert resp.status_code == 404


def test_status_endpoint_requires_viewer_role() -> None:
    resp = client.get("/api/deployments/any/status")
    assert resp.status_code == 401


def test_health_poller_updates_status_on_change() -> None:
    """
    Unit-test the poller internals: inject a provider whose get_status returns
    'error', run one poll cycle, verify the store is updated.
    """
    import asyncio
    from unittest.mock import MagicMock
    from app.healthpoller import _poll_one
    from app.models import DeploymentStatus
    from app.store import DeploymentStore
    from app.services import DeploymentService

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        tmp_store = DeploymentStore(pathlib.Path(tmp) / "deps.json")
        svc = DeploymentService(tmp_store)

        # Create a deployment
        from app.models import DeploymentRequest, CloudProvider
        req = DeploymentRequest(
            model_name="poller-test",
            artifact_path="model.ckpt",
            gpu="A100",
            region="us-west",
            cloud=CloudProvider.aws,
            replicas=1,
            min_replicas=1,
            max_replicas=2,
        )
        dep = svc.create_deployment(req)
        assert dep.status == DeploymentStatus.running

        # Mock a provider that returns "error"
        mock_provider = MagicMock()
        mock_provider.get_status.return_value = "error"

        def mock_get_provider(cloud: str):
            return mock_provider

        asyncio.get_event_loop().run_until_complete(
            _poll_one(dep.deployment_id, tmp_store, mock_get_provider)
        )

        updated = tmp_store.get(dep.deployment_id)
        assert updated is not None
        assert updated.status == DeploymentStatus.failed
        assert any("health-poller" in entry for entry in updated.audit_trail)


def test_health_poller_skips_on_runtime_error() -> None:
    """Poller must not crash when provider raises RuntimeError (SDK missing)."""
    import asyncio
    from unittest.mock import MagicMock
    from app.healthpoller import _poll_one
    from app.store import DeploymentStore
    from app.services import DeploymentService

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        tmp_store = DeploymentStore(pathlib.Path(tmp) / "deps.json")
        svc = DeploymentService(tmp_store)

        from app.models import DeploymentRequest, CloudProvider
        req = DeploymentRequest(
            model_name="skip-test",
            artifact_path="model.ckpt",
            gpu="A100",
            region="us-west",
            cloud=CloudProvider.azure,
            replicas=1,
            min_replicas=1,
            max_replicas=2,
        )
        dep = svc.create_deployment(req)
        original_status = dep.status

        mock_provider = MagicMock()
        mock_provider.get_status.side_effect = RuntimeError("SDK not installed")

        def mock_get_provider(cloud: str):
            return mock_provider

        # Should complete without raising
        asyncio.get_event_loop().run_until_complete(
            _poll_one(dep.deployment_id, tmp_store, mock_get_provider)
        )

        # Status should be unchanged
        still = tmp_store.get(dep.deployment_id)
        assert still is not None
        assert still.status == original_status


# ---------------------------------------------------------------------------
# Teardown (DELETE) and scale (PATCH) endpoints
# ---------------------------------------------------------------------------

def _create_dep(cloud: str = "aws", replicas: int = 2, max_replicas: int = 8) -> str:
    resp = client.post(
        "/api/deployments",
        headers=AUTH_HEADERS,
        json={
            "model_name": "lifecycle-model",
            "artifact_path": "model.ckpt",
            "gpu": "A100",
            "region": "us-west",
            "cloud": cloud,
            "replicas": replicas,
            "min_replicas": 1,
            "max_replicas": max_replicas,
        },
    )
    assert resp.status_code == 200
    return resp.json()["deployment_id"]


def test_scale_endpoint_updates_replicas() -> None:
    dep_id = _create_dep(replicas=2, max_replicas=8)

    resp = client.patch(
        f"/api/deployments/{dep_id}/scale",
        headers=AUTH_HEADERS,
        json={"replicas": 4},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["replicas"] == 4
    assert any("Scaled" in entry for entry in body["audit_trail"])


def test_scale_rejects_out_of_bounds() -> None:
    dep_id = _create_dep(replicas=2, max_replicas=4)

    resp = client.patch(
        f"/api/deployments/{dep_id}/scale",
        headers={"Authorization": "Bearer local-dev-token"},
        json={"replicas": 99},
    )
    assert resp.status_code == 400


def test_scale_requires_operator_role() -> None:
    dep_id = _create_dep()
    resp = client.patch(
        f"/api/deployments/{dep_id}/scale",
        headers={"Authorization": "Bearer local-viewer-token"},
        json={"replicas": 1},
    )
    assert resp.status_code == 403


def test_delete_removes_deployment() -> None:
    dep_id = _create_dep()

    resp = client.delete(f"/api/deployments/{dep_id}", headers=AUTH_HEADERS)
    assert resp.status_code == 204

    # Deployment should be gone
    get_resp = client.get(f"/api/deployments/{dep_id}", headers=AUTH_HEADERS)
    assert get_resp.status_code == 404


def test_delete_returns_404_for_unknown() -> None:
    resp = client.delete("/api/deployments/does-not-exist", headers=AUTH_HEADERS)
    assert resp.status_code == 404


def test_delete_requires_admin_role() -> None:
    dep_id = _create_dep()
    resp = client.delete(
        f"/api/deployments/{dep_id}",
        headers={"Authorization": "Bearer local-operator-token"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limit_triggers_on_write_endpoint() -> None:
    """POST /api/deployments is capped at 20/minute per token.
    Issuing 21 identical rapid requests from the same token must yield 429."""
    from app.main import limiter
    limiter._storage.reset()

    payload = {
        "model_name": "ratelimit-probe",
        "artifact_path": "model.ckpt",
        "gpu": "T4",
        "region": "us-east",
        "cloud": "aws",
        "replicas": 1,
        "min_replicas": 1,
        "max_replicas": 2,
    }
    # Use local-dev-token (admin ≥ operator role)
    headers = AUTH_HEADERS
    responses = [
        client.post("/api/deployments", headers=headers, json=payload)
        for _ in range(21)
    ]
    status_codes = [r.status_code for r in responses]
    assert 429 in status_codes, f"Expected a 429 among {set(status_codes)}"


def test_rate_limit_read_endpoint_allows_many_requests() -> None:
    """GET /api/overview is capped at 120/minute — 5 requests must all succeed."""
    from app.main import limiter
    limiter._storage.reset()

    for _ in range(5):
        resp = client.get("/api/overview", headers=AUTH_HEADERS)
        assert resp.status_code == 200


def test_rate_limit_different_tokens_have_independent_buckets() -> None:
    """Two different tokens each get their own 20/minute write quota."""
    from app.main import limiter
    limiter._storage.reset()

    payload = {
        "model_name": "bucket-probe",
        "artifact_path": "model.ckpt",
        "gpu": "T4",
        "region": "us-east",
        "cloud": "gcp",
        "replicas": 1,
        "min_replicas": 1,
        "max_replicas": 2,
    }
    token_a = AUTH_HEADERS  # local-dev-token
    token_b = {"Authorization": "Bearer local-operator-token"}
    # 21 requests with token A
    for _ in range(21):
        client.post("/api/deployments", headers=token_a, json=payload)
    # token B should still have capacity
    resp = client.post("/api/deployments", headers=token_b, json=payload)
    assert resp.status_code != 429, "Token B should not be rate-limited by token A's usage"


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

def _make_budget_service(global_limit: float | None = None, **cloud_limits: float):
    """Return a DeploymentService wired with a specific BudgetConfig."""
    import tempfile, json, pathlib
    from app.budget import BudgetConfig
    from app.services import DeploymentService
    from app.store import DeploymentStore

    tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text("{}", encoding="utf-8")
    tmp_store = DeploymentStore(tmp)
    budget = BudgetConfig(global_hourly=global_limit, cloud_hourly=cloud_limits)
    return DeploymentService(tmp_store, budget=budget)


def test_budget_endpoint_returns_status() -> None:
    resp = client.get("/api/budget", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert "global_current" in body
    assert "cloud_current" in body


def test_budget_endpoint_requires_operator() -> None:
    resp = client.get("/api/budget", headers={"Authorization": "Bearer local-viewer-token"})
    assert resp.status_code == 403


def test_budget_blocks_create_when_global_exceeded() -> None:
    from app.budget import BudgetExceededError
    from app.models import DeploymentRequest, CloudProvider

    svc = _make_budget_service(global_limit=0.01)  # $0.01/hr — any real deploy exceeds this
    req = DeploymentRequest(
        model_name="budget-test",
        artifact_path="model.ckpt",
        gpu="A100",
        region="us-east",
        cloud=CloudProvider.aws,
        replicas=1,
        min_replicas=1,
        max_replicas=2,
    )
    with pytest.raises(BudgetExceededError):
        svc.create_deployment(req)


def test_budget_allows_create_within_limit() -> None:
    from app.models import DeploymentRequest, CloudProvider

    svc = _make_budget_service(global_limit=9999.0)  # very generous
    req = DeploymentRequest(
        model_name="budget-ok",
        artifact_path="model.ckpt",
        gpu="T4",
        region="us-east",
        cloud=CloudProvider.gcp,
        replicas=1,
        min_replicas=1,
        max_replicas=4,
    )
    dep = svc.create_deployment(req)
    assert dep.deployment_id is not None


def test_budget_blocks_scale_when_cloud_limit_exceeded() -> None:
    from app.budget import BudgetExceededError
    from app.models import DeploymentRequest, CloudProvider

    svc = _make_budget_service(aws=5.0)  # $5/hr AWS cap
    # Deploy 1 A100 replica at ~$5.20/hr (just over the cap)
    req = DeploymentRequest(
        model_name="scale-budget-test",
        artifact_path="model.ckpt",
        gpu="A100",
        region="us-east",
        cloud=CloudProvider.aws,
        replicas=1,
        min_replicas=1,
        max_replicas=4,
    )
    # Initial deploy may succeed if the simulated cost ≤ 5.0.
    # Force the deployment into store and then try to scale up.
    from app.budget import BudgetConfig
    from app.models import DeploymentDetail, DeploymentStatus, RoutingStrategy, ModelVersion
    from datetime import datetime, timezone

    dep = DeploymentDetail(
        deployment_id="scale-budget-dep",
        model_name="scale-budget-test",
        status=DeploymentStatus.running,
        region="us-east",
        cloud=CloudProvider.aws,
        gpu="A100",
        endpoint="https://x.flexai.run/infer",
        runtime="custom",
        replicas=1,
        min_replicas=1,
        max_replicas=4,
        versions=[ModelVersion(version="v1", image_uri="img")],
        metrics={"estimated_hourly_cost": 4.9, "gpu_utilization": 0, "inference_latency_ms": 0, "throughput_rps": 0},
    )
    svc.store.upsert(dep)

    with pytest.raises(BudgetExceededError):
        svc.scale_deployment("scale-budget-dep", 4)  # 4x would cost ~$19.6, over $5 cap


def test_budget_create_returns_402_via_api() -> None:
    """The API must return 402 when budget enforcement blocks a deployment."""
    import os
    from app.main import limiter
    limiter._storage.reset()

    original = os.environ.get("FLEXAI_BUDGET_HOURLY_GLOBAL")
    os.environ["FLEXAI_BUDGET_HOURLY_GLOBAL"] = "0.001"  # effectively $0 — blocks everything
    try:
        # Recreate the service with the new env so budget config is reloaded
        from app import main as app_main
        from app.budget import BudgetConfig
        old_budget = app_main.service._budget
        app_main.service._budget = BudgetConfig.from_env()
        try:
            resp = client.post(
                "/api/deployments",
                headers=AUTH_HEADERS,
                json={
                    "model_name": "budget-block-test",
                    "artifact_path": "model.ckpt",
                    "gpu": "A100",
                    "region": "us-west",
                    "cloud": "aws",
                    "replicas": 1,
                    "min_replicas": 1,
                    "max_replicas": 2,
                },
            )
            assert resp.status_code == 402, f"Expected 402, got {resp.status_code}: {resp.text}"
        finally:
            app_main.service._budget = old_budget
    finally:
        if original is None:
            os.environ.pop("FLEXAI_BUDGET_HOURLY_GLOBAL", None)
        else:
            os.environ["FLEXAI_BUDGET_HOURLY_GLOBAL"] = original


# ---------------------------------------------------------------------------
# Rollback calls provider
# ---------------------------------------------------------------------------

def _make_deployment_with_versions(service, cloud: str = "aws") -> str:
    """Create a deployment and register a second version for rollback tests."""
    from app.models import DeploymentRequest, CloudProvider

    req = DeploymentRequest(
        model_name="rollback-model",
        artifact_path="model.ckpt",
        gpu="T4",
        region="us-east",
        cloud=CloudProvider(cloud),
        replicas=1,
        min_replicas=1,
        max_replicas=4,
    )
    dep = service.create_deployment(req)

    # Inject a second version so rollback has a valid target
    from app.models import ModelVersion
    dep.versions.append(ModelVersion(version="v0", image_uri="registry.flexai.local/aws/rollback-model:v0"))
    service.store.upsert(dep)
    return dep.deployment_id


def test_rollback_updates_status_and_version() -> None:
    """rollback() must set status=rolled_back and active_version to the target."""
    from app.budget import BudgetConfig
    from app.services import DeploymentService
    from app.store import DeploymentStore
    from app.models import RollbackRequest
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text("{}", encoding="utf-8")
    svc = DeploymentService(DeploymentStore(tmp), budget=BudgetConfig(global_hourly=9999.0, cloud_hourly={}))

    dep_id = _make_deployment_with_versions(svc)
    result = svc.rollback(dep_id, RollbackRequest(target_version="v0"))

    assert result is not None
    assert result.active_version == "v0"
    assert result.status.value == "rolled_back"
    assert any("v0" in entry for entry in result.audit_trail)


def test_rollback_logs_image_uri() -> None:
    """rollback() must log a message referencing the resolved image URI."""
    from app.budget import BudgetConfig
    from app.services import DeploymentService
    from app.store import DeploymentStore
    from app.models import RollbackRequest
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text("{}", encoding="utf-8")
    svc = DeploymentService(DeploymentStore(tmp), budget=BudgetConfig(global_hourly=9999.0, cloud_hourly={}))

    dep_id = _make_deployment_with_versions(svc)
    result = svc.rollback(dep_id, RollbackRequest(target_version="v0"))

    assert result is not None
    log_text = " ".join(result.logs)
    assert "v0" in log_text
    # The provider RuntimeError (no SDK env vars) should be caught gracefully —
    # status must still be rolled_back even though cloud call was skipped.
    assert result.status.value == "rolled_back"


def test_rollback_rejects_unknown_version() -> None:
    """rollback() must raise ValueError for a version that doesn't exist."""
    from app.budget import BudgetConfig
    from app.services import DeploymentService
    from app.store import DeploymentStore
    from app.models import RollbackRequest
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text("{}", encoding="utf-8")
    svc = DeploymentService(DeploymentStore(tmp), budget=BudgetConfig(global_hourly=9999.0, cloud_hourly={}))

    dep_id = _make_deployment_with_versions(svc)
    with pytest.raises(ValueError, match="Unknown version"):
        svc.rollback(dep_id, RollbackRequest(target_version="v999"))


def test_rollback_returns_none_for_missing_deployment() -> None:
    from app.budget import BudgetConfig
    from app.services import DeploymentService
    from app.store import DeploymentStore
    from app.models import RollbackRequest
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text("{}", encoding="utf-8")
    svc = DeploymentService(DeploymentStore(tmp), budget=BudgetConfig(global_hourly=9999.0, cloud_hourly={}))

    result = svc.rollback("nonexistent-id", RollbackRequest(target_version="v0"))
    assert result is None


def test_rollback_provider_called_with_correct_image() -> None:
    """Provider.rollback must be called with the correct image_uri for the target version."""
    from unittest.mock import patch, MagicMock
    from app.budget import BudgetConfig
    from app.services import DeploymentService
    from app.store import DeploymentStore
    from app.models import RollbackRequest
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text("{}", encoding="utf-8")
    svc = DeploymentService(DeploymentStore(tmp), budget=BudgetConfig(global_hourly=9999.0, cloud_hourly={}))
    dep_id = _make_deployment_with_versions(svc)

    mock_provider = MagicMock()
    with patch("app.services.get_provider", return_value=mock_provider):
        svc.rollback(dep_id, RollbackRequest(target_version="v0"))

    mock_provider.rollback.assert_called_once()
    call_kwargs = mock_provider.rollback.call_args
    # positional args: (deployment_id, meta, target_version, image_uri)
    args = call_kwargs[0]
    assert args[0] == dep_id
    assert args[2] == "v0"
    assert "v0" in args[3]  # image_uri contains the version tag


# ---------------------------------------------------------------------------
# Multi-region failover
# ---------------------------------------------------------------------------

def _svc_with_budget():
    from app.budget import BudgetConfig
    from app.services import DeploymentService
    from app.store import DeploymentStore
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text("{}", encoding="utf-8")
    return DeploymentService(DeploymentStore(tmp), budget=BudgetConfig(global_hourly=9999.0, cloud_hourly={}))


def _create_with_failover(svc, failover_regions: list) -> str:
    from app.models import DeploymentRequest, CloudProvider

    req = DeploymentRequest(
        model_name="failover-model",
        artifact_path="model.ckpt",
        gpu="T4",
        region="us-east",
        cloud=CloudProvider.aws,
        replicas=2,
        min_replicas=1,
        max_replicas=4,
        failover_regions=failover_regions,
    )
    dep = svc.create_deployment(req)
    return dep.deployment_id


def test_failover_standbys_created_in_requested_regions() -> None:
    svc = _svc_with_budget()
    primary_id = _create_with_failover(svc, ["eu-west", "ap-southeast"])

    all_deps = svc.store.list()
    # primary + 2 standbys = 3 total
    assert len(all_deps) == 3

    standbys = [d for d in all_deps if not d.is_primary]
    standby_regions = {d.region for d in standbys}
    assert standby_regions == {"eu-west", "ap-southeast"}


def test_failover_primary_has_group_id() -> None:
    svc = _svc_with_budget()
    primary_id = _create_with_failover(svc, ["eu-west"])

    primary = svc.store.get(primary_id)
    assert primary.failover_group_id is not None
    assert primary.is_primary is True


def test_failover_standbys_reference_same_group() -> None:
    svc = _svc_with_budget()
    primary_id = _create_with_failover(svc, ["eu-west"])
    primary = svc.store.get(primary_id)

    standbys = [d for d in svc.store.list() if not d.is_primary]
    assert all(d.failover_group_id == primary.failover_group_id for d in standbys)


def test_promote_failover_swaps_primary() -> None:
    svc = _svc_with_budget()
    primary_id = _create_with_failover(svc, ["eu-west"])

    group = svc.promote_failover(primary_id)

    # New primary must be in eu-west
    assert group.primary.region == "eu-west"
    assert group.primary.is_primary is True
    # Old primary demoted
    old_primary = svc.store.get(primary_id)
    assert old_primary.is_primary is False


def test_promote_failover_scales_new_primary_replicas() -> None:
    svc = _svc_with_budget()
    primary_id = _create_with_failover(svc, ["eu-west"])

    group = svc.promote_failover(primary_id)
    # Standby is scaled to the old primary's replica count (2)
    assert group.primary.replicas == 2


def test_promote_failover_raises_when_not_in_group() -> None:
    svc = _svc_with_budget()
    # Create a regular (non-failover) deployment
    from app.models import DeploymentRequest, CloudProvider

    req = DeploymentRequest(
        model_name="solo",
        artifact_path="model.ckpt",
        gpu="T4",
        region="us-east",
        cloud=CloudProvider.gcp,
        replicas=1,
        min_replicas=1,
        max_replicas=2,
    )
    dep = svc.create_deployment(req)

    with pytest.raises(ValueError, match="not part of a failover group"):
        svc.promote_failover(dep.deployment_id)


def test_promote_failover_raises_when_no_healthy_standby() -> None:
    svc = _svc_with_budget()
    primary_id = _create_with_failover(svc, ["eu-west"])

    # Mark standby as failed
    standbys = [d for d in svc.store.list() if not d.is_primary]
    for s in standbys:
        from app.models import DeploymentStatus
        s.status = DeploymentStatus.failed
        svc.store.upsert(s)

    with pytest.raises(RuntimeError, match="No healthy standby"):
        svc.promote_failover(primary_id)


def test_promote_failover_endpoint_returns_group() -> None:
    """POST /api/deployments/{id}/failover/promote → 200 with group payload (admin)."""
    from app import main as app_main
    from app.main import limiter
    limiter._storage.reset()

    svc = _svc_with_budget()
    primary_id = _create_with_failover(svc, ["eu-west"])

    old_service = app_main.service
    app_main.service = svc
    try:
        resp = client.post(
            f"/api/deployments/{primary_id}/failover/promote",
            headers={"Authorization": "Bearer local-dev-token"},  # admin
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "primary" in body
        assert "standbys" in body
        assert body["primary"]["region"] == "eu-west"
    finally:
        app_main.service = old_service


def test_promote_failover_endpoint_requires_admin() -> None:
    resp = client.post(
        "/api/deployments/any-id/failover/promote",
        headers={"Authorization": "Bearer local-viewer-token"},  # viewer → 403
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Autoscaler
# ---------------------------------------------------------------------------

def _make_running_dep(svc, gpu_util: float = 50.0, rps: float = 50.0,
                      replicas: int = 2, min_r: int = 1, max_r: int = 8):
    from app.models import DeploymentRequest, CloudProvider
    req = DeploymentRequest(
        model_name="as-model",
        artifact_path="model.ckpt",
        gpu="T4",
        region="us-east",
        cloud=CloudProvider.aws,
        replicas=replicas,
        min_replicas=min_r,
        max_replicas=max_r,
    )
    dep = svc.create_deployment(req)
    dep.metrics["gpu_utilization"] = gpu_util
    dep.metrics["throughput_rps"] = rps
    svc.store.upsert(dep)
    return dep


def test_autoscaler_scale_up_on_high_gpu() -> None:
    from app.autoscaler import AutoscalerConfig, evaluate_autoscale, _cooldown_tracker
    svc = _svc_with_budget()
    dep = _make_running_dep(svc, gpu_util=85.0, replicas=2, max_r=8)
    _cooldown_tracker.pop(dep.deployment_id, None)

    cfg = AutoscalerConfig(gpu_scale_up=75.0, gpu_scale_down=20.0,
                           rps_scale_up=100.0, rps_scale_down=5.0, cooldown_seconds=0)
    result = evaluate_autoscale(dep, cfg)
    assert result == 3


def test_autoscaler_scale_down_on_low_metrics() -> None:
    from app.autoscaler import AutoscalerConfig, evaluate_autoscale, _cooldown_tracker
    svc = _svc_with_budget()
    dep = _make_running_dep(svc, gpu_util=5.0, rps=1.0, replicas=4, min_r=1, max_r=8)
    _cooldown_tracker.pop(dep.deployment_id, None)

    cfg = AutoscalerConfig(gpu_scale_up=75.0, gpu_scale_down=20.0,
                           rps_scale_up=100.0, rps_scale_down=5.0, cooldown_seconds=0)
    result = evaluate_autoscale(dep, cfg)
    assert result == 3


def test_autoscaler_no_change_in_normal_band() -> None:
    from app.autoscaler import AutoscalerConfig, evaluate_autoscale, _cooldown_tracker
    svc = _svc_with_budget()
    dep = _make_running_dep(svc, gpu_util=50.0, rps=50.0, replicas=2, min_r=1, max_r=8)
    _cooldown_tracker.pop(dep.deployment_id, None)

    cfg = AutoscalerConfig(gpu_scale_up=75.0, gpu_scale_down=20.0,
                           rps_scale_up=100.0, rps_scale_down=5.0, cooldown_seconds=0)
    result = evaluate_autoscale(dep, cfg)
    assert result is None


def test_autoscaler_respects_max_replicas() -> None:
    from app.autoscaler import AutoscalerConfig, evaluate_autoscale, _cooldown_tracker
    svc = _svc_with_budget()
    dep = _make_running_dep(svc, gpu_util=95.0, replicas=4, min_r=1, max_r=4)
    _cooldown_tracker.pop(dep.deployment_id, None)

    cfg = AutoscalerConfig(gpu_scale_up=75.0, gpu_scale_down=20.0,
                           rps_scale_up=100.0, rps_scale_down=5.0, cooldown_seconds=0)
    result = evaluate_autoscale(dep, cfg)
    assert result is None  # already at max


def test_autoscaler_respects_min_replicas() -> None:
    from app.autoscaler import AutoscalerConfig, evaluate_autoscale, _cooldown_tracker
    svc = _svc_with_budget()
    dep = _make_running_dep(svc, gpu_util=1.0, rps=0.5, replicas=1, min_r=1, max_r=8)
    _cooldown_tracker.pop(dep.deployment_id, None)

    cfg = AutoscalerConfig(gpu_scale_up=75.0, gpu_scale_down=20.0,
                           rps_scale_up=100.0, rps_scale_down=5.0, cooldown_seconds=0)
    result = evaluate_autoscale(dep, cfg)
    assert result is None  # already at min


def test_autoscaler_respects_cooldown() -> None:
    from app.autoscaler import AutoscalerConfig, evaluate_autoscale, _cooldown_tracker
    from datetime import datetime, timezone
    svc = _svc_with_budget()
    dep = _make_running_dep(svc, gpu_util=95.0, replicas=2, max_r=8)

    # Inject a recent scale timestamp to simulate cooldown
    _cooldown_tracker[dep.deployment_id] = datetime.now(timezone.utc)

    cfg = AutoscalerConfig(gpu_scale_up=75.0, gpu_scale_down=20.0,
                           rps_scale_up=100.0, rps_scale_down=5.0, cooldown_seconds=300)
    result = evaluate_autoscale(dep, cfg)
    assert result is None  # suppressed by cooldown


def test_autoscaler_apply_updates_store_and_audit() -> None:
    from unittest.mock import MagicMock
    from app.autoscaler import apply_autoscale, _cooldown_tracker

    svc = _svc_with_budget()
    dep = _make_running_dep(svc, gpu_util=90.0, replicas=2, max_r=8)
    _cooldown_tracker.pop(dep.deployment_id, None)

    mock_provider = MagicMock()
    apply_autoscale(dep, 3, svc.store, lambda _cloud: mock_provider)

    updated = svc.store.get(dep.deployment_id)
    assert updated.replicas == 3
    assert any("autoscaler" in e for e in updated.audit_trail)
    mock_provider.scale.assert_called_once()


def test_autoscaler_scale_up_on_high_rps() -> None:
    from app.autoscaler import AutoscalerConfig, evaluate_autoscale, _cooldown_tracker
    svc = _svc_with_budget()
    # GPU is low but RPS is high — should still scale up
    dep = _make_running_dep(svc, gpu_util=10.0, rps=150.0, replicas=2, max_r=8)
    _cooldown_tracker.pop(dep.deployment_id, None)

    cfg = AutoscalerConfig(gpu_scale_up=75.0, gpu_scale_down=20.0,
                           rps_scale_up=100.0, rps_scale_down=5.0, cooldown_seconds=0)
    result = evaluate_autoscale(dep, cfg)
    assert result == 3


# ---------------------------------------------------------------------------
# Canary deployments
# ---------------------------------------------------------------------------

def _make_canary_dep(svc, canary_percent: int = 10):
    from app.models import DeploymentRequest, CloudProvider
    req = DeploymentRequest(
        model_name="canary-model",
        artifact_path="model.ckpt",
        gpu="T4",
        region="us-east",
        cloud=CloudProvider.aws,
        replicas=2,
        canary_percent=canary_percent,
    )
    return svc.create_deployment(req)


def test_canary_create_sets_routing_strategy() -> None:
    svc = _svc_with_budget()
    dep = _make_canary_dep(svc, canary_percent=15)
    assert dep.routing_strategy.value == "ab_test"
    assert dep.canary_percent == 15


def test_canary_create_zero_stays_single() -> None:
    svc = _svc_with_budget()
    dep = _make_canary_dep(svc, canary_percent=0)
    assert dep.routing_strategy.value == "single"
    assert dep.canary_percent == 0


def test_canary_create_audit_trail_records_split() -> None:
    svc = _svc_with_budget()
    dep = _make_canary_dep(svc, canary_percent=20)
    assert any("canary" in e.lower() for e in dep.audit_trail)


def test_canary_update_changes_percent() -> None:
    from app.models import CanaryUpdateRequest
    svc = _svc_with_budget()
    dep = _make_canary_dep(svc, canary_percent=10)

    updated = svc.update_canary(dep.deployment_id, CanaryUpdateRequest(canary_percent=30))
    assert updated.canary_percent == 30
    assert updated.routing_strategy.value == "ab_test"


def test_canary_update_to_zero_clears_canary() -> None:
    from app.models import CanaryUpdateRequest
    svc = _svc_with_budget()
    dep = _make_canary_dep(svc, canary_percent=20)

    updated = svc.update_canary(dep.deployment_id, CanaryUpdateRequest(canary_percent=0))
    assert updated.canary_percent == 0
    assert updated.routing_strategy.value == "single"


def test_canary_update_audit_trail() -> None:
    from app.models import CanaryUpdateRequest
    svc = _svc_with_budget()
    dep = _make_canary_dep(svc, canary_percent=10)

    updated = svc.update_canary(dep.deployment_id, CanaryUpdateRequest(canary_percent=50))
    assert any("[canary]" in e for e in updated.audit_trail)


def test_canary_update_not_found_returns_none() -> None:
    from app.models import CanaryUpdateRequest
    svc = _svc_with_budget()
    result = svc.update_canary("no-such-id", CanaryUpdateRequest(canary_percent=10))
    assert result is None


def test_canary_api_endpoint_returns_200() -> None:
    dep = client.post(
        "/api/deployments",
        json={
            "model_name": "canary-api-model",
            "artifact_path": "m.ckpt",
            "gpu": "T4",
            "region": "us-east",
            "cloud": "aws",
            "replicas": 1,
            "canary_percent": 10,
        },
        headers={"Authorization": "Bearer local-dev-token"},
    )
    assert dep.status_code == 200
    dep_id = dep.json()["deployment_id"]

    resp = client.patch(
        f"/api/deployments/{dep_id}/canary",
        json={"canary_percent": 25},
        headers={"Authorization": "Bearer local-dev-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["canary_percent"] == 25


def test_canary_api_requires_operator() -> None:
    resp = client.patch(
        "/api/deployments/any-id/canary",
        json={"canary_percent": 10},
        headers={"Authorization": "Bearer local-viewer-token"},
    )
    assert resp.status_code == 403


def test_ab_test_sets_canary_percent() -> None:
    from app.models import ABTestRequest
    svc = _svc_with_budget()
    dep = _make_canary_dep(svc, canary_percent=0)

    result = svc.create_ab_test(
        dep.deployment_id,
        ABTestRequest(
            challenger_version="v2",
            challenger_image_uri="registry/model:v2",
            challenger_weight=20,
        ),
    )
    assert result.canary_percent == 20
    assert result.routing_strategy.value == "ab_test"


# ---------------------------------------------------------------------------
# Service mesh
# ---------------------------------------------------------------------------

def _make_mesh_dep(svc, service_mesh: bool = True):
    from app.models import DeploymentRequest, CloudProvider
    req = DeploymentRequest(
        model_name="mesh-model",
        artifact_path="model.ckpt",
        gpu="T4",
        region="us-east",
        cloud=CloudProvider.aws,
        replicas=1,
        service_mesh=service_mesh,
    )
    return svc.create_deployment(req)


def test_mesh_create_sets_mesh_enabled_true() -> None:
    svc = _svc_with_budget()
    dep = _make_mesh_dep(svc, service_mesh=True)
    assert dep.mesh_enabled is True


def test_mesh_create_false_stays_disabled() -> None:
    svc = _svc_with_budget()
    dep = _make_mesh_dep(svc, service_mesh=False)
    assert dep.mesh_enabled is False


def test_mesh_create_true_records_audit() -> None:
    svc = _svc_with_budget()
    dep = _make_mesh_dep(svc, service_mesh=True)
    assert any("mesh" in e.lower() for e in dep.audit_trail)


def test_mesh_update_enable() -> None:
    from app.models import MeshUpdateRequest
    svc = _svc_with_budget()
    dep = _make_mesh_dep(svc, service_mesh=False)

    updated = svc.update_mesh(dep.deployment_id, MeshUpdateRequest(enabled=True))
    assert updated.mesh_enabled is True
    assert any("[mesh]" in e for e in updated.audit_trail)


def test_mesh_update_disable() -> None:
    from app.models import MeshUpdateRequest
    svc = _svc_with_budget()
    dep = _make_mesh_dep(svc, service_mesh=True)

    updated = svc.update_mesh(dep.deployment_id, MeshUpdateRequest(enabled=False))
    assert updated.mesh_enabled is False
    assert any("disabled" in e for e in updated.audit_trail)


def test_mesh_update_not_found_returns_none() -> None:
    from app.models import MeshUpdateRequest
    svc = _svc_with_budget()
    result = svc.update_mesh("no-such-id", MeshUpdateRequest(enabled=True))
    assert result is None


def test_mesh_api_endpoint_returns_200() -> None:
    dep = client.post(
        "/api/deployments",
        json={
            "model_name": "mesh-api-model",
            "artifact_path": "m.ckpt",
            "gpu": "T4",
            "region": "us-east",
            "cloud": "aws",
            "replicas": 1,
            "service_mesh": False,
        },
        headers={"Authorization": "Bearer local-dev-token"},
    )
    assert dep.status_code == 200
    dep_id = dep.json()["deployment_id"]

    resp = client.patch(
        f"/api/deployments/{dep_id}/mesh",
        json={"enabled": True},
        headers={"Authorization": "Bearer local-dev-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["mesh_enabled"] is True


def test_mesh_api_requires_operator() -> None:
    resp = client.patch(
        "/api/deployments/any-id/mesh",
        json={"enabled": True},
        headers={"Authorization": "Bearer local-viewer-token"},
    )
    assert resp.status_code == 403


def test_mesh_api_404_on_unknown_id() -> None:
    resp = client.patch(
        "/api/deployments/nonexistent-id/mesh",
        json={"enabled": True},
        headers={"Authorization": "Bearer local-dev-token"},
    )
    assert resp.status_code == 404
