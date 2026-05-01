import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .artifacts import (
    ArtifactRecord,
    PresignedUploadResponse,
    generate_presigned_upload,
    store_local_artifact,
)
from .budget import BudgetExceededError
from .healthpoller import run_health_poller
from .logstream import stream_logs
from .models import ABTestRequest, CanaryUpdateRequest, DeploymentRequest, MeshUpdateRequest, PlatformOverview, RollbackRequest
from .observability import setup_observability
from .providers import get_provider
from .security import AuthContext, read_auth_audit_events, require_role, _verify_token
from .services import DeploymentService
from .store import DeploymentStore

store = DeploymentStore(Path(__file__).resolve().parents[1] / "data" / "deployments.json")
service = DeploymentService(store)


def _rate_limit_key(request: Request) -> str:
    """Key rate limits by JWT subject when available, falling back to remote IP."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        # Use the raw token as the bucket key — avoids decoding overhead here.
        return auth_header[len("Bearer "):].strip() or get_remote_address(request)
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)


@asynccontextmanager
async def _lifespan(application: FastAPI):  # noqa: ARG001
    """Start background tasks on startup; cancel them on shutdown."""
    poller_task = asyncio.create_task(
        run_health_poller(store, get_provider)
    )
    try:
        yield
    finally:
        poller_task.cancel()
        try:
            await poller_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="FlexAI Deployment Engine", version="0.1.0", lifespan=_lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_observability(app)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/overview", response_model=PlatformOverview)
@limiter.limit("120/minute")
def overview(request: Request, _: AuthContext = Depends(require_role("viewer"))) -> PlatformOverview:
    return service.platform_overview()


@app.get("/api/deployments")
@limiter.limit("120/minute")
def list_deployments(request: Request, _: AuthContext = Depends(require_role("viewer"))):
    return service.list_deployments()


@app.post("/api/deployments")
@limiter.limit("20/minute")
def create_deployment(
    request: Request,
    body: DeploymentRequest,
    _: AuthContext = Depends(require_role("operator")),
):
    try:
        return service.create_deployment(body)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc


@app.get("/api/deployments/{deployment_id}")
@limiter.limit("120/minute")
def get_deployment(
    request: Request,
    deployment_id: str,
    _: AuthContext = Depends(require_role("viewer")),
):
    deployment = service.get_deployment(deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return deployment


@app.get("/api/deployments/{deployment_id}/status")
@limiter.limit("120/minute")
def get_deployment_status(
    request: Request,
    deployment_id: str,
    _: AuthContext = Depends(require_role("viewer")),
):
    """Lightweight status endpoint — returns status + timestamps only."""
    deployment = service.store.get(deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return {
        "deployment_id": deployment.deployment_id,
        "status": deployment.status,
        "updated_at": deployment.updated_at.isoformat(),
        "model_name": deployment.model_name,
        "cloud": deployment.cloud,
        "region": deployment.region,
    }


@app.post("/api/deployments/{deployment_id}/rollback")
@limiter.limit("20/minute")
def rollback(
    request: Request,
    deployment_id: str,
    body: RollbackRequest,
    _: AuthContext = Depends(require_role("admin")),
):
    try:
        deployment = service.rollback(deployment_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return deployment


class _ScaleRequest(BaseModel):
    replicas: int


@app.patch("/api/deployments/{deployment_id}/scale")
@limiter.limit("20/minute")
def scale_deployment(
    request: Request,
    deployment_id: str,
    body: _ScaleRequest,
    _: AuthContext = Depends(require_role("operator")),
):
    """Adjust replica count; delegates to the cloud provider adapter."""
    try:
        deployment = service.scale_deployment(deployment_id, body.replicas)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return deployment


@app.delete("/api/deployments/{deployment_id}", status_code=204)
@limiter.limit("20/minute")
def delete_deployment(
    request: Request,
    deployment_id: str,
    _: AuthContext = Depends(require_role("admin")),
):
    """Tear down cloud resources and remove the deployment record."""
    found = service.teardown_deployment(deployment_id)
    if not found:
        raise HTTPException(status_code=404, detail="Deployment not found")


@app.post("/api/deployments/{deployment_id}/ab-test")
@limiter.limit("20/minute")
def create_ab_test(
    request: Request,
    deployment_id: str,
    body: ABTestRequest,
    _: AuthContext = Depends(require_role("admin")),
):
    deployment = service.create_ab_test(deployment_id, body)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return deployment


@app.get("/api/budget")
@limiter.limit("120/minute")
def get_budget_status(
    request: Request,
    _: AuthContext = Depends(require_role("operator")),
):
    """Current hourly spend vs configured budget limits."""
    return service.budget_status()


@app.post("/api/deployments/{deployment_id}/failover/promote")
@limiter.limit("20/minute")
def promote_failover(
    request: Request,
    deployment_id: str,
    _: AuthContext = Depends(require_role("admin")),
):
    """Promote a healthy standby to primary for this deployment's failover group.

    Useful when the primary is degraded/failed and operator wants to manually
    trigger the failover rather than waiting for the health poller.
    """
    try:
        group = service.promote_failover(deployment_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "primary": group.primary,
        "standbys": group.standbys,
    }


@app.patch("/api/deployments/{deployment_id}/canary")
@limiter.limit("20/minute")
def update_canary(
    request: Request,
    deployment_id: str,
    body: CanaryUpdateRequest,
    _: AuthContext = Depends(require_role("operator")),
):
    """Adjust the live canary/A-B traffic split for a running deployment.

    Set ``canary_percent=0`` to promote baseline to 100% and clear the canary.
    """
    dep = service.update_canary(deployment_id, body)
    if dep is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return dep


@app.patch("/api/deployments/{deployment_id}/mesh")
@limiter.limit("20/minute")
def update_mesh(
    request: Request,
    deployment_id: str,
    body: MeshUpdateRequest,
    _: AuthContext = Depends(require_role("operator")),
):
    """Enable or disable service-mesh (mTLS, sidecar injection) for a deployment."""
    dep = service.update_mesh(deployment_id, body)
    if dep is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return dep


@app.get("/api/auth/audit")
@limiter.limit("60/minute")
def get_auth_audit_events(
    request: Request,
    limit: int = 100,
    decision: str | None = None,
    path_contains: str | None = None,
    _: AuthContext = Depends(require_role("admin")),
):
    return read_auth_audit_events(
        limit=limit,
        decision=decision,
        path_contains=path_contains,
    )


@app.get("/api/auth/whoami")
@limiter.limit("120/minute")
def whoami(request: Request, auth: AuthContext = Depends(require_role("viewer"))):
    return {
        "subject": auth.subject,
        "role": auth.role,
    }


# ---------------------------------------------------------------------------
# Artifact ingestion
# ---------------------------------------------------------------------------

class _PresignRequest(BaseModel):
    filename: str


@app.post("/api/artifacts/upload", response_model=ArtifactRecord)
async def upload_artifact(
    file: UploadFile,
    auth: AuthContext = Depends(require_role("operator")),
):
    """Multipart upload — stores the artifact on the local filesystem."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    record = store_local_artifact(
        filename=file.filename or "artifact",
        data=data,
        uploaded_by=auth.subject or "anonymous",
    )
    return record


@app.post("/api/artifacts/presign", response_model=PresignedUploadResponse)
def presign_artifact_upload(
    body: _PresignRequest,
    auth: AuthContext = Depends(require_role("operator")),
):
    """Return a presigned S3 POST URL (requires FLEXAI_S3_BUCKET env var)."""
    try:
        return generate_presigned_upload(
            uploaded_by=auth.subject or "anonymous",
            filename=body.filename,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# WebSocket — live log streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/deployments/{deployment_id}/logs")
async def ws_deployment_logs(websocket: WebSocket, deployment_id: str, token: str = "") -> None:
    """
    Stream live log lines for *deployment_id*.

    Authentication: pass the Bearer token as the ``?token=`` query parameter
    (browsers cannot set custom headers on WebSocket connections).
    Minimum role: viewer.
    """
    # Authenticate before accepting the socket so we send 403 cleanly.
    try:
        _verify_token(token, required_role="viewer")
    except HTTPException as exc:
        await websocket.close(code=1008, reason=str(exc.detail))
        return

    deployment = service.get_deployment(deployment_id)
    if not deployment:
        await websocket.close(code=1008, reason="Deployment not found")
        return

    await websocket.accept()
    try:
        async for line in stream_logs(deployment_id):
            await websocket.send_text(line)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 — catch-all to avoid leaking stack traces
        pass
