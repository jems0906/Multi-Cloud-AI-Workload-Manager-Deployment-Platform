"""Artifact ingestion: local multipart upload + optional S3 presigned URLs.

Environment variables
---------------------
FLEXAI_ARTIFACT_DIR      Local directory for uploaded artifacts (default: <backend>/data/artifacts)
FLEXAI_S3_BUCKET         S3 bucket name — enables presigned URL flow when set
FLEXAI_S3_PRESIGN_TTL    Presigned URL TTL in seconds (default: 3600)

When FLEXAI_S3_BUCKET is set, boto3 must be installed and AWS credentials must
be available (env vars, ~/.aws/credentials, or IAM instance role).
"""
from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _artifact_dir() -> Path:
    env = os.getenv("FLEXAI_ARTIFACT_DIR", "")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data" / "artifacts"


def _s3_bucket() -> str:
    return os.getenv("FLEXAI_S3_BUCKET", "")


def _presign_ttl() -> int:
    return int(os.getenv("FLEXAI_S3_PRESIGN_TTL", "3600"))


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class ArtifactRecord(BaseModel):
    artifact_id: str
    filename: str
    size_bytes: int
    sha256: str
    storage: str          # "local" | "s3"
    location: str         # local path or s3://bucket/key
    uploaded_at: str
    uploaded_by: str


class PresignedUploadResponse(BaseModel):
    artifact_id: str
    upload_url: str
    fields: dict[str, Any] = Field(default_factory=dict)
    expires_in: int = 3600
    s3_key: str = ""


# ---------------------------------------------------------------------------
# Local upload
# ---------------------------------------------------------------------------

def store_local_artifact(
    filename: str,
    data: bytes,
    uploaded_by: str,
) -> ArtifactRecord:
    artifact_dir = _artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_id = str(uuid.uuid4())
    sha256 = hashlib.sha256(data).hexdigest()

    # Sanitise filename — keep only the base name, replace spaces
    safe_name = Path(filename).name.replace(" ", "_")
    dest = artifact_dir / f"{artifact_id}_{safe_name}"
    dest.write_bytes(data)

    return ArtifactRecord(
        artifact_id=artifact_id,
        filename=safe_name,
        size_bytes=len(data),
        sha256=sha256,
        storage="local",
        location=str(dest),
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        uploaded_by=uploaded_by,
    )


# ---------------------------------------------------------------------------
# S3 presigned POST
# ---------------------------------------------------------------------------

def generate_presigned_upload(uploaded_by: str, filename: str) -> PresignedUploadResponse:
    """Generate a presigned S3 POST URL.

    Raises RuntimeError if boto3 is not installed or FLEXAI_S3_BUCKET is unset.
    """
    bucket = _s3_bucket()
    if not bucket:
        raise RuntimeError("FLEXAI_S3_BUCKET is not configured")

    try:
        import boto3  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for S3 presigned uploads. "
            "Install it with: pip install boto3"
        ) from exc

    artifact_id = str(uuid.uuid4())
    safe_name = Path(filename).name.replace(" ", "_")
    s3_key = f"artifacts/{uploaded_by}/{artifact_id}/{safe_name}"
    ttl = _presign_ttl()

    s3 = boto3.client("s3")
    presigned = s3.generate_presigned_post(
        Bucket=bucket,
        Key=s3_key,
        ExpiresIn=ttl,
    )

    return PresignedUploadResponse(
        artifact_id=artifact_id,
        upload_url=presigned["url"],
        fields=presigned.get("fields", {}),
        expires_in=ttl,
        s3_key=s3_key,
    )
