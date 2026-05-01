"""Observability bootstrap: Prometheus metrics + OpenTelemetry tracing.

Call `setup_observability(app)` once during application startup.

Environment variables
---------------------
FLEXAI_OTLP_ENDPOINT   gRPC OTLP endpoint for traces (default: disabled)
FLEXAI_SERVICE_NAME    service.name resource attribute (default: "flexai")
"""
from __future__ import annotations

import os

from fastapi import FastAPI


def setup_observability(app: FastAPI) -> None:
    """Attach Prometheus instrumentator and (optionally) OTel tracing to *app*."""
    _setup_prometheus(app)
    _setup_tracing(app)


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------

def _setup_prometheus(app: FastAPI) -> None:
    from prometheus_fastapi_instrumentator import Instrumentator

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/metrics", "/healthz"],
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# ---------------------------------------------------------------------------
# OpenTelemetry
# ---------------------------------------------------------------------------

def _setup_tracing(app: FastAPI) -> None:
    otlp_endpoint = os.getenv("FLEXAI_OTLP_ENDPOINT", "")
    service_name = os.getenv("FLEXAI_SERVICE_NAME", "flexai")

    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    else:
        # No OTLP endpoint configured — use a silent no-op exporter so tracing
        # infrastructure is still active (spans exist) without writing to stdout.
        from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

        class _NullExporter(SpanExporter):
            def export(self, spans):  # type: ignore[override]
                return SpanExportResult.SUCCESS

            def shutdown(self) -> None:
                pass

        exporter = _NullExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
