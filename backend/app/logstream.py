"""
logstream.py — Per-deployment async log streaming.

Each deployment gets a :class:`LogBus` that holds the last ``MAX_HISTORY``
lines (so late-connecting clients catch up) and broadcasts every new line
to all active WebSocket subscribers.

A background task (started lazily on first subscription) simulates
structured log lines for deployments that have no real provider log feed.
Real provider adapters can call ``LogBus.publish(deployment_id, line)``
to inject actual log lines.
"""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List

_logger = logging.getLogger(__name__)

MAX_HISTORY = 200          # lines kept per deployment
_EMIT_INTERVAL = 2.0       # seconds between simulated log lines

# ---------------------------------------------------------------------------
# Per-deployment bus
# ---------------------------------------------------------------------------

class LogBus:
    """Holds history + active subscriber queues for one deployment."""

    def __init__(self) -> None:
        self._history: List[str] = []
        self._queues: List[asyncio.Queue[str | None]] = []
        self._lock = asyncio.Lock()

    async def publish(self, line: str) -> None:
        """Append a log line and fan it out to all subscribers."""
        async with self._lock:
            self._history.append(line)
            if len(self._history) > MAX_HISTORY:
                self._history = self._history[-MAX_HISTORY:]
            for q in self._queues:
                await q.put(line)

    async def subscribe(self) -> tuple[List[str], asyncio.Queue[str | None]]:
        """Register a new subscriber; returns (history_snapshot, live_queue)."""
        async with self._lock:
            snapshot = list(self._history)
            q: asyncio.Queue[str | None] = asyncio.Queue()
            self._queues.append(q)
        return snapshot, q

    async def unsubscribe(self, q: asyncio.Queue[str | None]) -> None:
        async with self._lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass
        # Signal the consumer to stop if it's still waiting
        await q.put(None)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_buses: Dict[str, LogBus] = {}
_emitter_tasks: Dict[str, asyncio.Task] = {}  # type: ignore[type-arg]


def _bus_for(deployment_id: str) -> LogBus:
    if deployment_id not in _buses:
        _buses[deployment_id] = LogBus()
    return _buses[deployment_id]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def stream_logs(deployment_id: str) -> AsyncIterator[str]:
    """
    Async generator that yields log lines for *deployment_id*.

    Yields the recent history first, then live lines as they arrive.
    Raises ``StopAsyncIteration`` when the caller disconnects (caller
    should cancel / close the generator on disconnect).
    """
    bus = _bus_for(deployment_id)
    _ensure_emitter(deployment_id)

    history, q = await bus.subscribe()
    try:
        for line in history:
            yield line
        while True:
            line = await q.get()
            if line is None:
                break
            yield line
    finally:
        await bus.unsubscribe(q)


async def publish_log(deployment_id: str, line: str) -> None:
    """Inject a log line from a provider adapter or any other source."""
    await _bus_for(deployment_id).publish(line)


# ---------------------------------------------------------------------------
# Simulated log emitter
# ---------------------------------------------------------------------------

_LOG_TEMPLATES = [
    "INFO  [worker] processed batch in {ms}ms ({tok} tokens)",
    "INFO  [health] GPU utilization {gpu}% | memory {mem}% | temp {tmp}°C",
    "INFO  [router] routed request to replica {rep} (latency {lat}ms)",
    "DEBUG [autoscale] current replicas={rep}, load={load:.2f}, threshold=0.75",
    "INFO  [inference] model={model} p50={p50}ms p95={p95}ms p99={p99}ms",
    "INFO  [cache] hit_rate={hr:.1f}% evictions={ev}",
    "DEBUG [heartbeat] replica {rep} healthy",
]


def _sim_line(deployment_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    template = random.choice(_LOG_TEMPLATES)
    vals = {
        "ms": random.randint(8, 120),
        "tok": random.randint(64, 512),
        "gpu": random.randint(40, 99),
        "mem": random.randint(30, 90),
        "tmp": random.randint(45, 85),
        "rep": random.randint(0, 3),
        "lat": random.randint(5, 80),
        "load": random.uniform(0.1, 1.2),
        "model": deployment_id[:8],
        "p50": random.randint(5, 50),
        "p95": random.randint(50, 150),
        "p99": random.randint(100, 300),
        "hr": random.uniform(60, 99),
        "ev": random.randint(0, 50),
    }
    return f"{ts} {template.format(**vals)}"


async def _emitter_loop(deployment_id: str) -> None:
    _logger.debug("Log emitter started for deployment %s", deployment_id)
    try:
        while True:
            await asyncio.sleep(_EMIT_INTERVAL)
            bus = _buses.get(deployment_id)
            if bus is None or not bus._queues:
                # No active subscribers — keep looping but skip publishing.
                continue
            line = _sim_line(deployment_id)
            await bus.publish(line)
    except asyncio.CancelledError:
        _logger.debug("Log emitter stopped for deployment %s", deployment_id)


def _ensure_emitter(deployment_id: str) -> None:
    """Start the background emitter task if it isn't running yet."""
    task = _emitter_tasks.get(deployment_id)
    if task is None or task.done():
        try:
            loop = asyncio.get_event_loop()
            _emitter_tasks[deployment_id] = loop.create_task(
                _emitter_loop(deployment_id)
            )
        except RuntimeError:
            # No running event loop (e.g. during import in tests) — skip.
            pass
