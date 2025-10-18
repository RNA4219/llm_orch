"""Metrics logging with optional OpenTelemetry export.

Set ``ORCH_OTEL_METRICS_EXPORT`` to emit ``requests_total`` and ``latency_ms``
metrics alongside the JSONL audit log. ``MetricsLogger.flush`` forces an export
and ``configure_metric_reader`` swaps the underlying reader for tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Any, ClassVar, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.sdk.metrics.export import MetricReader

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
_FLAG = "ORCH_OTEL_METRICS_EXPORT"


def _env_var_as_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if not normalized:
        return default
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    return default


class _OtelMetrics:
    __slots__ = ("_reader", "_previous_provider", "_provider", "_requests_counter", "_latency_histogram", "_shutdown")

    def __init__(self, reader: Optional["MetricReader"] = None):
        from opentelemetry import metrics as otel_metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.resources import Resource

        self._reader = reader or InMemoryMetricReader()
        self._previous_provider = otel_metrics.get_meter_provider()
        provider = MeterProvider(
            resource=Resource.create({"service.name": "llm-orch"}),
            metric_readers=[self._reader],
        )
        otel_metrics.set_meter_provider(provider)
        self._provider = provider
        meter = otel_metrics.get_meter("orch.metrics")
        self._requests_counter = meter.create_counter(
            "requests_total", description="Total number of orchestrated requests."
        )
        self._latency_histogram = meter.create_histogram(
            "latency_ms", unit="ms", description="Request latency in milliseconds."
        )
        self._shutdown = False

    def record(self, payload: dict[str, Any]) -> None:
        attrs: dict[str, Any] = {}
        provider = payload.get("provider")
        if isinstance(provider, str) and provider:
            attrs["provider"] = provider
        status = payload.get("status")
        if isinstance(status, int) and not isinstance(status, bool):
            attrs["status"] = status
        ok = payload.get("ok")
        if isinstance(ok, bool):
            attrs["ok"] = ok
        self._requests_counter.add(1, attributes=attrs)
        latency = payload.get("latency_ms")
        if isinstance(latency, (int, float)):
            self._latency_histogram.record(float(latency), attributes=attrs)

    async def flush(self) -> None:
        if self._shutdown:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._provider.force_flush)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        from opentelemetry import metrics as otel_metrics

        self._provider.shutdown()
        otel_metrics.set_meter_provider(self._previous_provider)
        self._shutdown = True


class MetricsLogger:
    _otel_lock: ClassVar[threading.Lock] = threading.Lock()
    _otel_instance: ClassVar[Optional[_OtelMetrics]] = None
    _otel_error: ClassVar[bool] = False
    _custom_reader: ClassVar[Optional["MetricReader"]] = None

    def __init__(self, dirpath: str):
        self.dir = dirpath
        os.makedirs(self.dir, exist_ok=True)
        self._lock: Optional[asyncio.Lock] = None
        self._otel = self._ensure_otel()

    @classmethod
    def configure_metric_reader(cls, reader: Optional["MetricReader"]) -> None:
        with cls._otel_lock:
            if cls._otel_instance is not None:
                cls._otel_instance.shutdown()
            cls._otel_instance = None
            cls._custom_reader = reader
            cls._otel_error = False

    @classmethod
    def _ensure_otel(cls) -> Optional[_OtelMetrics]:
        if not _env_var_as_bool(_FLAG):
            return None
        with cls._otel_lock:
            if cls._otel_instance is not None:
                return cls._otel_instance
            if cls._otel_error:
                return None
            try:
                cls._otel_instance = _OtelMetrics(cls._custom_reader)
            except ImportError:
                cls._otel_error = True
                cls._otel_instance = None
            return cls._otel_instance

    def _file(self) -> str:
        return os.path.join(self.dir, f"requests-{time.strftime('%Y%m%d')}.jsonl")

    async def write(self, record: dict[str, Any]):
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            with open(self._file(), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        otel = self._otel or self._ensure_otel()
        if otel is not None:
            otel.record(record)

    async def flush(self) -> None:
        otel = self._otel or self._ensure_otel()
        if otel is not None:
            await otel.flush()
