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
from collections import defaultdict
from typing import Any, ClassVar, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.sdk.metrics.export import MetricReader  # type: ignore[import-not-found]
else:  # pragma: no cover
    class MetricReader:  # type: ignore[too-many-ancestors]
        """Runtime placeholder when OpenTelemetry is unavailable."""

        pass

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
_FLAG = "ORCH_OTEL_METRICS_EXPORT"
_MODE_FLAG = "ORCH_METRICS_EXPORT_MODE"
_PROM_FILE = "prometheus.prom"
_PROM_MODE = "prom"
_OTEL_MODE = "otel"
_BOTH_MODE = "both"
_MODE_VALUES = {_PROM_MODE, _OTEL_MODE, _BOTH_MODE}
_DEFAULT_MODE = _PROM_MODE
_HISTOGRAM_BUCKETS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


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


def _metrics_mode_from_env() -> str:
    raw = os.environ.get(_MODE_FLAG)
    if raw is not None:
        normalized = raw.strip().lower()
        if normalized in _MODE_VALUES:
            return normalized
    if _env_var_as_bool(_FLAG):
        return _BOTH_MODE
    return _DEFAULT_MODE


def _mode_includes_prom(mode: str) -> bool:
    return mode in (_PROM_MODE, _BOTH_MODE)


def _mode_includes_otel(mode: str) -> bool:
    return mode in (_OTEL_MODE, _BOTH_MODE)


def _new_histogram_state() -> dict[str, Any]:
    return {"buckets": [0] * (len(_HISTOGRAM_BUCKETS) + 1), "count": 0, "sum": 0.0}


class _PromMetrics:
    __slots__ = ("_dir", "_lock", "_counter", "_histogram")

    def __init__(self, dirpath: str) -> None:
        self._dir = dirpath
        self._lock = threading.Lock()
        self._counter: defaultdict[tuple[str, str, str], int] = defaultdict(int)
        self._histogram: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(_new_histogram_state)

    def record(self, payload: dict[str, Any]) -> None:
        provider = str(payload.get("provider") or "unknown")
        status = str(payload.get("status") or "0")
        ok_label = "true" if bool(payload.get("ok")) else "false"
        latency_seconds = max(float(payload.get("latency_ms") or 0.0) / 1000.0, 0.0)

        with self._lock:
            self._counter[(provider, status, ok_label)] += 1
            hist_state = self._histogram[(provider, ok_label)]
            buckets = hist_state["buckets"]
            for idx, bound in enumerate(_HISTOGRAM_BUCKETS):
                if latency_seconds <= bound:
                    buckets[idx] += 1
            buckets[-1] += 1
            hist_state["count"] += 1
            hist_state["sum"] += latency_seconds
            self._write_locked()

    def _render_locked(self) -> str:
        lines: list[str] = [
            "# HELP orch_requests_total Total number of orchestrator requests",
            "# TYPE orch_requests_total counter",
        ]
        for (provider, status, ok_label), value in sorted(self._counter.items()):
            lines.append(
                f'orch_requests_total{{provider="{provider}",status="{status}",ok="{ok_label}"}} {value}'
            )
        lines.append("# HELP orch_request_latency_seconds Request latency for orchestrated requests")
        lines.append("# TYPE orch_request_latency_seconds histogram")
        for (provider, ok_label), state in sorted(self._histogram.items()):
            buckets = state["buckets"]
            for idx, bound in enumerate(_HISTOGRAM_BUCKETS):
                le_value = format(bound, ".6g")
                count = buckets[idx]
                lines.append(
                    f'orch_request_latency_seconds_bucket{{provider="{provider}",ok="{ok_label}",le="{le_value}"}} {count}'
                )
            lines.append(
                f'orch_request_latency_seconds_bucket{{provider="{provider}",ok="{ok_label}",le="+Inf"}} {buckets[-1]}'
            )
            lines.append(
                f'orch_request_latency_seconds_count{{provider="{provider}",ok="{ok_label}"}} {state["count"]}'
            )
            lines.append(
                f'orch_request_latency_seconds_sum{{provider="{provider}",ok="{ok_label}"}} {state["sum"]}'
            )
        return "\n".join(lines) + "\n"

    def _write_locked(self) -> None:
        os.makedirs(self._dir, exist_ok=True)
        prom_path = os.path.join(self._dir, _PROM_FILE)
        tmp_path = f"{prom_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(self._render_locked())
        os.replace(tmp_path, prom_path)


class _OtelMetrics:
    __slots__ = ("_reader", "_previous_provider", "_provider", "_requests_counter", "_latency_histogram", "_shutdown")

    def __init__(self, reader: Optional["MetricReader"] = None):
        from opentelemetry import metrics as otel_metrics  # type: ignore[import-not-found]
        from opentelemetry.sdk.metrics import MeterProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]

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
        self._mode = _metrics_mode_from_env()
        self._prom = _PromMetrics(self.dir) if _mode_includes_prom(self._mode) else None
        self._otel = self._ensure_otel(self._mode)

    @classmethod
    def configure_metric_reader(cls, reader: Optional["MetricReader"]) -> None:
        with cls._otel_lock:
            if cls._otel_instance is not None:
                cls._otel_instance.shutdown()
            cls._otel_instance = None
            cls._custom_reader = reader
            cls._otel_error = False

    @classmethod
    def _ensure_otel(cls, mode: str | None = None) -> Optional[_OtelMetrics]:
        current_mode = mode or _metrics_mode_from_env()
        if not _mode_includes_otel(current_mode):
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

    async def write(self, record: dict[str, Any]) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            with open(self._file(), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        otel = self._otel or self._ensure_otel(self._mode)
        if otel is not None:
            otel.record(record)
        prom = self._prom
        if prom is not None:
            prom.record(record)

    async def flush(self) -> None:
        otel = self._otel or self._ensure_otel(self._mode)
        if otel is not None:
            await otel.flush()
