"""MetricsLogger OpenTelemetry integration tests."""

import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_otel_stub() -> None:
    if "opentelemetry" in sys.modules:
        return

    class MetricsData:
        def __init__(self, resource_metrics: Iterable["ResourceMetrics"]) -> None:
            self.resource_metrics = list(resource_metrics)

    class ResourceMetrics:
        def __init__(self, scope_metrics: Iterable["ScopeMetrics"]) -> None:
            self.scope_metrics = list(scope_metrics)

    class ScopeMetrics:
        def __init__(self, metrics: Iterable["_Metric"]) -> None:
            self.metrics = list(metrics)

    class SumDataPoint:
        def __init__(self, value: float, attributes: Dict[str, Any]):
            self.value = value
            self.attributes = attributes

    class HistogramDataPoint:
        def __init__(self, count: int, total: float, attributes: Dict[str, Any]):
            self.count = count
            self.sum = total
            self.attributes = attributes

    class Sum:
        def __init__(self, data_points: Iterable[SumDataPoint]):
            self.data_points = list(data_points)

    class Histogram:
        def __init__(self, data_points: Iterable[HistogramDataPoint]):
            self.data_points = list(data_points)

    class _Metric:
        def __init__(self, name: str, data: Any):
            self.name = name
            self.data = data

    class MetricReader:
        def __init__(self) -> None:
            self._metrics_data = MetricsData([])

        def _receive_metrics(self, data: MetricsData) -> None:
            self._metrics_data = data

    class InMemoryMetricReader(MetricReader):
        def get_metrics_data(self) -> MetricsData:
            return self._metrics_data

    class _CounterInstrument:
        def __init__(self, name: str) -> None:
            self.name = name
            self.records: List[Tuple[float, Dict[str, Any]]] = []

        def add(self, value: float, attributes: Dict[str, Any] | None = None) -> None:
            self.records.append((float(value), dict(attributes or {})))

    class _HistogramInstrument:
        def __init__(self, name: str) -> None:
            self.name = name
            self.records: List[Tuple[float, Dict[str, Any]]] = []

        def record(self, value: float, attributes: Dict[str, Any] | None = None) -> None:
            self.records.append((float(value), dict(attributes or {})))

    class _Meter:
        def __init__(self) -> None:
            self.counters: List[_CounterInstrument] = []
            self.histograms: List[_HistogramInstrument] = []

        def create_counter(self, name: str, **_: Any) -> _CounterInstrument:
            counter = _CounterInstrument(name)
            self.counters.append(counter)
            return counter

        def create_histogram(self, name: str, **_: Any) -> _HistogramInstrument:
            hist = _HistogramInstrument(name)
            self.histograms.append(hist)
            return hist

    class MeterProvider:
        def __init__(self, *, resource: Any | None = None, metric_readers: Iterable[MetricReader] | None = None) -> None:
            self._resource = resource
            self._metric_readers = list(metric_readers or [])
            self._meters: Dict[str, _Meter] = {}
            self._shutdown = False

        def get_meter(self, name: str) -> _Meter:
            return self._meters.setdefault(name, _Meter())

        def force_flush(self) -> bool:
            metrics: List[_Metric] = []
            for meter in self._meters.values():
                for counter in meter.counters:
                    if counter.records:
                        points = [SumDataPoint(value, attrs) for value, attrs in counter.records]
                        counter.records.clear()
                        metrics.append(_Metric(counter.name, Sum(points)))
                for hist in meter.histograms:
                    if hist.records:
                        points = [HistogramDataPoint(1, value, attrs) for value, attrs in hist.records]
                        hist.records.clear()
                        metrics.append(_Metric(hist.name, Histogram(points)))
            data = MetricsData([ResourceMetrics([ScopeMetrics(metrics)])]) if metrics else MetricsData([])
            for reader in self._metric_readers:
                reader._receive_metrics(data)
            return True

        def shutdown(self) -> bool:
            self._shutdown = True
            return True

    class Resource:
        def __init__(self, attributes: Dict[str, Any]):
            self.attributes = attributes

        @classmethod
        def create(cls, attributes: Dict[str, Any]) -> "Resource":
            return cls(attributes)

    otel_pkg = types.ModuleType("opentelemetry")
    metrics_mod = types.ModuleType("opentelemetry.metrics")
    metrics_mod._provider = MeterProvider(metric_readers=[])

    def _get_meter_provider() -> MeterProvider:
        return metrics_mod._provider

    def _set_meter_provider(provider: MeterProvider) -> None:
        metrics_mod._provider = provider

    def _get_meter(name: str) -> _Meter:
        return metrics_mod._provider.get_meter(name)

    metrics_mod.get_meter_provider = _get_meter_provider  # type: ignore[attr-defined]
    metrics_mod.set_meter_provider = _set_meter_provider  # type: ignore[attr-defined]
    metrics_mod.get_meter = _get_meter  # type: ignore[attr-defined]

    sdk_pkg = types.ModuleType("opentelemetry.sdk")
    metrics_pkg = types.ModuleType("opentelemetry.sdk.metrics")
    export_pkg = types.ModuleType("opentelemetry.sdk.metrics.export")
    resources_pkg = types.ModuleType("opentelemetry.sdk.resources")

    export_pkg.MetricReader = MetricReader  # type: ignore[attr-defined]
    export_pkg.InMemoryMetricReader = InMemoryMetricReader  # type: ignore[attr-defined]
    metrics_pkg.MeterProvider = MeterProvider  # type: ignore[attr-defined]
    resources_pkg.Resource = Resource  # type: ignore[attr-defined]

    otel_pkg.metrics = metrics_mod  # type: ignore[attr-defined]
    sdk_pkg.metrics = metrics_pkg  # type: ignore[attr-defined]
    sdk_pkg.resources = resources_pkg  # type: ignore[attr-defined]

    sys.modules["opentelemetry"] = otel_pkg
    sys.modules["opentelemetry.metrics"] = metrics_mod
    sys.modules["opentelemetry.sdk"] = sdk_pkg
    sys.modules["opentelemetry.sdk.metrics"] = metrics_pkg
    sys.modules["opentelemetry.sdk.metrics.export"] = export_pkg
    sys.modules["opentelemetry.sdk.resources"] = resources_pkg


try:  # pragma: no cover - executed during import
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # type: ignore[assignment]
except ModuleNotFoundError:  # pragma: no cover - fallback for test environment
    _install_otel_stub()
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # type: ignore[assignment]

from src.orch.metrics import MetricsLogger


def _sample_record() -> dict[str, Any]:
    return {
        "req_id": "req-1",
        "ts": 1.0,
        "task": "demo",
        "provider": "provider-a",
        "model": "model-x",
        "latency_ms": 123.0,
        "ok": True,
        "status": 200,
        "error": None,
        "usage_prompt": 1,
        "usage_completion": 2,
        "retries": 0,
    }


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_metrics_logger_records_opentelemetry_samples(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCH_METRICS_EXPORT_MODE", "both")
    reader = InMemoryMetricReader()
    MetricsLogger.configure_metric_reader(reader)

    try:
        logger = MetricsLogger(str(tmp_path))
        record = _sample_record()
        await logger.write(record)
        await logger.flush()

        metrics_data = reader.get_metrics_data()
        counter_points: list[Any] = []
        histogram_points: list[Any] = []
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name == "requests_total":
                        counter_points.extend(metric.data.data_points)
                    if metric.name == "latency_ms":
                        histogram_points.extend(metric.data.data_points)

        assert counter_points, "requests_total counter must have data points"
        assert histogram_points, "latency_ms histogram must have data points"

        assert any(getattr(dp, "value", 0) == 1 for dp in counter_points)
        assert any(
            getattr(dp, "count", 0) == 1 and pytest.approx(getattr(dp, "sum", 0.0)) == record["latency_ms"]
            for dp in histogram_points
        )
        prom_path = Path(tmp_path) / "prometheus.prom"
        assert prom_path.exists(), "Prometheus export should be written in both mode"
        prom_text = prom_path.read_text(encoding="utf-8")
        assert "orch_requests_total" in prom_text
        assert "orch_request_latency_seconds" in prom_text
    finally:
        MetricsLogger.configure_metric_reader(None)
        monkeypatch.delenv("ORCH_METRICS_EXPORT_MODE", raising=False)


@pytest.mark.anyio
async def test_metrics_logger_prometheus_only_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCH_OTEL_METRICS_EXPORT", raising=False)
    monkeypatch.setenv("ORCH_METRICS_EXPORT_MODE", "prom")
    MetricsLogger.configure_metric_reader(None)

    logger = MetricsLogger(str(tmp_path))
    record = _sample_record()
    await logger.write(record)

    prom_path = Path(tmp_path) / "prometheus.prom"
    assert prom_path.exists(), "Prometheus output should be generated in prom mode"
    prom_text = prom_path.read_text(encoding="utf-8")
    assert "provider=\"provider-a\"" in prom_text
    assert logger._otel is None


@pytest.mark.anyio
async def test_metrics_logger_otel_only_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCH_METRICS_EXPORT_MODE", "otel")
    reader = InMemoryMetricReader()
    MetricsLogger.configure_metric_reader(reader)

    try:
        logger = MetricsLogger(str(tmp_path))
        record = _sample_record()
        await logger.write(record)
        await logger.flush()

        prom_path = Path(tmp_path) / "prometheus.prom"
        assert not prom_path.exists(), "Prometheus file must not be created in otel-only mode"

        metrics_data = reader.get_metrics_data()
        assert metrics_data.resource_metrics, "OTel metrics should be recorded in otel-only mode"
    finally:
        MetricsLogger.configure_metric_reader(None)
        monkeypatch.delenv("ORCH_METRICS_EXPORT_MODE", raising=False)
