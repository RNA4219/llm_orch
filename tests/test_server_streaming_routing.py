import sys
from pathlib import Path
from typing import Any, Callable
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.router import RouteDef, RouteTarget  # noqa: E402

from tests.test_server_routes import capture_metric_records, load_app  # noqa: E402


def _http_status_error(status_code: int, *, message: str = "boom") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(status_code, text=message, request=request)
    return httpx.HTTPStatusError(message, request=request, response=response)


def _failing_stream(*_args: Any, **_kwargs: Any) -> Any:
    async def _generator() -> Any:
        if False:  # pragma: no cover - generator formality
            yield None
        raise _http_status_error(500)

    return _generator()


def _http_error_stream(status: int, message: str) -> Any:
    async def _generator() -> Any:
        if False:  # pragma: no cover - generator formality
            yield None
        raise _http_status_error(status, message=message)

    return _generator()


def _successful_stream(*_args: Any, **_kwargs: Any) -> Any:
    async def _generator() -> Any:
        yield {"event": "message", "data": {"id": "1"}}
        yield {"event": "message", "data": {"id": "2"}}

    return _generator()


def _partial_http_error_stream(*_args: Any, **_kwargs: Any) -> Any:
    async def _generator() -> Any:
        yield {"event": "message", "data": {"id": "1"}}
        raise _http_status_error(502)

    return _generator()


class _DummyGuard:
    def __init__(self) -> None:
        self.estimated_prompt_tokens: int | None = None

    def acquire(
        self, *, estimated_prompt_tokens: int | None = None
    ) -> "_DummyGuard":
        self.estimated_prompt_tokens = estimated_prompt_tokens
        return self

    async def __aenter__(self) -> "_DummyGuard":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _Registry:
    def __init__(self, mapping: dict[str, Any]) -> None:
        self._mapping = mapping

    def get(self, key: str) -> Any:
        return self._mapping[key]


def _provider(model: str, stream_factory: Callable[..., Any]) -> Any:
    return type("Provider", (), {"model": model, "chat_stream": staticmethod(stream_factory)})()


def _prepare_streaming(
    monkeypatch: pytest.MonkeyPatch,
    providers_mapping: dict[str, Any],
    order: list[str],
) -> tuple[TestClient, list[dict[str, object]], dict[str, object]]:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    records = capture_metric_records(server_module, monkeypatch)

    route = RouteDef(
        name="PLAN",
        strategy="priority",
        targets=[RouteTarget(provider=name) for name in order],
    ).ordered(order)

    planner_mock = Mock()
    planner_mock.plan.return_value = route
    planner_mock.record_success = Mock()
    planner_mock.record_failure = Mock()
    monkeypatch.setattr(server_module, "planner", planner_mock, raising=False)

    guards_mapping = {name: _DummyGuard() for name in providers_mapping}
    monkeypatch.setattr(server_module, "providers", _Registry(providers_mapping), raising=False)
    monkeypatch.setattr(server_module, "guards", _Registry(guards_mapping), raising=False)

    client = TestClient(app)
    body = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    return client, records, body


def test_streaming_fallback_uses_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    route = RouteDef(
        name="PLAN",
        strategy="priority",
        targets=[
            RouteTarget(provider="frontier_primary"),
            RouteTarget(provider="frontier_backup"),
        ],
    ).ordered(["frontier_primary", "frontier_backup"])

    planner_mock = Mock()
    planner_mock.plan.return_value = route
    planner_mock.record_success = Mock()
    planner_mock.record_failure = Mock()
    monkeypatch.setattr(server_module, "planner", planner_mock, raising=False)

    providers_mapping = {
        "frontier_primary": type(
            "PrimaryProvider",
            (),
            {"model": "primary", "chat_stream": staticmethod(_failing_stream)},
        )(),
        "frontier_backup": type(
            "BackupProvider",
            (),
            {"model": "backup", "chat_stream": staticmethod(_successful_stream)},
        )(),
    }
    guards_mapping = {
        "frontier_primary": _DummyGuard(),
        "frontier_backup": _DummyGuard(),
    }
    monkeypatch.setattr(server_module, "providers", _Registry(providers_mapping), raising=False)
    monkeypatch.setattr(server_module, "guards", _Registry(guards_mapping), raising=False)

    client = TestClient(app)
    body = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        content = "".join(response.iter_text())

    assert planner_mock.plan.called
    assert "[DONE]" in content


def test_streaming_skips_provider_without_chat_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    route = RouteDef(
        name="PLAN",
        strategy="priority",
        targets=[
            RouteTarget(provider="frontier_primary"),
            RouteTarget(provider="frontier_backup"),
        ],
    ).ordered(["frontier_primary", "frontier_backup"])

    planner_mock = Mock()
    planner_mock.plan.return_value = route
    planner_mock.record_success = Mock()
    planner_mock.record_failure = Mock()
    monkeypatch.setattr(server_module, "planner", planner_mock, raising=False)

    providers_mapping = {
        "frontier_primary": type(
            "PrimaryProvider",
            (),
            {"model": "primary"},
        )(),
        "frontier_backup": type(
            "BackupProvider",
            (),
            {"model": "backup", "chat_stream": staticmethod(_successful_stream)},
        )(),
    }
    guards_mapping = {
        "frontier_primary": _DummyGuard(),
        "frontier_backup": _DummyGuard(),
    }
    monkeypatch.setattr(server_module, "providers", _Registry(providers_mapping), raising=False)
    monkeypatch.setattr(server_module, "guards", _Registry(guards_mapping), raising=False)

    client = TestClient(app)
    body = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        content = "".join(response.iter_text())

    assert planner_mock.plan.called
    planner_mock.record_failure.assert_called_once_with("frontier_primary")
    planner_mock.record_success.assert_called_once_with("frontier_backup")
    assert "[DONE]" in content


@pytest.mark.parametrize(
    "providers_mapping, order, expected_status, expected_message, final_provider, failure_expectation",
    [
        (
            {
                "frontier_primary": _provider(
                    "primary",
                    lambda *_args, **_kwargs: _http_error_stream(500, "primary failure"),
                ),
                "frontier_backup": _provider(
                    "backup",
                    lambda *_args, **_kwargs: _http_error_stream(503, "secondary failure"),
                ),
            },
            ["frontier_primary", "frontier_backup"],
            503,
            "secondary failure",
            "frontier_backup",
            ("frontier_primary", 500, "primary failure"),
        ),
        (
            {
                "frontier_primary": _provider(
                    "primary",
                    lambda *_args, **_kwargs: _http_error_stream(429, "rate limited"),
                ),
            },
            ["frontier_primary"],
            429,
            "rate limited",
            "frontier_primary",
            ("frontier_primary", 429, "rate limited"),
        ),
    ],
)
def test_streaming_failure_metrics(
    providers_mapping: dict[str, Any],
    order: list[str],
    expected_status: int,
    expected_message: str,
    final_provider: str,
    failure_expectation: tuple[str, int, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, records, body = _prepare_streaming(monkeypatch, providers_mapping, order)
    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == expected_status
    payload = response.json()
    assert payload["error"]["message"] == expected_message
    if expected_status == 429:
        assert "application/json" in response.headers.get("content-type", "")

    assert records
    failure_provider, failure_status, failure_message = failure_expectation
    failure_record = next(
        record
        for record in records
        if record["provider"] == failure_provider and record.get("ok") is False
    )
    assert failure_record["status"] == failure_status
    assert failure_record["error"] == failure_message

    final_record = records[-1]
    assert final_record["provider"] == final_provider
    assert final_record["status"] == expected_status
    assert final_record["error"] == expected_message
