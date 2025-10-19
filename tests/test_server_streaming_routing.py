import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.router import RouteDef, RouteTarget  # noqa: E402

from tests.test_server_routes import load_app  # noqa: E402


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def _failing_stream(*_args: Any, **_kwargs: Any) -> Any:
    async def _generator() -> Any:
        yield {"event": "message", "data": {"id": "primary"}}
        raise _http_status_error(500)

    return _generator()


def _successful_stream(*_args: Any, **_kwargs: Any) -> Any:
    async def _generator() -> Any:
        yield {"event": "message", "data": {"id": "1"}}
        yield {"event": "message", "data": {"id": "2"}}

    return _generator()


def _error_stream(*_args: Any, **_kwargs: Any) -> Any:
    async def _generator() -> Any:
        yield {"event": "message", "data": {"id": "primary"}}
        from src.orch.providers import UnsupportedContentBlockError

        raise UnsupportedContentBlockError("unsupported")

    return _generator()


class _DummyGuard:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _Registry:
    def __init__(self, mapping: dict[str, Any]) -> None:
        self._mapping = mapping

    def get(self, key: str) -> Any:
        return self._mapping[key]


def _make_provider(model: str, stream: Any | None = None) -> Any:
    attrs: dict[str, Any] = {"model": model}
    if stream is not None:
        attrs["chat_stream"] = staticmethod(stream)
    return type(f"{model.title()}Provider", (), attrs)()


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
        "frontier_primary": _make_provider("primary", _failing_stream),
        "frontier_backup": _make_provider("backup", _successful_stream),
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
    planner_mock.record_failure.assert_any_call("frontier_primary")
    planner_mock.record_success.assert_called_with("frontier_backup")
    assert "primary" not in content
    assert "[DONE]" in content


def test_streaming_midstream_error(monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    route = RouteDef(
        name="PLAN",
        strategy="priority",
        targets=[RouteTarget(provider="frontier_primary")],
    ).ordered(["frontier_primary"])

    planner_mock = Mock()
    planner_mock.plan.return_value = route
    planner_mock.record_success = Mock()
    planner_mock.record_failure = Mock()
    monkeypatch.setattr(server_module, "planner", planner_mock, raising=False)

    providers_mapping = {
        "frontier_primary": _make_provider("primary", _error_stream),
    }
    guards_mapping = {"frontier_primary": _DummyGuard()}
    monkeypatch.setattr(server_module, "providers", _Registry(providers_mapping), raising=False)
    monkeypatch.setattr(server_module, "guards", _Registry(guards_mapping), raising=False)

    client = TestClient(app)
    body = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["message"] == "unsupported"
    planner_mock.record_success.assert_not_called()


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
        "frontier_primary": _make_provider("primary"),
        "frontier_backup": _make_provider("backup", _successful_stream),
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


def test_streaming_primary_http_error_without_backup(monkeypatch: pytest.MonkeyPatch) -> None:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]

    route = RouteDef(
        name="PLAN",
        strategy="priority",
        targets=[RouteTarget(provider="frontier_primary")],
    ).ordered(["frontier_primary"])

    planner_mock = Mock()
    planner_mock.plan.return_value = route
    planner_mock.record_success = Mock()
    planner_mock.record_failure = Mock()
    monkeypatch.setattr(server_module, "planner", planner_mock, raising=False)

    providers_mapping = {
        "frontier_primary": type(
            "PrimaryProvider",
            (),
            {"model": "primary", "chat_stream": staticmethod(_partial_http_error_stream)},
        )(),
    }
    guards_mapping = {
        "frontier_primary": _DummyGuard(),
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
        status_code = response.status_code

    assert status_code == 502
    payload = json.loads(content)
    assert payload["error"]["message"] == "boom"
    assert payload["error"]["type"] == "provider_server_error"
    assert planner_mock.plan.called
    planner_mock.record_failure.assert_called_once_with("frontier_primary")
    planner_mock.record_success.assert_not_called()
