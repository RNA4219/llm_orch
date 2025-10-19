from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.orch.router import RouteDef, RouteTarget

from tests.test_server_routes import load_app
from tests.test_server_streaming_routing import (
    _DummyGuard,
    _Registry,
    _http_status_error,
)


def _partial_failure_stream(*_args: Any, **_kwargs: Any) -> Any:
    async def _generator() -> Any:
        yield {"event": "message", "data": {"id": "primary-1"}}
        raise _http_status_error(502)

    return _generator()


def _backup_stream(*_args: Any, **_kwargs: Any) -> Any:
    async def _generator() -> Any:
        yield {"event": "message", "data": {"id": "backup-1"}}
        yield {"event": "message", "data": {"id": "backup-2"}}

    return _generator()


def test_streaming_failover_after_partial_chunk(
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
            {"model": "primary", "chat_stream": staticmethod(_partial_failure_stream)},
        )(),
        "frontier_backup": type(
            "BackupProvider",
            (),
            {"model": "backup", "chat_stream": staticmethod(_backup_stream)},
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

    planner_mock.record_failure.assert_called_once_with("frontier_primary")
    planner_mock.record_success.assert_called_once_with("frontier_backup")
    assert "backup-1" in content
    assert "[DONE]" in content
