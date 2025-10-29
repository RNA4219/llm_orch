from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from starlette.requests import Request

StreamFn = Callable[..., AsyncIterator[Any]]


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


def load_app(dummy_env: str | None = None) -> FastAPI:
    module_name = "src.orch.server"
    sys.modules.pop(module_name, None)
    sys.modules.pop("src.orch", None)
    if dummy_env is None:
        os.environ.pop("ORCH_USE_DUMMY", None)
    else:
        os.environ["ORCH_USE_DUMMY"] = dummy_env
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    return module.app


def ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _parse_sse_payload(payload: str) -> list[tuple[str | None, str]]:
    events: list[tuple[str | None, str]] = []
    for chunk in filter(None, payload.split("\n\n")):
        event_name: str | None = None
        data_text = ""
        for line in chunk.split("\n"):
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                data_text = line[6:]
        events.append((event_name, data_text))
    return events


def _collect_sse_events(
    monkeypatch: MonkeyPatch,
    stream_fn: StreamFn,
    *,
    guard_registry: Any | None = None,
) -> list[tuple[str | None, str]]:
    ensure_project_root_on_path()
    from src.orch.router import RouteDef, RouteTarget

    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    model_name = "mock-provider"

    class _Guard:
        async def __aenter__(self) -> None:  # pragma: no cover - trivial context manager
            return None

        async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial context manager
            return None

    class _Registry:
        def __init__(self, mapping: dict[str, Any]) -> None:
            self._mapping = mapping

        def get(self, key: str) -> Any:
            return self._mapping[key]

    route = RouteDef(
        name="PLAN",
        strategy="priority",
        targets=[RouteTarget(provider=model_name)],
    ).ordered([model_name])

    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    planner_stub = SimpleNamespace(
        plan=lambda _task, *, sticky_key=None: route,
        record_success=_noop,
        record_failure=_noop,
    )

    monkeypatch.setattr(server_module, "planner", planner_stub, raising=False)
    monkeypatch.setattr(
        server_module,
        "providers",
        _Registry({model_name: SimpleNamespace(model=model_name, chat_stream=staticmethod(stream_fn))}),
        raising=False,
    )
    if guard_registry is None:
        guard_registry = _Registry({model_name: _Guard()})
    monkeypatch.setattr(server_module, "guards", guard_registry, raising=False)

    client = TestClient(app)
    body = {"model": model_name, "messages": [{"role": "user", "content": "hello"}], "stream": True}

    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        assert response.status_code == 200
        payload = "".join(response.iter_text())

    return _parse_sse_payload(payload)


def test_streaming_events_emit_spec_names(monkeypatch: MonkeyPatch) -> None:
    ensure_project_root_on_path()
    from src.orch.types import ProviderStreamChunk

    async def _stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[ProviderStreamChunk]:
        yield ProviderStreamChunk(event_type="message_start", delta={"role": "assistant"})
        yield ProviderStreamChunk(event_type="delta", delta={"content": "Hel"})
        yield ProviderStreamChunk(event_type="usage", usage={"prompt_tokens": 3, "completion_tokens": 1})
        yield ProviderStreamChunk(event_type="message_stop", finish_reason="stop")

    events = _collect_sse_events(monkeypatch, _stream)

    names = {name for name, _ in events if name}
    assert {"chat.completion.chunk", "telemetry.usage", "done"} <= names

    for name, data_text in events:
        if name is None or data_text == "[DONE]":
            continue
        parsed = json.loads(data_text)
        if name == "done":
            assert parsed == {}
        else:
            assert isinstance(parsed, dict)


@dataclass
class _Chunk:
    event_type: str | None = None
    event: str | None = None
    delta: dict[str, Any] | None = None
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None


def test_streaming_structured_events_are_normalized(monkeypatch: MonkeyPatch) -> None:
    async def _stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        yield _Chunk(event_type="message_start", delta={"role": "assistant"}, raw={"ignored": True})
        yield {"event_type": "delta", "delta": {"content": "Hel"}}
        yield {"event_type": "usage", "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
        yield _Chunk(event="response.completed", finish_reason="stop")

    events = _collect_sse_events(monkeypatch, _stream)

    named_events = [item for item in events if item[0]]
    assert [name for name, _ in named_events] == [
        "chat.completion.chunk",
        "chat.completion.chunk",
        "telemetry.usage",
        "done",
    ]

    assert events[-1] == (None, "[DONE]")

    for name, data_text in named_events:
        parsed = json.loads(data_text) if data_text else {}
        if name == "done":
            assert parsed == {}
        else:
            assert "event_type" not in parsed
            assert "raw" not in parsed


def test_streaming_done_emitted_once_for_empty_stream(monkeypatch: MonkeyPatch) -> None:
    ensure_project_root_on_path()
    from src.orch.types import ProviderStreamChunk
    async def _stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[ProviderStreamChunk]:
        if False:  # pragma: no cover - satisfy async generator semantics
            yield ProviderStreamChunk()

    events = _collect_sse_events(monkeypatch, _stream)

    assert events == [(None, "[DONE]")]


def test_streaming_emits_without_guard(monkeypatch: MonkeyPatch) -> None:
    ensure_project_root_on_path()
    from src.orch.types import ProviderStreamChunk

    class _MissingGuards:
        def get(self, key: str) -> Any:  # pragma: no cover - exercised via test
            raise KeyError(key)

    async def _stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[ProviderStreamChunk]:
        yield ProviderStreamChunk(event_type="delta", delta={"content": "hi"})

    events = _collect_sse_events(
        monkeypatch,
        _stream,
        guard_registry=_MissingGuards(),
    )

    assert events[-1] == (None, "[DONE]")
    named = [name for name, _ in events if name]
    assert named[0] == "chat.completion.chunk"


@pytest.mark.anyio
async def test_streaming_cleanup_reraises_cancelled_error(
    monkeypatch: MonkeyPatch,
) -> None:
    ensure_project_root_on_path()
    load_app("1")
    server_module = sys.modules["src.orch.server"]

    from src.orch.router import RouteDef, RouteTarget
    from src.orch.types import ChatMessage, ChatRequest, ProviderStreamChunk

    model_name = "mock-provider"

    cancel_blocker = asyncio.Event()

    async def _stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[ProviderStreamChunk]:
        yield ProviderStreamChunk(event_type="delta", delta={"content": "hi"})
        await cancel_blocker.wait()

    class _Guard:
        async def __aenter__(self) -> None:  # pragma: no cover - trivial context manager
            return None

        async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial context manager
            return None

    class _Registry:
        def __init__(self, mapping: dict[str, Any]) -> None:
            self._mapping = mapping

        def get(self, key: str) -> Any:
            return self._mapping[key]

    route = RouteDef(
        name="PLAN",
        strategy="priority",
        targets=[RouteTarget(provider=model_name)],
    ).ordered([model_name])

    planner_stub = SimpleNamespace(
        plan=lambda _task, *, sticky_key=None: route,
        record_success=lambda *_args, **_kwargs: None,
        record_failure=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(server_module, "planner", planner_stub, raising=False)
    monkeypatch.setattr(
        server_module,
        "providers",
        _Registry({model_name: SimpleNamespace(model=model_name, chat_stream=staticmethod(_stream))}),
        raising=False,
    )
    monkeypatch.setattr(
        server_module,
        "guards",
        _Registry({model_name: _Guard()}),
        raising=False,
    )
    monkeypatch.setattr(server_module.metrics, "write", AsyncMock(), raising=False)

    captured_response: dict[str, Any] = {}

    class _CapturingStreamingResponse:
        def __init__(self, iterable: AsyncIterator[bytes], media_type: str) -> None:
            captured_response["iterable"] = iterable
            self.iterable = iterable
            self.media_type = media_type
            self.headers: dict[str, str] = {}

    monkeypatch.setattr(
        server_module,
        "StreamingResponse",
        _CapturingStreamingResponse,
        raising=False,
    )

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "method": "POST",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "headers": [],
            "query_string": b"",
            "client": ("testclient", 0),
            "server": ("testserver", 80),
            "scheme": "http",
        },
        _receive,
    )

    body = ChatRequest(
        model=model_name,
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
    )

    await server_module.chat_completions(request, body)
    iterable = captured_response["iterable"]

    chunk = await iterable.__anext__()
    assert b"data:" in chunk

    with pytest.raises(asyncio.CancelledError):
        await iterable.aclose()

    cancel_blocker.set()
