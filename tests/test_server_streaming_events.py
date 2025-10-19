from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.router import RouteDef, RouteTarget
from src.orch.types import ProviderStreamChunk
from tests.test_server_routes import load_app


StreamFn = Callable[..., AsyncIterator[Any]]


def _parse_sse_payload(payload: str) -> list[tuple[str | None, str]]:
    events: list[tuple[str | None, str]] = []
    for chunk in filter(None, payload.split("\n\n")):
        event_name: str | None = None
        data_text = ""
        for line in chunk.split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_text = line[len("data: ") :]
        events.append((event_name, data_text))
    return events


def _collect_sse_events(monkeypatch: MonkeyPatch, stream_fn: StreamFn) -> list[tuple[str | None, str]]:
    app = load_app("1")
    server_module = sys.modules["src.orch.server"]
    model_name = "mock-provider"

    class _Guard:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, exc_type, exc, tb) -> None:
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
    planner_stub = type(
        "Planner",
        (),
        {
            "plan": staticmethod(lambda _task: route),
            "record_success": staticmethod(lambda _provider: None),
            "record_failure": staticmethod(lambda _provider, now=None: None),
        },
    )()
    monkeypatch.setattr(server_module, "planner", planner_stub, raising=False)
    monkeypatch.setattr(
        server_module,
        "providers",
        _Registry({model_name: type("Provider", (), {"model": model_name, "chat_stream": staticmethod(stream_fn)})()}),
        raising=False,
    )
    monkeypatch.setattr(server_module, "guards", _Registry({model_name: _Guard()}), raising=False)

    client = TestClient(app)
    body = {"model": model_name, "messages": [{"role": "user", "content": "hello"}], "stream": True}

    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        assert response.status_code == 200
        payload = "".join(response.iter_text())

    return _parse_sse_payload(payload)


def test_streaming_events_emit_spec_names(monkeypatch: MonkeyPatch) -> None:
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


def test_streaming_dict_events_are_mapped(monkeypatch: MonkeyPatch) -> None:
    async def _stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"event_type": "message_start", "delta": {"role": "assistant"}}
        yield {"event_type": "delta", "delta": {"content": "Hel"}}
        yield {"event_type": "usage", "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
        yield {"event_type": "message_stop", "finish_reason": "stop"}

    events = _collect_sse_events(monkeypatch, _stream)

    named_events = [name for name, _ in events if name]
    assert named_events[:4] == [
        "chat.completion.chunk",
        "chat.completion.chunk",
        "telemetry.usage",
        "done",
    ]

    assert any(name is None and data == "[DONE]" for name, data in events)

    for name, data_text in events:
        if name is None:
            assert data_text == "[DONE]"
            continue
        if data_text == "":
            continue
        parsed = json.loads(data_text)
        if name == "done":
            assert parsed == {}
        else:
            assert "event_type" not in parsed
            assert "raw" not in parsed


def test_streaming_dataclass_events_use_event_field(monkeypatch: MonkeyPatch) -> None:
    @dataclass
    class _Chunk:
        event_type: str | None = None
        event: str | None = None
        delta: dict[str, Any] | None = None
        usage: dict[str, int] | None = None
        finish_reason: str | None = None
        raw: dict[str, Any] | None = None

    async def _stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[_Chunk]:
        yield _Chunk(event_type="message_start", delta={"role": "assistant"}, raw={"ignored": True})
        yield _Chunk(event_type="delta", delta={"content": "Hel"})
        yield _Chunk(event_type="usage", usage={"prompt_tokens": 2, "completion_tokens": 1})
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

