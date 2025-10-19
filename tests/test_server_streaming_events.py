from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.router import RouteDef, RouteTarget
from src.orch.types import ProviderStreamChunk
from tests.test_server_routes import load_app


def test_streaming_events_emit_spec_names(monkeypatch) -> None:
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

    async def _stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[ProviderStreamChunk]:
        yield ProviderStreamChunk(event_type="message_start", delta={"role": "assistant"})
        yield ProviderStreamChunk(event_type="delta", delta={"content": "Hel"})
        yield ProviderStreamChunk(event_type="usage", usage={"prompt_tokens": 3, "completion_tokens": 1})
        yield ProviderStreamChunk(event_type="message_stop", finish_reason="stop")

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
        _Registry({model_name: type("Provider", (), {"model": model_name, "chat_stream": staticmethod(_stream)})()}),
        raising=False,
    )
    monkeypatch.setattr(server_module, "guards", _Registry({model_name: _Guard()}), raising=False)

    client = TestClient(app)
    body = {"model": model_name, "messages": [{"role": "user", "content": "hello"}], "stream": True}

    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        assert response.status_code == 200
        payload = "".join(response.iter_text())

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

