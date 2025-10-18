import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.providers import OllamaProvider  # noqa: E402
from src.orch.router import ProviderDef  # noqa: E402


def make_provider(base_url: str = "http://localhost:11434") -> OllamaProvider:
    provider_def = ProviderDef(
        name="ollama",
        type="ollama",
        base_url=base_url,
        model="llama3",
        auth_env=None,
        rpm=60,
        concurrency=1,
    )
    return OllamaProvider(provider_def)


def test_ollama_top_p_option(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider()
    post_calls: list[dict[str, Any]] = []

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        payload = kwargs.get("json", {})
        post_calls.append({"url": url, "json": payload})
        request = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            json={"message": {"content": "ok"}, "done": True},
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    async def invoke() -> None:
        await provider.chat(
            model="llama3",
            messages=[{"role": "user", "content": "ping"}],
            top_p=0.3,
        )

    asyncio.run(invoke())

    assert post_calls
    options = post_calls[0]["json"]["options"]
    assert options["top_p"] == 0.3


def test_ollama_response_format_json_sets_format(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider()
    post_calls: list[dict[str, Any]] = []

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        payload = kwargs.get("json", {})
        post_calls.append({"url": url, "json": payload})
        request = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            json={"message": {"content": "ok"}, "done": True},
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    async def invoke() -> None:
        await provider.chat(
            model="llama3",
            messages=[{"role": "user", "content": "ping"}],
            response_format={"type": "json_object"},
        )

    asyncio.run(invoke())

    assert post_calls
    payload = post_calls[0]["json"]
    assert payload.get("format") == "json"


def test_ollama_chat_stream_normalizes_jsonl(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider()

    stream_lines = [
        json.dumps({"model": "llama3", "message": {"role": "assistant", "content": "Hel"}, "done": False}),
        json.dumps({"model": "llama3", "message": {"role": "assistant", "content": "lo"}, "done": False}),
        json.dumps(
            {
                "model": "llama3",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                    ],
                },
                "done": False,
            }
        ),
        json.dumps(
            {
                "model": "llama3",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3,
                "eval_count": 5,
            }
        ),
    ]

    captured: dict[str, Any] = {}

    class DummyStreamResponse:
        def __init__(self) -> None:
            self._lines = list(stream_lines)

        async def __aenter__(self) -> "DummyStreamResponse":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self) -> Any:
            for line in self._lines:
                yield line

    class DummyAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "DummyAsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def stream(self, method: str, url: str, **kwargs: Any) -> DummyStreamResponse:
            captured["method"] = method
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            return DummyStreamResponse()

    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)

    async def collect() -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        async for chunk in provider.chat_stream(
            model="llama3", messages=[{"role": "user", "content": "ping"}]
        ):
            results.append(chunk)
        return results

    chunks = asyncio.run(collect())

    assert captured["json"]["stream"] is True
    assert [chunk["event"] for chunk in chunks] == ["chunk", "chunk", "chunk", "chunk"]
    assert chunks[0]["data"]["choices"][0]["delta"] == {"role": "assistant", "content": "Hel"}
    assert chunks[1]["data"]["choices"][0]["delta"] == {"content": "lo"}
    assert chunks[2]["data"]["choices"][0]["delta"] == {
        "tool_calls": [
            {"type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ]
    }
    final_choice = chunks[-1]["data"]["choices"][0]
    assert final_choice.get("finish_reason") == "stop"
    assert chunks[-1]["data"]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
    }
