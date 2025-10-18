import asyncio
import json
from pathlib import Path
import sys
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


def _require_stream_chunk_type() -> type[Any]:
    try:
        from src.orch.types import ProviderStreamChunk  # noqa: WPS433 (local import for tests)
    except Exception as exc:  # pragma: no cover - fail-fast assertion
        pytest.fail(f"ProviderStreamChunk が未実装です: {exc}")
    return ProviderStreamChunk


class _MockStreamResponse:
    def __init__(self, url: str, status_code: int, lines: list[str]) -> None:
        self._lines = list(lines)
        self.status_code = status_code
        self.request = httpx.Request("POST", url)

    async def __aenter__(self) -> "_MockStreamResponse":
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def aiter_lines(self) -> Any:
        for line in self._lines:
            await asyncio.sleep(0)
            yield line


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


def test_ollama_stream_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider()
    chunk_type = _require_stream_chunk_type()
    if not hasattr(provider, "stream_chat"):
        pytest.fail("OllamaProvider.stream_chat が未実装です")

    def run(events: list[dict[str, Any]]) -> tuple[list[Any], list[dict[str, Any]]]:
        lines = [json.dumps(event) for event in events]

        async def fake_stream(
            self: httpx.AsyncClient,
            method: str,
            url: str,
            **kwargs: Any,
        ) -> _MockStreamResponse:
            return _MockStreamResponse(url, 200, lines)

        monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

        async def gather() -> list[Any]:
            stream = provider.stream_chat  # type: ignore[attr-defined]
            items: list[Any] = []
            async for chunk in stream(
                model="llama3",
                messages=[{"role": "user", "content": "ping"}],
            ):
                items.append(chunk)
            return items

        raw_chunks = asyncio.run(gather())
        return raw_chunks, [
            chunk.model_dump(mode="python", exclude_none=True)  # type: ignore[call-arg]
            for chunk in raw_chunks
        ]

    normal_events = [
        {"event": "delta", "delta": {"content": part}}
        for part in ("hel", "lo ", "world")
    ]
    normal_events.append(
        {"event": "done", "done_reason": "stop", "usage": {"prompt_tokens": 5, "completion_tokens": 7}}
    )
    raw_chunks, chunk_dicts = run(normal_events)
    assert raw_chunks and all(isinstance(chunk, chunk_type) for chunk in raw_chunks)
    assert "".join(choice.get("delta", {}).get("content", "") for chunk in chunk_dicts for choice in chunk.get("choices", [])) == "hello world"
    assert any(choice.get("finish_reason") == "stop" for choice in chunk_dicts[-1].get("choices", []))
    assert next(chunk.get("usage") for chunk in chunk_dicts if chunk.get("usage")) == {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
    }
    rate_limit_events = [
        {"event": "delta", "delta": {"content": "hi"}},
        {"event": "error", "status_code": 429, "error": {"message": "rate limited"}, "done": True},
    ]
    _, rate_chunks = run(rate_limit_events)
    abort_chunk = None
    for chunk in rate_chunks:
        if (
            chunk.get("status_code") == 429
            or chunk.get("event") == "abort"
            or chunk.get("type") == "abort"
            or any(choice.get("finish_reason") == "rate_limit" for choice in chunk.get("choices", []))
        ):
            abort_chunk = chunk
            break
    assert abort_chunk, "429 中断イベントが発火していません"
    assert "rate" in abort_chunk.get("error", {}).get("message", "").lower()
