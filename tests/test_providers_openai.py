import asyncio
from collections.abc import AsyncGenerator
from typing import Any, TypedDict

import httpx
import pytest

from src.orch.providers import OpenAICompatProvider
from src.orch.router import ProviderDef
from src.orch.types import (
    ProviderChatResponse,
    ProviderStreamChunk as ProviderStreamChunkModel,
    chat_response_from_provider,
    provider_chat_response_from_stream,
)


class ProviderStreamChunkDict(TypedDict, total=False):
    index: int
    delta: dict[str, Any]
    finish_reason: str | None


def run_chat(
    provider: OpenAICompatProvider,
    monkeypatch: pytest.MonkeyPatch,
    request_model: str = "gpt-4o",
    upstream_response: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], ProviderChatResponse]:
    post_calls: list[dict[str, Any]] = []
    response: ProviderChatResponse | None = None

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        headers: dict[str, str] = kwargs.get("headers", {})
        payload: dict[str, Any] = kwargs.get("json", {})
        post_calls.append({"url": url, "headers": headers, "json": payload})
        request = httpx.Request("POST", url, headers=headers)
        return httpx.Response(
            status_code=200,
            json=upstream_response
            or {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    async def invoke() -> None:
        nonlocal response
        response = await provider.chat(
            model=request_model,
            messages=[{"role": "user", "content": "ping"}],
        )

    asyncio.run(invoke())
    assert response is not None
    return post_calls, response


def make_provider(base_url: str, defn_model: str = "gpt-4o") -> OpenAICompatProvider:
    provider_def = ProviderDef(
        name="openai",
        type="openai",
        base_url=base_url,
        model=defn_model,
        auth_env="OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )
    return OpenAICompatProvider(provider_def)


def test_no_authorization_header_when_auth_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAICompatProvider(
        ProviderDef(
            name="openai-no-auth",
            type="openai",
            base_url="https://api.openai.com",
            model="gpt-4o",
            auth_env=None,
            rpm=60,
            concurrency=1,
        )
    )

    post_calls, _ = run_chat(provider, monkeypatch)

    assert post_calls
    assert "Authorization" not in post_calls[0]["headers"]


def test_openai_base_url_uses_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com")

    post_calls, _ = run_chat(provider, monkeypatch)

    assert post_calls
    assert post_calls[0]["url"] == "https://api.openai.com/v1/chat/completions"


def test_openai_preserves_existing_chat_completions_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider(
        "https://example.ai/OpenAI/deployments/foo/Chat/Completions"
    )

    post_calls, _ = run_chat(provider, monkeypatch)

    assert post_calls
    assert (
        post_calls[0]["url"]
        == "https://example.ai/OpenAI/deployments/foo/Chat/Completions"
    )


def test_openai_chat_response_preserves_finish_reason_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com")
    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{\"key\": \"value\"}"},
        }
    ]
    upstream_response = {
        "model": "gpt-4o",
        "choices": [
            {
                "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }

    post_calls, response = run_chat(provider, monkeypatch, upstream_response=upstream_response)

    assert post_calls
    assert response.content is None
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == tool_calls
    assert response.model == "gpt-4o"
    assert response.usage_prompt_tokens == 3
    assert response.usage_completion_tokens == 4

    payload = chat_response_from_provider(response)
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["tool_calls"] == tool_calls
    assert "content" not in payload["choices"][0]["message"]


def test_openai_chat_response_preserves_function_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com")
    function_call = {"name": "lookup", "arguments": "{\"key\": \"value\"}"}
    upstream_response = {
        "model": "gpt-4o",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": function_call,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }

    post_calls, response = run_chat(provider, monkeypatch, upstream_response=upstream_response)

    assert post_calls
    assert response.content is None
    assert response.function_call == function_call

    payload = chat_response_from_provider(response)
    message = payload["choices"][0]["message"]
    assert message["function_call"] == function_call
    assert "content" not in message


def test_provider_chat_response_from_stream_merges_chunks() -> None:
    chunks = [
        ProviderStreamChunkModel.model_validate({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]}),
        ProviderStreamChunkModel.model_validate({"choices": [{"index": 0, "delta": {"content": "lo"}}]}),
        ProviderStreamChunkModel.model_validate(
            {
                "choices": [
                    {"index": 0, "delta": {"content": " world"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            }
        ),
    ]

    provider_response = provider_chat_response_from_stream("gpt-4o", chunks)
    assert provider_response.model == "gpt-4o"
    assert provider_response.finish_reason == "stop"
    assert provider_response.usage_prompt_tokens == 5
    assert provider_response.usage_completion_tokens == 7

    payload = chat_response_from_provider(provider_response)
    assert payload["choices"][0]["message"]["content"] == "Hello world"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}


def test_openai_chat_response_preserves_list_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com")
    content_blocks = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]
    upstream_response = {
        "model": "gpt-4o",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content_blocks,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }

    post_calls, response = run_chat(provider, monkeypatch, upstream_response=upstream_response)

    assert post_calls
    assert response.content == content_blocks


def test_openai_chat_response_handles_multiple_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com")
    upstream_response = {
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "first"},
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": "second"},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6},
    }

    _, response = run_chat(provider, monkeypatch, upstream_response=upstream_response)

    assert response.choices is not None
    assert len(response.choices) == 2
    assert response.choices[0]["message"]["content"] == "first"
    assert response.choices[1]["message"]["content"] == "second"

    payload = chat_response_from_provider(response)
    assert len(payload["choices"]) == 2
    assert payload["choices"][0]["message"]["content"] == "first"
    assert payload["choices"][1]["message"]["content"] == "second"

def test_chat_response_from_provider_keeps_multiple_choices() -> None:
    provider_response = ProviderChatResponse(
        model="gpt-4o",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "first"},
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": "second"},
                "finish_reason": "stop",
            },
        ],
    )

    payload = chat_response_from_provider(provider_response)

    assert [choice["message"]["content"] for choice in payload["choices"]] == [
        "first",
        "second",
    ]


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "https://api.openai.com/v1/",
        "https://proxy.example.com/custom/v1",
    ],
)
def test_base_url_with_version_suffix_is_not_duplicated(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider(base_url)

    post_calls, _ = run_chat(provider, monkeypatch)

    assert post_calls
    assert post_calls[0]["url"].endswith("/chat/completions")
    assert "/v1/v1/" not in post_calls[0]["url"]


def test_groq_base_url_keeps_openai_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.groq.com/openai/v1")

    post_calls, _ = run_chat(provider, monkeypatch)

    assert post_calls
    assert post_calls[0]["url"] == "https://api.groq.com/openai/v1/chat/completions"


def test_openai_chat_response_uses_requested_model_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com", defn_model="")

    post_calls, response = run_chat(provider, monkeypatch, request_model="gpt-4.1-mini")

    assert post_calls
    assert response.model == "gpt-4.1-mini"


class FakeOpenAIStreamClient:
    def __init__(self, chunks: list[ProviderStreamChunkDict], *, error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stop_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[ProviderStreamChunkDict, None]:
        self.calls.append({"model": model, "messages": messages, "stop_event": stop_event})
        if self._error is not None:
            if stop_event is not None:
                stop_event.set()
            raise self._error
        for chunk in self._chunks:
            yield chunk
        if stop_event is not None:
            stop_event.set()


@pytest.mark.asyncio
async def test_openai_stream_chat_emits_expected_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com")
    expected_chunks: list[ProviderStreamChunkDict] = [
        {"index": 0, "delta": {"role": "assistant"}},
        {"index": 0, "delta": {"content": "hel"}},
        {"index": 0, "delta": {"content": "lo"}},
        {"index": 0, "delta": {}, "finish_reason": "stop"},
    ]
    fake_client = FakeOpenAIStreamClient(expected_chunks)
    monkeypatch.setattr(provider, "_openai_client", fake_client, raising=False)
    stop_event = asyncio.Event()

    async def collect() -> list[ProviderStreamChunkDict]:
        results: list[ProviderStreamChunkDict] = []
        async for chunk in provider.stream_chat(  # type: ignore[attr-defined]
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            stop_event=stop_event,
        ):
            results.append(chunk)
        return results

    collected = await collect()

    assert fake_client.calls
    assert collected == expected_chunks
    assert stop_event.is_set()


@pytest.mark.asyncio
async def test_openai_stream_chat_stops_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com")
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    error = httpx.HTTPStatusError("Too Many Requests", request=request, response=response)
    stop_event = asyncio.Event()
    fake_client = FakeOpenAIStreamClient([], error=error)
    monkeypatch.setattr(provider, "_openai_client", fake_client, raising=False)

    async def consume() -> None:
        async for _ in provider.stream_chat(  # type: ignore[attr-defined]
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            stop_event=stop_event,
        ):
            pass

    with pytest.raises(httpx.HTTPStatusError):
        await consume()

    assert stop_event.is_set(), "429発生時はストップイベントが発火する前提"


def test_openai_chat_stream_normalizes_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com")
    captured_calls: list[dict[str, Any]] = []

    class FakeStreamResponse:
        def __init__(self, lines: list[str], *, headers: dict[str, str] | None = None) -> None:
            self._lines = lines
            self.headers = headers or {}
            self.status_code = 200

        async def aiter_lines(self) -> AsyncGenerator[str, None]:
            for line in self._lines:
                yield line

        def raise_for_status(self) -> None:
            return None

    class FakeStreamContext:
        def __init__(self, response: FakeStreamResponse) -> None:
            self._response = response

        async def __aenter__(self) -> FakeStreamResponse:
            return self._response

        async def __aexit__(self, exc_type, exc: BaseException | None, tb: Any) -> None:
            return None

    lines = [
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}",
        "",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}",
        "",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}",
        "",
        "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7}}",
        "",
        "data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7}}",
        "",
        "data: {\"error\":{\"message\":\"overload\",\"type\":\"server_error\"}}",
        "",
        "data: [DONE]",
        "",
    ]

    def fake_stream(self: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> FakeStreamContext:
        captured_calls.append({"method": method, "url": url, "json": kwargs.get("json")})
        assert method == "POST"
        payload = kwargs.get("json")
        assert isinstance(payload, dict)
        assert payload.get("stream") is True
        return FakeStreamContext(
            FakeStreamResponse(lines, headers={"retry-after": "2"}),
        )

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    async def collect() -> list[ProviderStreamChunkModel]:
        results: list[ProviderStreamChunkModel] = []
        async for chunk in provider.chat_stream(  # type: ignore[attr-defined]
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        ):
            assert isinstance(chunk, ProviderStreamChunkModel)
            results.append(chunk)
        return results

    chunks = asyncio.run(collect())

    assert captured_calls
    assert all(call["url"].endswith("/v1/chat/completions") for call in captured_calls)

    assert [chunk.event_type for chunk in chunks[:4]] == [
        "chunk",
        "chunk",
        "chunk",
        "chunk",
    ]
    assert [chunk.delta for chunk in chunks[:3]] == [
        {"role": "assistant"},
        {"content": "Hel"},
        {"content": "lo"},
    ]
    assert chunks[3].finish_reason == "stop"
    assert chunks[3].usage == {"prompt_tokens": 5, "completion_tokens": 7}
    assert chunks[4].usage == {"prompt_tokens": 5, "completion_tokens": 7}
    assert chunks[5].event_type == "error"
    assert chunks[5].error == {
        "message": "overload",
        "type": "server_error",
        "retry_after": 2,
    }

    response = provider_chat_response_from_stream("gpt-4o", chunks)

    assert response.content == "Hello"
    assert response.finish_reason == "stop"
    assert response.usage_prompt_tokens == 5
    assert response.usage_completion_tokens == 7


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param(
            [
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}",
                "data: [DONE]",
            ],
            id="no-separator",
        ),
        pytest.param(
            [
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}",
                "",
                "",
                "data: [DONE]",
            ],
            id="blank-lines-before-done",
        ),
        pytest.param(
            [
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}",
                "",
                "data: [DONE]",
                "",
                "",
            ],
            id="trailing-empties-after-done",
        ),
    ],
)
def test_openai_chat_stream_handles_done_without_separator(
    monkeypatch: pytest.MonkeyPatch,
    lines: list[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com")

    class FakeStreamResponse:
        def __init__(self, lines: list[str]) -> None:
            self._lines = lines
            self.headers: dict[str, str] = {}
            self.status_code = 200

        async def aiter_lines(self) -> AsyncGenerator[str, None]:
            for line in self._lines:
                yield line

        def raise_for_status(self) -> None:
            return None

    class FakeStreamContext:
        def __init__(self, response: FakeStreamResponse) -> None:
            self._response = response

        async def __aenter__(self) -> FakeStreamResponse:
            return self._response

        async def __aexit__(self, exc_type, exc: BaseException | None, tb: Any) -> None:
            return None

    def fake_stream(self: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> FakeStreamContext:
        assert method == "POST"
        assert kwargs.get("json", {}).get("stream") is True
        return FakeStreamContext(FakeStreamResponse(lines))

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    async def collect() -> list[ProviderStreamChunkModel]:
        results: list[ProviderStreamChunkModel] = []
        async for chunk in provider.chat_stream(  # type: ignore[attr-defined]
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        ):
            results.append(chunk)
        return results

    chunks = asyncio.run(collect())

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.event_type == "chunk"
    assert chunk.delta == {"content": "hi"}
    assert chunk.finish_reason == "stop"
    assert chunk.usage == {"prompt_tokens": 3, "completion_tokens": 2}
