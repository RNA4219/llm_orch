from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
import pytest

from src.orch.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAICompatProvider,
    UnsupportedContentBlockError,
)
from src.orch.router import ProviderDef
from src.orch.types import (
    ProviderChatResponse,
    ProviderStreamChunk as ProviderStreamChunkModel,
)


@dataclass(slots=True)
class ProviderCase:
    name: str
    provider: Any
    auth_env: str | None


_PROVIDER_CONFIGS = (
    (
        "openai",
        OpenAICompatProvider,
        {
            "type": "openai",
            "base_url": "https://api.openai.com",
            "model": "gpt-4o",
            "auth_env": "OPENAI_API_KEY",
        },
    ),
    (
        "anthropic",
        AnthropicProvider,
        {
            "type": "anthropic",
            "base_url": "https://api.anthropic.com",
            "model": "claude-3-sonnet",
            "auth_env": "ANTHROPIC_API_KEY",
        },
    ),
    (
        "ollama",
        OllamaProvider,
        {
            "type": "ollama",
            "base_url": "http://localhost:11434",
            "model": "llama3",
            "auth_env": None,
        },
    ),
)


@pytest.fixture(params=_PROVIDER_CONFIGS, name="provider_case")
def provider_case_fixture(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> ProviderCase:
    name, provider_cls, defn_kwargs = request.param
    provider_def = ProviderDef(
        name=name,
        type=defn_kwargs["type"],
        base_url=defn_kwargs["base_url"],
        model=defn_kwargs["model"],
        auth_env=defn_kwargs["auth_env"],
        rpm=60,
        concurrency=1,
    )
    provider = provider_cls(provider_def)
    auth_env = defn_kwargs["auth_env"]
    if auth_env:
        monkeypatch.setenv(auth_env, "test-token")
    return ProviderCase(name=name, provider=provider, auth_env=auth_env)


def test_provider_chat_minimal(
    provider_case: ProviderCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = provider_case.provider
    post_calls: list[dict[str, Any]] = []

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        headers = kwargs.get("headers") or {}
        payload = kwargs.get("json") or {}
        post_calls.append({"url": url, "headers": headers, "json": payload})
        request = httpx.Request("POST", url, headers=headers)
        if provider_case.name == "openai":
            upstream = {
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }
        elif provider_case.name == "anthropic":
            upstream = {
                "model": "claude-3-sonnet",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }
        else:
            upstream = {
                "message": {"content": "hello", "tool_calls": []},
                "done": True,
                "done_reason": "stop",
            }
        return httpx.Response(status_code=200, json=upstream, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    async def invoke() -> ProviderChatResponse:
        return await provider.chat(
            model=provider.defn.model,
            messages=[{"role": "user", "content": "ping"}],
        )

    response = asyncio.run(invoke())

    assert post_calls
    assert isinstance(response, ProviderChatResponse)
    if provider_case.name == "openai":
        assert response.content == "hello"
        assert response.finish_reason == "stop"
        assert response.usage_prompt_tokens == 1
        assert response.usage_completion_tokens == 2
    elif provider_case.name == "anthropic":
        assert response.content == "hello"
        assert response.finish_reason == "stop"
        assert response.usage_prompt_tokens == 1
        assert response.usage_completion_tokens == 2
    else:
        assert response.content == "hello"
        assert response.finish_reason == "stop"
        assert response.tool_calls == []


def test_provider_chat_stream_minimal(
    provider_case: ProviderCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = provider_case.provider

    class DummyStream:
        def __init__(self, lines: list[str]):
            self._lines = lines
            self.headers: dict[str, str] = {}

        async def __aenter__(self) -> "DummyStream":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self) -> AsyncIterator[str]:
            for line in self._lines:
                yield line

    def build_lines() -> list[str]:
        if provider_case.name == "openai":
            return [
                "event: chunk",
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}",
                "",
                "event: completion",
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}",
                "",
            ]
        if provider_case.name == "anthropic":
            return [
                "data: {\"type\":\"message_start\",\"message\":{\"role\":\"assistant\"}}",
                "",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}",
                "",
                "data: {\"type\":\"message_delta\",\"delta\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}",
                "",
                "data: {\"type\":\"message_stop\",\"stop_reason\":\"end_turn\"}",
                "",
            ]
        return [
            '{"message":{"role":"assistant","content":"hi"},"done":false}',
            '{"done":true,"done_reason":"stop"}',
        ]

    def fake_stream(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> DummyStream:
        return DummyStream(build_lines())

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    chunks: list[Any] = []

    async def consume() -> None:
        async for chunk in provider.chat_stream(
            model=provider.defn.model,
            messages=[{"role": "user", "content": "ping"}],
        ):
            chunks.append(chunk)

    asyncio.run(consume())

    assert chunks
    if provider_case.name == "openai":
        assert isinstance(chunks[0], ProviderStreamChunkModel)
        assert chunks[0].event_type == "chunk"
        assert chunks[1].event_type == "completion"
        assert chunks[1].usage == {"prompt_tokens": 1, "completion_tokens": 2}
    elif provider_case.name == "anthropic":
        first = chunks[0]
        assert isinstance(first, dict)
        assert first["event"] == "message_start"
        assert chunks[1]["event"] == "delta"
        assert chunks[3]["event"] == "message_stop"
    else:
        assert isinstance(chunks[0], dict)
        assert chunks[0]["event"] == "chunk"
        assert chunks[-1]["data"]["choices"][0]["finish_reason"] == "stop"


def test_provider_chat_tool_calls(
    provider_case: ProviderCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = provider_case.provider
    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        headers = kwargs.get("headers") or {}
        request = httpx.Request("POST", url, headers=headers)
        if provider_case.name == "openai":
            upstream = {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        elif provider_case.name == "anthropic":
            upstream = {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "lookup",
                        "input": {"key": "value"},
                    }
                ],
                "stop_reason": "tool_use",
            }
        else:
            upstream = {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                "done": True,
                "done_reason": "tool_calls",
            }
        return httpx.Response(status_code=200, json=upstream, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    async def invoke() -> ProviderChatResponse:
        return await provider.chat(
            model=provider.defn.model,
            messages=[{"role": "user", "content": "ping"}],
        )

    response = asyncio.run(invoke())

    assert response.finish_reason in {"tool_calls", "stop"}
    assert response.tool_calls is not None
    assert response.tool_calls[0]["function"]["name"] == "lookup"


def test_anthropic_unsupported_content_block(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AnthropicProvider(
        ProviderDef(
            name="anthropic",
            type="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-3-sonnet",
            auth_env="ANTHROPIC_API_KEY",
            rpm=60,
            concurrency=1,
        )
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "token")
    with pytest.raises(UnsupportedContentBlockError):
        provider._prepare_chat_request(
            provider.defn.model,
            messages=[{"role": "assistant", "content": [{"type": "image", "text": "bad"}]}],
            temperature=0.2,
            max_tokens=128,
            tools=None,
            tool_choice=None,
            function_call=None,
            top_p=None,
            frequency_penalty=None,
            presence_penalty=None,
            logit_bias=None,
            response_format=None,
            extra_options=None,
            stream=False,
        )
