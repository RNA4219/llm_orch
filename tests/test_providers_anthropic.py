import asyncio
import sys
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.providers import AnthropicProvider
from src.orch.router import ProviderDef
from src.orch.types import ProviderChatResponse


def run_chat(
    provider: AnthropicProvider,
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, str]],
    request_model: str = "claude-3-sonnet",
) -> tuple[dict[str, Any], ProviderChatResponse]:
    captured: dict[str, Any] = {}

    class DummyAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "DummyAsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
            captured["call_count"] = captured.get("call_count", 0) + 1
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            request = httpx.Request("POST", url, headers=headers)
            return httpx.Response(
                status_code=200,
                json={
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                },
                request=request,
            )

    async def invoke() -> ProviderChatResponse:
        monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)
        return await provider.chat(model=request_model, messages=messages)

    response = asyncio.run(invoke())
    return captured, response


def test_anthropic_payload_maps_openai_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_def = ProviderDef(
        name="anthropic",
        type="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-3-sonnet",
        auth_env="ANTHROPIC_API_KEY",
        rpm=60,
        concurrency=1,
    )
    provider = AnthropicProvider(provider_def)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    captured, response = run_chat(provider, monkeypatch, messages)

    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["system"] == "you are helpful"
    messages_payload = cast(list[dict[str, Any]], request_json["messages"])
    first_content = cast(list[dict[str, str]], messages_payload[0]["content"])
    assert all(block["type"] == "text" for block in first_content)
    assert messages_payload == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]

    assert response.content == "ok"
    assert response.usage_prompt_tokens == 1
    assert response.usage_completion_tokens == 2


def test_anthropic_chat_response_uses_requested_model_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_def = ProviderDef(
        name="anthropic",
        type="anthropic",
        base_url="https://api.anthropic.com",
        model="",
        auth_env="ANTHROPIC_API_KEY",
        rpm=60,
        concurrency=1,
    )
    provider = AnthropicProvider(provider_def)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    messages = [{"role": "user", "content": "hello"}]

    _, response = run_chat(
        provider,
        monkeypatch,
        messages,
        request_model="claude-3-5-haiku",
    )

    assert response.model == "claude-3-5-haiku"


def test_anthropic_chat_omits_api_key_when_no_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_def = ProviderDef(
        name="anthropic",
        type="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-3-sonnet",
        auth_env=None,
        rpm=60,
        concurrency=1,
    )
    provider = AnthropicProvider(provider_def)

    messages = [{"role": "user", "content": "hello"}]

    captured, _ = run_chat(provider, monkeypatch, messages)

    request_headers = cast(dict[str, str], captured["headers"])
    assert "x-api-key" not in request_headers


def test_anthropic_chat_respects_versioned_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_def = ProviderDef(
        name="anthropic",
        type="anthropic",
        base_url="https://api.anthropic.com/v1",
        model="claude-3-sonnet",
        auth_env="ANTHROPIC_API_KEY",
        rpm=60,
        concurrency=1,
    )
    provider = AnthropicProvider(provider_def)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    messages = [{"role": "user", "content": "hello"}]

    captured, _ = run_chat(provider, monkeypatch, messages)

    assert captured["url"] == "https://api.anthropic.com/v1/messages"
