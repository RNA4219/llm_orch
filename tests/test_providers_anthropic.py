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
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Be precise."},
                {"type": "text", "text": "Stay calm."},
            ],
        },
        {"role": "system", "content": "Respond in Japanese."},
        {
            "role": "user",
            "content": [
                "Hello",
                {"type": "text", "text": "there"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi!"}],
        },
        {
            "role": "tool",
            "content": "Ignored",
        },
        {"role": "user", "content": "How are you?"},
    ]

    response = await provider.chat(model="claude-3", messages=messages, temperature=0.3, max_tokens=256)

    payload = captured["json"]

    assert payload["model"] == "claude-3"
    assert (
        payload["system"]
        == "Be precise.\n\nStay calm.\n\nRespond in Japanese."
    )
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 256
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "there"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
    ]

    class DummyAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "DummyAsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            request = httpx.Request("POST", url, headers=headers)
            return httpx.Response(
                status_code=200,
                json={
                    "model": "claude-3-sonnet",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                },
                request=request,
            )

    async def run_chat() -> None:
        monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)
        response = await provider.chat(model="claude-3-sonnet", messages=messages)

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

    asyncio.run(run_chat())
