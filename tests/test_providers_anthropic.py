import json
import sys
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.orch.providers import AnthropicProvider
from src.orch.router import ProviderDef


def test_anthropic_chat_formats_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "claude-3",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    transport = httpx.MockTransport(handler)

    original_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    provider = AnthropicProvider(
        ProviderDef(
            name="anthropic",
            type="anthropic",
            base_url="https://example.com",
            model="claude-3",
            auth_env=None,
            rpm=60,
            concurrency=1,
        )
    )

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

    async def invoke() -> Any:
        return await provider.chat(
            model="claude-3",
            messages=messages,
            temperature=0.3,
            max_tokens=256,
        )

    response = anyio.run(invoke)

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

    assert response.content == "ok"
    assert response.usage_prompt_tokens == 10
    assert response.usage_completion_tokens == 5
