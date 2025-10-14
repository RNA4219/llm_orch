import asyncio
from typing import Any

import httpx
import pytest

from src.orch.providers import OpenAICompatProvider
from src.orch.router import ProviderDef


class DummyAsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.post_calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> "DummyAsyncClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
        self.post_calls.append({"url": url, "headers": headers, "json": json})
        request = httpx.Request("POST", url, headers=headers)
        return httpx.Response(
            status_code=200,
            json={
                "model": json["model"],
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
            request=request,
        )


def run_chat(provider: OpenAICompatProvider, monkeypatch: pytest.MonkeyPatch) -> DummyAsyncClient:
    dummy_client = DummyAsyncClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: dummy_client)

    async def invoke() -> None:
        await provider.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "ping"}],
        )

    asyncio.run(invoke())
    return dummy_client


def make_provider(base_url: str) -> OpenAICompatProvider:
    provider_def = ProviderDef(
        name="openai",
        type="openai",
        base_url=base_url,
        model="gpt-4o",
        auth_env="OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )
    return OpenAICompatProvider(provider_def)


def test_openai_base_url_uses_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com/v1")

    dummy_client = run_chat(provider, monkeypatch)

    assert dummy_client.post_calls
    assert dummy_client.post_calls[0]["url"] == "https://api.openai.com/v1/chat/completions"


def test_groq_base_url_keeps_openai_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.groq.com/openai/v1")

    dummy_client = run_chat(provider, monkeypatch)

    assert dummy_client.post_calls
    assert dummy_client.post_calls[0]["url"] == "https://api.groq.com/openai/v1/chat/completions"
