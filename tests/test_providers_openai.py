import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.providers import OpenAICompatProvider
from src.orch.router import ProviderDef


def run_chat(provider: OpenAICompatProvider, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    post_calls: list[dict[str, Any]] = []

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        headers: dict[str, str] = kwargs.get("headers", {})
        payload: dict[str, Any] = kwargs.get("json", {})
        post_calls.append({"url": url, "headers": headers, "json": payload})
        request = httpx.Request("POST", url, headers=headers)
        return httpx.Response(
            status_code=200,
            json={
                "model": payload.get("model", provider.model),
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    async def invoke() -> None:
        await provider.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "ping"}],
        )

    asyncio.run(invoke())
    return post_calls


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
    provider = make_provider("https://api.openai.com")

    post_calls = run_chat(provider, monkeypatch)

    assert post_calls
    assert post_calls[0]["url"] == "https://api.openai.com/v1/chat/completions"


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

    post_calls = run_chat(provider, monkeypatch)

    assert post_calls
    assert post_calls[0]["url"].endswith("/chat/completions")
    assert "/v1/v1/" not in post_calls[0]["url"]


def test_groq_base_url_keeps_openai_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.groq.com/openai/v1")

    post_calls = run_chat(provider, monkeypatch)

    assert post_calls
    assert post_calls[0]["url"] == "https://api.groq.com/openai/v1/chat/completions"
