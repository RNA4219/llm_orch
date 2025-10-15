import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.providers import OpenAICompatProvider  # noqa: E402
from src.orch.router import ProviderDef  # noqa: E402
from src.orch.types import ProviderChatResponse  # noqa: E402


def run_chat(
    provider: OpenAICompatProvider,
    monkeypatch: pytest.MonkeyPatch,
    request_model: str = "gpt-4o",
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
            json={
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


def test_openai_base_url_uses_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = make_provider("https://api.openai.com")

    post_calls, _ = run_chat(provider, monkeypatch)

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
