import asyncio
from typing import Any

import httpx
import pytest

from src.orch.providers import OllamaProvider
from src.orch.router import ProviderDef


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
