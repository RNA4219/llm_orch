import asyncio
import sys
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orch.providers import OpenAICompatProvider
from src.orch.router import ProviderDef


def test_openai_compat_appends_single_v1_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_def = ProviderDef(
        name="openai",
        type="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        auth_env="OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )
    provider = OpenAICompatProvider(provider_def)

    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    captured: dict[str, Any] = {}

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
                    "model": "gpt-4o",
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                },
                request=request,
            )

    async def run_chat() -> None:
        monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)
        response = await provider.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "ping"}],
        )

        assert captured["url"] == "https://api.openai.com/v1/chat/completions"
        request_json = cast(dict[str, Any], captured["json"])
        assert request_json["stream"] is False
        assert response.content == "ok"

    asyncio.run(run_chat())


def test_openai_compat_preserves_perplexity_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_def = ProviderDef(
        name="perplexity",
        type="openai",
        base_url="https://api.perplexity.ai",
        model="sonar",
        auth_env="PERPLEXITY_API_KEY",
        rpm=60,
        concurrency=1,
    )
    provider = OpenAICompatProvider(provider_def)

    monkeypatch.setenv("PERPLEXITY_API_KEY", "secret")

    captured: dict[str, Any] = {}

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
                    "model": "sonar",
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                },
                request=request,
            )

    async def run_chat() -> None:
        monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)
        response = await provider.chat(
            model="sonar",
            messages=[{"role": "user", "content": "ping"}],
        )

        assert captured["url"] == "https://api.perplexity.ai/chat/completions"
        request_json = cast(dict[str, Any], captured["json"])
        assert request_json["stream"] is False
        assert response.content == "ok"

    asyncio.run(run_chat())
