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


def _run_chat_and_capture(
    provider_def: ProviderDef,
    env_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Any]:
    provider = OpenAICompatProvider(provider_def)
    monkeypatch.setenv(env_name, "secret")

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
                    "model": provider_def.model,
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                },
                request=request,
            )

    async def run_chat() -> Any:
        monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)
        return await provider.chat(
            model=provider_def.model,
            messages=[{"role": "user", "content": "ping"}],
        )

    response = asyncio.run(run_chat())
    return captured, response


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

    captured, response = _run_chat_and_capture(provider_def, "OPENAI_API_KEY", monkeypatch)

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["stream"] is False
    assert response.content == "ok"


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

    captured, response = _run_chat_and_capture(provider_def, "PERPLEXITY_API_KEY", monkeypatch)

    assert captured["url"] == "https://api.perplexity.ai/chat/completions"
    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["stream"] is False
    assert response.content == "ok"


def test_openai_compat_preserves_query_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_def = ProviderDef(
        name="azure-openai",
        type="openai",
        base_url="https://example.openai.azure.com/openai/deployments/foo?api-version=2024-02-01",
        model="gpt-4o",
        auth_env="AZURE_OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )

    captured, response = _run_chat_and_capture(provider_def, "AZURE_OPENAI_API_KEY", monkeypatch)

    assert (
        captured["url"]
        == "https://example.openai.azure.com/openai/deployments/foo/chat/completions?api-version=2024-02-01"
    )
    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["stream"] is False
    assert response.content == "ok"


def test_openai_compat_azure_sets_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_def = ProviderDef(
        name="azure-openai",
        type="openai",
        base_url="https://example.openai.azure.com/openai/deployments/foo",
        model="gpt-4o",
        auth_env="AZURE_OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )

    captured, _ = _run_chat_and_capture(provider_def, "AZURE_OPENAI_API_KEY", monkeypatch)

    headers = cast(dict[str, str], captured["headers"])
    assert headers["api-key"] == "secret"
    assert "Authorization" not in headers


def test_openai_compat_azure_with_port_uses_api_key_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_def = ProviderDef(
        name="azure-openai",
        type="openai",
        base_url="https://example.openai.azure.com:443/openai/deployments/foo",
        model="gpt-4o",
        auth_env="AZURE_OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )

    captured, _ = _run_chat_and_capture(provider_def, "AZURE_OPENAI_API_KEY", monkeypatch)

    headers = cast(dict[str, str], captured["headers"])
    assert headers["api-key"] == "secret"
    assert "Authorization" not in headers


def test_openai_compat_async_client_post_receives_query_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_def = ProviderDef(
        name="azure-openai",
        type="openai",
        base_url="https://example.openai.azure.com/openai/deployments/foo?api-version=2024-02-01",
        model="gpt-4o",
        auth_env="AZURE_OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )

    captured, _ = _run_chat_and_capture(provider_def, "AZURE_OPENAI_API_KEY", monkeypatch)

    assert (
        captured["url"]
        == "https://example.openai.azure.com/openai/deployments/foo/chat/completions?api-version=2024-02-01"
    )
