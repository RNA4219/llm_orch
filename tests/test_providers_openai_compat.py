import asyncio
from typing import Any, cast

import httpx
import pytest

from src.orch.providers import OpenAICompatProvider
from src.orch.router import ProviderDef


def _run_chat_and_capture(
    provider_def: ProviderDef,
    env_name: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_url: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    function_call: dict[str, Any] | str | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> tuple[dict[str, Any], Any]:
    provider = OpenAICompatProvider(provider_def)
    monkeypatch.setenv(env_name, "secret")

    captured: dict[str, Any] = {}

    call_count = 0

    class DummyAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "DummyAsyncClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if expected_url is not None:
                assert url == expected_url
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["call_count"] = call_count
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
            tools=tools,
            tool_choice=tool_choice,
            function_call=function_call,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
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


def test_openai_compat_respects_chat_completions_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_def = ProviderDef(
        name="openai",
        type="openai",
        base_url="https://api.openai.com/v1/chat/completions",
        model="gpt-4o",
        auth_env="OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )

    captured, response = _run_chat_and_capture(
        provider_def,
        "OPENAI_API_KEY",
        monkeypatch,
        expected_url="https://api.openai.com/v1/chat/completions",
    )

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["stream"] is False
    assert response.content == "ok"


def test_openai_compat_respects_chat_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_def = ProviderDef(
        name="openai",
        type="openai",
        base_url="https://api.openai.com/v1/chat",
        model="gpt-4o",
        auth_env="OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )

    captured, response = _run_chat_and_capture(
        provider_def,
        "OPENAI_API_KEY",
        monkeypatch,
        expected_url="https://api.openai.com/v1/chat/completions",
    )

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["call_count"] == 1
    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["stream"] is False
    assert response.content == "ok"


def test_openai_compat_async_client_post_uses_chat_suffix_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_def = ProviderDef(
        name="openai",
        type="openai",
        base_url="https://api.openai.com/v1/chat",
        model="gpt-4o",
        auth_env="OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )

    captured, _ = _run_chat_and_capture(
        provider_def,
        "OPENAI_API_KEY",
        monkeypatch,
        expected_url="https://api.openai.com/v1/chat/completions",
    )

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["url"].count("/chat") == 1
    assert captured["url"].count("/v1") == 1


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


def test_openai_compat_async_client_post_with_azure_chat_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_def = ProviderDef(
        name="azure-openai",
        type="openai",
        base_url="https://example.openai.azure.com/openai/deployments/foo/chat",
        model="gpt-4o",
        auth_env="AZURE_OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )

    captured, _ = _run_chat_and_capture(
        provider_def,
        "AZURE_OPENAI_API_KEY",
        monkeypatch,
        expected_url="https://example.openai.azure.com/openai/deployments/foo/chat/completions",
    )

    assert (
        captured["url"]
        == "https://example.openai.azure.com/openai/deployments/foo/chat/completions"
    )
    assert captured["url"].count("/chat") == 1


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


def test_openai_compat_handles_mixed_case_openai_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_def = ProviderDef(
        name="azure-openai",
        type="openai",
        base_url="https://example.openai.azure.com/OpenAI/deployments/foo",
        model="gpt-4o",
        auth_env="AZURE_OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )

    captured, response = _run_chat_and_capture(
        provider_def,
        "AZURE_OPENAI_API_KEY",
        monkeypatch,
        expected_url="https://example.openai.azure.com/OpenAI/deployments/foo/chat/completions",
    )

    assert (
        captured["url"]
        == "https://example.openai.azure.com/OpenAI/deployments/foo/chat/completions"
    )
    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["stream"] is False
    assert response.content == "ok"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.openai.azure.com/openai/deployments/foo",
        "https://example.openai.azure.com:443/openai/deployments/foo",
        "https://example.openai.azure.us/openai/deployments/foo",
        "https://example.openai.azure.cn/openai/deployments/foo",
    ],
)
def test_openai_compat_azure_variants_use_api_key_header(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_def = ProviderDef(
        name="azure-openai",
        type="openai",
        base_url=base_url,
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


def test_openai_compat_includes_tools_in_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_def = ProviderDef(
        name="openai",
        type="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        auth_env="OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup data",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "lookup"}}

    captured, _ = _run_chat_and_capture(
        provider_def,
        "OPENAI_API_KEY",
        monkeypatch,
        tools=tools,
        tool_choice=tool_choice,
    )

    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["tools"] == tools
    assert request_json["tool_choice"] == tool_choice


def test_openai_compat_includes_function_call_in_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_def = ProviderDef(
        name="openai",
        type="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        auth_env="OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )

    function_call = {"name": "lookup", "arguments": "{}"}

    captured, _ = _run_chat_and_capture(
        provider_def,
        "OPENAI_API_KEY",
        monkeypatch,
        function_call=function_call,
    )

    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["function_call"] == function_call


def test_openai_compat_includes_sampling_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_def = ProviderDef(
        name="openai",
        type="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        auth_env="OPENAI_API_KEY",
        rpm=60,
        concurrency=1,
    )

    captured, _ = _run_chat_and_capture(
        provider_def,
        "OPENAI_API_KEY",
        monkeypatch,
        top_p=0.1,
        frequency_penalty=1.5,
        presence_penalty=-0.25,
    )

    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["top_p"] == 0.1
    assert request_json["frequency_penalty"] == 1.5
    assert request_json["presence_penalty"] == -0.25
