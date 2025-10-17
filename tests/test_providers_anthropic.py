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
from src.orch.types import ProviderChatResponse, chat_response_from_provider


def run_chat(
    provider: AnthropicProvider,
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, Any]],
    request_model: str = "claude-3-sonnet",
    response_payload: dict[str, Any] | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    function_call: dict[str, Any] | None = None,
    **chat_kwargs: Any,
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
                json=
                response_payload
                or {
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                },
                request=request,
            )

    async def invoke() -> ProviderChatResponse:
        monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)
        return await provider.chat(
            model=request_model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            function_call=function_call,
            **chat_kwargs,
        )

    response = asyncio.run(invoke())
    return captured, response


def build_anthropic_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_url: str = "https://api.anthropic.com",
) -> AnthropicProvider:
    provider_def = ProviderDef(
        name="anthropic",
        type="anthropic",
        base_url=base_url,
        model="claude-3-sonnet",
        auth_env="ANTHROPIC_API_KEY",
        rpm=60,
        concurrency=1,
    )
    provider = AnthropicProvider(provider_def)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    return provider


def test_anthropic_chat_normalizes_structured_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(monkeypatch)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "alpha"},
                {"type": "text", "text": "beta"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ],
        },
    ]

    captured, response = run_chat(provider, monkeypatch, messages)

    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["system"] == "alphabeta"

    messages_payload = cast(list[dict[str, Any]], request_json["messages"])
    assert messages_payload == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "helloworld"}],
        }
    ]

    assert response.content == "ok"


def test_anthropic_payload_maps_openai_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = build_anthropic_provider(monkeypatch)

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


def test_anthropic_payload_includes_supported_sampling_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(monkeypatch)

    captured, _ = run_chat(
        provider,
        monkeypatch,
        messages=[{"role": "user", "content": "hello"}],
        top_p=0.3,
        frequency_penalty=1.2,
        presence_penalty=-0.4,
    )

    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["top_p"] == 0.3
    assert "frequency_penalty" not in request_json
    assert "presence_penalty" not in request_json


def test_anthropic_payload_sets_max_output_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = build_anthropic_provider(monkeypatch)

    captured, _ = run_chat(
        provider,
        monkeypatch,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=321,
    )

    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["max_output_tokens"] == 321
    assert "max_tokens" not in request_json


def test_anthropic_base_url_with_messages_adds_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(
        monkeypatch, base_url="https://api.anthropic.com/messages"
    )

    captured, _ = run_chat(
        provider,
        monkeypatch,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert captured["url"] == "https://api.anthropic.com/v1/messages"


def test_anthropic_base_url_custom_messages_adds_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(
        monkeypatch, base_url="https://api.anthropic.com/custom/messages"
    )

    captured, _ = run_chat(
        provider,
        monkeypatch,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert captured["url"] == "https://api.anthropic.com/custom/v1/messages"


def test_anthropic_payload_includes_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = build_anthropic_provider(monkeypatch)
    tools = [
        {
            "name": "lookup",
            "description": "Lookup data",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
    ]
    tool_choice = {"type": "tool", "name": "lookup"}

    captured, _ = run_chat(
        provider,
        monkeypatch,
        messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        tool_choice=tool_choice,
    )

    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["tools"] == tools
    assert request_json["tool_choice"] == tool_choice


def test_anthropic_function_call_none_disables_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(monkeypatch)
    tools = [
        {
            "name": "lookup",
            "description": "Lookup data",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
    ]

    captured, _ = run_chat(
        provider,
        monkeypatch,
        messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        tool_choice={"type": "tool", "name": "lookup"},
        function_call="none",
    )

    request_json = cast(dict[str, Any], captured["json"])
    assert "tools" not in request_json
    assert request_json.get("tool_choice") == "none"


def test_anthropic_payload_applies_function_call_name(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = build_anthropic_provider(monkeypatch)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup data",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    captured, _ = run_chat(
        provider,
        monkeypatch,
        messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        function_call={"name": "lookup"},
    )

    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["tool_choice"] == {"type": "tool", "name": "lookup"}


def test_anthropic_payload_normalizes_tool_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = build_anthropic_provider(monkeypatch)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup data",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        }
    ]

    captured, _ = run_chat(
        provider,
        monkeypatch,
        messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "lookup"}},
    )

    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["tool_choice"] == {"type": "tool", "name": "lookup"}

    captured_auto, _ = run_chat(
        provider,
        monkeypatch,
        messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        tool_choice="auto",
    )

    request_json_auto = cast(dict[str, Any], captured_auto["json"])
    assert request_json_auto["tool_choice"] == "auto"


def test_anthropic_payload_maps_function_call_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(monkeypatch)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "function_call": {
                "name": "lookup",
                "arguments": "{\"q\": \"weather\"}",
            },
        },
    ]

    captured, _ = run_chat(provider, monkeypatch, messages)

    request_json = cast(dict[str, Any], captured["json"])
    anthropic_messages = cast(list[dict[str, Any]], request_json["messages"])

    assert anthropic_messages == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "function_call_1",
                    "name": "lookup",
                    "input": {"q": "weather"},
                }
            ],
        },
    ]


def test_anthropic_payload_maps_tool_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = build_anthropic_provider(monkeypatch)
    tool_content = [{"type": "output_text", "text": "done"}]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "calling tool", "tool_calls": []},
        {"role": "tool", "tool_call_id": "call-1", "content": "completed"},
        {"role": "tool", "tool_call_id": "call-2", "content": tool_content},
    ]

    captured, _ = run_chat(provider, monkeypatch, messages)

    request_json = cast(dict[str, Any], captured["json"])
    messages_payload = cast(list[dict[str, Any]], request_json["messages"])

    assert messages_payload[0]["content"][0]["text"] == "hello"
    assert messages_payload[1]["role"] == "assistant"
    tool_messages = messages_payload[-2:]
    assert [m["role"] for m in tool_messages] == ["user", "user"]
    assert [m["content"][0]["tool_use_id"] for m in tool_messages] == ["call-1", "call-2"]
    assert tool_messages[0]["content"][0]["content"] == [{"type": "text", "text": "completed"}]
    assert tool_messages[1]["content"][0]["content"] == tool_content


def test_anthropic_tool_message_accepts_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(monkeypatch)
    tool_content = [{"type": "output_text", "text": "done"}]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "call", "tool_calls": []},
        {"role": "tool", "tool_call_id": "call-1", "content": tool_content},
    ]

    response_payload = {
        "content": [
            {"type": "tool_result", "tool_use_id": "call-1", "content": tool_content}
        ]
    }

    captured, response = run_chat(
        provider,
        monkeypatch,
        messages,
        response_payload=response_payload,
    )

    request_json = cast(dict[str, Any], captured["json"])
    tool_messages = cast(list[dict[str, Any]], request_json["messages"])

    assert tool_messages[-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call-1",
                "content": tool_content,
            }
        ],
    }
    assert response.content == "done"


@pytest.mark.parametrize(
    ("stop_reason", "expected_finish_reason"),
    [
        ("tool_use", "tool_calls"),
        ("max_tokens", "length"),
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
    ],
)
def test_anthropic_chat_normalizes_stop_reason(
    monkeypatch: pytest.MonkeyPatch,
    stop_reason: str,
    expected_finish_reason: str,
) -> None:
    provider = build_anthropic_provider(monkeypatch)

    response_payload: dict[str, Any] = {
        "content": [],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    if stop_reason == "tool_use":
        response_payload["content"] = [
            {
                "type": "tool_use",
                "id": "call-1",
                "name": "lookup",
                "input": {"q": "weather"},
            }
        ]

    _, response = run_chat(
        provider,
        monkeypatch,
        messages=[{"role": "user", "content": "hello"}],
        response_payload=response_payload,
    )

    assert response.finish_reason == expected_finish_reason
    if stop_reason == "tool_use":
        assert response.tool_calls is not None
        assert response.tool_calls[0]["function"]["name"] == "lookup"


def test_anthropic_tool_message_wraps_string_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(monkeypatch)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "call", "tool_calls": []},
        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
    ]

    captured, _ = run_chat(provider, monkeypatch, messages)

    request_json = cast(dict[str, Any], captured["json"])
    tool_messages = cast(list[dict[str, Any]], request_json["messages"])

    assert tool_messages[-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call-1",
                "content": [{"type": "text", "text": "done"}],
            }
        ],
    }


def test_anthropic_tool_message_accepts_single_output_text_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(monkeypatch)
    tool_block = {"type": "output_text", "text": "done"}
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "call", "tool_calls": []},
        {"role": "tool", "tool_call_id": "call-1", "content": tool_block},
    ]

    captured, _ = run_chat(provider, monkeypatch, messages)

    request_json = cast(dict[str, Any], captured["json"])
    tool_messages = cast(list[dict[str, Any]], request_json["messages"])

    assert tool_messages[-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call-1",
                "content": [tool_block],
            }
        ],
    }


def test_anthropic_payload_errors_on_tool_without_id(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = build_anthropic_provider(monkeypatch)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "no id"},
    ]

    with pytest.raises(ValueError, match="tool_call_id"):
        run_chat(provider, monkeypatch, messages)


def test_anthropic_payload_normalizes_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(monkeypatch)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "alpha"},
                {"type": "text", "text": "beta"},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "hi"}],
        },
    ]

    captured, _ = run_chat(provider, monkeypatch, messages)

    request_json = cast(dict[str, Any], captured["json"])
    assert request_json["system"] == "alphabeta"

    messages_payload = cast(list[dict[str, Any]], request_json["messages"])
    assert messages_payload == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]


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


@pytest.mark.parametrize(
    ("base_url", "expected"),
    (
        ("https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"),
        ("https://api.anthropic.com/v1/messages", "https://api.anthropic.com/v1/messages"),
    ),
)
def test_anthropic_chat_base_url_handles_version_suffix(
    monkeypatch: pytest.MonkeyPatch, base_url: str, expected: str
) -> None:
    provider_def = ProviderDef(
        name="anthropic",
        type="anthropic",
        base_url=base_url,
        model="claude-3-sonnet",
        auth_env="ANTHROPIC_API_KEY",
        rpm=60,
        concurrency=1,
    )
    provider = AnthropicProvider(provider_def)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    messages = [{"role": "user", "content": "hello"}]

    captured, _ = run_chat(provider, monkeypatch, messages)

    assert captured["call_count"] == 1
    assert captured["url"] == expected

def test_anthropic_chat_omits_api_key_when_no_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_anthropic_payload_maps_tool_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = build_anthropic_provider(monkeypatch)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"Tokyo\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "sunny",
        },
        {"role": "assistant", "content": "done"},
    ]

    captured, _ = run_chat(provider, monkeypatch, messages)

    payload = cast(dict[str, Any], captured["json"])
    anthropic_messages = cast(list[dict[str, Any]], payload["messages"])

    assert anthropic_messages == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"location": "Tokyo"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [{"type": "text", "text": "sunny"}],
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]


def test_anthropic_payload_errors_on_tool_without_id(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = build_anthropic_provider(monkeypatch)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "result"},
    ]

    with pytest.raises(ValueError, match="tool_call_id"):
        run_chat(provider, monkeypatch, messages)


def test_anthropic_chat_maps_tool_use_stop_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = build_anthropic_provider(monkeypatch)

    messages = [{"role": "user", "content": "hello"}]
    response_payload = {
        "content": [
            {
                "type": "tool_use",
                "id": "call_42",
                "name": "get_weather",
                "input": {"location": "Tokyo"},
            },
            {"type": "text", "text": "Working on it."},
        ],
        "stop_reason": "tool_use",
        "model": "claude-3-sonnet",
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }

    _, response = run_chat(
        provider,
        monkeypatch,
        messages,
        response_payload=response_payload,
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == [
        {
            "id": "call_42",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"},
        }
    ]
    assert response.content == "Working on it."

    openai_response = chat_response_from_provider(response)
    choice = openai_response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == response.tool_calls
    assert choice["message"]["content"] == "Working on it."


def test_anthropic_tool_only_response_sets_content_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(monkeypatch)

    messages = [{"role": "user", "content": "hello"}]
    response_payload = {
        "content": [
            {
                "type": "tool_use",
                "id": "call_42",
                "name": "get_weather",
                "input": {"location": "Tokyo"},
            }
        ],
        "stop_reason": "tool_use",
        "model": "claude-3-sonnet",
    }

    _, response = run_chat(
        provider,
        monkeypatch,
        messages,
        response_payload=response_payload,
    )

    assert response.content is None
    openai_response = chat_response_from_provider(response)
    choice = openai_response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"].get("content") is None


def test_anthropic_tool_use_only_message_omits_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = build_anthropic_provider(monkeypatch)

    messages = [{"role": "user", "content": "hello"}]
    response_payload = {
        "content": [
            {
                "type": "tool_use",
                "id": "call_7",
                "name": "get_weather",
                "input": {"location": "Tokyo"},
            }
        ],
        "stop_reason": "tool_use",
        "model": "claude-3-sonnet",
    }

    _, response = run_chat(
        provider,
        monkeypatch,
        messages,
        response_payload=response_payload,
    )

    assert response.content is None

    openai_response = chat_response_from_provider(response)
    message = openai_response["choices"][0]["message"]
    assert "content" not in message
    assert message.get("content") is None
