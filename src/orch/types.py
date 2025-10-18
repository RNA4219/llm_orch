from collections import defaultdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]], None]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    response_format: Optional[Dict[str, Any]] = None


class ProviderChatChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int | None = None
    message: dict[str, Any] | str | None = None
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None
    finish_reason: str | None = None

    def __getitem__(self, key: str) -> Any:
        data = self.model_dump(mode="python", exclude_none=True)
        if key in data:
            return data[key]
        raise KeyError(key)


class ProviderStreamChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    delta: dict[str, Any] | str | None = None
    role: Literal["system", "user", "assistant", "tool"] | None = None
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None
    finish_reason: str | None = None
    message: dict[str, Any] | None = None


class ProviderStreamChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    choices: list[ProviderStreamChoice] = Field(default_factory=list)
    usage: dict[str, int] | None = None


class ProviderChatResponse(BaseModel):
    status_code: int = 200
    model: str
    content: str | list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None
    usage_prompt_tokens: Optional[int] = 0
    usage_completion_tokens: Optional[int] = 0
    choices: list[ProviderChatChoice] | None = None


class ProviderStreamChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: str | None = None
    type: str | None = None
    status_code: int | None = None
    model: str | None = None
    choices: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None
    error: dict[str, Any] | None = None


def chat_response_from_provider(p: ProviderChatResponse) -> dict[str, Any]:
    import time
    import uuid

    def normalize_message(raw_message: dict[str, Any] | None) -> dict[str, Any]:
        if raw_message is None:
            return {"role": "assistant"}
        normalized = {
            key: value for key, value in raw_message.items() if value is not None
        }
        if "role" not in normalized:
            normalized["role"] = "assistant"
        return normalized

    def normalize_choice(raw_choice: dict[str, Any], index: int) -> dict[str, Any]:
        choice_payload = dict(raw_choice)
        choice_payload.pop("index", None)
        raw_message = choice_payload.pop("message", None)
        flattened_fields: dict[str, Any] = {}
        for field in ("content", "tool_calls", "function_call"):
            if field in choice_payload:
                value = choice_payload.pop(field)
                if value is not None and field not in flattened_fields:
                    flattened_fields[field] = value

        message_payload: Any
        if isinstance(raw_message, dict):
            message_payload = dict(raw_message)
            for field, value in flattened_fields.items():
                message_payload.setdefault(field, value)
        elif raw_message is None:
            message_payload = flattened_fields or None
        else:
            message_payload = dict(flattened_fields)
            message_payload.setdefault("content", raw_message)

        if isinstance(message_payload, dict):
            normalized_message = normalize_message(message_payload)
        elif message_payload is None:
            normalized_message = normalize_message(None)
        else:
            normalized_message = normalize_message({"content": message_payload})

        choice_payload["message"] = normalized_message
        choice_payload["index"] = index
        if choice_payload.get("finish_reason") is None:
            choice_payload["finish_reason"] = "stop"
        return choice_payload

    fallback_message = {
        key: value
        for key, value in {
            "role": "assistant",
            "content": p.content,
            "tool_calls": p.tool_calls,
            "function_call": p.function_call,
        }.items()
        if value is not None
    }
    serialized_choices: list[dict[str, Any]] = (
        [choice.model_dump(mode="json", exclude_none=True) for choice in p.choices]
        if p.choices
        else []
    )
    if not serialized_choices:
        serialized_choices = [
            {
                "message": fallback_message,
                "finish_reason": p.finish_reason or "stop",
            }
        ]

    normalized_choices = [
        normalize_choice(raw_choice, new_index)
        for new_index, raw_choice in enumerate(serialized_choices)
    ]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": p.model,
        "choices": normalized_choices,
        "usage": {
            "prompt_tokens": p.usage_prompt_tokens or 0,
            "completion_tokens": p.usage_completion_tokens or 0,
            "total_tokens": (p.usage_prompt_tokens or 0)
            + (p.usage_completion_tokens or 0),
        },
    }
