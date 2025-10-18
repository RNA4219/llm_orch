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

    event_type: str | None = None
    index: int | None = None
    delta: dict[str, Any] | str | None = None
    finish_reason: str | None = None
    choices: list[ProviderStreamChoice] = Field(default_factory=list)
    usage: dict[str, int] | None = None
    error: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


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


def provider_chat_response_from_stream(
    model: str,
    chunks: Iterable[ProviderStreamChunk],
    *,
    default_finish_reason: str = "stop",
) -> ProviderChatResponse:
    states: dict[int, dict[str, Any]] = defaultdict(lambda: {"message": {"role": "assistant"}, "segments": [], "finish": None})
    prompt_tokens = completion_tokens = 0

    def ingest(state: dict[str, Any], value: Any) -> None:
        if value is None:
            return
        if isinstance(value, list):
            state["segments"].append([item for item in value if item is not None])
        else:
            state["segments"].append(value)

    def apply_mapping(state: dict[str, Any], payload: dict[str, Any]) -> None:
        if (role := payload.get("role")) and isinstance(role, str):
            state["message"]["role"] = role
        if "content" in payload:
            ingest(state, payload.get("content"))
        if (tool_calls := payload.get("tool_calls")) and isinstance(tool_calls, list):
            state["message"]["tool_calls"] = tool_calls
        if (function_call := payload.get("function_call")) and isinstance(function_call, dict):
            state["message"]["function_call"] = function_call
        if (finish := payload.get("finish_reason")) and isinstance(finish, str):
            state["finish"] = finish

    for chunk in chunks:
        usage = chunk.usage or {}
        if (value := usage.get("prompt_tokens")) and isinstance(value, int):
            prompt_tokens = max(prompt_tokens, value)
        if (value := usage.get("completion_tokens")) and isinstance(value, int):
            completion_tokens = max(completion_tokens, value)
        for choice in chunk.choices:
            state = states[choice.index]
            if choice.role is not None:
                state["message"]["role"] = choice.role
            if isinstance(choice.message, dict):
                apply_mapping(state, choice.message)
            if isinstance(choice.delta, dict):
                apply_mapping(state, choice.delta)
            elif isinstance(choice.delta, str):
                ingest(state, choice.delta)
            ingest(state, choice.content)
            if choice.tool_calls is not None:
                state["message"]["tool_calls"] = choice.tool_calls
            if choice.function_call is not None:
                state["message"]["function_call"] = choice.function_call
            if choice.finish_reason is not None:
                state["finish"] = choice.finish_reason

    _ = states[0]
    provider_choices: list[ProviderChatChoice] = []
    for index in sorted(states):
        state = states[index]
        message = dict(state["message"])
        segments = state["segments"]
        if segments:
            if any(isinstance(segment, (dict, list)) for segment in segments):
                flattened: list[Any] = []
                for segment in segments:
                    flattened.extend(segment if isinstance(segment, list) else [segment])
                message["content"] = flattened
            else:
                message["content"] = "".join(str(segment) for segment in segments)
        finish = state["finish"] or default_finish_reason
        provider_choices.append(ProviderChatChoice(index=index, message=message, finish_reason=finish))

    first_choice = provider_choices[0] if provider_choices else ProviderChatChoice(index=0)
    first_message = first_choice.message if isinstance(first_choice.message, dict) else {}

    return ProviderChatResponse(
        model=model,
        content=first_message.get("content"),
        finish_reason=first_choice.finish_reason,
        tool_calls=first_message.get("tool_calls"),
        function_call=first_message.get("function_call"),
        usage_prompt_tokens=prompt_tokens or 0,
        usage_completion_tokens=completion_tokens or 0,
        choices=provider_choices,
    )


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
