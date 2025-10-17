from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


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


class ProviderChatResponse(BaseModel):
    status_code: int = 200
    model: str
    content: str | list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None
    usage_prompt_tokens: Optional[int] = 0
    usage_completion_tokens: Optional[int] = 0
    choices: list[dict[str, Any]] | None = None


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
        choice_payload["index"] = choice_payload.get("index", index)
        raw_message = choice_payload.get("message")
        if isinstance(raw_message, dict):
            normalized_message = normalize_message(raw_message)
        elif raw_message is None:
            normalized_message = {"role": "assistant"}
        else:
            normalized_message = normalize_message({"content": raw_message})
        choice_payload["message"] = normalized_message
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
    choices_source = p.choices or [
        {
            "index": 0,
            "message": fallback_message,
            "finish_reason": p.finish_reason or "stop",
        }
    ]
    normalized_choices = [
        normalize_choice(
            choice if isinstance(choice, dict) else {"message": choice}, index
        )
        for index, choice in enumerate(choices_source)
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
