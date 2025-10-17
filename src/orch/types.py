from typing import Any, Dict, List, Literal, Optional, Union

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


class ProviderChatResponse(BaseModel):
    status_code: int = 200
    model: str
    content: str | list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None
    usage_prompt_tokens: Optional[int] = 0
    usage_completion_tokens: Optional[int] = 0
    additional_message_fields: dict[str, Any] = Field(default_factory=dict)


def chat_response_from_provider(p: ProviderChatResponse) -> dict[str, Any]:
    import time
    import uuid

    message_payload: dict[str, Any] = {"role": "assistant"}
    if p.additional_message_fields:
        message_payload.update(p.additional_message_fields)
    if p.content is not None:
        message_payload["content"] = p.content
    if p.tool_calls is not None:
        message_payload["tool_calls"] = p.tool_calls
    if p.function_call is not None:
        message_payload["function_call"] = p.function_call

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": p.model,
        "choices": [
            {
                "index": 0,
                "message": message_payload,
                "finish_reason": p.finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": p.usage_prompt_tokens or 0,
            "completion_tokens": p.usage_completion_tokens or 0,
            "total_tokens": (p.usage_prompt_tokens or 0)
            + (p.usage_completion_tokens or 0),
        },
    }
